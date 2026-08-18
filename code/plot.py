"""
plot.py — visualize training curves and evaluation results for the CE vs SupCon experiment.

Usage:
    # Training curves only
    python plot.py --ce_metrics results/ce/metrics.json --con_metrics results/contrastive/metrics.json

    # Full comparison (requires --save_preds npz files from test.py)
    python plot.py \
        --ce_metrics  results/ce/metrics.json \
        --con_metrics results/contrastive/metrics.json \
        --ce_preds    results/ce/preds.npz \
        --con_preds   results/contrastive/preds.npz \
        --out_dir     figures/

All arguments are optional — pass only what you have and the relevant plots will be generated.
"""

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import softmax
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_score, recall_score, roc_auc_score,
)

CLASS_NAMES = ["AD", "MCI", "NC"]


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_metrics(path):
    with open(path) as f:
        return json.load(f)


def load_preds(path):
    data = np.load(path)
    preds = data['predictions']
    labels = data['labels']
    logits = data['logits'] if 'logits' in data else None
    return preds, labels, logits


# ── Plot helpers ──────────────────────────────────────────────────────────────

def plot_training_curves(ce_metrics, con_metrics, out_dir):
    """Loss and accuracy curves for CE and/or SupCon runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for metrics, label, ls in [(ce_metrics, 'CE', '-'), (con_metrics, 'SupCon', '--')]:
        if metrics is None:
            continue
        epochs     = [m['epoch']      for m in metrics]
        train_loss = [m['train_loss'] for m in metrics]
        val_loss   = [m['val_loss']   for m in metrics]
        train_acc  = [m['train_acc']  for m in metrics]
        val_acc    = [m['val_acc']    for m in metrics]

        axes[0].plot(epochs, train_loss, ls,       label=f'{label} train')
        axes[0].plot(epochs, val_loss,   ls, alpha=0.5, label=f'{label} val')
        axes[1].plot(epochs, train_acc,  ls,       label=f'{label} train')
        axes[1].plot(epochs, val_acc,    ls, alpha=0.5, label=f'{label} val')

    for ax, title, ylabel in zip(axes, ['Loss', 'Accuracy'], ['Loss', 'Accuracy']):
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(out_dir, 'training_curves.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_confusion_matrix(preds, labels, title, out_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_roc_curves(logits, labels, title, out_path):
    if logits is None:
        print(f"Skipping ROC for '{title}': no logits in .npz  "
              f"(re-run test.py --save_preds with updated test.py)")
        return
    probs = softmax(logits, axis=1)
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, name in enumerate(CLASS_NAMES):
        binary = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name}  AUC={roc_auc:.3f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def _per_class_metrics(preds, labels, logits):
    probs = softmax(logits, axis=1) if logits is not None else None
    result = {}
    for i, cls in enumerate(CLASS_NAMES):
        mask = (labels == i)
        acc_i  = float((preds[mask] == labels[mask]).mean()) if mask.sum() > 0 else float('nan')
        prec_i = float(precision_score(labels, preds, labels=[i], average='micro', zero_division=0))
        rec_i  = float(recall_score(labels, preds, labels=[i], average='micro', zero_division=0))
        if probs is not None:
            try:
                auc_i = float(roc_auc_score((labels == i).astype(int), probs[:, i]))
            except ValueError:
                auc_i = float('nan')
        else:
            auc_i = float('nan')
        result[cls] = {'acc': acc_i, 'prec': prec_i, 'rec': rec_i, 'auc': auc_i}
    return result


def plot_comparison_bar(ce_preds, ce_labels, ce_logits,
                         con_preds, con_labels, con_logits, out_dir):
    """Side-by-side bar chart comparing CE vs SupCon per class across 4 metrics."""
    ce_stats  = _per_class_metrics(ce_preds,  ce_labels,  ce_logits)
    con_stats = _per_class_metrics(con_preds, con_labels, con_logits)

    metric_keys   = ['acc',      'prec',       'rec',     'auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall',  'AUC']
    x     = np.arange(len(CLASS_NAMES))
    width = 0.35

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, mkey, mlabel in zip(axes, metric_keys, metric_labels):
        ce_vals  = [ce_stats[cls][mkey]  for cls in CLASS_NAMES]
        con_vals = [con_stats[cls][mkey] for cls in CLASS_NAMES]
        bars_ce  = ax.bar(x - width / 2, ce_vals,  width, label='CE')
        bars_con = ax.bar(x + width / 2, con_vals, width, label='SupCon')
        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_ylim(0, 1.05)
        ax.set_title(mlabel)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        # value labels on top of bars
        for bar in list(bars_ce) + list(bars_con):
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

    fig.suptitle('CE vs SupCon — Per-Class Performance', fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot training curves and evaluation results")
    parser.add_argument('--model', type=str, default='resnet10', choices=['resnet10', 'resnet18'],
                        help='Which model to plot results for. Determines default result and figure paths.')
    parser.add_argument('--ce_metrics',  type=str, default=None,
                        help='Override path to CE metrics.json (default: results/ce[_resnet18]/metrics.json)')
    parser.add_argument('--con_metrics', type=str, default=None,
                        help='Override path to SupCon metrics.json')
    parser.add_argument('--ce_preds',   type=str, default=None,
                        help='Override path to CE predictions .npz')
    parser.add_argument('--con_preds',  type=str, default=None,
                        help='Override path to SupCon predictions .npz')
    parser.add_argument('--out_dir',    type=str, default=None,
                        help='Override output directory (default: figures/resnet10 or figures/resnet18)')
    args = parser.parse_args()

    suffix = '' if args.model == 'resnet10' else '_resnet18'
    if args.ce_metrics  is None: args.ce_metrics  = f'results/ce{suffix}/metrics.json'
    if args.con_metrics is None: args.con_metrics = f'results/contrastive{suffix}/metrics.json'
    if args.ce_preds    is None: args.ce_preds    = f'results/ce{suffix}/preds.npz'
    if args.con_preds   is None: args.con_preds   = f'results/contrastive{suffix}/preds.npz'
    if args.out_dir     is None: args.out_dir     = f'figures/{args.model}'

    os.makedirs(args.out_dir, exist_ok=True)

    ce_metrics  = load_metrics(args.ce_metrics)  if os.path.exists(args.ce_metrics)  else None
    con_metrics = load_metrics(args.con_metrics) if os.path.exists(args.con_metrics) else None

    ce_preds = ce_labels = ce_logits = None
    con_preds = con_labels = con_logits = None
    if os.path.exists(args.ce_preds):
        ce_preds, ce_labels, ce_logits = load_preds(args.ce_preds)
    if os.path.exists(args.con_preds):
        con_preds, con_labels, con_logits = load_preds(args.con_preds)

    # Training curves (either or both runs)
    if ce_metrics or con_metrics:
        plot_training_curves(ce_metrics, con_metrics, args.out_dir)

    # Per-model confusion matrix + ROC
    if ce_preds is not None:
        plot_confusion_matrix(ce_preds, ce_labels, 'Confusion Matrix — CE',
                              os.path.join(args.out_dir, 'cm_ce.png'))
        plot_roc_curves(ce_logits, ce_labels, 'ROC Curves — CE',
                        os.path.join(args.out_dir, 'roc_ce.png'))
    if con_preds is not None:
        plot_confusion_matrix(con_preds, con_labels, 'Confusion Matrix — SupCon',
                              os.path.join(args.out_dir, 'cm_supcon.png'))
        plot_roc_curves(con_logits, con_labels, 'ROC Curves — SupCon',
                        os.path.join(args.out_dir, 'roc_supcon.png'))

    # CE vs SupCon bar chart (only when both are available)
    if ce_preds is not None and con_preds is not None:
        plot_comparison_bar(ce_preds, ce_labels, ce_logits,
                            con_preds, con_labels, con_logits,
                            args.out_dir)

    print("Done.")


if __name__ == '__main__':
    main()
