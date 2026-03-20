# losses/supcon_loss.py
#
# Supervised Contrastive Loss (SupCon)
# =====================================
# Based on: "Supervised Contrastive Learning" (Khosla et al., NeurIPS 2020)
# https://arxiv.org/abs/2004.11362
#
# Core idea:
#   Given a batch of (view1, view2, label) triplets, the model produces
#   L2-normalized projection embeddings for each view. The loss pulls
#   embeddings from the same class closer together, and pushes
#   embeddings from different classes apart — using ALL in-batch samples
#   as negatives, not just a fixed pair.
#
# How it differs from vanilla triplet loss:
#   - Uses all same-class samples as positives (not just one anchor-positive pair)
#   - Uses temperature-scaled cosine similarity
#   - Works naturally with multiple views (e.g. two augmentations per scan)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    Args:
        temperature (float): Scales the cosine similarity logits.
                             Lower = sharper distribution (more confident).
                             Recommended starting range: 0.07 - 0.2.
                             For medical imaging (subtle class differences),
                             0.1-0.2 often works better than the 0.07 default
                             used in natural image papers.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features (Tensor): Shape (2*B, proj_dim).
                               The first B rows are view1 embeddings,
                               the last B rows are view2 embeddings.
                               Must already be L2-normalized (unit vectors).
            labels (Tensor):   Shape (B,). Class index for each sample.
                               Will be repeated internally to match (2*B,).

        Returns:
            Scalar loss value.
        """
        device = features.device
        batch_size = labels.shape[0]           # B (number of MRI scans)
        n = features.shape[0]                  # 2*B (both views concatenated)

        # ── Step 1: Build the positive mask ──────────────────────────────────
        # labels_rep: (2*B,) — repeat each label for both views
        labels_rep = labels.repeat(2)

        # positive_mask[i, j] = True  if samples i and j share the same class
        #                      = False otherwise
        # Shape: (2*B, 2*B)
        positive_mask = (labels_rep.unsqueeze(0) == labels_rep.unsqueeze(1))

        # Remove the diagonal: a sample is not its own positive
        diag_mask = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~diag_mask

        # ── Step 2: Compute pairwise cosine similarities ──────────────────────
        # features are already L2-normalized, so dot product = cosine similarity
        # Shape: (2*B, 2*B)
        sim = torch.matmul(features, features.T) / self.temperature

        # ── Step 3: Numerically stable softmax denominator ────────────────────
        # Subtract row-max for stability (standard log-sum-exp trick)
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Exponentiate, zeroing out the diagonal (self-similarity)
        exp_sim = torch.exp(sim) * ~diag_mask     # (2*B, 2*B)

        # log of the denominator for each anchor row
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

        # ── Step 4: Compute the per-anchor loss ───────────────────────────────
        # For each anchor i, average the log-probability over all its positives
        log_prob = sim - log_denom               # (2*B, 2*B)

        # Only sum over positive pairs; divide by number of positives per anchor
        num_positives = positive_mask.sum(dim=1).clamp(min=1)   # (2*B,)
        loss_per_anchor = -(log_prob * positive_mask).sum(dim=1) / num_positives

        return loss_per_anchor.mean()
