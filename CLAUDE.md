# Alzheimer's MRI Classification

3D brain MRI classification into Alzheimer's Disease, Mild Cognitive Impairment, and Normal Cognition using 3D ResNet / Swin Transformer backbones with optional supervised contrastive learning. Runs on an HPC cluster (SLURM).

## Project Structure

- `datasets/MyDataSet.py` — All dataset classes (standard, contrastive, multi-task)
- `models/resnet.py` — 3D ResNet with dual-head (classification + contrastive projector)
- `models/resnet_org.py` — Original baseline ResNet without contrastive support
- `losses/supcon_loss.py` — Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
- `medicalnet_model.py` — Model factory, pretrained weight loading, param group splitting
- `train.py` — Main training loop
- `test.py` — Evaluation: per-class accuracy, AUC, precision, recall

## Data Format

Images are `.npz` files with key `"image_mr"` (3D float array, shape `(D, H, W)`).
Labels come from a JSON manifest:

```json
[{"id": "001", "label": "Normal Cognition"}, ...]
```

Label mapping: `"Alzheimer's Disease"→0`, `"Mild Cognitive Impairment"→1`, `"Normal Cognition"→2`

## Training Modes

**Standard classification:**

```bash
python train.py --backbone resnet --model_name resnet10 --lr 1e-4 --batch_size 8 --epochs 100
```

**With supervised contrastive loss:**

```bash
python train.py --contrastive --lambda_con 0.5 --temperature 0.1 ...
```

When `--contrastive` is enabled, `ContrastiveDataset` returns two augmented views per scan and `SupConLoss` is added alongside cross-entropy.

## Model Architecture

`ResNet.forward(x, return_embedding=False)`:

- `False` → 3-class logits (classification)
- `True` → L2-normalized 128-dim embedding via projector MLP (for SupCon)

## Augmentations (MRI-safe)

Applied only during contrastive training in `ContrastiveDataset._augment()`:

- Random axis flips (50% per axis)
- Additive Gaussian noise (σ=0.05)
- Intensity scaling (0.9–1.1×)

## Known Issues

- Output paths in `train.py` are hardcoded to `/scratch/user/baileyyeah/...` — update before running
- `medicalnet_model.py` line 3 imports `swintransformer` and `resnet_20_head` which have no corresponding files in `models/` — this causes an `ImportError` on import even if those code paths aren't used. Files were likely on the original developer's cluster but never committed. Avoid using `generate_model_swin()` or `model_name='resnet50_atrophy'` until resolved.
