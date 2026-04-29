"""
losses.py — Label-Smoothing Cross-Entropy with per-class alpha weighting

Why replace Focal Loss with Label Smoothing?

Focal Loss is designed for DETECTION tasks where there are thousands of easy
background anchors swamping a handful of object anchors. Our problem is
different: we have ~4800 training samples and a model with 25M parameters
that is severely overfitting (train_loss=0.018 vs val_loss=0.64).

In that regime, Focal Loss makes the problem WORSE:
  • It concentrates gradient on "hard" examples — which in an overfit regime
    are almost always the samples the model hasn't memorised yet.
  • It drives the model toward very confident (low-entropy) predictions on
    training data, accelerating memorisation.

Label Smoothing fixes this at the root:
  • Instead of target = [0, 1, 0], it uses [ε/K, 1-ε(K-1)/K, ε/K].
  • The model can never reach zero loss even on training examples it has seen
    100 times → gradient stays informative → memorisation slows.
  • Per-class alpha weights handle class imbalance the same way focal alpha did.

Recommended ε = 0.10–0.15 for 3-class problems with noisy labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCE(nn.Module):
    """
    Label-smoothing cross-entropy with optional per-class alpha weighting.

    Args:
        alpha     : (num_classes,) per-class weights. Inverse-frequency normalised.
        smoothing : Label smoothing factor ε. 0.0 = standard CE. Default 0.12.
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        alpha     : torch.Tensor,
        smoothing : float = 0.12,
        reduction : str   = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha.float())
        self.smoothing = smoothing
        self.reduction = reduction
        self.num_classes = alpha.shape[0]

    def forward(
        self,
        logits  : torch.Tensor,   # (B, C)  raw logits
        targets : torch.Tensor,   # (B,)    integer labels  OR  (B, C) soft labels (Mixup)
    ) -> torch.Tensor:

        log_probs = F.log_softmax(logits, dim=1)  # (B, C)

        # ── Build smoothed target distribution ────────────────────
        if targets.dim() == 1:
            # Hard labels → one-hot then smooth
            smooth_val   = self.smoothing / self.num_classes
            one_hot      = torch.zeros_like(log_probs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_target = one_hot * (1.0 - self.smoothing) + smooth_val
            # Per-sample alpha weight from the TRUE class
            alpha_t = self.alpha[targets]          # (B,)
        else:
            # Soft labels from Mixup → already in [0,1], sum to 1
            # Smooth: push all targets slightly toward uniform
            smooth_val   = self.smoothing / self.num_classes
            smooth_target = targets * (1.0 - self.smoothing) + smooth_val
            # Alpha weight from the DOMINANT class in the Mixup blend
            dominant = targets.argmax(dim=1)
            alpha_t  = self.alpha[dominant]        # (B,)

        # ── Per-sample loss ───────────────────────────────────────
        # KL divergence between smooth target and model prediction
        per_sample = -(smooth_target * log_probs).sum(dim=1)  # (B,)
        weighted   = alpha_t * per_sample                      # (B,)

        if self.reduction == "mean":
            return weighted.mean()
        elif self.reduction == "sum":
            return weighted.sum()
        return weighted

    def extra_repr(self):
        return f"smoothing={self.smoothing}, num_classes={self.num_classes}"


def build_loss(
    class_counts : list,
    smoothing    : float = 0.12,
    device       : str   = "cuda",
) -> LabelSmoothingCE:
    """
    Build LabelSmoothingCE with alpha = inverse-normalised class frequency.

    alpha_c = (1 / count_c) / sum_i(1 / count_i)

    Args:
        class_counts : [count_class0, count_class1, count_class2]
        smoothing    : Label smoothing ε (default 0.12)
        device       : Target device

    Returns:
        LabelSmoothingCE module already on `device`.
    """
    counts = torch.tensor(class_counts, dtype=torch.float32)
    if (counts == 0).any():
        raise ValueError("class_counts must not contain zeros.")
    inv   = 1.0 / counts
    alpha = inv / inv.sum()

    print(f"[Loss] LabelSmoothingCE  ε={smoothing}")
    print(f"[Loss] Class counts : {class_counts}")
    print(f"[Loss] Alpha weights : {[f'{a:.3f}' for a in alpha.tolist()]}")

    return LabelSmoothingCE(alpha=alpha.to(device), smoothing=smoothing)


# ── Keep backward-compatible name for anything that imports build_focal_loss ──
def build_focal_loss(class_counts, gamma=2.0, device="cuda"):
    """
    Compatibility shim. Ignores gamma, uses LabelSmoothingCE instead.
    Prevents ImportError if old code still calls build_focal_loss().
    """
    print("[Loss] Note: build_focal_loss() now calls build_loss() (LabelSmoothingCE).")
    return build_loss(class_counts, smoothing=0.12, device=device)
