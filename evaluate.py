"""
evaluate.py — With Test-Time Augmentation (TTA)

TTA: run each test image through N augmented views, average the softmax
probabilities, then take the argmax. Free performance gain at inference —
no retraining needed.

For junction classification, safe TTA augmentations are:
  - Original (no aug)
  - Horizontal flip
  - Slight brightness shift (+10%)
  - Slight contrast shift (+10%)

Vertical flip and rotation are NOT used as TTA because junction
topology is orientation-sensitive.

Usage:
    python evaluate.py --split val
    python evaluate.py --split test
    python evaluate.py --split test --tta        ← enables TTA
    python evaluate.py --split test --tta --tta_n 6
"""

import argparse, os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from dataset import JunctionDataset, CLASS_NAMES
from model   import FusionJunctionClassifier

ROOT = r"D:\DATASETS\new_dataset_waymo\new_dataset"

SPLIT_PATHS = {
    "val" : {
        "excel"    : os.path.join(ROOT, "val",  "val_encoded.xlsx"),
        "rgb_dir"  : os.path.join(ROOT, "val",  "val_frames"),
        "mask_dir" : os.path.join(ROOT, "val",  "val_masks", "color"),
    },
    "test" : {
        "excel"    : os.path.join(ROOT, "test", "test_encoded.xlsx"),
        "rgb_dir"  : os.path.join(ROOT, "test", "test_frames"),
        "mask_dir" : os.path.join(ROOT, "test", "test_masks", "color"),
    },
}


def load_model(ckpt_path, model_name="nvidia/mit-b2", device="cuda"):
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = FusionJunctionClassifier(num_classes=3, pretrained_name=model_name)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    print(f"[Eval] Loaded: {ckpt_path}")
    print(f"       Phase={ckpt.get('phase','?')} | "
          f"epoch={ckpt.get('epoch','?')} | "
          f"macro_f1={ckpt.get('macro_f1', 0):.4f}")
    return model


# ── TTA augmentation views ────────────────────────────────────────────
def tta_views(rgb: torch.Tensor, mask: torch.Tensor):
    """
    Returns list of (rgb_aug, mask_aug) pairs for TTA.
    Geometric augs applied identically to rgb and mask.
    Photometric augs applied to rgb only (masks are semantic colour maps).
    """
    views = [
        (rgb, mask),                                           # original
        (TF.hflip(rgb), TF.hflip(mask)),                      # H-flip
        (TF.adjust_brightness(rgb, 1.1), mask),               # brighter
        (TF.adjust_contrast(rgb, 1.1),   mask),               # more contrast
        (TF.adjust_brightness(rgb, 0.9), mask),               # dimmer
        (TF.adjust_saturation(rgb, 1.15), mask),              # more saturated
    ]
    return views


@torch.no_grad()
def predict(model, loader, device, use_tta=False, tta_n=4):
    """
    Standard or TTA inference.
    tta_n controls how many TTA views to use (max 6, default 4).
    """
    all_labels, all_preds, all_probs = [], [], []

    for rgb, mask, labels in loader:
        rgb  = rgb.to(device,  non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        if use_tta:
            views     = tta_views(rgb, mask)[:tta_n]
            prob_sum  = None
            for rgb_v, mask_v in views:
                with autocast():
                    logits = model(rgb_v, mask_v)
                probs = torch.softmax(logits, dim=1)
                prob_sum = probs if prob_sum is None else prob_sum + probs
            avg_probs = (prob_sum / len(views)).cpu().numpy()
        else:
            with autocast():
                logits = model(rgb, mask)
            avg_probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_preds.extend(avg_probs.argmax(axis=1))
        all_labels.extend(labels.numpy())
        all_probs.extend(avg_probs)

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrices(labels, preds, save_path="confusion_matrix.png"):
    cm      = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    for ax, data, fmt, title in zip(
        axes, [cm, cm_norm], ["d", ".2%"],
        ["Absolute counts", "Row-normalised (recall per class)"]
    ):
        im = ax.imshow(data, cmap="Blues", vmin=0)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = data[i, j]
                ax.text(j, i, format(v, fmt), ha="center", va="center",
                        color="white" if (v / data.max()) > 0.55 else "black",
                        fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Eval] Saved → {save_path}")


def plot_confidence_histograms(labels, probs, save_path="confidence_hist.png"):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Prediction confidence per class", fontsize=13)
    for c, ax in enumerate(axes):
        mask    = labels == c
        conf    = probs[mask, c]
        correct = probs[mask].argmax(1) == c
        ax.hist(conf[correct],  bins=20, alpha=0.7, color="steelblue", label="Correct")
        ax.hist(conf[~correct], bins=20, alpha=0.7, color="tomato",    label="Wrong")
        ax.set_title(CLASS_NAMES[c]); ax.set_xlabel("Confidence"); ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[Eval] Saved → {save_path}")


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths  = SPLIT_PATHS[args.split]
    model  = load_model(args.checkpoint, model_name=args.model_name, device=device)

    ds = JunctionDataset(
        excel_path = paths["excel"],
        rgb_dir    = paths["rgb_dir"],
        mask_dir   = paths["mask_dir"],
        split      = "val",
        img_size   = args.img_size,
    )
    # Batch size 1 is needed for TTA (so each sample's views stay isolated)
    bs = 1 if args.tta else args.batch_size
    loader = DataLoader(ds, batch_size=bs, shuffle=False,
                        num_workers=4, pin_memory=True)

    mode = f"TTA (n={args.tta_n})" if args.tta else "standard"
    print(f"[Eval] {args.split} set: {len(ds)} samples | mode: {mode}\n")

    labels, preds, probs = predict(
        model, loader, device,
        use_tta=args.tta, tta_n=args.tta_n,
    )

    print("── Classification Report ─────────────────────────────────")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))
    mf1 = f1_score(labels, preds, average="macro")
    wf1 = f1_score(labels, preds, average="weighted")
    print(f"  Macro-F1    : {mf1:.4f}  ← primary metric")
    print(f"  Weighted-F1 : {wf1:.4f}")

    plot_confusion_matrices(labels, preds)
    plot_confidence_histograms(labels, probs)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=os.path.join(
        ROOT, "checkpoints_fusion", "best_fusion_model.pt"))
    p.add_argument("--split",      default="val", choices=["val", "test"])
    p.add_argument("--batch_size", type=int,  default=8)
    p.add_argument("--img_size",   type=int,  default=384)
    p.add_argument("--model_name", default="nvidia/mit-b2")
    p.add_argument("--tta",        action="store_true",
                   help="Enable test-time augmentation")
    p.add_argument("--tta_n",      type=int, default=4,
                   help="Number of TTA views (max 6)")
    run_evaluation(p.parse_args())
