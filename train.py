"""
train.py — Anti-overfitting edition

Problem: train_loss=0.018 vs val_loss=0.640 → 35× gap → severe memorisation.
The 0.709 ceiling is an overfitting ceiling, not a capacity ceiling.

Changes vs previous version:
  1. Mixup augmentation (α=0.3) applied in the training loop
     Blends pairs of (rgb, mask, label) samples — the model sees infinite
     interpolations rather than the same ~4800 images repeatedly.
     Works with soft-label LabelSmoothingCE (both Mixup targets are passed in).

  2. LabelSmoothingCE replaces FocalLoss (see losses.py)
     Prevents overconfident logits on memorised samples.

  3. Stronger regularisation
     dropout: 0.4 → 0.55 (applied at MODEL level via CFG)
     weight_decay: 0.01 → 0.05

  4. Best-checkpoint reload between phases (kept from previous fix)
  5. use_sampler=False (kept from previous fix)
  6. Corrected LRs (kept from previous fix)

Everything else (architecture, joint transforms, phase structure) unchanged.
"""

import os, math, time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from dataset import build_dataloaders, JunctionDataset
from model   import FusionJunctionClassifier
from losses  import build_loss


# ════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════
ROOT = r"D:\DATASETS\new_dataset_waymo\new_dataset"

CFG = {
    # ── Paths ──────────────────────────────────────────────────────
    "train_excel"    : os.path.join(ROOT, "train", "train_encoded.xlsx"),
    "train_rgb_dir"  : os.path.join(ROOT, "train", "train_frames"),
    "train_mask_dir" : os.path.join(ROOT, "train", "train_masks", "color"),
    "val_excel"      : os.path.join(ROOT, "val",   "val_encoded.xlsx"),
    "val_rgb_dir"    : os.path.join(ROOT, "val",   "val_frames"),
    "val_mask_dir"   : os.path.join(ROOT, "val",   "val_masks", "color"),
    "save_dir"       : os.path.join(ROOT, "checkpoints_fusion"),

    # ── Data ───────────────────────────────────────────────────────
    "img_size"    : 384,
    "num_workers" : 4,
    "num_classes" : 3,
    "class_names" : ["No junction", "T-junction", "X-junction"],
    "class_counts": None,

    # ── Model ──────────────────────────────────────────────────────
    "model_name"   : "nvidia/mit-b2",
    "mask_feat_dim": 256,
    "dropout"      : 0.55,    # was 0.40 — stronger to combat overfitting

    # ── Loss ───────────────────────────────────────────────────────
    "label_smoothing": 0.12,  # ε for LabelSmoothingCE

    # ── Mixup ──────────────────────────────────────────────────────
    "mixup_alpha": 0.3,       # Beta(α,α) distribution for Mixup
    "mixup_prob" : 0.5,       # Apply Mixup to 50% of batches

    # ── Training ───────────────────────────────────────────────────
    "batch_size"       : 4,
    "grad_accum_steps" : 8,

    "phase_epochs" : {"phase1": 10, "phase2": 20, "phase3": 20},
    "phase_lr"     : {
        "phase1" : 3e-4,
        "phase2" : 5e-5,
        "phase3" : 1e-5,
    },

    "weight_decay"  : 0.05,   # was 0.01 — much stronger L2
    "warmup_epochs" : 3,
    "patience"      : 8,

    "seed"   : 42,
    "device" : "cuda" if torch.cuda.is_available() else "cpu",
}


# ════════════════════════════════════════════════════════════════════
#  MIXUP
# ════════════════════════════════════════════════════════════════════
def mixup_batch(rgb, mask, labels, num_classes, alpha=0.3):
    """
    Applies Mixup to a batch of (rgb, mask, labels).

    Mixup: given two samples (x_i, y_i) and (x_j, y_j),
      x_mix = λ·x_i + (1-λ)·x_j
      y_mix = λ·y_i + (1-λ)·y_j   (soft label)
    where λ ~ Beta(alpha, alpha).

    Both the RGB tensor and mask tensor are blended with the SAME λ,
    preserving the semantic alignment between the two streams.

    Returns (rgb_mix, mask_mix, soft_labels) where soft_labels is (B, C).
    """
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)           # ensure dominant sample has λ > 0.5

    B   = rgb.size(0)
    idx = torch.randperm(B, device=rgb.device)

    rgb_mix  = lam * rgb  + (1 - lam) * rgb[idx]
    mask_mix = lam * mask + (1 - lam) * mask[idx]

    # Convert integer labels to one-hot soft labels
    y_a = F.one_hot(labels,          num_classes).float()
    y_b = F.one_hot(labels[idx],     num_classes).float()
    soft_labels = lam * y_a + (1 - lam) * y_b    # (B, C)

    return rgb_mix, mask_mix, soft_labels


# Need F for one_hot in mixup
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════
#  UTILITIES
# ════════════════════════════════════════════════════════════════════
def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def print_gpu_info():
    if not torch.cuda.is_available():
        print("[GPU] No CUDA — CPU only."); return
    p  = torch.cuda.get_device_properties(0)
    gb = p.total_memory / 1024**3
    print(f"[GPU] {p.name}  |  VRAM: {gb:.1f} GB")
    if gb < 7:
        print("[GPU] Under 7 GB — img_size=384, batch=4.")


def cosine_lr_with_warmup(optimizer, epoch, warmup_epochs, total_epochs, base_lr):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / max(1, warmup_epochs)
    else:
        t  = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# ════════════════════════════════════════════════════════════════════
#  CHECKPOINT HELPERS
# ════════════════════════════════════════════════════════════════════
def save_best(model, epoch, phase_name, macro_f1, cfg):
    os.makedirs(cfg["save_dir"], exist_ok=True)
    path = os.path.join(cfg["save_dir"], "best_fusion_model.pt")
    torch.save({
        "epoch": epoch, "phase": phase_name,
        "model_state_dict": model.state_dict(),
        "macro_f1": macro_f1, "cfg": cfg,
    }, path)
    return path


def reload_best(model, cfg):
    path = os.path.join(cfg["save_dir"], "best_fusion_model.pt")
    if not os.path.exists(path):
        print("[Checkpoint] No best checkpoint yet."); return 0.0
    ckpt = torch.load(path, map_location=cfg["device"])
    model.load_state_dict(ckpt["model_state_dict"])
    f1   = ckpt.get("macro_f1", 0.0)
    print(f"[Checkpoint] Reloaded: phase={ckpt.get('phase','?')} "
          f"epoch={ckpt.get('epoch','?')} macro_f1={f1:.4f}")
    return f1


# ════════════════════════════════════════════════════════════════════
#  EVALUATION  (no Mixup — clean val pass)
# ════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, class_names):
    model.eval()
    ce         = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for rgb, mask, labels in loader:
        rgb    = rgb.to(device,    non_blocking=True)
        mask   = mask.to(device,   non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(rgb, mask)
            loss   = ce(logits, labels)
        total_loss += loss.item()
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    report   = classification_report(all_labels, all_preds,
                                     target_names=class_names, digits=3)
    cm       = confusion_matrix(all_labels, all_preds)
    return macro_f1, total_loss / len(loader), report, cm


# ════════════════════════════════════════════════════════════════════
#  TRAINING PHASE  (with Mixup)
# ════════════════════════════════════════════════════════════════════
def run_phase(
    model, train_loader, val_loader, criterion,
    phase_name, num_epochs, base_lr, cfg, best_f1
):
    device      = cfg["device"]
    accum       = cfg["grad_accum_steps"]
    mixup_alpha = cfg["mixup_alpha"]
    mixup_prob  = cfg["mixup_prob"]
    num_classes = cfg["num_classes"]

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr, weight_decay=cfg["weight_decay"],
    )
    scaler       = GradScaler()
    patience_ctr = 0

    print(f"\n{'═'*65}")
    print(f"  {phase_name}")
    print(f"  epochs={num_epochs}  lr={base_lr}  "
          f"eff_batch={cfg['batch_size'] * accum}  "
          f"mixup_α={mixup_alpha}  wd={cfg['weight_decay']}")
    print(f"  Trainable: {model.count_trainable():,} params")
    print(f"{'═'*65}")

    for epoch in range(num_epochs):
        model.train()
        lr = cosine_lr_with_warmup(
            optimizer, epoch, cfg["warmup_epochs"], num_epochs, base_lr
        )
        running_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (rgb, mask, labels) in enumerate(train_loader):
            rgb    = rgb.to(device,    non_blocking=True)
            mask   = mask.to(device,   non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # ── Mixup (applied with probability mixup_prob) ───────
            use_mixup = (np.random.rand() < mixup_prob)
            if use_mixup:
                rgb, mask, soft_labels = mixup_batch(
                    rgb, mask, labels, num_classes, alpha=mixup_alpha
                )
                target = soft_labels   # (B, C) soft
            else:
                target = labels        # (B,)   hard

            with autocast():
                logits = model(rgb, mask)
                loss   = criterion(logits, target) / accum

            scaler.scale(loss).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accum

        scaler.step(optimizer); scaler.update(); optimizer.zero_grad()

        avg_train_loss = running_loss / len(train_loader)
        macro_f1, val_loss, report, cm = evaluate(
            model, val_loader, device, cfg["class_names"]
        )
        elapsed = time.time() - t0

        # ── Watch the train/val gap ────────────────────────────────
        gap_flag = " ← GAP OK" if val_loss / (avg_train_loss + 1e-9) < 10 else " ← OVERFIT"
        print(f"  E{epoch+1:02d}/{num_epochs} | "
              f"train={avg_train_loss:.4f}  val={val_loss:.4f}{gap_flag}  "
              f"macro_f1={macro_f1:.4f}  lr={lr:.2e}  ({elapsed:.0f}s)")

        if macro_f1 > best_f1:
            best_f1      = macro_f1
            patience_ctr = 0
            ckpt_path    = save_best(model, epoch, phase_name, macro_f1, cfg)
            print(f"\n  ✓ New best  macro_f1={macro_f1:.4f}  → {ckpt_path}")
            print(f"\n{report}")
            print(f"  Confusion matrix:\n{cm}\n")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"\n  Early stop: {patience_ctr} epochs without improvement.\n")
                break

    return best_f1


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
def main():
    set_seed(CFG["seed"])
    print_gpu_info()

    # ── Auto-read real class counts ────────────────────────────────
    tmp = JunctionDataset(
        CFG["train_excel"], CFG["train_rgb_dir"], CFG["train_mask_dir"],
        split="train",
    )
    CFG["class_counts"] = tmp.class_counts()
    print(f"[Train] Class counts (No/T/X): {CFG['class_counts']}")
    del tmp

    # ── Dataloaders ───────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        train_excel    = CFG["train_excel"],
        train_rgb_dir  = CFG["train_rgb_dir"],
        train_mask_dir = CFG["train_mask_dir"],
        val_excel      = CFG["val_excel"],
        val_rgb_dir    = CFG["val_rgb_dir"],
        val_mask_dir   = CFG["val_mask_dir"],
        batch_size     = CFG["batch_size"],
        img_size       = CFG["img_size"],
        num_workers    = CFG["num_workers"],
        use_sampler    = False,
    )

    # ── Model ─────────────────────────────────────────────────────
    # dropout=0.55 is passed here — model.py uses this in the MLP head
    model = FusionJunctionClassifier(
        num_classes     = CFG["num_classes"],
        pretrained_name = CFG["model_name"],
        mask_feat_dim   = CFG["mask_feat_dim"],
        dropout         = CFG["dropout"],         # 0.55
    ).to(CFG["device"])
    model.model_summary()

    # ── Loss ──────────────────────────────────────────────────────
    criterion = build_loss(
        class_counts = CFG["class_counts"],
        smoothing    = CFG["label_smoothing"],
        device       = CFG["device"],
    )

    best_f1 = 0.0

    # ── Phase 1: Head only ────────────────────────────────────────
    model.freeze_all_encoders()
    best_f1 = run_phase(
        model, train_loader, val_loader, criterion,
        phase_name = "Phase 1 — Head only",
        num_epochs = CFG["phase_epochs"]["phase1"],
        base_lr    = CFG["phase_lr"]["phase1"],
        cfg=CFG, best_f1=best_f1,
    )

    # ── Phase 2 ───────────────────────────────────────────────────
    print("\n[Phase transition] Reloading best before Phase 2...")
    best_f1 = reload_best(model, CFG)
    model.unfreeze_phase2(n_rgb_stages=2)
    best_f1 = run_phase(
        model, train_loader, val_loader, criterion,
        phase_name = "Phase 2 — Mask CNN + SegFormer top 2 stages",
        num_epochs = CFG["phase_epochs"]["phase2"],
        base_lr    = CFG["phase_lr"]["phase2"],
        cfg=CFG, best_f1=best_f1,
    )

    # ── Phase 3 ───────────────────────────────────────────────────
    print("\n[Phase transition] Reloading best before Phase 3...")
    best_f1 = reload_best(model, CFG)
    model.unfreeze_all()
    best_f1 = run_phase(
        model, train_loader, val_loader, criterion,
        phase_name = "Phase 3 — Full fine-tune",
        num_epochs = CFG["phase_epochs"]["phase3"],
        base_lr    = CFG["phase_lr"]["phase3"],
        cfg=CFG, best_f1=best_f1,
    )

    print(f"\n{'═'*65}")
    print(f"  Training complete.  Best macro-F1: {best_f1:.4f}")
    print(f"  Checkpoint → {CFG['save_dir']}\\best_fusion_model.pt")
    print(f"{'═'*65}")


if __name__ == "__main__":
    main()
