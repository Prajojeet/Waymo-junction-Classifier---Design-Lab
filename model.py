"""
model.py — FusionJunctionClassifier

Architecture:
                    RGB (3, H, W)              Mask (3, H, W)
                         │                          │
              ┌──────────▼───────────┐   ┌──────────▼───────────┐
              │   SegFormer-B2       │   │   Mask CNN (4 layers) │
              │   (MiT encoder)      │   │   BN + ReLU + Pool    │
              │   Cityscapes pretrain│   │   trained from scratch│
              └──────────┬───────────┘   └──────────┬───────────┘
                         │                          │
              Multi-scale pool → 1024d    Global avg pool → 256d
                         │                          │
                         └──────────┬───────────────┘
                               cat → 1280d
                                    │
                         ┌──────────▼──────────┐
                         │  MLP Fusion Head     │
                         │  LN → Linear(512)    │
                         │  GELU → Drop(0.4)    │
                         │  Linear(128)         │
                         │  GELU → Drop(0.2)    │
                         │  Linear(3) → logits  │
                         └─────────────────────┘

Why 4-layer CNN for masks (not ResNet-18)?
  • Colour masks are flat-coloured semantic maps — no texture, no fine detail.
  • ResNet-18 has ImageNet priors tuned for photographic texture, wasting capacity.
  • A 4-layer BN-CNN trains from scratch and reaches saturation quickly on flat maps.
  • Much lighter: ~1.2 M params vs ResNet-18's 11 M → saves ~350 MB VRAM.

VRAM estimate on RTX 4050 (6 GB) with AMP + grad checkpoint:
  SegFormer-B2 : ~3.2 GB  (with checkpointing)
  Mask CNN     : ~0.3 GB
  Activations  : ~1.5 GB  (batch 4 × 384×384)
  Total        : ~5.0 GB  → safe headroom
"""

import torch
import torch.nn as nn
from transformers import SegformerModel


# ── SegFormer hidden dims per variant ────────────────────────────────
_MIT_HIDDEN_DIMS = {
    "nvidia/mit-b0": [32,  64,  160, 256],
    "nvidia/mit-b1": [64,  128, 320, 512],
    "nvidia/mit-b2": [64,  128, 320, 512],
    "nvidia/mit-b3": [64,  128, 320, 512],
}


# ══════════════════════════════════════════════════════════════════════
#  MASK ENCODER — lightweight 4-layer CNN
# ══════════════════════════════════════════════════════════════════════
class MaskCNN(nn.Module):
    """
    4-layer CNN for colour segmentation masks.

    Input  : (B, 3, H, W)  — normalised colour mask
    Output : (B, 256)      — global-average-pooled feature vector

    Spatial progression at img_size=384:
        384 → 192 → 96 → 48 → 24 → GlobalAvgPool → (B, 256)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # /2 → 192

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # /2 → 96

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                               # /2 → 48

            # Block 4 — out_dim channels
            nn.Conv2d(128, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),                       # → (B, 256, 1, 1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x).flatten(1)          # (B, 256)


# ══════════════════════════════════════════════════════════════════════
#  FUSION MODEL
# ══════════════════════════════════════════════════════════════════════
class FusionJunctionClassifier(nn.Module):
    """
    Dual-stream late-fusion classifier.

    Args:
        num_classes      : Output classes (default 3).
        pretrained_name  : HuggingFace MiT encoder name.
        mask_feat_dim    : Output dim of MaskCNN (default 256).
        dropout          : Dropout in MLP head (default 0.4).
    """

    def __init__(
        self,
        num_classes     : int = 3,
        pretrained_name : str = "nvidia/mit-b2",
        mask_feat_dim   : int = 256,
        dropout         : float = 0.4,
    ):
        super().__init__()
        self.pretrained_name = pretrained_name

        # ── RGB branch: SegFormer encoder ────────────────────────
        self.rgb_encoder = SegformerModel.from_pretrained(pretrained_name)
        self.rgb_encoder.gradient_checkpointing_enable()
        rgb_dims    = _MIT_HIDDEN_DIMS[pretrained_name]
        rgb_out_dim = sum(rgb_dims)                # 1024 for B2

        # ── Mask branch: lightweight CNN ──────────────────────────
        self.mask_encoder = MaskCNN(out_dim=mask_feat_dim)   # 256-d

        # ── Fusion MLP head ───────────────────────────────────────
        fused_dim = rgb_out_dim + mask_feat_dim        # 1024 + 256 = 1280

        self.head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
        self._init_head()

    def _init_head(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────
    def forward(
        self,
        rgb  : torch.Tensor,   # (B, 3, H, W) — normalised RGB frame
        mask : torch.Tensor,   # (B, 3, H, W) — normalised colour mask
    ) -> torch.Tensor:

        # RGB stream → multi-scale pooling → 1024-d
        rgb_out = self.rgb_encoder(
            pixel_values=rgb, output_hidden_states=True
        )
        rgb_feat = torch.cat(
            [hs.mean(dim=[-2, -1]) for hs in rgb_out.hidden_states],
            dim=1,
        )                                           # (B, 1024)

        # Mask stream → CNN → 256-d
        mask_feat = self.mask_encoder(mask)         # (B, 256)

        # Late fusion: simple concatenation
        fused = torch.cat([rgb_feat, mask_feat], dim=1)   # (B, 1280)

        return self.head(fused)                     # (B, 3)

    # ══════════════════════════════════════════════════════════════
    #  PHASE-WISE FREEZE / UNFREEZE
    # ══════════════════════════════════════════════════════════════
    def freeze_all_encoders(self):
        """
        Phase 1 — freeze BOTH encoders, train head only.
        ~0.9 M trainable params. Very fast — validates that the head
        can learn from frozen multi-modal features.
        """
        for p in self.rgb_encoder.parameters():
            p.requires_grad = False
        for p in self.mask_encoder.parameters():
            p.requires_grad = False
        print(f"[Model] Both encoders frozen. "
              f"Trainable: {self.count_trainable():,} params")

    def unfreeze_phase2(self, n_rgb_stages: int = 2):
        """
        Phase 2 — unfreeze:
          • Mask CNN fully (it was trained from scratch — needs gradient flow)
          • Top n_rgb_stages of SegFormer (encode high-level road topology)

        Keeps early SegFormer stages (edge detectors) frozen to preserve
        Cityscapes pretraining priors.
        """
        # Mask CNN: fully unfreeze
        for p in self.mask_encoder.parameters():
            p.requires_grad = True

        # SegFormer: unfreeze top n stages
        enc         = self.rgb_encoder.encoder
        total_stages = len(enc.block)
        for i in range(total_stages - n_rgb_stages, total_stages):
            for p in enc.block[i].parameters():
                p.requires_grad = True
            for p in enc.layer_norm[i].parameters():
                p.requires_grad = True

        print(f"[Model] Phase 2: mask CNN unfrozen + top {n_rgb_stages} "
              f"SegFormer stages unfrozen. "
              f"Trainable: {self.count_trainable():,} params")

    def unfreeze_all(self):
        """
        Phase 3 — full fine-tune at very low LR (1e-5).
        Allows early SegFormer stages to adapt to driving-scene specifics.
        """
        for p in self.parameters():
            p.requires_grad = True
        print(f"[Model] All params unfrozen. "
              f"Trainable: {self.count_trainable():,} params")

    # ── Utilities ─────────────────────────────────────────────────
    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_summary(self):
        total = self.count_total()
        train = self.count_trainable()
        rgb_p = sum(p.numel() for p in self.rgb_encoder.parameters())
        msk_p = sum(p.numel() for p in self.mask_encoder.parameters())
        hd_p  = sum(p.numel() for p in self.head.parameters())
        print(f"\n[Model] FusionJunctionClassifier ({self.pretrained_name})")
        print(f"        RGB encoder (SegFormer-B2) : {rgb_p:>12,} params")
        print(f"        Mask encoder (CNN-4)        : {msk_p:>12,} params")
        print(f"        Fusion MLP head             : {hd_p:>12,} params")
        print(f"        ─────────────────────────────────────────")
        print(f"        Total                       : {total:>12,} params")
        print(f"        Trainable (current phase)   : {train:>12,} params\n")
