# Junction Type Classifier for Autonomous Vehicle Safety

**Course:** AI69002 — Design Laboratory, IIT Kharagpur (Spring 2025–26)  
**Guide:** Prof. Somdyuti Paul, Department of AI, IIT Kharagpur  

---

## Overview

This repository implements a **two-stream late-fusion deep learning classifier** that identifies the type of road junction visible in a front-facing camera frame captured by an autonomous vehicle. The classifier distinguishes three categories:

| Label | Class ID | Description |
|-------|----------|-------------|
| No Junction | 0 | Open road segment with no intersection visible |
| T-Junction | 1 | Three-way intersection |
| X-Junction | 2 | Four-way intersection |

Junction type classification is a prerequisite for computing **surrogate safety measures (SSMs)** - such as Time-to-Collision and Post-Encroachment Time in autonomous driving pipelines, which motivated this project.

---

## Architecture

The model processes two co-registered input modalities per frame:

```
RGB Frame (3, H, W)               Segmentation Mask (3, H, W)
        │                                      │
┌───────▼──────────────┐           ┌───────────▼──────────┐
│  SegFormer-B2 Encoder│           │   4-Layer Mask CNN   │
│  (Cityscapes pretrain│           │   (trained from init)│
│  Multi-scale pooling │           │   AdaptiveAvgPool    │
└───────┬──────────────┘           └───────────┬──────────┘
        │  1024-dim                             │  256-dim
        └──────────────┬────────────────────────┘
                  cat → 1280-dim
                       │
           ┌───────────▼───────────┐
           │    MLP Fusion Head    │
           │  LN → 512 → 128 → 3  │
           └───────────────────────┘
```

**RGB branch:** MiT-B2 encoder from SegFormer, pretrained on Cityscapes. All four hierarchical feature maps are globally average-pooled and concatenated, giving the classifier access to both fine-grained (lane markings, road edges) and high-level (road topology shape) representations simultaneously.

**Mask branch:** Lightweight four-layer CNN with BatchNorm and ReLU, trained from random initialisation. A custom CNN is preferred over a pretrained ResNet for this modality because colour-coded segmentation masks are flat semantic maps with no photographic texture, ImageNet priors are irrelevant.

**Fusion:** Simple concatenation followed by a LayerNorm-gated MLP head with Dropout(0.55).

---

## Repository Structure

```
junction_fusion/
├── dataset.py        # JunctionDataset + JointTransform + DataLoader factory
├── model.py          # FusionJunctionClassifier (SegFormer-B2 + MaskCNN)
├── losses.py         # LabelSmoothingCE with per-class alpha weighting
├── train.py          # 3-phase graduated training loop with Mixup
├── evaluate.py       # Evaluation with optional Test-Time Augmentation (TTA)
└── requirements.txt  # Python dependencies 
└── Full implementatin in notebook format.ipynb # For full implementation in jupyter notebook format for Google Colab, Kaggle, etc 
```

---

## Dataset Layout

The code expects the following directory structure, matching the Waymo-derived dataset used in this project:

```
new_dataset/
├── train/
│   ├── train_frames/          # RGB .png images
│   ├── train_masks/
│   │   └── color/             # Colour-coded segmentation masks (.png)
│   └── train_encoded.xlsx     # col 0 = filename, col 1 = label string
├── val/
│   ├── val_frames/
│   ├── val_masks/color/
│   └── val_encoded.xlsx
└── test/
    ├── test_frames/
    ├── test_masks/color/
    └── test_encoded.xlsx
```

**Label strings in the Excel file:** `"No junction"`, `"T-junction"`, `"X-junction"`  
**Image format:** `.png` (the filename in the Excel may have any extension; the loader forces `.png`)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Prajojeet/Waymo-junction-Classifier---Design-Lab.git

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| torch | ≥ 2.0.0 | Core deep learning framework |
| torchvision | ≥ 0.15.0 | Image transforms |
| transformers | ≥ 4.35.0 | SegFormer (MiT-B2) encoder |
| timm | ≥ 0.9.0 | Required by transformers for SegFormer |
| scikit-learn | ≥ 1.3.0 | Metrics (macro-F1, classification report) |
| pandas / openpyxl | ≥ 2.0.0 | Excel label file reading |

---

## Training

### 1. Configure paths

Open `train.py` and set the `ROOT` variable to your dataset root:

```python
ROOT = r"D:\DATASETS\new_dataset_waymo\new_dataset"  # Windows
# ROOT = "/mnt/datasets/new_dataset"                 # Linux / Colab
```

### 2. Run training

```bash
python train.py
```

Training proceeds automatically through three phases:

| Phase | Trainable Components | Learning Rate | Epochs |
|-------|---------------------|---------------|--------|
| 1 | MLP head only (both encoders frozen) | 3e-4 | 10 |
| 2 | Mask CNN + SegFormer top 2 stages | 5e-5 | 20 |
| 3 | All parameters | 1e-5 | 20 |

The globally best checkpoint (by macro-F1 on the validation set) is saved to `checkpoints_fusion/best_fusion_model.pt` and **reloaded at the start of each phase** to prevent performance regression from phase-boundary learning rate spikes.

---

## Evaluation

```bash
# Evaluate on validation set (standard)
python evaluate.py --split val

# Evaluate on test set
python evaluate.py --split test
```

The evaluation script produces:
- Per-class precision / recall / F1 report
- Macro and weighted F1 scores
- Confusion matrix (absolute counts + row-normalised recall view)
- Per-class confidence histograms

---

## Results

Best model performance on the validation set (1,450 samples):

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| No Junction | 0.859 | 0.749 | 0.801 | 742 |
| T-Junction | 0.543 | 0.649 | 0.591 | 262 |
| X-Junction | 0.782 | 0.859 | 0.818 | 446 |
| **Macro Avg** | **0.728** | **0.752** | **0.737** | 1450 |
| Weighted Avg | 0.778 | 0.765 | 0.768 | 1450 |

**Accuracy:** 76.5%

The T-Junction class shows the lowest F1, consistent with its geometric variability (three distinct approach directions) and its disproportionate representation in annotation boundary errors.

---

## Annotation Notes

Manual inspection of frames where model uncertainty was high revealed systematic annotation inconsistencies, particularly at junction sequence boundaries. This is identified as the primary performance bottleneck. Human re-annotation with video context access (rather than frame-by-frame static annotation) is planned as the foundation for the continuation of this project.

---

## Citation

If you use this code or the methodology in your research, please cite:

```
Prajojeet Pradhan 22CE3AI15. "Junction Type Classification in Traffic Scenes for 
Surrogate Safety Measure Estimation in Autonomous Vehicles." 
Design Laboratory Report, AI69002, IIT Kharagpur, Spring 2025-26.
Supervisor: Prof. Somdyuti Paul.
```
