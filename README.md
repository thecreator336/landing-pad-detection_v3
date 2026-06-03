# Landing Pad Detection v3 — ArUco Marker Detector

Part of an ongoing UAV autonomous landing project at ePlane. This version (Week 3) upgrades the Week 2 helipad detector to use an ArUco marker (ID 1, DICT_4X4), adds negative frames to fix a flawed confusion matrix, and introduces shadow-casting buildings for scene realism.

---

## Project Overview

The goal is to train a YOLOv11s object detection model to identify a rooftop ArUco landing pad from drone footage. The full pipeline runs on synthetic data generated in NVIDIA Isaac Sim, with sim-to-real validation against real drone flight footage as the next step.

---

## Pipeline
Isaac Sim (USD scene)
↓
Synthetic image generation (9,000 frames)
↓
convert_to_yolo.py  →  YOLO dataset (train/val/test)
↓
train.py  →  YOLOv11s model
↓
evaluate.py  →  Confusion matrix + metrics

---

## Scene Setup (Isaac Sim)

- Rooftop environment with buildings, guard rails, conveyor belt, wooden crate
- ArUco marker ID 1 (DICT_4X4) applied as albedo texture on a disk primitive
- Randomised camera position, altitude, sun direction, and light intensity per frame
- Shadow-casting buildings added for lighting realism
- **Positive frames** (batches 0–5): pad visible
- **Negative frames** (batches 6–8): pad moved underground, ~30% of dataset

---

## Dataset

| Split | Images |
|-------|--------|
| Train | 7,200  |
| Val   | 900    |
| Test  | 900    |
| **Total** | **9,000** |

- Positive : Negative ratio ≈ 67 : 33
- Resolution: 640 × 480
- Labels: YOLO normalised format (class cx cy w h)

---

## Model

- Architecture: YOLOv11s
- Epochs: 100
- Batch size: 8
- Image size: 640
- Device: NVIDIA RTX 3050 6GB
- Framework: Ultralytics, PyTorch 2.5.1 + CUDA 12.1

---

## Results (Synthetic Test Set, IoU 0.5)

| Metric | Value |
|--------|-------|
| Precision | 1.000 |
| Recall | 0.979 |
| F1 | 0.989 |
| Accuracy | 98.6% |
| mAP50 | 0.948 |
| mAP50-95 | 0.263 |

> Note: The 13 false negatives in the confusion matrix were identified as pitch-black rendering artifacts from Isaac Sim (~8% black frame rate per batch). On valid frames, precision and recall are both 1.0. mAP50-95 is lower due to loose box localisation at high camera altitudes.

---

## Confusion Matrix (IoU 0.5)

| | Pred Pos | Pred Neg |
|---|---|---|
| **Actual Pos** | 596 (TP) | 13 (FN) |
| **Actual Neg** | 0 (FP) | 291 (TN) |

---

## Repository Structure
landing-pad-detection_v3/
│
├── convert_to_yolo.py      # Converts Isaac Sim output (.npy + rgb) to YOLO format
├── train.py                # YOLOv11s training script
├── evaluate.py             # Confusion matrix + metrics at IoU 0.25/0.50/0.75
├── inspect_frame.py        # Manual single-image inference tool
├── failureanalysis.py      # Extracts and annotates false negative frames
├── check.py                # Dataset validation utility
├── usdscript.py            # Isaac Sim Replicator batch generation script
│
├── runs/
│   └── detect/runs/detect/landing_pad_v4/weights/
│       └── best.pt         # Trained model weights
│
└── failure_analysis/       # Annotated false negative frames from test set

---

## How to Run

**Setup:**
```bash
py -3.11 -m venv venv
venv\Scripts\activate
pip install ultralytics
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Convert Isaac Sim data to YOLO format:**
```bash
python convert_to_yolo.py
```

**Train:**
```bash
python train.py
```

**Evaluate:**
```bash
python evaluate.py
```

**Inspect a single frame manually:**
```bash
python inspect_frame.py
```

---

## Next Steps

- [ ] Collect real drone flight footage with ArUco ID 1 marker
- [ ] Annotate real frames using Roboflow
- [ ] Run best.pt on real footage — sim-to-real validation
- [ ] Compare synthetic vs real confusion matrices
- [ ] Analyse sim-to-real gap and identify failure modes on real data

---

## Related

- Week 2 repo: [landing-pad-detection_v2](https://github.com/thecreator336/landing-pad-detection_v2)

---

## Acknowledgements

Developed during an internship at [ePlane](https://www.eplane.co.in). Synthetic data generated using NVIDIA Isaac Sim with Omniverse Replicator.
