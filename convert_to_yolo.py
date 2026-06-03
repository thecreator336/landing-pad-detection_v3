import os
import json
import shutil
import random
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INPUT_ROOT  = "C:/landing_pad_3"
OUTPUT_ROOT = "C:/yolo_dataset_3"
BATCHES     = 9
IMG_W       = 640
IMG_H       = 480
TRAIN_RATIO = 0.8
VAL_RATIO   = 0.1
TEST_RATIO  = 0.1
# ─────────────────────────────────────────────────────────────────────────────

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_ROOT, "images", split), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "labels", split), exist_ok=True)

all_samples = []

for batch_idx in range(BATCHES):
    batch_dir = os.path.join(INPUT_ROOT, f"batch_{batch_idx:04d}")

    if not os.path.exists(batch_dir):
        print(f"Skipping batch {batch_idx} — folder not found")
        continue

    for img_file in sorted(os.listdir(batch_dir)):
        if not img_file.endswith(".png"):
            continue

        img_path  = os.path.join(batch_dir, img_file)
        frame_num = img_file.replace("rgb_", "").replace(".png", "")
        npy_file  = f"bounding_box_2d_tight_{frame_num}.npy"
        npy_path  = os.path.join(batch_dir, npy_file)

        label_lines = []

        if os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True)
            for entry in data:
                x_min = int(entry["x_min"])
                y_min = int(entry["y_min"])
                x_max = int(entry["x_max"])
                y_max = int(entry["y_max"])

                # Skip invalid boxes
                if x_max <= x_min or y_max <= y_min:
                    continue

                x_center = ((x_min + x_max) / 2) / IMG_W
                y_center = ((y_min + y_max) / 2) / IMG_H
                width    = (x_max - x_min) / IMG_W
                height   = (y_max - y_min) / IMG_H

                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width    = max(0, min(1, width))
                height   = max(0, min(1, height))

                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        all_samples.append((img_path, label_lines))

print(f"Total samples collected: {len(all_samples)}")
positive = sum(1 for _, l in all_samples if len(l) > 0)
negative = sum(1 for _, l in all_samples if len(l) == 0)
print(f"  Positive: {positive}")
print(f"  Negative: {negative}")

# Shuffle and split
random.seed(42)
random.shuffle(all_samples)

n_total = len(all_samples)
n_train = int(n_total * TRAIN_RATIO)
n_val   = int(n_total * VAL_RATIO)

splits = {
    "train": all_samples[:n_train],
    "val":   all_samples[n_train:n_train + n_val],
    "test":  all_samples[n_train + n_val:]
}

for split, samples in splits.items():
    for i, (img_path, label_lines) in enumerate(samples):
        dst_img = os.path.join(OUTPUT_ROOT, "images", split, f"{split}_{i:05d}.png")
        shutil.copy(img_path, dst_img)

        dst_lbl = os.path.join(OUTPUT_ROOT, "labels", split, f"{split}_{i:05d}.txt")
        with open(dst_lbl, "w") as f:
            f.write("\n".join(label_lines))

    print(f"{split}: {len(samples)} samples")

# Write dataset yaml
yaml_path = os.path.join(OUTPUT_ROOT, "dataset.yaml")
with open(yaml_path, "w") as f:
    f.write(f"path: {OUTPUT_ROOT}\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n")
    f.write("nc: 1\n")
    f.write("names: ['landing_pad']\n")

print(f"✓ Dataset saved to {OUTPUT_ROOT}")
print(f"  dataset.yaml written")

