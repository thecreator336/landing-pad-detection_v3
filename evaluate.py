import os
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

if __name__ == "__main__":

    # ─── CONFIG ───────────────────────────────────────────────────────────────
    MODEL_PATH   = r"C:\Users\epity\OneDrive\Documents\Landingrecog_3\runs\detect\runs\detect\landing_pad_v4\weights\best.pt"
    TEST_IMAGES  = r"C:/yolo_dataset_3/images/test"
    TEST_LABELS  = r"C:/yolo_dataset_3/labels/test"
    OUTPUT_DIR   = r"C:\Users\epity\OneDrive\Documents\Landingrecog_3\eval_results"
    IOU_THRESHOLDS = [0.25, 0.50, 0.75]
    CONF_THRESHOLD = 0.25
    # ─────────────────────────────────────────────────────────────────────────

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    print(f"Model loaded: {MODEL_PATH}")

    image_files = sorted([f for f in os.listdir(TEST_IMAGES) if f.endswith(".png")])
    print(f"Test images: {len(image_files)}")

    results_per_iou = {iou: {"TP": 0, "FP": 0, "FN": 0, "TN": 0} for iou in IOU_THRESHOLDS}
    sample_images = []

    for img_file in image_files:
        img_path   = os.path.join(TEST_IMAGES, img_file)
        label_path = os.path.join(TEST_LABELS, img_file.replace(".png", ".txt"))

        # Load ground truth
        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.read().strip().splitlines()
            for line in lines:
                if line.strip() == "":
                    continue
                parts = line.strip().split()
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                gt_boxes.append([cx, cy, w, h])

        # Run inference
        preds = model(img_path, conf=CONF_THRESHOLD, verbose=False)[0]
        pred_boxes = []
        if preds.boxes is not None and len(preds.boxes) > 0:
            for box in preds.boxes:
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w  = x2 - x1
                h  = y2 - y1
                pred_boxes.append([cx, cy, w, h])

        # Save sample images (first 16)
        if len(sample_images) < 16:
            sample_images.append((img_path, gt_boxes, pred_boxes))

        # Evaluate at each IoU threshold
        for iou_thresh in IOU_THRESHOLDS:
            has_gt   = len(gt_boxes) > 0
            has_pred = len(pred_boxes) > 0

            if not has_gt and not has_pred:
                results_per_iou[iou_thresh]["TN"] += 1
                continue

            if not has_gt and has_pred:
                results_per_iou[iou_thresh]["FP"] += 1
                continue

            if has_gt and not has_pred:
                results_per_iou[iou_thresh]["FN"] += 1
                continue

            # Both have boxes — compute IoU
            matched = False
            for gt in gt_boxes:
                for pred in pred_boxes:
                    # Convert to x1y1x2y2
                    gt_x1   = gt[0] - gt[2] / 2
                    gt_y1   = gt[1] - gt[3] / 2
                    gt_x2   = gt[0] + gt[2] / 2
                    gt_y2   = gt[1] + gt[3] / 2
                    pr_x1   = pred[0] - pred[2] / 2
                    pr_y1   = pred[1] - pred[3] / 2
                    pr_x2   = pred[0] + pred[2] / 2
                    pr_y2   = pred[1] + pred[3] / 2

                    ix1 = max(gt_x1, pr_x1)
                    iy1 = max(gt_y1, pr_y1)
                    ix2 = min(gt_x2, pr_x2)
                    iy2 = min(gt_y2, pr_y2)

                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    gt_area = gt[2] * gt[3]
                    pr_area = pred[2] * pred[3]
                    union   = gt_area + pr_area - inter

                    iou = inter / union if union > 0 else 0

                    if iou >= iou_thresh:
                        matched = True
                        break
                if matched:
                    break

            if matched:
                results_per_iou[iou_thresh]["TP"] += 1
            else:
                results_per_iou[iou_thresh]["FN"] += 1
                results_per_iou[iou_thresh]["FP"] += 1

    # Print results
    print("\n─── Evaluation Results ───")
    for iou_thresh in IOU_THRESHOLDS:
        r = results_per_iou[iou_thresh]
        TP, FP, FN, TN = r["TP"], r["FP"], r["FN"], r["TN"]
        P  = TP / (TP + FP) if (TP + FP) > 0 else 0
        R  = TP / (TP + FN) if (TP + FN) > 0 else 0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0
        print(f"\nIoU threshold: {iou_thresh}")
        print(f"  TP={TP}, FP={FP}, FN={FN}, TN={TN}")
        print(f"  Precision={P:.3f}, Recall={R:.3f}, F1={F1:.3f}")

    # Save confusion matrix plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, iou_thresh in enumerate(IOU_THRESHOLDS):
        r = results_per_iou[iou_thresh]
        matrix = np.array([
            [r["TP"], r["FN"]],
            [r["FP"], r["TN"]]
        ])
        ax = axes[idx]
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"IoU {iou_thresh}")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Pos", "Pred Neg"])
        ax.set_yticklabels(["True Pos", "True Neg"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    print(f"\n✓ Confusion matrix saved to {OUTPUT_DIR}")

    # Save sample images
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for idx, (img_path, gt_boxes, pred_boxes) in enumerate(sample_images):
        ax = axes[idx // 4][idx % 4]
        img = Image.open(img_path)
        ax.imshow(img)
        W, H = img.size
        for gt in gt_boxes:
            cx, cy, w, h = gt
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            ax.add_patch(patches.Rectangle((x1, y1), w*W, h*H, linewidth=2, edgecolor="green", facecolor="none"))
        for pred in pred_boxes:
            cx, cy, w, h = pred
            x1 = (cx - w/2) * W
            y1 = (cy - h/2) * H
            ax.add_patch(patches.Rectangle((x1, y1), w*W, h*H, linewidth=2, edgecolor="red", facecolor="none"))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_predictions.png"))
    print(f"✓ Sample predictions saved to {OUTPUT_DIR}")