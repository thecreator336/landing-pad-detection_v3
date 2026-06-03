# failure_analysis.py
# Saves all false negative frames to a folder for visual inspection
# Run: python failure_analysis.py

from ultralytics import YOLO
import cv2
import os
import json

MODEL_PATH   = r"C:\Users\epity\OneDrive\Documents\Landingrecog_3\runs\detect\runs\detect\landing_pad_v4\weights\best.pt"
TEST_IMG_DIR = r"C:\yolo_dataset_3\images\test"
TEST_LBL_DIR = r"C:\yolo_dataset_3\labels\test"
OUTPUT_DIR   = r"C:\Users\epity\OneDrive\Documents\Landingrecog_3\failure_analysis"
CONF         = 0.25
IOU_THRESH   = 0.50

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

def load_gt_boxes(label_path, img_w, img_h):
    """Load YOLO label file and convert to pixel coords"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = int((cx - w/2) * img_w)
            y1 = int((cy - h/2) * img_h)
            x2 = int((cx + w/2) * img_w)
            y2 = int((cy + h/2) * img_h)
            boxes.append((x1, y1, x2, y2))
    return boxes

def compute_iou(boxA, boxB):
    ax1,ay1,ax2,ay2 = boxA
    bx1,by1,bx2,by2 = boxB
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    areaA = (ax2-ax1)*(ay2-ay1)
    areaB = (bx2-bx1)*(by2-by1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0

image_files = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png','.jpg'))]

fn_count = 0
tp_count = 0
summary  = []

for fname in sorted(image_files):
    img_path = os.path.join(TEST_IMG_DIR, fname)
    lbl_path = os.path.join(TEST_LBL_DIR, fname.replace('.png','.txt').replace('.jpg','.txt'))

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    gt_boxes   = load_gt_boxes(lbl_path, w, h)
    is_positive = len(gt_boxes) > 0  # has a ground truth pad

    if not is_positive:
        continue  # skip negatives — we only want false negatives here

    results  = model.predict(source=img_path, conf=CONF, verbose=False)
    pred_boxes = []
    if results[0].boxes:
        for box in results[0].boxes:
            x1,y1,x2,y2 = [int(v) for v in box.xyxy[0]]
            conf_score = float(box.conf[0])
            pred_boxes.append((x1,y1,x2,y2,conf_score))

    # Check if GT box was matched
    matched = False
    best_iou = 0.0
    for gt in gt_boxes:
        for pred in pred_boxes:
            iou = compute_iou(gt, pred[:4])
            if iou > best_iou:
                best_iou = iou
            if iou >= IOU_THRESH:
                matched = True

    if matched:
        tp_count += 1
        continue  # correct detection, skip

    # --- FALSE NEGATIVE --- save it
    fn_count += 1
    vis = img.copy()

    # Draw ground truth in green (dashed effect via two rects)
    for gt in gt_boxes:
        cv2.rectangle(vis, (gt[0],gt[1]), (gt[2],gt[3]), (0,200,0), 2)
        cv2.putText(vis, "GT", (gt[0], gt[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,0), 2)

    # Draw predictions in red (if any — near-miss case)
    for pred in pred_boxes:
        cv2.rectangle(vis, (pred[0],pred[1]), (pred[2],pred[3]), (0,0,220), 2)
        cv2.putText(vis, f"pred {pred[4]:.2f}", (pred[0], pred[1]-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,220), 2)

    # Overlay info
    label = f"FN #{fn_count} | best IoU: {best_iou:.2f} | preds: {len(pred_boxes)}"
    cv2.putText(vis, label, (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    cv2.putText(vis, label, (10,22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1)

    out_name = f"fn_{fn_count:03d}_iou{best_iou:.2f}_{fname}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), vis)

    summary.append({
        "file": fname,
        "best_iou": round(best_iou, 4),
        "num_predictions": len(pred_boxes),
        "prediction_confs": [round(p[4],3) for p in pred_boxes]
    })

# Save summary JSON
with open(os.path.join(OUTPUT_DIR, "fn_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nDone.")
print(f"  True Positives : {tp_count}")
print(f"  False Negatives: {fn_count}")
print(f"  Saved to       : {OUTPUT_DIR}")
print(f"\nFalse Negative breakdown:")
for s in summary:
    tag = "NO DETECTION" if s['num_predictions'] == 0 else f"near-miss conf={s['prediction_confs']}"
    print(f"  {s['file']}  IoU={s['best_iou']}  {tag}")