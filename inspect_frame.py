# inspect_frame.py
# Run: python inspect_frame.py
# Uses the same venv as your project (ultralytics, opencv already installed)

from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\Users\epity\OneDrive\Documents\Landingrecog_3\runs\detect\runs\detect\landing_pad_v4\weights\best.pt"
CONF_THRESHOLD = 0.25

model = YOLO(MODEL_PATH)

print("Landing Pad Inspector — type 'q' to quit\n")

while True:
    img_path = input("Image path (or 'q' to quit): ").strip().strip('"')
    if img_path.lower() == 'q':
        break

    results = model.predict(source=img_path, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    img = cv2.imread(img_path)
    if img is None:
        print(f"  Could not load image: {img_path}\n")
        continue

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        print("  No detections.\n")
    else:
        print(f"  {len(boxes)} detection(s):")
        for i, box in enumerate(boxes):
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            cx = (x1 + x2) / 2 / img.shape[1]
            cy = (y1 + y2) / 2 / img.shape[0]
            w  = (x2 - x1) / img.shape[1]
            h  = (y2 - y1) / img.shape[0]
            print(f"    [{i+1}] {label}  conf={conf:.3f}  box=({x1},{y1})->({x2},{y2})")
            print(f"         YOLO normalised: {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}")

            # Draw on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    print()
    cv2.imshow("Inspection", img)
    print("  (press any key in the image window to continue)\n")
    cv2.waitKey(0)
    cv2.destroyAllWindows()