from ultralytics import YOLO
import torch

if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo11s.pt")

    model.train(
        data="C:/yolo_dataset_3/dataset.yaml",
        epochs=100,
        batch=8,
        workers=2,
        device=0,
        imgsz=640,
        project="runs/detect",
        name="landing_pad_v4",
        exist_ok=True,
    )
    