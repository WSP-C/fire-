from ultralytics import YOLO
import torch

if __name__ == '__main__':
    print("cuda available:", torch.cuda.is_available())
    print("current device:", torch.cuda.current_device())
    print("device name:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data=r"D:\03-Work\fire\datasets\data.yaml",
        workers=2,
        epochs=100,
        device=0,
        batch=32,
        deterministic=False,
        imgsz=640,
        patience=15,
        cache=True,
        project="runs/detect",
        name="train_baseline_yolov8n"
    )
