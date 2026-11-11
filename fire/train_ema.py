import torch

from ultralytics import YOLO

if __name__ == "__main__":
    cuda_available = torch.cuda.is_available()
    print("cuda available:", cuda_available)
    if cuda_available:
        print("current device:", torch.cuda.current_device())
        print("device name:", torch.cuda.get_device_name(0))

    model = YOLO(r"D:\03-Work\fire\ultralytics\cfg\models\v8\yolov8n-ema.yaml")
    model.load(r"D:\03-Work\fire\yolov8n.pt")

    model.train(
        data=r"D:\03-Work\fire\datasets\data.yaml",
        workers=2,
        epochs=100,
        device=0 if cuda_available else "cpu",
        batch=32,
        deterministic=True,
        seed=0,
        imgsz=640,
        patience=15,
        cache=True,
        project="runs/detect",
        name="train_ema_yolov8n_residual",
    )
