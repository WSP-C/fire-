from ultralytics import YOLO

model = YOLO('yolov8n.pt')
# Train the model
model.train(data=r'D:\03-Work\fire\datasets\data.yaml', workers=0, epochs=100, device=0, batch=32, deterministic=False)
