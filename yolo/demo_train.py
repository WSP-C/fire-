from ultralytics import YOLO

# model = YOLO('yolov8n.pt')
# model = YOLO('yolov8-simam.yaml').load('yolov8n.pt')
model = YOLO(r'D:\03-Work\yolo\ultralytics\cfg\models\v8\yolov8n-ema.yaml').load('yolov8n.pt')
# Train the model
model.train(data=r'D:\03-Work\yolo\datasets\data.yaml', workers=0, epochs=100, device=0, batch=32, deterministic=False)
