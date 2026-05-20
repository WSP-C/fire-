from ultralytics import YOLO
# 加载模型
yolo = YOLO("./yolov8n.pt ", task="detect")
# 进行推理
result = yolo(source="D:\\03-Work\\yolo\\ultralytics\\assets\\bus.jpg", save=True)

# 或者如果你想直接保存结果，可以加 save=True
# result = model(source="./ultralytics/assets/bus.jpg", save=True)