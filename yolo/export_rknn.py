from ultralytics import YOLO

# 1. 加载你的模型文件
model = YOLO(r'D:\03-Work\yolo\runs\detect\train3\weights\best.pt')

# 2. 导出为 ONNX 格式，设置 opset=12 保证 RK3588 兼容性
success = model.export(format='onnx', opset=12)

print(f"导出结果: {success}")