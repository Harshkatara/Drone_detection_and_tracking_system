from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="DRONES_NEW.v1i.yolov8/data.yaml",
    epochs=20,
    imgsz=640,
    batch=8,
    device="mps"   # Apple Silicon GPU
)