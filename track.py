from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

model.track(
    source=0,
    show=True,
    tracker="bytetrack.yaml",
    persist=True
)