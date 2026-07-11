from ultralytics import YOLO

#loading best trained model
model = YOLO("runs/detect/train/weights/best.pt")

#now validatein on the validation dataset
metrics = model.val(
    data="DRONES_NEW.v1i.yolov8/data.yaml"
)

print(metrics)