print("Step 1: Import complete")

from ultralytics import YOLO
import cv2

print("Step 2: Loading model...")
model = YOLO("runs/detect/train/weights/best.pt")

print("Step 3: Opening camera...")
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

print("Step 4: Camera opened =", cap.isOpened())

# Assumptions
KNOWN_WIDTH = 30.0      # Drone width in cm (adjust later)
FOCAL_LENGTH = 700.0    # Calibration value



while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    for box in results[0].boxes:

        x1, y1, x2, y2 = box.xyxy[0]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        pixel_width = x2 - x1

        if pixel_width > 0:

            distance_cm = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width
            distance_m = distance_cm / 100
            text = f"Distance: {distance_m:.2f}m"
            cv2.rectangle(annotated, (10, 10), (280, 60), (0, 0, 0), -1)
            cv2.putText(
                annotated,
                text,
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Drone Detection + Distance", annotated)

    key = cv2.waitKey(1)

    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()