from ultralytics import YOLO
import cv2
import time
import math
import serial

arduino = serial.Serial("COM7",115200)

time.sleep(2) #waiting for aurdino to reset
# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Open camera
cap = cv2.VideoCapture(1)

# Drone parameters
KNOWN_WIDTH = 0.30  # meters (30 cm drone)
FOCAL_LENGTH = 350  # calibration value

# Tracking storage
previous_positions = {}
previous_times = {}
speed_history = {}

center_x = -1
center_y = -1

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        conf=0.6
    )

    annotated = results[0].plot()

    if (
        results[0].boxes is not None
        and results[0].boxes.id is not None
    ):

        ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(results[0].boxes, ids):

            x1, y1, x2, y2 = box.xyxy[0]

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            pixel_width = x2 - x1

            if pixel_width <= 0:
                continue



            # Distance estimation
            distance_m = (KNOWN_WIDTH * FOCAL_LENGTH) / pixel_width

            current_time = time.time()

            direction = "STABLE"
            speed_mps = 0.0

            if track_id in previous_positions:

                prev_x, prev_y = previous_positions[track_id]

                dx = center_x - prev_x
                dy = center_y - prev_y

                # Direction
                if abs(dx) > abs(dy):

                    if dx > 5:
                        direction = "RIGHT"

                    elif dx < -5:
                        direction = "LEFT"

                else:

                    if dy > 5:
                        direction = "DOWN"

                    elif dy < -5:
                        direction = "UP"

                dt = current_time - previous_times[track_id]

                if dt > 0:

                    pixel_distance = math.sqrt(
                        dx ** 2 + dy ** 2
                    )
                    

                    frame_width = frame.shape[1]

                    # Approx camera FOV
                    scene_width_m = (
                        2
                        * distance_m
                        * math.tan(math.radians(35))
                    )

                    meters_per_pixel = (
                        scene_width_m / frame_width
                    )

                    real_distance = (
                        pixel_distance
                        * meters_per_pixel
                    )

                    speed_mps = (
                        real_distance / dt
                    )

                    # Smooth speed
                    if track_id not in speed_history:
                        speed_history[track_id] = []

                    speed_history[track_id].append(
                        speed_mps
                    )

                    if len(speed_history[track_id]) > 10:
                        speed_history[track_id].pop(0)

                    speed_mps = (
                        sum(speed_history[track_id])
                        / len(speed_history[track_id])
                    )

            previous_positions[track_id] = (
                center_x,
                center_y
            )

            previous_times[track_id] = current_time

            # Threat level
            threat = "LOW"

            if distance_m < 3:
                threat = "HIGH"

            elif distance_m < 7:
                threat = "MEDIUM"

            # HUD Panel
            cv2.rectangle(
                annotated,
                (10, 10),
                (500, 150),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                annotated,
                f"Drone ID: {track_id}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Distance: {distance_m:.2f} m",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Direction: {direction}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Speed: {speed_mps:.2f} m/s",
                (20, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated,
                f"Threat: {threat}",
                (300, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

    cv2.imshow(
        "AI Drone Surveillance System",
        annotated
    )


    data = f"{center_x},{center_y}\n"
    last_send = 0
    if time.time() - last_send > 0.05:
        arduino.write(f"{center_x},{center_y}\n".encode())
        last_send = time.time()

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()