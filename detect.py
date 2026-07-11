
from ultralytics import YOLO
import cv2
import serial
import time

arduino = serial.Serial("COM7",9600)

time.sleep(2) #waiting for aurdino to reset


#loading the model
model = YOLO("runs/detect/train/weights/best.pt")

#streaming
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Error: cannot open camera.")
    exit()

print("camera started...")
print("press esc to exit")

# frame reading 
while True:
    ret, frame = cap.read() # ret is the bollean value which tells wether getting frames succesfully or not
    
    if not ret:
        break


    results = model(frame) # passing frames to model
    # what is it doing 
    # frame -> yolo -> drone detected -> bounding box -> confidence
    # results are stored in -> results[0].boxes


    for box in results[0].boxes:
        # annotated_frame = results[0].plot()
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2   
        print(f"center: ({center_x},{center_y})")

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1),(x2, y2),(0, 255, 0),2)

        # Draw center point
        cv2.circle(frame,(center_x, center_y),5,(0, 0, 255),1)

    cv2.imshow("Drone Detection", frame)

       
    data = f"{center_x},{center_y}\n"

    arduino.write(data.encode())
        # while arduino.in_waiting>0:
        #     print(arduino.readline())

    key = cv2.waitKey(1)

    if key == 27:      # ESC key
        print("Closing...")
        break

arduino.close()
cap.release()
cv2.destroyAllWindows()
