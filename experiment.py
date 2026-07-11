# code to transfer information to hardware
import serial
import time

def arduinoConnect():
    arduino = serial.Serial("COM7",9600)

    time.sleep(2) #waiting for aurdino to reset

    while True: 

        x = input("x= ")
        y = input("y= ")
        data = f"{x},{y}\n"
        if x == "0" or y == "0":
            print("Ending connection...")
            break

        arduino.write(data.encode())
        while arduino.in_waiting>0:
            print(arduino.readline())
    
    arduino.close()
