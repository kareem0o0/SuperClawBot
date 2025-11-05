import serial
import time

# Change the port according to your setup
arduino = serial.Serial(port='/dev/ttyACM1', baudrate=9600, timeout=1)
time.sleep(2)  # wait for Arduino to initialize

while True:
    angle = input("Enter angle (0-180), or q to quit: ").strip()
    if angle.lower() == 'q':
        break
    if angle.isdigit() and 0 <= int(angle) <= 180:
        arduino.write(angle.encode())  # send the number
        arduino.write(b'\n')           # send newline so parseInt() can detect end
        print(f"Sent angle: {angle}")
    else:
        print("Invalid input. Please enter a number between 0 and 180.")
    arduino.write(angle.encode())
    time.sleep(0.1) 
# in servo the ranges are (11-170)
# 50 ->  86 ------ 84 -> 86
# 100 -> 160 ------ 100 ->102
