import serial

ser = serial.Serial("COM3", 9600) # aadjust port name to the one showed in the Arduino IDE, Baud Rate should stay the same
arduino_data = []

while True:
    data = ser.readline().decode().strip()
    arduino_data.append(data)
    print(arduino_data)