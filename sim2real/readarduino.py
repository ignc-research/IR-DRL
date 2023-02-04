import serial.tools.list_ports
#This code reads out the serial output that the arduino sensors are giving us. 


 #Warning: Do not run Arduino IDE parallel to this. It will block the COM Port and denies access to reading it.
# Verify and upload the code VIA Arduino IDE then make sure to close it.

#TODO: Dont just print the data save it in a yaml filme
ports = serial.tools.list_ports.comports()
serialInst = serial.Serial()

portsList = []

#shows all possible Ports
for onePort in ports:
    portsList.append(str(onePort))
    print(str(onePort))

#can also be hard coded, if it is always clear which Port is used. I personally wouldnt recomment doing it.
val = input("Select Port: COM")

for x in range(0,len(portsList)):
    if portsList[x].startswith("COM" + str(val)):
        portVar = "COM" + str(val)
        print(portVar)

#initialises serial connection to Arduino through chosen port
serialInst.baudrate = 9600
serialInst.port = portVar
serialInst.open()


while True:
    if serialInst.in_waiting:
        packet = int(serialInst.readline().decode('utf').rstrip('\n'))
        print(packet)
        if packet < 50:
            print("STOP. TOO CLOSE")
    #TODO: Save it in Yaml File
    #TODO: Stop at a certain distance