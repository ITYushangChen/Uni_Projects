import stream 
import serial as ser

def main(): 
    stream.stream()

    if ser.read():
        ser.flushInput()
        ser.flushOutput()
        ser.close()

if __name__ == '__main__': 
    main()


