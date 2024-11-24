import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.fft as fft
import statistics
import scipy.signal
from serial.tools import list_ports
from scipy.io.wavfile import write
from matplotlib.animation import FuncAnimation


def read_arduino(ser,inputBufferSize):
#   data = ser.readline(inputBufferSize)
    data = ser.read(inputBufferSize)
    out =[(int(data[i])) for i in range(0,len(data))]
    return out

def process_data(data):
    data_in = np.array(data)
    result = []
    i = 1
    while i < len(data_in)-1:
        if data_in[i] > 127:
            # Found beginning of frame
            # Extract one sample from 2 bytes
            intout = (np.bitwise_and(data_in[i],127))*128
            i = i + 1
            intout = intout + data_in[i]
            result = np.append(result,intout)
        i=i+1
    return result

# use this to find ports
# ports = list_ports.comports()
# for port in ports:
#     print(port)

# take continuous data stream 
print("Starting Continuous Stream")

baudrate = 230400
cport = "/dev/cu.usbserial-DM02IZ0A"  # set the correct port before you run it
ser = serial.Serial(port=cport, baudrate=baudrate) 

inputBufferSize = 5000 # keep betweein 2000-20000
ser.timeout = inputBufferSize/20000.0  # set read timeout, 20000 is one second

total_time = 5; # time in seconds [[1 s = 20000 buffer size]]
max_time = total_time; # time plotted in window [s]
N_loops = 20000.0/inputBufferSize*total_time

T_acquire = inputBufferSize/20000.0    # length of time that data is acquired for 
N_max_loops = max_time/T_acquire    # total number of loops to cover desire time window


# # create the figure and axis objects
fig1, ax1 = plt.subplots()

# function that draws each frame of the animation
def animate(i):
    for k in range(0,int(N_loops)):
        data = read_arduino(ser,inputBufferSize)
        data_temp = process_data(data)
        if k <= N_max_loops:
            if k==0:
                data_plot = data_temp
            else:
                data_plot = np.append(data_plot,data_temp)
            t = (min(k+1,N_max_loops))*inputBufferSize/20000.0*np.linspace(0,1,len(data_plot))
        t = (min(k+1,N_max_loops))*inputBufferSize/20000.0*np.linspace(0,1,len(data_plot))
        ax1.clear()
        ax1.plot(t, data_plot)
        ax1.set_ylim([0,1200])
    datasave = np.column_stack((t,data_plot))
    write("Test1.wav", 10000, datasave.astype(np.int16))
    
# run the animation
ani = FuncAnimation(fig1, animate, interval=1, repeat=False)
plt.show()

# fig2, ax2 = plt.subplots()
# # plot the data and customize
# ax2.plot(t,data_plot)
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Y')
# ax2.set_title('Title')

# save and show the plot
# fig2.savefig('static_plot.png')
# plt.show()

if ser.read():
    ser.flushInput()
    ser.flushOutput()
    ser.close()