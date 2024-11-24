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


def read_arduino(ser,input_buffer_size):
#   data = ser.readline(input_buffer_size)
    data = ser.read(input_buffer_size)
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

def filter_out(f_min, f_max, freq_ft, data_ft): # this method will reduce amplitude of data_ft to 0 for all f outside of range(fmin, fmax)
    done_min = False 
    done_max = False
    index_min = 0
    index_max = 0
    for i in range(0,len(freq_ft)): 
        if freq_ft[i] >= f_min and done_min == False: 
            index_min =i
            done_min = True
        if freq_ft[i] >= f_max and done_max == False: 
            index_max = i 
            done_max = True
    data_ft_clean = data_ft
    data_ft_clean[index_max:]= 0
    data_ft_clean[0:index_min] = 0

    return data_ft_clean 

def fft_clean(data, sample_rate,frequency_range): 
    f_min = frequency_range[0]
    f_max = frequency_range[1]

    data_ft = np.fft.fft(data)
    freq = np.linspace(0, sample_rate,len(data_ft))

    data_ft_clean = filter_out(f_min,f_max,freq, data_ft)
    data_clean = np.fft.ifft(data_ft_clean)
    
    return data_clean , [data_ft, freq]

# use this to find ports
# ports = list_ports.comports()
# for port in ports:
#     print(port)

# Data Streaming over interval total_time: 

baudrate = 230400
cport = "/dev/cu.usbserial-DM02IZ0A"  # set the correct port before you run it
ser = serial.Serial(port=cport, baudrate=baudrate) 

input_buffer_size = 10000 # keep betweein 2000-20000
ser.timeout = input_buffer_size/20000.0  # set read timeout, 20000 is one second

total_time = 10; # time in seconds [[1 s = 20000 buffer size]]
window_time = 10; # time plotted in window [s]
n_loops = 20000.0/input_buffer_size*total_time
print(n_loops)

time_acquire = input_buffer_size/20000.0    # length of time that data is acquired for 
n_window_loops = window_time/time_acquire    # total number of loops to cover desire time window
print(n_window_loops)

# create the figure and axis objects
print("Starting Continuous Stream")
for k in range(0,int(n_loops)):
    print("Collecting Data")
    data = read_arduino(ser,input_buffer_size)
    data_temp = process_data(data)

    data_temp = [x - 510 for x in data_temp]

    if k <= n_window_loops:
        if k==0:
            data_plot = data_temp
        else:
            data_plot = np.append(data_plot,data_temp)
        t = (min(k+1,n_window_loops))*input_buffer_size/20000.0*np.linspace(0,1,len(data_plot))
    t = (min(k+1,n_window_loops))*input_buffer_size/20000.0*np.linspace(0,1,len(data_plot))
print("Finished")

# Details of movement:
number_of_movements = "5"
direction_of_movement = "left"
iteration = "1"
file_name = "%s_%s_%s" %(number_of_movements,direction_of_movement,iteration)
electrode_pos = "Possition = left_right_temple"

sample_rate = 10000
frequency_range = [0,10]
print(data_plot)
print(len(data_plot))
data_clean, data_ft = fft_clean(data_plot, sample_rate,frequency_range)
t2 = np.linspace(0, len(data_clean), len(data_clean))/sample_rate
datasave = np.column_stack((t2,data_clean))
write("%s.wav" %file_name, sample_rate, datasave.astype(np.int16))

# plot the data and customize
fig, ax = plt.subplots()
ax.plot(t,data_plot)
# ax.set_ylim([300,800])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Y')
ax.set_title("Unfiltered Data: %s" %(file_name))

fig1, ax1 = plt.subplots()
ax1.plot(t2,data_clean)
# ax1.set_ylim([300,800])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Y')
ax1.set_title('Filtered Data in range %i to %i Hz: %s'%(frequency_range[0],frequency_range[1], file_name))

# fig2, ax2 = plt.subplots()
# ax2.plot(data_ft[0],data_ft[1])
# ax2.set_xlim([0,100])
# ax2.set_ylim([0,100])
# ax2.set_xlabel('Time (s)')
# ax2.set_ylabel('Y')
# ax2.set_title('Title')

fig1.savefig("%s.png" %(file_name))
plt.show()

if ser.read():
    ser.flushInput()
    ser.flushOutput()
    ser.close()