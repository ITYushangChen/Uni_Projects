import pyautogui
#import pygame
from serial.tools import list_ports
import serial
import time
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute

def load_classifier(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def On_Off(program_status):
    """
    :param program_status: on/off
    :return: switch the status when user double blinks
    """
    return not program_status


def goLeft():
    pyautogui.press('left')


def goRight():
    pyautogui.press('right')
# def play_music(file_path):
#     pygame.mixer.init()
#     pygame.mixer.music.load(file_path)
#     pygame.mixer.music.play()
#
#     while pygame.mixer.music.get_busy():
#         pygame.time.Clock().tick(10)  # This line helps to wait for the music to finish playing

def load_all_classifiers(filenames):
    classifiers = {}
    for filename in filenames:
        classifier_name = filename[:-4].replace("_", " ")
        classifiers[classifier_name] = load_classifier(filename)
        print(f"{classifier_name} classifier loaded.")
    return classifiers


def stream():
    classifier_filenames = [
    "Generated_Files/Random_Forest_classifier.pkl",
    "Generated_Files/SVM_classifier.pkl",
    "Generated_Files/Logistic_Regression_classifier.pkl",
    "Generated_Files/Decision_Tree_classifier.pkl",
    "Generated_Files/Gradient_Boosting_classifier.pkl",]
    program_status = False
    baudrate = 230400
    cport = "/dev/cu.usbserial-DM02IZ0A"  # set the correct port before you run it
    ser = serial.Serial(port=cport, baudrate=baudrate)

    input_buffer_size = 10000 # keep betweein 2000-20000
    ser.timeout = input_buffer_size/20000.0  # set read timeout, 20000 is one second



    window_time = 2; # time plotted in window [s]
    time_acquire = input_buffer_size/20000.0    # length of time that data is acquired for
    n_window_loops = window_time/time_acquire

    tick1 = time.time()
    print("Starting Data Stream")
    classifiers = load_all_classifiers(classifier_filenames)

    classifier = classifiers["Generated Files/Random Forest classifier"]
    total_loop = 0

    start = 0

    data_ls = []

    while True:
        try:
            open = '../sound/open.wav'
            pause = '../sound/pause.ogg'
            print("Collecting data, loop iteration %i" %(total_loop))

            data = read_arduino(ser,input_buffer_size)
            data_temp = process_data(data)
            if len(data_temp) == 0:
                print("Dataset Empty, restarting loop...")
                total_loop = 0
                continue

            data_temp = data_temp - 512
            # print("Average = %f" %(abs(np.average(data_temp))))

            # t_temp = input_buffer_size/20000.0*np.linspace(0,1,len(data_temp))
            # sigma_gauss = 25
            data_filtered_temp = fft_clean(data_temp, 10000, [0,10])
            # data_filtered_temp = process_gaussian_fft(t_temp,data_temp,sigma_gauss)

            if total_loop == 0:
                data_raw_total = data_temp
                data_filtered_total = data_filtered_temp
                data_raw_window = data_temp
                data_filtered_window = data_filtered_temp
            else:
                data_filtered_total = np.append(data_filtered_total,data_filtered_temp)
                data_raw_total = np.append(data_raw_total,data_temp)

            t_filtered_total = (total_loop+1)*time_acquire*np.linspace(0,1,len(data_filtered_total))
            t_raw_total = (total_loop+1)*time_acquire*np.linspace(0,1,len(data_raw_total))

            data_last_window_filt, t_last_window_filt = get_last_window(t_filtered_total, data_filtered_total, window_time)
            data_last_window_raw, t_last_window_raw = get_last_window(t_raw_total, data_raw_total, window_time)
            if total_loop !=0:
                print(data_last_window_filt)

            #### ADD CLASSIFICATION METHOD HERE ####
            zero_crossing_point = len(np.where(np.diff(np.sign(data_last_window_filt)))[0])
            if zero_crossing_point < 10:
                predcit_class = predict(data_last_window_filt, classifier)
                print("Prediction: %s" %(predcit_class))
                if (predcit_class == "R"):
                    goRight()
                elif (predcit_class == "L"):
                    goLeft()
                elif (predcit_class == "B"):
                    # if program_status:
                    #     play_music(pause)
                    # else:
                    #     play_music(open)
                    On_Off(program_status)
            ########################################
            total_loop = total_loop+1
        except KeyboardInterrupt:
            break

    print("Finished")
    tick2 = time.time()
    time_taken = tick2 - tick1
    print("Time taken: ",time_taken)

    fig1, ax1 = plt.subplots()
    ax1.clear()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (nd)")
    ax1.set_title("Last %i second of stream" %(window_time))
    ax1.plot(t_last_window_raw,data_last_window_raw, label = "Raw data")
    ax1.plot(t_last_window_filt,data_last_window_filt, label = "FFT filtered data")
    ax1.legend()
    ax1.set_ylim([-300, 300])

    fig2, ax2 = plt.subplots()
    ax2.clear()
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude (nd)")
    ax2.set_title("Stream")
    ax2.plot(t_raw_total, data_raw_total, label = "Raw data")
    ax2.plot(t_filtered_total, data_filtered_total, label = "FFT filtered data")
    ax2.legend()
    ax2.set_ylim([-300, 300])
    plt.show()

def read_arduino(ser,input_buffer_size):
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
    
    return data_clean

def process_gaussian_fft(t,data_t,sigma_gauss):
    nfft = len(data_t) # number of points
    print("data_t length = %i" %(len(data_t)))

    dt = t[1]-t[0]  # time interval
    maxf = 1/dt     # maximum frequency
    df = 1/np.max(t)   # frequency interval
    f_fft = np.arange(-maxf/2,maxf/2+df,df)          # define frequency domain
    print("f_fft length = %i" %(len(f_fft)))

    ## DO FFT
    data_f = np.fft.fftshift(np.fft.fft(data_t)) # FFT of data
    print('data_f length = %i'%(len(data_f)))

    ## GAUSSIAN FILTER
    #    sigma_gauss = 25  # width of gaussian - defined in the function
    gauss_filter = np.exp(-(f_fft)**2/sigma_gauss**2)   # gaussian filter used
    print("gauss filt length = %i" %(len(gauss_filter)))
    print(len(gauss_filter))
    data_f_filtered= data_f*gauss_filter    # gaussian filter spectrum in frquency domain
    data_t_filtered = np.fft.ifft(np.fft.ifftshift(data_f_filtered))    # bring filtered signal in time domain
    return data_t_filtered

def find_ports():
    ports = list_ports.comports()
    for port in ports:
        print(port)

def get_last_window(time_array, data_array, window): 
    start_index= 0
    data_last_window = data_array
    time_last_window = time_array
    for i in range(len(time_array)-1, 0, -1): 
        if (time_array[-1] - time_array[i]) > window: 
            start_index = i
            break
    data_last_window = data_last_window[start_index:]
    time_last_window = time_last_window[start_index:]
    return data_last_window, time_last_window

def load_all_classifiers(filenames):
    classifiers = {}
    for filename in filenames:
        classifier_name = filename[:-4].replace("_", " ")
        classifiers[classifier_name] = load_classifier(filename)
        print(f"{classifier_name} classifier loaded.")
    return classifiers

def predict(signal, classifier):
    df = pd.DataFrame({'id': 0, 'time': np.arange(len(signal)), 'value': signal.astype(float)})
    extracted_features = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters())
    impute(extracted_features)
    predicted_class = classifier.predict(extracted_features)
    return predicted_class

def load_classifier(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)