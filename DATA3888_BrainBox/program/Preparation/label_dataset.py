from scipy.io import wavfile
import numpy as np
import pandas as pd
from decimal import Decimal
import os
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import EfficientFCParameters,MinimalFCParameters



def eye_movement_ZC(Y, time, windowSize=0.5, thresholdEvents=20, downSampleRate=50):
    # print(f"Original Y: {Y}")
    #
    # Y = [elem - 510 for elem in Y]
    # print(f"Modified Y: {Y}")
    # with open('list.txt', 'w') as fp:
    #     for item in Y:
    #         # write each item on a new line
    #         fp.write("%s " % item)
    #     print('Done')
    ind = np.arange(0, np.where(time == np.round(time[-1] - windowSize, 4))[0][0] + 1, downSampleRate)

    timeMiddle = time[ind] + windowSize / 2
    testStat = np.empty(len(ind), dtype=int)
    for i in range(len(ind)):
        Y_subset = Y[(time >= time[ind[i]]) & (time < time[ind[i]] + windowSize)]
        if np.ndim(Y_subset) > 1:
            Y_subset = Y_subset[:, 1]
        testStat[i] = sum(Decimal(int(Y_subset[i])) * Decimal(int(Y_subset[i + 1])) <= 0 for i in range(len(Y_subset) - 1))
    predictedEvent = np.where(testStat < thresholdEvents)[0]
    eventTimes = timeMiddle[predictedEvent]
    gaps = np.where(np.diff(eventTimes) > windowSize)[0]
    if len(eventTimes) == 0:
        return None
    event_time_interval = [np.min(eventTimes)]
    for i in range(len(gaps)):
        event_time_interval.extend([eventTimes[gaps[i]], eventTimes[gaps[i] + 1]])
    event_time_interval.append(np.max(eventTimes))
    event_time_interval = np.reshape(event_time_interval, (-1, 2))

    predictedEventTimes = np.full(len(Y), False)
    for i in range(event_time_interval.shape[0]):
        predictedEventTimes[(event_time_interval[i, 0] <= time) & (event_time_interval[i, 1] >= time)] = True

    num_event = len(gaps) + 1
    movement_list = []
    y_values = []
    for i in range(event_time_interval.shape[0]):
        interval_start, interval_end = event_time_interval[i]
        interval_indices = (time >= interval_start) & (time <= interval_end)
        interval_Y = Y[interval_indices]
        y_values.append(interval_Y)
        movement = LR_detection(interval_Y)
        movement_list.append(movement)
    return {
        "num_event": num_event,
        "predictedEventTimes": predictedEventTimes,
        "predictedInterval": event_time_interval,
        "labels": movement_list,
        "signals": y_values
    }

def LR_detection(seq):
    maxval = np.max(seq)
    minval = np.min(seq)
    movement = "L" if maxval < -minval else "R"
    return movement
def record_all_training(files,method):
    ls_signals = []
    ls_labels = []
    ls_intervals = []
    for i in range(len(files)):
        wave = files[i]
        print(wave)
        if wave[-3:] != "wav":
            continue
        window_size, Y = wavfile.read(wave)
        timeSeq = []
        for i in range(len(Y)):
            timeSeq.append(i / window_size)
        timeSeq = np.array(timeSeq)
        if np.ndim(Y) > 1:
            Y = Y[:, 1]
            Y = Y-500
        Y = np.array(Y)
        if method == "zc":
            result = eye_movement_ZC(Y=Y, time=timeSeq)
        # elif method == "max":
        #     result = eye_movement_max(Y=Y,time = timeSeq)
        if result == None:
            continue
        ls_signals.append(result["signals"])
        ls_labels.append(result["labels"])
        ls_intervals.append(result["predictedInterval"])
    ls_labels = [item for sublist in ls_labels for item in sublist]
    ls_signals = [item for sublist in ls_signals for item in sublist]
    ls_intervals = [item for sublist in ls_intervals for item in sublist]
    return {
        "ls_signals":ls_signals,
        "ls_labels":ls_labels,
        "ls_intervals": ls_intervals
    }

def make_matrix_tsfresh(signals, labels):
    df_list = []
    for i, signal in enumerate(signals):
        temp_df = pd.DataFrame({'id': i, 'time': np.arange(len(signal)), 'value': signal.astype(float)})
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    extracted_features = extract_features(df, column_id='id', column_sort='time',default_fc_parameters=MinimalFCParameters())
    impute(extracted_features)
    extracted_features['Label'] = labels
    extracted_features.to_csv("../Generated_Files/matrix.csv", index=False)
    return extracted_features

if __name__ == '__main__':
    # Load the classifier
    # path = "../../datasets/zoe_spiker/Length3"
    # file_ls = []
    # wave_file_ls = os.listdir(path)
    # for i in range(len(wave_file_ls)):
    #     file_path = path + "/" + wave_file_ls[i]
    #     file_ls.append(file_path)
    # results = record_all_training(files = file_ls,method="zc")
    #
    # if results != None:
    #     matrix = make_matrix_tsfresh(results["ls_signals"], results["ls_labels"])
    #     print(matrix)


    path = "Data"
    file_ls = []
    wave_file_ls = os.listdir(path)
    for i in range(len(wave_file_ls)):
        file_path = path + "/" + wave_file_ls[i]
        file_ls.append(file_path)
    results = record_all_training(files = file_ls,method="zc")
    print(results['ls_intervals'])