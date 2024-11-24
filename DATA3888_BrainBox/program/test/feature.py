from os import listdir
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from decimal import Decimal
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tsfresh.feature_extraction import extract_features, MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import make_forecasting_frame
from tsfresh.utilities.dataframe_functions import impute
import featuretools as ft
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import argrelextrema




# 1. Read all the training files
#path = "../datasets/collected"
#path = "../datasets/zoe_spiker/Test/"

# Test set
# path_2 = "../datasets/Test_2"
# file_ls_2 = []
# wave_file_ls_2 = os.listdir(path_2)
# print(wave_file_ls_2)
# for i in range(len(wave_file_ls_2)):
#     file_path_2 = path_2 + "/" + wave_file_ls_2[i]
#     file_ls_2.append(file_path_2)
# print(file_ls_2)
#
# window_size, Y = wavfile.read(file_ls_2[0])
#
# print(window_size,Y)

# test_File = "../datasets/zoe_spiker/Length3/LLL_z.wav"
# window_size, Y = wavfile.read(test_File)
# print(window_size,Y)

# 2. Using simple classifier to make all the labels

def LR_detection(seq):
    maxval = np.max(seq)
    minval = np.min(seq)
    movement = "L" if maxval < -minval else "R"
    return movement

def LR_detection_max(seq):
    max = 0
    min = 0
    for i in seq:
        if i > 550:
            max += 1
        if i < 450:
            min += 1
    movement = "R" if min > max else "L"
    return movement

def is_nested(my_list):
    for element in my_list:
        if type(element) == list:
            return True
    return False

def eye_movement_ZC(Y, time, windowSize=0.5, thresholdEvents=20, downSampleRate=50):
    Y = Y - 500
    ind = np.arange(0, np.where(time == np.round(time[-1] - windowSize, 4))[0][0] + 1, downSampleRate)
    Y = Y - np.average(Y)
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

def eye_movement_max(Y, time, windowSize=0.5, thresholdEvents=[540,460], downSampleRate=50):
    ind = np.arange(0, np.where(time == np.round(time[-1] - windowSize, 4))[0][0] + 1, downSampleRate)
    if np.ndim(Y) > 1:
        Y = Y[:,1]
    timeMiddle = time[ind] + windowSize / 2
    testStat_max = np.empty(len(ind), dtype = int)
    testStat_min = np.empty(len(ind), dtype = int)
    for i in range(len(ind)):
        Y_subset = Y[(time >= time[ind[i]]) & (time < time[ind[i]] + windowSize)]
        if np.ndim(Y_subset) > 1:
            Y_subset = Y_subset[:, 1]
        testStat_max[i] = max(Y_subset)
        testStat_min[i] = min(Y_subset)
    first_condition = testStat_max < thresholdEvents[1]
    second_condition = testStat_min > thresholdEvents[0]
    predictedEvent = np.where(np.logical_or(first_condition, second_condition))[0]
    movement_list = []

    eventTimes = timeMiddle[predictedEvent]
    gaps = np.where(np.diff(eventTimes) > windowSize)[0]

    event_time_interval = [np.min(eventTimes)]
    for i in range(len(gaps)):
        event_time_interval.extend([eventTimes[gaps[i]], eventTimes[gaps[i] + 1]])
    event_time_interval.append(np.max(eventTimes))
    event_time_interval = np.reshape(event_time_interval, (-1, 2))

    predictedEventTimes = np.full(len(Y), False)
    for i in range(event_time_interval.shape[0]):
        predictedEventTimes[(event_time_interval[i, 0] <= time) & (event_time_interval[i, 1] >= time)] = True

    num_event = len(gaps) + 1
    y_values = []
    for i in range(event_time_interval.shape[0]):
        interval_start, interval_end = event_time_interval[i]
        interval_indices = (time >= interval_start) & (time <= interval_end)
        interval_Y = Y[interval_indices]
        y_values.append(interval_Y)
        movement = LR_detection_max(interval_Y)
        movement_list.append(movement)

    return {
        "num_event": num_event,
        "predictedEventTimes": predictedEventTimes,
        "predictedInterval": event_time_interval,
        "labels": movement_list,
        "signals": y_values
    }

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
        elif method == "max":
            result = eye_movement_max(Y=Y,time = timeSeq)
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

# make a feature matrix for the following classifier
def make_matrix(signals, labels):
    mean_ls = []
    sd_ls = []
    zero_crossing = []
    # entropy = []
    # lumpiness = []
    # flat_spots = []
    for i in range(len(signals)):
        mean = np.mean(signals[i])
        zero_crossing.append(len(np.where(np.diff(np.sign(signals[i])))[0]))
        sd = np.std(signals[i])
        mean_ls.append(mean)
        sd_ls.append(sd)
        #entropy.append(ts_entropy(signals))
        # lumpiness.append(ts_lumpiness(signals))
        # flat_spots.append(ts_flat_spots(signals))
    # , 'Signals': signals}
    dependent_vars = pd.DataFrame({'Mean': mean_ls, 'SD': sd_ls})
    feature_matrix = pd.concat([pd.Series(labels, name='Label'), dependent_vars], axis=1)
    return feature_matrix


def make_matrix_tsfresh(signals, labels):
    df_list = []
    for i, signal in enumerate(signals):
        temp_df = pd.DataFrame({'id': i, 'time': np.arange(len(signal)), 'value': signal.astype(float)})
        df_list.append(temp_df)

    df = pd.concat(df_list, ignore_index=True)
    extracted_features = extract_features(df, column_id='id', column_sort='time',default_fc_parameters=EfficientFCParameters())
    impute(extracted_features)
    extracted_features['Label'] = labels

    return extracted_features


def left_right_classifier_knn(feature_matrix):
    X = feature_matrix.drop('Label', axis=1)
    y = feature_matrix['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, cm, knn, X_test, y_test
def extract_features_new_data(new_files):
    results = record_all_training(files=new_files,method = "zc")
    new_data_matrix = make_matrix_tsfresh(results["ls_signals"], results["ls_labels"])
    return new_data_matrix

def classify_new_data(knn_classifier, new_data_matrix):
    new_X = new_data_matrix.drop('Label', axis=1)
    new_y_pred = knn_classifier.predict(new_X)
    return new_y_pred


def streaming_condition(wave_file, knn_classifier, window_size=None, increment=None):
    print(wave_file)
    sample_rate, Y = wavfile.read(wave_file)

    Y = Y[:, 1] if Y.ndim > 1 else Y
    Y = Y - np.average(Y)
    xtime = np.arange(len(Y)) / sample_rate

    if window_size is None:
        window_size = sample_rate
    if increment is None:
        increment = window_size // 3

    lower_interval = 0
    max_time = np.max(xtime) * sample_rate
    return_str = ""
    while max_time > lower_interval + window_size:


        upper_interval = lower_interval + window_size
        interval = Y[lower_interval:upper_interval]
        mean = np.mean(interval)
        sd = np.std(interval)
        zero_crossing_point = len(np.where(np.diff(np.sign(interval)))[0])
        # entropy = ts_entropy(interval)
        # lumpiness = ts_lumpiness(interval)
        # flat_spots = ts_flat_spots(interval)

        ## check whether there is an event
        if zero_crossing_point < 40:
            temp_df = pd.DataFrame({'id': 0, 'time': np.arange(len(interval)), 'value': interval.astype(float)})
            extracted_features = extract_features(temp_df, column_id='id', column_sort='time',
                                                  default_fc_parameters=EfficientFCParameters())
            impute(extracted_features)
            features = extracted_features

            predicted = knn_classifier.predict(features)

            if predicted[0] == 'L' or predicted[0] == 'R' or predicted[0] == "B":
                return_str += predicted[0]
            lower_interval += increment
        else:
            lower_interval += increment

    return return_str


if __name__ == '__main__':
    # Load the classifier
    path = "../datasets/clean_data/Train_data"
    file_ls = []
    wave_file_ls = os.listdir(path)
    for i in range(len(wave_file_ls)):
        file_path = path + "/" + wave_file_ls[i]
        file_ls.append(file_path)
    results = record_all_training(files = file_ls,method="zc")

    if results != None:
        matrix = make_matrix_tsfresh(results["ls_signals"], results["ls_labels"])
        print(matrix)


    accuracy, cm, knn_classifier, X_test, y_test = left_right_classifier_knn(feature_matrix = matrix)
    #
    knn_classifier = KNeighborsClassifier(n_neighbors=3)
    X_train, X_test, y_train, y_test = train_test_split(matrix.drop('Label', axis=1), matrix['Label'], test_size=0.2,
                                                        random_state=42)
    knn_classifier.fit(X_train, y_train)
    print(accuracy)

    # Load new, unlabeled data
    # path_test = "../datasets/clean_data/Right"
    # test_file_ls = []
    # test_wave_file_ls = os.listdir(path_test)
    # for i in range(len(test_wave_file_ls)):
    #     test_file_path = path_test + "/" + test_wave_file_ls[i]
    #     test_file_ls.append(test_file_path)
    # # new_data_matrix = extract_features_new_data(test_file_ls)
    # predicted_labels = streaming_condition("../datasets/clean_data/Left/clean5_left_1", knn_classifier)
    # print(predicted_labels)



