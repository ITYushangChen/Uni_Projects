import pyautogui
import pygame
import pickle
import pandas as pd
from scipy.io import wavfile
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters,MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
import numpy as np


def classifier(wave_file):
    """
    :param wave_file: The relative path of the wave file
    :return: action_ls: A list of String indicating either eye movement or blink
    """
    action_ls = {}
    return action_ls


def predict(signal, classifier):
    df = pd.DataFrame({'id': 0, 'time': np.arange(len(signal)), 'value': signal.astype(float)})
    extracted_features = extract_features(df, column_id='id', column_sort='time', default_fc_parameters=MinimalFCParameters())
    impute(extracted_features)
    predicted_class = classifier.predict(extracted_features)
    return predicted_class



Left_Eye_Move = False
Right_Eye_Move= False
Blink_Blink= False

# Status = False indicates the program pauses for user to read.
# Otherwise, True if user wants to go left/right/zoomin/out
status = False


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


def zoomIn():
    pyautogui.hotkey('command', '+')


def zoomOut():
    pyautogui.hotkey('command', '-')
def play_music(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)  # This line helps to wait for the music to finish playing


classifier_filenames = [
    "KNN_classifier.pkl",
    "Random_Forest_classifier.pkl",
    "SVM_classifier.pkl",
    "Logistic_Regression_classifier.pkl",
    "Decision_Tree_classifier.pkl",
    "Gradient_Boosting_classifier.pkl",
]
def load_classifier(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def load_all_classifiers(filenames):
    classifiers = {}
    for filename in filenames:
        classifier_name = filename[:-4].replace("_", " ")
        classifiers[classifier_name] = load_classifier(filename)
        print(f"{classifier_name} classifier loaded.")
    return classifiers
if __name__ == '__main__':
    print("!")
    # open = '../sound/open.wav'
    # pause = '../sound/pause.ogg'
    # play_music(open)
    # play_music(pause)
    # try:
    #     while True:
    #         #
    #         if Left_Eye_Move:
    #             goLeft()
    #         if Right_Eye_Move:
    #             goRight()
    #         if Blink_Blink:
    #             status = On_Off(status)
    #
    #         # zoomIn()
    #         # zoomOut()
    #
    # except KeyboardInterrupt:
    #     print("Exit.....")
    window_size, Y = wavfile.read("../datasets/zoe_spiker/Length8/LLRLRLRL_z.wav")
    classifiers = load_all_classifiers(classifier_filenames)
    classifier = classifiers["Random Forest classifier"]
    predcit_class = predict(Y, classifier)
    print("Prediction: %s" %(predcit_class))


