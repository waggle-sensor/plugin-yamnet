"""
model_interface.py
"""

######################
# Import modules
######################

#import librosa
#import sounddevice as sd
import argparse
import numpy as np
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import io
import time
import csv

from utils import *

######################
# YAMNetInterface
######################

class YAMNetInterface():
    """
    A class to interface with the YAMNet tflite model

    ...

    Attributes
    ----------
    TOP_K : int
        Number of top k predictions to track
    DURATION_S : int
        Length in seconds of audio clips
    MODE : str
        Two operating modes, a or b
    WATCH_SOUNDS : [str]
        List of sounds to watch for if in mode b
    MODEL_PATH : str
        Path for YAMNet tflite model
    SAMPLERATE_HZ: int
        Sample rate of audio, must be 16000 to work with YAMNet

    Methods
    -------
    predict(y)
        Takes np.array of sound wave and predicts a vector of softmax scores over classes
    scale_data_yamnet(y)
        Scales data between [-1,1], requirement of YAMNet
    predictMode(y)
        Wrapper function for prediction
    getTopK(yh)
        Takes predictions and return top-k classes
    load_class_names()
        Loads list of classes model was trained on and can predict
    getAudioModel()
        Utility to declare tflite model of YAMNet
    """

    def __init__(self,args):
        self.TOP_K = args.TOP_K
        self.DURATION_S = args.DURATION_S
        self.MODE = args.MODE
        self.WATCH_SOUNDS = args.WATCH_SOUNDS
        self.MODEL_PATH = "model_data/lite-model_yamnet_tflite_1.tflite"
        self.SAMPLERATE_HZ = 16000 # requirement of YAMNet

        # Load model
        self.interpreter, self.waveform_input_index, self.scores_output_index, self.embeddings_output_index, self.spectrogram_output_index = self.getAudioModel()

        # Load class names
        self.class_names = self.load_class_names()

    def predict(self,y):
        """ Perform prediction with YAMNet tflite model
        Args:
            y (np.array): sound wave in form of np.array with shape (length,)

        Returns:
            yh_k (list): top k classes from predictions
            yh_conf (list): corresponding softmax values for top predictions
        """
        # Scale between -1 and 1
        y = self.scale_data_yamnet(y)

        # Make a prediction
        scores, embeddings, spectrogram = self.predictModel(y)
        yh_k, yh_conf = self.getTopK(scores)
        return yh_k, yh_conf

    @staticmethod
    def scale_data_yamnet(y):
        """ Perform prediction with YAMNet tflite model
        Args:
            y (np.array): sound wave in form of np.array with shape (length,)

        Returns:
            y (np.array): sound wave scaled between [-1,1]
        """
        return 2.0 * (y - np.min(y)) / np.ptp(y) - 1

    def predictModel(self, y):
        """ Perform prediction with YAMNet tflite model
        Args:
            y (np.array): sound wave in form of np.array with shape (length,)

        Returns:
            scores (np.array): softmax values for all classes of YAMNet for given data
            embeddings (np.array): feature embedding of input data
            spectrogram (np.array): a spectrogram of the input
        """
        self.interpreter.resize_tensor_input(self.waveform_input_index, [len(y)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, y)
        self.interpreter.invoke()
        scores, embeddings, spectrogram = (
            self.interpreter.get_tensor(self.scores_output_index),
            self.interpreter.get_tensor(self.embeddings_output_index),
            self.interpreter.get_tensor(self.spectrogram_output_index),
        )
        return scores, embeddings, spectrogram

    def getTopK(self, yh):
        """ Perform prediction with YAMNet tflite model
        Args:
            yh (np.array): scores of prediction

        Returns:
            (list): top k class predictions from the input
        """
        yh = yh.mean(axis=0)
        yh_max_id = yh.argsort()[-self.TOP_K:][::-1]
        return [self.class_names[k] for k in yh_max_id], [yh[k] for k in yh_max_id]

    def load_class_names(self):
        """Returns list of class names corresponding to score vector."""
        with open("model_data/yamnet_class_map.csv", "r") as file:
            class_map_csv_text = file.read()
        class_map_csv = io.StringIO(class_map_csv_text)
        class_names = [
            display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)
        ]
        class_names = class_names[1:]  # Skip CSV header
        return class_names

    def getAudioModel(self):
        """ Returns tflite model of YAMNet and dependencies """
        interpreter = tflite.Interpreter(model_path=self.MODEL_PATH)
        input_details = interpreter.get_input_details()
        waveform_input_index = input_details[0]["index"]
        output_details = interpreter.get_output_details()
        scores_output_index = output_details[0]["index"]
        embeddings_output_index = output_details[1]["index"]
        spectrogram_output_index = output_details[2]["index"]
        return (
            interpreter,
            waveform_input_index,
            scores_output_index,
            embeddings_output_index,
            spectrogram_output_index,
        )
