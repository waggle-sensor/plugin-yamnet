
"""
main.py
Example: python3 main.py --DURATION_S 10 --TOP_K 5
"""

"""

from waggle.data import AudioFolder, Microphone

def main():
    # can now read test audio data from a folder for testing
    dataset = AudioFolder("audio_test_data")
    for sample in dataset:
        print(sample.data)
        print(sample.samplerate)
    # can now record audio data from a microphone
    microphone = Microphone(samplerate=22050)
    sample = microphone.record(duration=5)
    print(sample.data)


"""

######################
# Import waggle modules
######################

#from waggle import plugin
#from waggle.data import AudioFolder, Microphone
import argparse
import logging
import time

######################
# Import main modules
######################

import librosa
import sounddevice as sd

import numpy as np
import tensorflow as tf
import io
import csv

from model_interface import * 
from utils import *

######################
# Globals
######################

SAMPLERATE_HZ = 16000 # requirement of YAMNet

######################
# Main
#####################

def main():
    # Get parse args
    args = get_parser()

    # Declare logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')
    logging.info("starting plugin. sample rate of audio is {} with duration of {} seconds".format(SAMPLERATE_HZ,args.DURATION_S))

    # Declare model interface
    model_interface = YAMNetInterface(args.TOP_K,args.DURATION_S)


    #####################

    # Init plugin
    #plugin.init()
    #microphone = Microphone(samplerate=SAMPLERATE_HZ)

    #while True:
    #    sample = microphone.record(duration=args.DURATION_S)
    #    scores, embeddings, spectrogram = predictModel(interpreter, waveform_input_index, scores_output_index, embeddings_output_index, spectrogram_output_index, sample)
    #    yh_k, yh_conf = getTopK(scores,class_names,args.TOP_K)

         # Publish to plugin
    #    for i in range(args.TOP_K):
    #        plugin.publish("rank."+str(i+1)+".class", yh_k[i])
    #        plugin.publish("rank."+str(i+1)+".prob", yh_conf[i])


######################

if __name__ == '__main__':
    main()
