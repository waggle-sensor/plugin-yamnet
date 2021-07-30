
"""
main.py
Example: python3 main.py --DURATION_S 10 --TOP_K 5 --MIC_PATH dummy_path
"""

######################
# Import waggle modules
######################

from waggle import plugin
from waggle.data import AudioFolder, Microphone
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
    model_interface = YAMNetInterface(args)

    #####################

    # Init plugin
    plugin.init()
    microphone = Microphone(samplerate=SAMPLERATE_HZ)

    while True:
        sample = microphone.record(duration=args.DURATION_S)
        yh_k, yh_conf = model_interface.predict(sample)

         # Publish to plugin
        if args.MODE == 'a':
            for i in range(args.TOP_K):
                plugin.publish("rank."+str(i+1)+".class", yh_k[i])
                plugin.publish("rank."+str(i+1)+".prob", yh_conf[i])
        elif args.MODE == 'b':
            matched_sounds = list(set(args.WATCH_SOUNDS).intersection(set(yh_k)))
            plugin.publish(matched_sounds)


######################

if __name__ == '__main__':
    main()
