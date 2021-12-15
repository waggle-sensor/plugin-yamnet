
"""
main.py
Example: python3 main.py --DURATION_S 10 --TOP_K 5
"""

######################
# Import waggle modules
######################

from waggle import plugin
from waggle.data.audio import AudioFolder, Microphone
import argparse
import logging
import time

######################
# Import main modules
######################

#import librosa
#import sounddevice as sd

import numpy as np
#import tensorflow as tf
import io
import csv

from model_interface import *
from utils import *

import wave

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
    if not args.INPUT_FILE:
        logging.info("this run will record from microphone")
        microphone = Microphone(samplerate=SAMPLERATE_HZ)

    sampling_countdown = -1
    if args.SAMPLING_INTERVAL >= 0 and not args.INPUT_FILE:
        logging.info("sampling enabled -- occurs every %sth inferencing", args.SAMPLING_INTERVAL)
        sampling_countdown = args.SAMPLING_INTERVAL

    counter = 0
    while counter < args.ITERATIONS:
        do_sampling = False
        if sampling_countdown > 0:
            sampling_countdown -= 1
        elif sampling_countdown == 0:
            do_sampling = True
            sampling_countdown = args.SAMPLING_INTERVAL

        logging.info(f'Sampling...')
        # NOTE: PyWaggle's Microphone is (SAMPLINGRATE_HZ*DURATION, 1) matrix, but the model
        #       requires a vector. Squeezing the sample makes it work
        if args.INPUT_FILE:
            my_file = wave.open(args.INPUT_FILE, 'rb')
            data = np.array(list(my_file.readframes(my_file.getnframes())), dtype="float32")
        else:
            sample = microphone.record(duration=args.DURATION_S)
            data = np.squeeze(sample.data)

        logging.info(f'Inferencing...')
        yh_k, yh_conf = model_interface.predict(data)

         # Publish to plugin
        if args.MODE == 'a':
            for i in range(args.TOP_K):
                class_name = yh_k[i].replace(' ', '').replace(',', '').replace('(','').replace(')','').lower()
                class_confidence = float(yh_conf[i])
                if args.PUBLISH:
                    logging.info(f'Publishing...')
                    if not args.INPUT_FILE:
                        plugin.publish(f'env.detection.sound.{class_name}.prob', class_confidence, timestamp=sample.timestamp)
                    else:
                        plugin.publish(f'env.detection.sound.{class_name}.prob', class_confidence)

                logging.info(f'env.detection.sound.{class_name}.prob: {class_confidence}')
        elif args.MODE == 'b':
            for k, conf in zip(yh_k, yh_conf):
                if k in args.WATCH_SOUNDS:
                    class_name = k.replace(' ', '').replace(',', '').replace('(','').replace(')','').lower()
                    class_confidence = float(conf)
                    if args.PUBLISH:
                        logging.info(f'Publishing...')
                        if not args.INPUT_FILE:
                            plugin.publish(f'env.detection.sound.{class_name}.prob', class_confidence, timestamp=sample.timestamp)
                        else:
                            plugin.publish(f'env.detection.sound.{class_name}.prob', class_confidence)


                    logging.info(f'env.detection.sound.{class_name}.prob: {class_confidence}')
        
        if do_sampling:
            # NOTE: PyWaggle 0.46.3 does not support mp3 as file extension << mp3 is not a free licensed software!
            sample.save(f'sample.flac')
            plugin.upload_file(f'sample.flac')
            logging.info("uploaded sample")

        time.sleep(args.INTERVAL)
        counter += 1

######################

if __name__ == '__main__':
    main()
