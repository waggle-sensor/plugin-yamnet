"""
python3 test.py
"""

######################
# Import main modules
######################

import unittest
import argparse
import logging
import time

import librosa
import sounddevice as sd

import numpy as np
import tensorflow as tf
import io
import csv

from utils import *
from model_interface import YAMNetInterface

######################
# Globals
######################

SAMPLERATE_HZ = 16000 # requirement of YAMNet

######################
# Unit test
######################

class MyTestCase(unittest.TestCase):
    def test_yamnet_sample(self):
        ''' Test a single prediction '''
        print("Single prediction test")

        # Get parse args
        args = get_parser()

        # Declare model interface
        model_interface = YAMNetInterface(args)

        # Get sample with open_data_source
        y, _ = librosa.load("sample_data/street_music_sample.wav",SAMPLERATE_HZ)

        # Make a prediction
        yh_k, yh_conf = model_interface.predict(y)

        for i in range(args.TOP_K):
            print("rank."+str(i+1)+".class", yh_k[i])
            print("rank."+str(i+1)+".prob", yh_conf[i])

        # Check outputs
        self.assertEqual(yh_k[0],'Music')
        self.assertEqual(yh_k[1],'Hip hop music')
        self.assertEqual(yh_k[2],'Electronic music')

    def test_local_mode_a(self):
        ''' Test for mode a done with local mic '''
        print("Local live stream test of mode a")

        # Get parse args
        args = get_parser()

        # Declare model interface
        model_interface = YAMNetInterface(args)

        def predict_sound(indata,frames,time,status):
            indata = np.squeeze(indata,axis=-1)
            yh_k, yh_conf = model_interface.predict(indata)
            print(yh_k[0], yh_conf[0])
        start_time = time.time()
        with sd.InputStream(samplerate=SAMPLERATE_HZ,
                            dtype='float32',
                            blocksize=SAMPLERATE_HZ*2,
                            callback=predict_sound):
            sd.sleep(10000)
        print("--- %s seconds ---" % (time.time() - start_time))


    def test_local_mode_b(self):
        ''' Test for mode b done with a local mic '''
        print("Local live stream mode b")

        # Get parse args
        args = get_parser()
        args.DURATION_S = 5
        args.MODE = 'b'
        args.WATCH_SOUNDS = ['Music','Ice cream truck, ice cream van']
        args.TOP_K = 30

        # Declare model interface
        model_interface = YAMNetInterface(args)

        def predict_sound(indata,frames,time,status):
            indata = np.squeeze(indata,axis=-1)
            yh_k, yh_conf = model_interface.predict(indata)
            matched_sounds = list(set(args.WATCH_SOUNDS).intersection(set(yh_k)))
            print(matched_sounds)

        start_time = time.time()
        with sd.InputStream(samplerate=SAMPLERATE_HZ,
                            dtype='float32',
                            blocksize=SAMPLERATE_HZ*2,
                            callback=predict_sound):
            sd.sleep(30000)
        print("--- %s seconds ---" % (time.time() - start_time))



######################

if __name__ == "__main__":
    unittest.main()
