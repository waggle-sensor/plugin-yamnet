"""
utils.py
"""

######################
# Import modules
######################

import shutil
import gdown
import librosa
import sounddevice as sd
import argparse
import numpy as np
import tensorflow as tf
import io
import time
import csv

######################
# Globals
######################

SAMPLERATE_HZ = 16000 # requirement of YAMNet

######################

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DURATION_S", default=10, type=int, help="Duration of audio clip in seconds"
    )
    parser.add_argument(
        "--TOP_K", default=3, type=int, help="Number of top predictions to store"
    )
    return parser.parse_args()
