"""
utils.py
"""

######################
# Import modules
######################

import shutil
#import librosa
#import sounddevice as sd
import argparse
import numpy as np
import io
import time
import csv

######################
# Globals
######################

SAMPLERATE_HZ = 16000 # requirement of YAMNet

######################

def get_parser():
    """ Declares parse arguments
    Returns:
        args : argparse.ArgumentParser().parse_args
    """


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--DURATION_S", default=2, type=int, help="Duration of audio clip in seconds"
    )
    parser.add_argument(
        "--TOP_K", default=1, type=int, help="Number of top predictions to store"
    )
    parser.add_argument(
        "--MODE", default='a', type=str, help="Either a or b to pick mode"
    )
    parser.add_argument(
        "--WATCH_SOUNDS", nargs='+', type=str, help="List of sounds to watch for"
    )
    parser.add_argument(
        "--INTERVAL", default=10, type=int, help="Time interval for inferencing in seconds"
    )
    parser.add_argument(
        "--SAMPLING_INTERVAL", default=-1, type=int, help="Sample audio every i-th inferencing"
    )
    parser.add_argument(
        "--ITERATIONS", default=1, type=int, help="Number of inferencing iterations in this run"
    )
    parser.add_argument(
        "--INPUT_FILE", default='', type=str, help="Path to the input file, if provided (default: none)"
    )
    parser.add_argument(
        "--PUBLISH", action='store_true', help="Publish the classifications made by yamned (default: false)"
    )
    
    args = parser.parse_args()

    assert args.MODE == 'a' or args.MODE == 'b', "Invalid mode passed, must be either a or b"

    return args
