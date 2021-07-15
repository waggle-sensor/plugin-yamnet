
######################
# Import main modules
######################

import unittest
import librosa
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf

from main import getAudioModel, getTopK, LogMelSpectMesh, predictModel

######################
# Globals
######################

CLASS_LABELS = ['air conditioner','car horn','children playing',\
                'dog barking','drilling', 'engine','gun shot',\
               'jackhammer','siren','street music']

######################
# Unit test
######################

class MyTestCase(unittest.TestCase):
    def test_audio_func(self):
        data = AudioFolder('test_path')
        for sample in data:
            sample.data

    def test_something(self):
        self.assertTrue(1 == 1)

if __name__ == "__main__":
    unittest.main()
