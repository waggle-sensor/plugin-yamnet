
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
from PIL import Image

import numpy as np
import tensorflow as tf
import io
import csv

######################
# Globals
######################

SAMPLERATE_HZ = 16000 # requirement of YAMNet

######################
# Utils
#####################

# Find the name of the class with the top score when mean-aggregated across frames.
def class_names_from_csv(class_map_csv_text):
      """Returns list of class names corresponding to score vector."""
      class_map_csv = io.StringIO(class_map_csv_text)
      class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
      class_names = class_names[1:]  # Skip CSV header
      return class_names

def getAudioModel(model_path='lite-model_yamnet_tflite_1.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    input_details = interpreter.get_input_details()
    waveform_input_index = input_details[0]['index']
    output_details = interpreter.get_output_details()
    scores_output_index = output_details[0]['index']
    embeddings_output_index = output_details[1]['index']
    spectrogram_output_index = output_details[2]['index']

    return interpreter, waveform_input_index, scores_output_index, embeddings_output_index, spectrogram_output_index

def getTopK(yh,class_names,k=1):
    yh = yh.mean(axis=0)
    yh_max_id  = yh.argsort()[-k:][::-1]
    return [class_names[k] for k in yh_max_id], [yh[k] for k in yh_max_id]

def predictModel(interpreter, waveform_input_index, scores_output_index, embeddings_output_index, spectrogram_output_index, y):
    interpreter.resize_tensor_input(waveform_input_index, [len(y)], strict=True)
    interpreter.allocate_tensors()
    interpreter.set_tensor(waveform_input_index, y)
    interpreter.invoke()
    scores, embeddings, spectrogram = (
        interpreter.get_tensor(scores_output_index),
        interpreter.get_tensor(embeddings_output_index),
        interpreter.get_tensor(spectrogram_output_index))
    return scores, embeddings, spectrogram

######################
# Main
#####################

def main():
    # Get parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("--DURATION_S", default=10, type=int, help="Duration of audio clip in seconds")
    parser.add_argument("--TOP_K", default=3, type=int, help="Number of top predictions to store")
    args = parser.parse_args()

    # Declare logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    logging.info("starting plugin. sample rate of audio is {} with duration of {} seconds".format(args.DURATION_S, SAMPLERATE_HZ))

    #####################

    # Load model
    interpreter, waveform_input_index, scores_output_index, embeddings_output_index, spectrogram_output_index = getAudioModel()

    # Load class names
    class_names = class_names_from_csv(open('yamnet_class_map.csv').read())

    # Get sample with open_data_source
    y, _ = librosa.load("street_music_sample.wav",SAMPLERATE_HZ)

    # Scale between -1 and 1
    y= 2.*(y - np.min(y))/np.ptp(y)-1

    # Make a prediction
    scores, embeddings, spectrogram = predictModel(interpreter, waveform_input_index, scores_output_index, embeddings_output_index, spectrogram_output_index, y)
    yh_k, yh_conf = getTopK(scores,class_names,args.TOP_K)

    for i,k in enumerate(yh_k):
        print("Rank: {} | Class: {} | Score: {:.4f}".format(i+1, k, yh_conf[i]))

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


if __name__ == '__main__':
    main()
