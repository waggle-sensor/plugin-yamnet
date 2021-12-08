# Science

The task of identifying onsets and offsets of target class activities in general audio signals is called Sound event detection (SED) [1].
The most common SED application is one in which the method takes as an input an audio signal, and outputs temporal activity for target classes like “car passing by”, “footsteps”, “people talking”, “gunshot”, etc [1, 2].
The time resolution of the activity of classes can vary among different methods and datasets, but typically is used 0.02 sec [3].
Furthermore, activities of classes can overlap (polyphonic SED) or not (monophonic SED).
The typical aplications in which SED can be employed are, for instance, wildlife monitoring and bird activity detection, home monitoring, autonomous vehicles, and surveillance, among others.


# AI@Edge

We are using a DNN, called **YAMNet**. [YAMNet](https://www.tensorflow.org/hub/tutorials/yamnet) is based on the VGG architecture, using depthwise separable convolutions. The amount of parameters of the YAMNET is 3.7M [3]. 
YAMNet predicts 521 audio event [classes](https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv) from the AudioSet-YouTube corpus which it was trained on. It employs the [Mobilenet v1](https://arxiv.org/pdf/1704.04861.pdf) depthwise-separable convolution architecture.

This plugin can be used to either evaluate all data through an audio stream or be set to scan for particular sounds. Due to the licensing on the AudioSet dataset, the raw training data is unavailable. However, the model provided outputs class scores, embedding vectors, and spectrograms of the input data.  For more specific audio classification tasks, this model can be used for transfer learning, and the outputted feature embeddings can be used to train another model. 



# Using the code

### A: Prediction stream for top-k classes
This mode runs "n" iterations and logs predictions for top-k classes in each iteration.

## This plugin has two modes

```python
#python3 main.py --DURATION_S 5 --TOP_K 3 --MIC_PATH dummy_path --MODE a --ITERATIONS 10
args = get_parser()
model_interface = YAMNetInterface(args)
```

### B: Scan for sounds in top-k classes
This mode runs "n" iterations and only logs if within top-k predicted classes are declared sound is matched in each iteration.

```python
#python3 main.py --DURATION_S 10 --TOP_K 5 --MIC_PATH dummy_path --MODE b WATCH_SOUNDS Music --ITERATIONS 10
args = get_parser()
model_interface = YAMNetInterface(args)
```

# Arguments

* **DURATION_S** (default = 2): length of audio clips in seconds
* **TOP_K** (default = 3): number of argmax predictions to record
* **MIC_PATH** (default = None): path to mic usesd for plugin
* **MODE** (default = a): either mode a or b, refer below
* **WATCH_SOUNDS** (default = None): list of classes to watch for, must be in mode b, seperate each term by a comma
* **INTERVAL** (default = 300): number os seconds of silence after each iteration
* **ITERATIONS** (default = 1): number of inference iterations on the run




# References

[1] E. C¸ akir, G. Parascandolo, T. Heittola, H. Huttunen, and T. Virtanen, "Convolutional recurrent neural networks for polyphonic sound event detection," IEEE/ACM Transactions on Audio, Speech and Language Processing, vol. 25, no. 6, pp. 1291–1303, Jun. 2017.

[2] K. Drossos, S. Gharib, P. Magron, and T. Virtanen, "Language modelling for sound event detection with teacher forcing and scheduled sampling," in Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), Oct. 2019.

[3]Konstantinos Drossos, Stylianos I. Mimilakis, Shayan Gharib, Yanxiong Li, Tuomas Virtanen, "Sound Event Detection with Depthwise Separable and Dilated Convolutions." IEEE World Congress on Computational Intelligence (WCCI) / International Joint Conference on Neural Networks (IJCNN), 2020. 



# Credits

- Image credit:
  * Wikimedia Commons License



