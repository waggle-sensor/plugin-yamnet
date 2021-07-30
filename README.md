# yamnet_plugin
SAGE plugin for YAMNet audio classification model 

## Plugin meta-data
The needed meta-data (class names, model weights, and sample data) can be downloaded [here](https://drive.google.com/drive/folders/1J2WtW7TWzq_4uv8UQ8Tl8ZVfFmC1TLFb?usp=sharing).


## Modes

### A: Prediction stream for top-k classes
This mode constantly logs predictions for top-k classes.

### B: Scan for sounds in top-k classes
This mode only logs if within top-k predicted classes are declared sound is matched.

