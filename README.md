# Sequential Operations: RNN used to learn how to perform arithemtic operations

## Requirements
Make sure you have installed TensorFlow and Matplotlib on Python.

## Data
There are three datasets used by this program MNIST_DATA, SYMBOLS_DATA and CONVNET_DATA. The first data set comes with TensorFlow, the second one is already present in this repository, and the third one (the pre-trained convolutional network used to identify digits) can be downloaded from https://www.dropbox.com/s/xdayqobr27o90if/convnet.pkl?dl=0.

## Execution
The bash script `run` implements the logic on how to execute the agent. It holds all the possible arguments that can be passed to the script. After setting up all the parameters as neede, simply run:

```
chmod +x run
./run > stdout 2> stderr &
```

This will execute the Python script `run.py` and store the standard output and the standard error on the stdout and stderr files correspondingly.
