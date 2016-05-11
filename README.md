# Sequential Operations: RNN used to learn how to perform arithemtic operations

## Requirements
Make sure you have installed TensorFlow and Matplotlib on Python.

## Data
There are three datasets used by this program MNIST_DATA, SYMBOLS_DATA and CONVNET_DATA. The first data set comes with TensorFlow, the second one is already present in this repository, and the third one (the pre-trained convolutional network used to identify digits) can be downloaded with the following command:

```wget https://www.dropbox.com/s/xdayqobr27o90if/convnet.pkl.```

## Execution
The bash script `run` on the `scripts` folder executes the Python script `run.py` which loads data, creates, trains, and tests the agent. The basch script holds all the possible arguments that can be passed to the script.

After setting up as needed all the parameters on the `run` script, make sure the sources folder `sequential-operations` is at the same level as the `run` script, and simply execute the following two commands:

```
chmod +x run
./run > stdout 2> stderr &
```

This will execute the Python script `run.py` and store the standard output and the standard error on the `stdout` and `stderr` files correspondingly.
