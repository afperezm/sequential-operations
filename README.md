# Sequential Operations: RNN used to learn how to perform arithemtic operations

## Requirements
Make sure you have installed TensorFlow and Matplotlib on Python.

## Data
There are three sets of data used by this program MNIST_DATA, SYMBOLS_DATA and CONVNET_DATA. The first set comes with TensorFlow, the second one is already present in this repository, and the third one (the pre-trained convolutional network used to identify digits) can be downloaded from Dropbox with the following URL:

```
https://www.dropbox.com/s/xdayqobr27o90if/convnet.pkl
```

## Execution
The bash script `run` on the `scripts` folder executes the Python script `run.py` which loads data, creates, trains, and tests the agent. The bash script holds all the possible arguments that can be passed to the script.

After setting up as needed all the parameters on the `run` script, make sure the sources folder `sequential-operations` is at the same level as the `run` script, and simply execute the following two commands:

```
chmod +x run
./run > stdout 2> stderr &
```

This will execute the Python script `run.py` and store the standard output and the standard error on the `stdout` and `stderr` files correspondingly.

## Plotting performance
This project includes the function `plot_performance` in the Python script `seqops/plot.py` which can be used to plot agent's performance in terms of minibatch error and training accuracy. The input for the `plot_performance` function is the standard output of the `run.py` script.

Here is a snippet of code showing how to use the `plot_performance` function:

```
from seqops.plot import plot_performance
plot_performance("stdout", "minibatch_error.png", "training_accuracy.png")
```

## Plotting generated sequences
This project includes the function `plot_sequence` in the Python script `seqops/plot.py` which can be used to plot a generated sequence.

First we load digits and symbols data:

```
from seqops.data import load_data

train_images_file="./SYMBOLS_DATA/DATA_TRAIN_IMAGES.npy"
train_labels_file="./SYMBOLS_DATA/DATA_TRAIN_LABELS.npy"

digits, digit_labels, symbols, symbol_labels = load_data(train_images_file, train_labels_file)
```

Then we generate a random sequence:

```
from seqops.data import generate_sequence

sequence, result, operands = generate_sequence(digits, digit_labels, symbols, symbol_labels, 1, 1, 1)
```

Print operands and result:

```
import numpy as np
np.hstack((operands, result))
>>> array([[  3.,   0.,   2.,   5.]])
```

And finally we print the generated sequence to a PNG iamge file:

```
from seqops.plot import plot_sequence
plot_sequence(sequence, 0, "sequence.png")
```
