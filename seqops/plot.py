import numpy as np
import matplotlib.pyplot as plt

def plot_sequence(sequence, seqIdx, filename):
    
    assert(len(sequence.shape) == 5)
    assert(seqIdx >=0)
    
    assert(sequence.shape[0] > seqIdx)
    assert(sequence.shape[1] > 0)
    assert(sequence.shape[2] == 28)
    assert(sequence.shape[3] == 28)
    assert(sequence.shape[4] == 1)
    
    operation = np.array([])
    
    stripLength = sequence.shape[1]
    
    for elemIdx in range(stripLength):
        elem = sequence[seqIdx, elemIdx, :, :, 0]
        b = elem.min()
        elem = elem + b
        w = elem.max()
        elem = elem / w
        elem = 255 * elem
        if operation.size == 0:
            operation = elem
        else:
            operation = np.hstack((operation, elem))
    
    plt.imshow(255 - operation, cmap='gray')
    plt.savefig(filename)

def parse_performance(data_fname):
    
    data = np.loadtxt(fname=data_fname, dtype="string")
    
    assert(len(data.shape) == 2)
    assert(data.shape[0] > 0)
    assert(data.shape[1] == 3)
    
    performance = np.zeros((data.shape[0], 2))
    
    for rowIdx in range(data.shape[0]):
        # Minibatch error
        performance[rowIdx, 1] = float(data[rowIdx, 1].split("=")[1])
        # Training accuracy
        performance[rowIdx, 2] = float(data[rowIdx, 2].split("=")[1])
    
    return performance

def plot_performance(minibatch_error_plot_fname, training_accuracy_fname):

    plt.plot(performance[:, 0], performance[:, 1])
    plt.xlabel("Iteration")
    plt.ylabel("Minibatch error")
    plt.savefig(minibatch_error_plot_fname)
    plt.clf()

    plt.plot(performance[:, 0], performance[:, 2])
    plt.xlabel("Iteration")
    plt.ylabel("Training accuracy")

    plt.savefig(training_accuracy_fname)
    plt.clf()
