import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from random import choice
from random import randrange

def load_data(train_images_file, train_labels_file):
    
    mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)
    
    symbols = np.load(train_images_file)
    symbol_labels = np.load(train_labels_file)
    
    digits = mnist.train.images
    digit_labels = mnist.train.labels
    
    # Find indices of symbols labeled as addition or subtraction
    idx = np.sum(symbol_labels[:,0:2], axis=1)
    
    # Obtain labels of symbols labeled as addition or subtraction
    add_sub_symbol_labels = symbol_labels[idx>0, 0:2]
    
    # Obtain symbols labeled as addition or subtraction
    add_sub_symbols = symbols[idx>0, :, :]
    
    return [digits, digit_labels, add_sub_symbols, add_sub_symbol_labels]

def generate_sequence(Dg, Et, Sm, Op, sequenceLength, operandOneLength, operandTwoLength):
    """
    Generates a random sequence of 'sequenceLength' triplets of operands and operators.
    """
    
    # Initialize return variables
    sequence = np.zeros((sequenceLength, operandOneLength + 1 + operandTwoLength, Sm.shape[1], Sm.shape[2], 1))
    result = np.zeros((sequenceLength, 1))
    operands = np.zeros((sequenceLength, 3))
    
    non_zero_indices = [i for i in range(Et.shape[0]) if Et[i, 0] == 0]
    
    for k in range(sequenceLength):
        
        # Generate random operator index
        operatorIdx = randrange(0, Sm.shape[0])
        # Store operator image
        operator = Sm[operatorIdx,:,:]
        sequence[k, operandOneLength, :, :, 0] = operator
        # Find operator
        y = np.argmax(Op[operatorIdx])
        
        x1 = 0
        
        for digitIdx in range(operandOneLength):
            # Generate random first operand index
            if operandOneLength > 1 and digitIdx == 0:
              j1 = choice(non_zero_indices)
            else:
              j1 = randrange(0, Dg.shape[0])
            # Reshape first operand image
            digitImg = np.reshape(Dg[j1,:], [Sm.shape[1], Sm.shape[2]])
            # Store first operand image
            sequence[k, digitIdx, :, :, 0] = digitImg
            # Find first operand
            x1 += np.argmax(Et[j1]) * pow(10, operandOneLength - 1 - digitIdx)
        
        x2 = 0
        
        for digitIdx in range(operandTwoLength):
            # Generate random second operand index
            if operandTwoLength > 1 and digitIdx == 0:
              j2 = choice(non_zero_indices)
            else:
              j2 = randrange(0, Dg.shape[0])
            # Reshape second operand image
            digitImg = np.reshape(Dg[j2,:], [Sm.shape[1], Sm.shape[2]])
            # Store second operand image
            sequence[k, operandOneLength + 1 + digitIdx, :, :, 0] = digitImg
            # Find second operand
            x2 += np.argmax(Et[j2]) * pow(10, operandTwoLength - 1 - digitIdx)
        
        # Compute operation result
        if (y == 0):
            # addition
            result[k] = x1 + x2
        elif (y == 1):
            # subtraction
            result[k] = x1 - x2
        
        operands[k,0] = x1
        operands[k,1] = y
        operands[k,2] = x2
        
    return sequence, result, operands

