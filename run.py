import argparse
from seqops.data import generate_sequence
from seqops.data import load_data
from seqops.learner import RecurrentNeuralLearner
from seqops.trainer import AgentTrainer

## In[0]: Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("train_images_file", help="training images file in NumPy standard binary file format")
parser.add_argument("train_labels_file", help="training labels file in NumPy standard binary file format")
parser.add_argument("convnet_file", help="convolutional network file in packing list file format")
parser.add_argument("learning_rate", help="learning rate", type=float)
parser.add_argument("batch_size", help="batch size", type=int)
parser.add_argument("n_input", help="output of the convnet and input to the RNN", type=int)
parser.add_argument("n_hidden", help="number of features in the hidden layer", type=int)
parser.add_argument("n_first_digit_length", help="length of the first operand", type=int)
parser.add_argument("n_second_digit_length", help="length of the second operand", type=int)
parser.add_argument("training_iters", help="number of training iterations", type=int)
parser.add_argument("display_step", help="how often must be shown training results", type=int)
parser.add_argument("test_images_file", help="testing images file in NumPy standard binary file format")
parser.add_argument("test_labels_file", help="testing labels file in NumPy standard binary file format")
args = parser.parse_args()

## In[1]: Load training data

#train_images_file = "./SYMBOLS_DATA/DATA_TRAIN_IMAGES.npy"
train_images_file = args.train_images_file
#train_labels_file = "./SYMBOLS_DATA/DATA_TRAIN_LABELS.npy"
train_labels_file = args.train_labels_file

print "- Loading training data"

digits, digit_labels, symbols, symbol_labels = load_data(train_images_file, train_labels_file)

print "  Finished"

sequence, result, operands = generate_sequence(digits, digit_labels, symbols, symbol_labels,  20, 1, 1)

print "- Size of sequence tensor:", sequence.shape

## In[2]: Create agent

#convnet_file = "./CONVNET_DATA/convnet.pkl"
convnet_file = args.convnet_file
# learning_rate = 0.001
learning_rate = args.learning_rate
#batch_size = 64
batch_size = args.batch_size
#n_input = 1024
n_input = args.n_input
#n_hidden = 512
n_hidden = args.n_hidden
#n_first_digit_length = 1
n_first_digit_length = args.n_first_digit_length
#n_second_digit_length = 1
n_second_digit_length = args.n_second_digit_length
n_steps = n_first_digit_length + 1 + n_second_digit_length

assert(n_steps <= 11)

print "- Creating agent"

agent = RecurrentNeuralLearner(convnet_file, learning_rate, batch_size, n_input, n_hidden, n_steps)
agent.createNetwork()

print "  Finished"

## In[3]: Train agent

#training_iters = 10000
training_iters = args.training_iters
#display_step = 500
display_step = args.display_step

print "- Training agent"

trainer = AgentTrainer(agent, training_iters, display_step, n_first_digit_length, n_second_digit_length)
trainer.train(digits, digit_labels, symbols, symbol_labels)

print "  Finished"

## In[4]: Test agent

#test_images_file = "./SYMBOLS_DATA/DATA_TEST_IMAGES.npy"
test_images_file = args.test_images_file
#test_labels_file = "./SYMBOLS_DATA/DATA_TEST_LABELS.npy"
test_labels_file = args.test_labels_file

print "- Loading testing data"

digits, digit_labels, symbols, symbol_labels = load_data(test_images_file, test_labels_file)

print "  Finished"

print "- Testing agent"

batch_ys, batch_ops, pred = trainer.test(digits, digit_labels, symbols, symbol_labels)

print '  GT', batch_ys, batch_ops
print '  Pred', pred

