from data import generate_sequence
from data import load_data
from learner import RecurrentNeuralLearner
from trainer import AgentTrainer

## In[1]: Load training data

train_images_file = "./DATA_TRAIN_IMAGES.npy"
train_labels_file = "./DATA_TRAIN_LABELS.npy"

print "- Loading training data"

digits, digit_labels, symbols, symbol_labels = load_data(train_images_file, train_labels_file)

print "  Finished"

sequence, result, operands = generate_sequence(digits, digit_labels, symbols, symbol_labels,  20, 1, 1)

print "- Size of sequence tensor:", sequence.shape

## In[2]: Create agent

convnet_file = "convnet.pkl"
learning_rate = 0.001
batch_size = 64
n_input = 1024
n_hidden = 512
n_first_digit_length = 1
n_second_digit_length = 1
n_steps = n_first_digit_length + 1 + n_second_digit_length

assert(n_steps <= 11)

print "- Creating agent"

agent = RecurrentNeuralLearner(convnet_file, learning_rate, batch_size, n_input, n_hidden, n_steps)
agent.createNetwork()

print "  Finished"

## In[3]: Train agent

training_iters = 1000
display_step = 50

print "- Training agent"

trainer = AgentTrainer(agent, training_iters, display_step, n_first_digit_length, n_second_digit_length)
trainer.train(digits, digit_labels, symbols, symbol_labels)

print "  Finished"

## In[4]: Test agent

#test_images_file = "./DATA_TEST_IMAGES.npy"
#test_labels_file = "./DATA_TEST_LABELS.npy"

#print "- Loading testing data"

#digits, digit_labels, symbols, symbol_labels = load_data(test_images_file, test_labels_file)

#print "  Finished"

#print "- Testing agent"

#batch_ys, batch_ops, pred = trainer.test(digits, digit_labels, symbols, symbol_labels)

#print '  GT', batch_ys, batch_ops
#print '  Pred', pred

