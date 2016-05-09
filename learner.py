import cPickle as pickle
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell

class RecurrentNeuralLearner(object):
    
    def __init__(self, convnet_file, learning_rate, batch_size, n_input, n_hidden, n_steps):
        self.convnet_file = convnet_file
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_steps = n_steps
        self.convnet = pickle.load(open(self.convnet_file))
    
    def createNetwork(self):
        
        # Define hidden layer weights
        weights = {
            'hidden': tf.Variable(tf.random_normal([self.n_input, self.n_hidden])),
            'out1': tf.Variable(tf.truncated_normal([self.n_hidden,512])),
            'out2': tf.Variable(tf.truncated_normal([512, 1]))
        }
        
        biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden])),
            'out1': tf.Variable(tf.zeros([512])),
            'out2': tf.Variable(tf.zeros([1]))
        }
        
        self.x = tf.placeholder("float", [None, self.n_steps, 28, 28, 1])
        self.y = tf.placeholder("float", [None, 1])
        # Tensorflow LSTM cell requires 2 x n_hidden length (state & cell)
        self.istate = tf.placeholder("float", [None, 2 * self.n_hidden])
        
        self.pred = self._buildRecurrentNeuralNetwork(self.x, self.istate, weights, biases)
        
        # Define loss using squared mean
        self.cost = tf.reduce_mean(tf.nn.l2_loss(self.pred - self.y))
        
        # Define optimizer as Adam Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
        # Evaluate model
        correct_pred = tf.equal(tf.round(self.pred), tf.round(self.y))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    def _buildRecurrentNeuralNetwork(self, _X, _istate, _weights, _biases):
        
        # Reshape _X to prepare input to hidden activation by permuting first
        # and second dimensions.
        #
        # Input shape:  (batch_size, n_steps, img_width, img_height, n_color_channels)
        # Output shape: (n_steps, batch_size, img_width, img_height, n_color_channels)
        _X = tf.transpose(_X, [1, 0, 2, 3, 4])
        
        sequence = []
        
        for seqIdx in range(self.n_steps):
            if seqIdx > 0:
                sequence.append(self._buildConvnet(_X[0,:,:,:,:]))
            elif seqIdx > 1:
                sequence.append(self._buildConvnet(_X[1,:,:,:,:]))
            elif seqIdx > 2:
                sequence.append(self._buildConvnet(_X[2,:,:,:,:]))
            elif seqIdx > 3:
                sequence.append(self._buildConvnet(_X[3,:,:,:,:]))
            elif seqIdx > 4:
                sequence.append(self._buildConvnet(_X[4,:,:,:,:]))
            elif seqIdx > 5:
                sequence.append(self._buildConvnet(_X[5,:,:,:,:]))
            elif seqIdx > 6:
                sequence.append(self._buildConvnet(_X[6,:,:,:,:]))
            elif seqIdx > 7:
                sequence.append(self._buildConvnet(_X[7,:,:,:,:]))
            elif seqIdx > 8:
                sequence.append(self._buildConvnet(_X[8,:,:,:,:]))
            elif seqIdx > 9:
                sequence.append(self._buildConvnet(_X[9,:,:,:,:]))
            elif seqIdx > 10:
                sequence.append(self._buildConvnet(_X[10,:,:,:,:]))
        
        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(self.n_hidden, forget_bias=1.0)
        
        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, sequence, initial_state=_istate)
        
        # Linear activation
        # Get inner loop last output
        out1 = tf.nn.relu(tf.matmul(outputs[-1], _weights['out1']) + _biases['out1'])
        out2 = tf.matmul(out1, _weights['out2']) + _biases['out2']
        
        return out2
    
    def _buildConvnet(self, x_image):
        
        W_conv1 = self._buildWeightVariable([5, 5, 1, 32], init = self.convnet['Wc1'])
        b_conv1 = self._buildBiasVariable([32], init = self.convnet['bc1'])
        
        W_conv2 = self._buildWeightVariable([5, 5, 32, 64], init = self.convnet['Wc2'])
        b_conv2 = self._buildBiasVariable([64], init = self.convnet['bc2'])
        
        W_fc1 = self._buildWeightVariable([7 * 7 * 64, 1024], init = self.convnet['Wfc1'])
        b_fc1 = self._buildBiasVariable([1024], init = self.convnet['bfc1'])
        
        #W_fc2 = weight_variable([1024, 10], init=self.convnet['Wfc2'])
        #b_fc2 = bias_variable([10], init=self.convnet['bfc2'])
        
        h_conv1 = tf.nn.relu(self._peform2DConvolution(x_image, W_conv1) + b_conv1)
        h_pool1 = self._performMaxPooling(h_conv1)
        
        h_conv2 = tf.nn.relu(self._peform2DConvolution(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self._performMaxPooling(h_conv2)
        
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        
        return h_fc1
    
    def _buildWeightVariable(self, shape, init=None):
      """
      Builds the weights variable for the convolutional network.
      """
      if init is None:
          # Builds a variable from a random sequence of given shape with mean 0.0 standard deviation 0.1
          initial = tf.truncated_normal(shape, stddev=0.1)
          return tf.Variable(initial)
      else:
          # Builds a variable from a given tensor
          return tf.Variable(init)
    
    def _buildBiasVariable(self, shape, init=None):
      """
      Builds the bias variable for the convolutional network.
      """
      if init is None:
          # Builds a variable from constant sequence of given shape with value 0.1
          initial = tf.constant(0.1, shape=shape)
          return tf.Variable(initial)
      else:
        # Builds a variable from a given tensor
        return tf.Variable(init)
    
    def _peform2DConvolution(self, x, W):
      """
      Applies a 2-D convolutional filter for the given input 'x' using weights 'W'.
      """
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def _performMaxPooling(self, x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

