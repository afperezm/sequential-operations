# coding: utf-8

# In[1]:

import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np
#get_ipython().magic(u'matplotlib inline')
#import numpy as np
from random import randint
#import matplotlib.pyplot as plt
import random
from random import randrange
from tensorflow.examples.tutorials.mnist import input_data

# In[2]:

mnist = input_data.read_data_sets('MNIST_DATA', one_hot = True)
sesion = tf.InteractiveSession()

# In[3]:

simbolos = np.load('./DATA_TRAIN_IMAGES.npy')
simbolos_label = np.load('./DATA_TRAIN_LABELS.npy')
digitos = mnist.train.images
digitos_label = mnist.train.labels
#digitos.shape
#simbolos.shape

# In[4]:

#print (simbolos_label[3])
#plt.imshow(simbolos[3])

# In[5]:

# Find indices of symbols labeled as addition or subtraction
idx = np.sum(simbolos_label[:,0:2], axis=1)
# Obtain labels of symbols labeled as addition or subtraction
simbolos_label_sr = simbolos_label[idx>0,0:2]
# Obtain symbols labeled as addition or subtraction
simbolos_sr = simbolos[idx>0,:,:]
# Show shape of symbols labeled as addition or subtraction
#simbolos_sr.shape

# In[6]:

def generateSequence(Dg, Et, Sm, Op, sequenceLength, digitOneLength, digitTwoLength):
    """
    Generates a random sequence of 'sequenceLength' triplets of operands and operators.
    """
    
    operationLength = digitOneLength + 1 + digitTwoLength
    
    # Initialize return variables
    sequence = np.zeros((sequenceLength, operationLength, Sm.shape[1], Sm.shape[2], 1))
    result = np.zeros((sequenceLength, 1))
    operands = np.zeros((sequenceLength, 3))
    
    for k in range(sequenceLength):
        
        x1 = 0
        x2 = 0
        
        for elemIdx in range(operationLength):
            if elemIdx < digitOneLength:
                for digitOneIdx in range(digitOneLength):
                    # Generate random first operand index
                    j1 = randrange(0, Dg.shape[0])
                    # Reshape first operand image
                    digitOne = np.reshape(Dg[j1,:], [Sm.shape[1], Sm.shape[2]])
                    # Store first operand image
                    sequence[k, elemIdx, :, :, 0] = digitOne
                    # Find first operand
                    x1 += np.argmax(Et[j1])
            elif elemIdx > digitOneLength:
                for digitTwoIdx in range(digitOneLength):
                    # Generate random second operand index
                    j2 = randrange(0, Dg.shape[0])
                    # Reshape second operand image
                    digitTwo = np.reshape(Dg[j2,:], [Sm.shape[1], Sm.shape[2]])
                    # Store second operand image
                    sequence[k, elemIdx, :, :, 0] = digitTwo
                    # Find second operand
                    x2 += np.argmax(Et[j2])
        
        # Generate random operator index
        operatorIdx = randrange(0, Sm.shape[0])
        # Store operator image
        operator = Sm[operatorIdx,:,:]
        sequence[k, elemIdx, :, :, 0] = operator
        # Find operator
        y = np.argmax(Op[operatorIdx])
        
        # Compute operation result
        if (y == 0):
            # addition
            result[k]= x1 + x2
        elif (y == 1):
            # subtraction
            result[k]= x1 - x2
        
        operands[k,0] = x1
        operands[k,1] = y
        operands[k,2] = x2
        
        #print (x1,Et[j1],x2,Et[j2],y,Op[i],result[k])
        
    return sequence, result, operands

#seq, res, ops = generateSequence(digitos, digitos_label, simbolos_sr, simbolos_label_sr,  20, 1, 1)

# In[7]:

#sid = 8
#l0 = seq[sid,0,:,:,0]
#l1 = seq[sid,1,:,:,0]
#l2 = seq[sid,2,:,:,0]

#plt.matshow(l0)
#plt.matshow(l1)
#plt.matshow(l2)
#print 'Size of sequence tensor:', seq.shape
#print res[sid]

# In[8]:

def weight_variable(shape, init=None):
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

def bias_variable(shape, init=None):
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

# solo los valores en cero empezar en cero lejano 
# incia busqu de parametros en punto aleatorio 

# maxpooling de 2x2.
#funciones de tensor
#filtros W
#entradas 
#tamño de paso strides  que tanto me muevo en las posiciones del vecrotr que entra 
#doc con2d

def conv2d(x, W):
  """
  Applies a 2-D convolutional filter for the given input 'x' using weights 'W'.
  """
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# procesamiwento de todos los canales en cada uno hace transformacion convolucionales tien un w y un b

# despuesd etener reformada la imagen 
# se activa pagtron el filtro y el pixel filtro pesos 
# remplaza por
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')# COMO DEFINIO EL MAX POOLING
#solo declaraciones 

## CARGAR CONVNET PRE-ENTRENADA
import cPickle as pickle
CONVNET = pickle.load(open('convnet.pkl'))
print '- Loaded convolutional network'

# particiones de la imagen ------------------- ????????????????????????????
# La convolución computará 32 característica para cada sector [5, 5] 
# 1 canales de entrada 
# numero de canales de salida
# tensor de peso [5, 5, 1, 32]
W_conv1 = weight_variable([5, 5, 1, 32], init=CONVNET['Wc1']) # PRIMERA CONVOLUCION 
b_conv1 = bias_variable([32], init=CONVNET['bc1'])

# segunda red convolucional 
#64 características para cada parche 5x5.
W_conv2 = weight_variable([5, 5, 32, 64], init=CONVNET['Wc2'])
b_conv2 = bias_variable([64], init=CONVNET['bc2'])

# POR QUE SE REDUCE A 7X7 LA IMAGEN???
#añadimos una capa plenamente conectado con 1024 neuronas para permitir el procesamiento en la imagen entera
#Nos remodelar el tensor de la capa de la agrupación en un lote de vectores
#se multiplica por una matriz de pesos, añadir un sesgo, y aplicar una regla.
W_fc1 = weight_variable([7 * 7 * 64, 1024], init=CONVNET['Wfc1'])
b_fc1 = bias_variable([1024], init=CONVNET['bfc1'])

## Capa de clasificacion de la Convnet
W_fc2 = weight_variable([1024, 10], init=CONVNET['Wfc2'])
b_fc2 = bias_variable([10], init=CONVNET['bfc2'])

def CNN(x_image):
    #para aplicar hay que modelar x a un tensor de 4d 2,3,  alto y ancho y ultimo canal de color 
    #x_image = tf.placeholder(tf.float32, shape=[None, 28, 28, 1]) # variables simbolicas para manipulacion 
    # x es un tensor  2d  punto flotante , 784 dimensiones de una unica imagen 
    # no vector se coamonadan pixeles en 2d
    #- muchas imagen pixeles, 1 canal de color
    #se aplica convoluvion a x_image, se añadeel sesgo  y se aplica relu y a esta misma el maxpooling
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # derivada
    h_pool1 = max_pool_2x2(h_conv1)
    # Operacion capa 2
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    # canales empiezan a tener patrones 
    #capa lineal 
    # Operacion capa 3
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    # toma todos loa  valres de la conv linelamente
    #h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2
    #return tf.stop_gradient(h_fc2)
    #return tf.stop_gradient( tf.nn.softmax(h_fc2) )
    return h_fc1

#keep_prob = tf.placeholder(tf.float32)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#dropout evita malas solucions , durante entrenamiento , ignora algunos full connected, conservo % que indque la proalidad


# In[9]:

learning_rate = 0.001
training_iters = 1000000
batch_size = 64
display_step = 10

# In[10]:

n_input = 1024 # salida de la red convolucional y entrada a la RNN
n_first_digit_length = 2
n_second_digit_length = 1
n_steps = n_first_digit_length + 1 + n_second_digit_length # timesteps: tamaño de secuencias
n_hidden = 512 # hidden layer num of features || tamano de la memoria

assert(n_steps <= 11)

# In[11]:

# tf Graph input
x = tf.placeholder("float", [None, n_steps, 28, 28, 1])
# Tensorflow LSTM cell requires 2x n_hidden length (state & cell)
istate = tf.placeholder("float", [None, 2*n_hidden])
y = tf.placeholder("float", [None, 1])

# In[12]:

# Define weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out1': tf.Variable(tf.truncated_normal([n_hidden,512])),
    'out2': tf.Variable(tf.truncated_normal([512, 1]))
}

biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out1': tf.Variable(tf.zeros([512])),
    'out2': tf.Variable(tf.zeros([1]))
}

# In[13]:

# _X tensor de secuencias: (20,3,28,28,1)
def RNN(_X, _istate, _weights, _biases):
    
    # input shape: (batch_size, n_steps, 28, 28, 1)
    _X = tf.transpose(_X, [1, 0, 2, 3, 4])  # permute n_steps and batch_size
    # input shape: (n_steps=3, batch_size=20, 28, 28, 1)
    # Reshape to prepare input to hidden activation
    #_X = tf.reshape(_X, [-1, n_input]) # (n_steps*batch_size, n_input)
    # Linear activation  ==> convolutional net
    #_X = tf.matmul(_X, _weights['hidden']) + _biases['hidden']
    
    sequence = []
    
    for seqIdx in range(n_steps):
        if seqIdx > 0:
            sequence.append(CNN(_X[0,:,:,:,:]))
        elif seqIdx > 1:
            sequence.append(CNN(_X[1,:,:,:,:]))
        elif seqIdx > 2:
            sequence.append(CNN(_X[2,:,:,:,:]))
        elif seqIdx > 3:
            sequence.append(CNN(_X[3,:,:,:,:]))
        elif seqIdx > 4:
            sequence.append(CNN(_X[4,:,:,:,:]))
        elif seqIdx > 5:
            sequence.append(CNN(_X[5,:,:,:,:]))
        elif seqIdx > 6:
            sequence.append(CNN(_X[6,:,:,:,:]))
        elif seqIdx > 7:
            sequence.append(CNN(_X[7,:,:,:,:]))
        elif seqIdx > 8:
            sequence.append(CNN(_X[8,:,:,:,:]))
        elif seqIdx > 9:
            sequence.append(CNN(_X[9,:,:,:,:]))
        elif seqIdx > 10:
            sequence.append(CNN(_X[10,:,:,:,:]))
    
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    #_X = tf.split(0, n_steps, _X) # n_steps * (batch_size, n_hidden)
    
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, sequence, initial_state=_istate)
    
    # Linear activation
    # Get inner loop last output
    out1 = tf.nn.relu( tf.matmul(outputs[-1], _weights['out1']) + _biases['out1'] )
    out2 = tf.matmul(out1, _weights['out2']) + _biases['out2'] 
    return out2

# In[14]:

pred = RNN(x, istate, weights, biases)

# In[15]:

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.l2_loss(pred - y)) # Error cuadratico
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# In[16]:

# Evaluate model
correct_pred = tf.equal(tf.round(pred), tf.round(y))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
prediccion = pred

# In[17]:

# Initializing the variables
init = tf.initialize_all_variables()

# In[18]:

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    print "- Optimization started"
    while step < training_iters:
        batch_xs, batch_ys, batch_ops = generateSequence(digitos, digitos_label, simbolos_sr, simbolos_label_sr, batch_size, n_first_digit_length, n_second_digit_length)
        # Fit training using batch data
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*n_hidden))})
        if step % display_step == 0:
            # Calculate batch accuracy
            loss, acc, prd = sess.run([cost, accuracy, pred], feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*n_hidden))})
            # Calculate batch loss
            #loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*n_hidden))})
            print "  Iteration=" + str(step) + ", Minibatch Error=" + "{:.6f}".format(np.sqrt(loss)) + ", Training Accuracy=" + "{:.5f}".format(acc)
            #print CNN(batch_xs[0,0:3,:,:,:].astype('float32')).eval(),batch_ops[0],batch_ys[0],prd[0]
            print " ", batch_ops[0], batch_ys[0], prd[0]
        # Increase step
        step += 1
    print "- Optimization finished"
    batch_xs, batch_ys, batch_ops = generateSequence(digitos, digitos_label, simbolos_sr, simbolos_label_sr, batch_size, n_first_digit_length, n_second_digit_length)
    # Calculate batch accuracy
    pred = sess.run(pred, feed_dict={x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2*n_hidden))})
    print '  GT', batch_ys, batch_ops
    print '  Pred', pred

# In[ ]:

# no se donde se identifica el simbolo 
# para que se hacen simbolos si no se identifican 
#no comprendo donde los trato 

