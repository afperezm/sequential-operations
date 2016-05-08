
# coding: utf-8

# In[38]:

import tensorflow as tf # se importa tensor flow
import matplotlib.pyplot as plt
import numpy as np


# In[2]:

# funcion  para importar la data de tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_DATA', one_hot=True)


# In[3]:

#inicio session interactiva
sess = tf.InteractiveSession()


# In[4]:

# Construccion del modelo softmax  con una sola capa 
#************se construyen los nodos de la imagen de entrada*********************
# x imagenes de entrada y y imagenes de salida 
#
x = tf.placeholder(tf.float32, shape=[None, 784]) # variables simbolicas para manipulacion 
# x es un tensor  2d  punto flotante , 784 dimensiones de una unica imagen 
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# y tensor 2d punto flotante , clases de salida 10 dimensiones indica la clase del digito a la cual la imagen pertenece
# el shaspe Inserta un marcador de posición para un tensor que será siempre alimentado.


# In[5]:

#definciion de pesos w y un sesgo b tensorflow tiene la propieda de agragarlos a una variable que vive en todo el grafico con 
# tf.variable  donde se inicilizan como tensores llenos de ceros (W y B) 
W = tf.Variable(tf.zeros([784,10]))
# b es un vector de 10 dimensiones por que se tienen 10 clases
b = tf.Variable(tf.zeros([10]))
# w es una matriz de 784x10 (784 entidades y 10 de salida)


# In[6]:

sess.run(tf.initialize_all_variables()) # inicializacion de variables 


# In[7]:

# funcion de coste (hasta ahora solo es un modelo de regresion)
#**************importante************
# se hace uso de una funcion lienal 
# 1) se multiplican las imagenes puestas anteriormente en un vector (kernel 3) por la matriz de pesos w mas el sesgo b (bias)**
# y se calcula la probabilidad softmax que asigna la probalidad de pertenencia a cada clase
# normalliza los datos de salida para que todos me sumen 1
y = tf.nn.softmax(tf.matmul(x,W) + b)


# In[8]:

# cross entropy reduce la funcion de coste con el entrenamiento intentar conseguir valores de los parámetros
#W y b que minimicen el valor de la métrica que indica cuan malo es el modelo.
# tf.reduce_sum se apliza a travez de todas las clases y tf.reduce_mean la media de todas las sumas
# Donde y es la distribución de probabilidad precedida y la y la distribución
#real (obtenida a partir del etiquetado de los datos de entrada) lo que busca es que ambas sean iguales 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# se calcula el logaritmo de cada elemento y con la función que
#nos provee TensorFlow tf.log y después se multiplica por cada
#elemento y_ . Finalmente, con tf.reduce_sum se suman todos los
#elementos del tensor (más adelante veremos que las imágenes se miran
#por paquetes, y en este caso el valor de cross-entropy corresponde a la
#suma de las cross-entropy de las imágenes de un paquete y no a la de
#una sola).


# In[9]:

# uso del gradiente descediente 
train_step = tf.train.GradientDescentOptimizer(0.10).minimize(cross_entropy)
#La taza de aprendizaje es una constante que indica qué tan grande haremos un paso en la dirección opuesta al gradiente
#GradientDescentOptimizer(learning rate) 
# TensorFlow tiene una variedad de algoritmos de optimización orden interna
# con una longitud de paso de 0,5, para minimiazar el cross entropy.
# elñ gradiente calcular los pasos de actualización de parámetros, y aplicar medidas de actualización de los parámetros.
# se aproxima de forma negativa al gradiente busca el minimo global 
#en que punto empieza mi gradiente descendiente ????? cuanto puedo ir avanzadano 


# In[10]:

# rabdi de cada 1000 iteraciones entrenamiento se genera  por la ejecucion repetida de cada paso
for i in range(1000):
  batch = mnist.train.next_batch(50) # 50 ejemplos 
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})# feed_dict reemplaza los tensores de marcador de posición
     #x e Y con los ejemplos
    


# In[11]:

#tf.argmax es una función extremadamente útil que le da el índice de la entrada más alta de un tensor a lo largo de un eje
accert_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# etiqueta que retona el sistema tf.argmax(y,1), etiqueta verdadera tf.argmax(y_,1)
# retorna bool
#verifica
# 2 funciones de perdida o cosot guia para incognitas desempeñ mide solo que tan bueno fue el soluciones  


# In[12]:

# se determina la media de los puntos flotantes y luego se toma la media 
accuracy = tf.reduce_mean(tf.cast(accert_prediction, tf.float32)) 
print (accuracy.eval(feed_dict ={x : mnist.test.images,y_ : mnist.test.labels}))
# cuantas veces fue acertada
# como arreglar el modelo ? red convolucional 


# In[13]:

# construccion de red convolucional


# In[14]:

# no comprendo muy bien 
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
# no ceros valores aleatorio a una matriz truncate normal deviacion shape

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
# todos los valores en 0.1

# solo los valores en cero empezar en cero lejano 
# incia busqu de parametros en punto aleatorio 


# In[15]:

# maxpooling de 2x2.
#funciones de tensor
#filtros W
#entradas 
#tamño de paso strides  que tanto me muevo en las posiciones del vecrotr que entra 
#doc con2d

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# procesamiwento de todos los canales en cada uno hace transformacion convolucionales tien un w y un b

# despuesd etener reformada la imagen 
# se activa pagtron el filtro y el pixel filtro pesos 
# remplaza por
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1], padding='SAME')# COMO DEFINIO EL MAX POOLING
#solo declaraciones 


# In[16]:

# particiones de la imagen ------------------- ????????????????????????????
# La convolución computará 32 característica para cada sector [5, 5] 
# 1 canales de entrada 
# numero de canales de salida
# tensor de peso [5, 5, 1, 32]
W_conv1 = weight_variable([5, 5, 1, 32]) # PRIMERA CONVOLUCION 
b_conv1 = bias_variable([32])


# In[17]:

#para aplicar hay que modelar x a un tensor de 4d 2,3 alto y ancho y ultimo canal de color 
x_image = tf.reshape(x, [-1,28,28,1])
# no vector se coamonadan pixeles en 2d
#- muchas imagen pixeles, 1 canal de color


# In[18]:

#se aplica convoluvion a x_image, se añadeel sesgo  y se aplica relu y a esta misma el maxpooling
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # derivada
h_pool1 = max_pool_2x2(h_conv1)


# In[19]:

# segunda red convolucional 
#64 características para cada parche 5x5.
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
# canales empiezan a tener patrones 


# In[20]:

# POR QUE SE REDUCE A 7X7 LA IMAGEN???
#añadimos una capa plenamente conectado con 1024 neuronas para permitir el procesamiento en la imagen entera
#Nos remodelar el tensor de la capa de la agrupación en un lote de vectores
#se multiplica por una matriz de pesos, añadir un sesgo, y aplicar una regla.
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
#capa lineal 
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# toma todos loa  valres de la conv linelamente


# In[21]:

# NO COMPRENDO MUY BIEN ?????????????
#Para minimizar el overfitting, se aplica dropout depues de la capa de lectura
# se crea un marcador de posicion para la probabilidad de salida para que mantenga el porcentaje de entrenamiento
#maneja el escalado de las salidas

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#dropout evita malas solucions , durante entrenamiento , ignora algunos full connected, conservo % que indque la proalidad


# In[22]:

# sea aplica softmax a la capa de regresion 
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 2 capa fullconnec que me lo envia en alguna de las clases

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# In[23]:

# se usa ADAM  en vez de gradiente descenienmte y se almacenan los registros en cada iteración número100.
#El valor keep_prob se utiliza para controlar la tasa de
#perdida utilizado cuando se entrena la red neuronal. lo cual lleva que cada conexión
#entre las capas (en este caso entre la última capa densamente conectado y la capa de lectura) sólo será
#utilizada con una probabilidad de 0,5 cuando el entrenamiento. Esto reduce sobreajuste.

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))#funcion de costo 
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy) 
accert_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))# funcion de desempeño
accuracy = tf.reduce_mean(tf.cast(accert_prediction, tf.float32))# funcion de desempeño 
sess.run(tf.initialize_all_variables())
for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

## SAVE MODEL
import cPickle as pickle
pickle.dump({'Wc1':W_conv1.eval(), 'bc1':b_conv1.eval(), 
             'Wc2':W_conv2.eval(), 'bc2':b_conv2.eval(), 
             'Wfc1':W_fc1.eval(), 'bfc1':b_fc1.eval(),
             'Wfc2':W_fc2.eval(), 'bfc2':b_fc2.eval()}, 
             open('convnet.pkl','wb'))

# In[26]:


