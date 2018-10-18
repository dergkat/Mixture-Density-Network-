"""SIMPLE DATA FITTING WITH TENSORFLOW"""
"""http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
""" y = 7.0sin(0.75x) + 0.5x + E """
#   E - Is Standard Guassian Random Noise 
NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5,10.5,(1,NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0 + x_data * 0.5 + r_data * 1.0)

# TensorFlow place holders as variables
x = tf.placeholder(dtype=tf.float32, shape=[None,1])
y = tf.placeholder(dtype=tf.float32, shape=[None,1])

# We will define this simple neural network one-hidden layer and 20 nodes:
""" Y = W_{out} \tanh( W X + b) + b_{out} """

NHIDDEN = 20
W = tf.Variable(tf.random_normal([1,NHIDDEN] , stddev = 1.0 , dtype = tf.float32))
b = tf.Variable(tf.random_normal([1,NHIDDEN] , stddev = 1.0 , dtype = tf.float32))
W_out = tf.Variable(tf.random_normal([NHIDDEN,1] , stddev = 1.0 , dtype = tf.float32))
b_out = tf.Variable(tf.random_normal([1,1] , stddev = 1.0 , dtype = tf.float32))

hidden_layer = tf.nn.tanh(tf.matmul(x,W) + b)
y_out = tf.matmul(hidden_layer,W_out) + b_out

"""Define Loss Function"""
lossfunc = tf.nn.l2_loss(y_out-y)

"""Define a training operator that will minimise the loss function later."""

"""RMSProp gradient descent optimisation method."""
train_op = tf.train.RMSPropOptimizer(learning_rate = 0.1, decay = 0.8).minimize(lossfunc)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())



"""We will run gradient descent for 1000 times
to minimise the loss function 
with the data fed in via a dictionary."""

NEPOCH = 1000
for i in range(NEPOCH):
  sess.run(train_op,feed_dict={x: x_data, y: y_data})

x_test = np.float32(np.arange(-10.5,10.5,0.1))
x_test = x_test.reshape(x_test.size,1)
y_test = sess.run(y_out,feed_dict={x: x_test})

plt.figure(figsize=(8, 8))
plt.plot(x_data,y_data,'ro', x_test,y_test,'bo',alpha=0.3)
plt.show()

sess.close()
