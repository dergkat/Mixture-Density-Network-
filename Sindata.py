"""SIMPLE DATA FITTING WITH TENSORFLOW"""
"""http://blog.otoro.net/2015/11/24/mixture-density-networks-with-tensorflow/"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math

""" y = 7.0sin(0.75x) + 0.5x + E """
""" E - Is Standard Guassian Random Noise """

""" Generate the sinusoidal data to train a neural net to fit later """

NSAMPLE = 1000
x_data = np.float32(np.random.uniform(-10.5,10.5,(1,NSAMPLE))).T
r_data = np.float32(np.random.normal(size=(NSAMPLE,1)))
y_data = np.float32(np.sin(0.75*x_data)*7.0 + x_data * 0.5 + r_data * 1.0)

 
"""Plot the graph"""
plt.figure(figsize = (8,8))
plot_out = plt.plot(x_data,y_data,'ro',alpha = 0.3)
plt.show()