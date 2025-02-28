import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import blocks
import MBQL
import time
from datetime import timedelta
import math
import os
from scipy.stats import ortho_group
from numpy import genfromtxt
from get_iris_data import *
import time
from sklearn import linear_model, datasets


# Taking iris data from sklearn
from sklearn import datasets
# 

from sklearn.cross_validation import train_test_split
import sklearn

sess = tf.InteractiveSession()

model, qubits, thetas = MBQL.square_8_2()
  
def get_batch(bsize):
    batch = [0]*2
    r=np.random.randint(0,y_train.shape[0])
    batch[0] = np.squeeze(np.array([[np.exp(-1J*x_train[r])], [np.exp(1J*x_train[r])]]).astype('complex64'))
    batch[1] = np.array([y_train[r]]).astype('float32')    
    return batch

def get_train_accuracy():
    train_accuracy = 0.0    
    for i in range(y_train.shape[0]):
        batch = [0]*2
        r=i
        batch[0] = np.squeeze(np.array([[np.exp(-1J*x_train[r])], [np.exp(1J*x_train[r])]]).astype('complex64'))
        batch[1] = np.array([y_train[r]]).astype('float32')  
        train_accuracy = train_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3], 
                                          qubits[4]: batch[0][:,4], 
                                          qubits[5]: batch[0][:,5], 
                                          qubits[6]: batch[0][:,6], 
                                          qubits[7]: batch[0][:,7], 
                                          y_truth: batch[1]})/float(y_train.shape[0])
    return train_accuracy


def get_test_accuracy():
    test_accuracy = 0.0
    for i in range(y_test.shape[0]):
        batch = [0]*2
        r=i
        batch[0] = np.squeeze(np.array([[np.exp(-1J*x_test[r])], [np.exp(1J*x_test[r])]]).astype('complex64'))
        batch[1] = np.array([y_test[r]]).astype('float32')  
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3], 
                                          qubits[4]: batch[0][:,4], 
                                          qubits[5]: batch[0][:,5], 
                                          qubits[6]: batch[0][:,6], 
                                          qubits[7]: batch[0][:,7], 
                                          y_truth: batch[1]})/float(y_test.shape[0])
    return test_accuracy


y_truth = tf.placeholder(tf.float32, shape=[1])

## Prep the data
# Let's trying building
#buildDataFromIris()
# Reading in the data
data = genfromtxt('tf_train.csv',delimiter=',')  # Training data
test_data = genfromtxt('tf_test.csv',delimiter=',')  # Test data

x_train=np.array([ i[1::] for i in data])
y_train,y_train_onehot = convertOneHot(data)



x_min_train = np.min(x_train,0)
x_train = x_train-x_min_train
x_max_train = np.max(x_train,0) # Normalize

# Doing a similiar conversion for the test data.
x_test=np.array([ i[1::] for i in test_data])
y_test,y_test_onehot = convertOneHot(test_data)

x_test = x_test-x_min_train

x_test = (x_test/x_max_train)
x_train = (x_train/x_max_train)

    ## Linear regression result





#renorm:
x_test=x_test*np.pi/2 #+ np.pi/4
x_train=x_train*np.pi/2 #+ np.pi/4

error = np.abs(model - y_truth)
correct_prediction = tf.equal(model>0.5, y_truth>0.5)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
train_step = tf.train.MomentumOptimizer(3e-3,0.9).minimize(error)
#train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(error)
sess.run(tf.global_variables_initializer())


iters = 2000
bsize = 1
for i in range(iters):
    batch = get_batch(bsize)
    train_step.run(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3], 
                                          qubits[4]: batch[0][:,4], 
                                          qubits[5]: batch[0][:,5], 
                                          qubits[6]: batch[0][:,6], 
                                          qubits[7]: batch[0][:,7], 
                              y_truth: batch[1]})
    if i % 10 == 0:
        train_accuracy = get_train_accuracy()
        print('step %d, training accuracy %g' % (i, train_accuracy))
        
    if i % 100 == 0:
        test_accuracy = get_test_accuracy()
        print('step %d, testing accuracy %g' % (i, test_accuracy))    
        
print('Test acc is:' + str(get_test_accuracy()))


thetas_np=np.zeros(len(thetas))
for i in range(len(thetas)):
    
    thetas_np[i]=thetas[i][0].eval()

