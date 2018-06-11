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

model, guess, qubits, thetas = MBQL.nn_4()


def sol(n):
    if n == 0:
        return np.reshape(np.array([1,0,0,0]).astype('float32'),(1,4))
    elif n == 1:
        return np.reshape(np.array([0,1,0,0]).astype('float32'),(1,4))
    elif n == 2:
        return np.reshape(np.array([0,0,1,0]).astype('float32'),(1,4))
    else:
        return('ERROR, invalid label')
  
def get_batch(bsize):
    batch = [0]*3
    r=np.random.randint(0,y_train.shape[0])
    batch[0] = np.squeeze(np.array([[np.exp(-1J*x_train[r])], [np.exp(1J*x_train[r])]]).astype('complex64'))
    batch[1] = sol(y_train[r])
    batch[2] = y_train[r] #np.array(y_train[r]).astype('float32')
    return batch

def get_train_accuracy():
    train_accuracy = 0.0    
    for i in range(y_train.shape[0]):
        batch = [0]*3
        r=i
        batch[0] = np.squeeze(np.array([[np.exp(-1J*x_train[r])], [np.exp(1J*x_train[r])]]).astype('complex64'))
        batch[1] = sol(y_train[r])
        batch[2] = y_train[r] #np.array(y_train[r]).astype('float32')
        train_accuracy = train_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3],
                                          y_truth: batch[1],
                                          y_truth2: batch[2]})/float(y_train.shape[0])
    return train_accuracy

def get_thing():
    return checker.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3],
                                          y_truth: batch[1],
                                          y_truth2: batch[2]})#/float(y_test.shape[0])
def get_thing2():
    return checker2.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3],
                                          y_truth: batch[1],
                                          y_truth2: batch[2]})#/float(y_test.shape[0])

def get_test_accuracy():
    test_accuracy = 0.0
    for i in range(y_test.shape[0]):
        batch = [0]*3
        r=i
        batch[0] = np.squeeze(np.array([[np.exp(-1J*x_test[r])], [np.exp(1J*x_test[r])]]).astype('complex64'))
        batch[1] = sol(y_test[r])
        batch[2] = y_test[r] # np.array(y_test[r]).astype('float32')
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={qubits[0]: batch[0][:,0], 
                                          qubits[1]: batch[0][:,1], 
                                          qubits[2]: batch[0][:,2], 
                                          qubits[3]: batch[0][:,3],
                                          y_truth: batch[1],
                                          y_truth2: batch[2]})/float(y_test.shape[0])
                                          #y_truth2: batch[2]]}
    return test_accuracy


y_truth = tf.placeholder(tf.float32, shape = [1,4])
y_truth2 = tf.placeholder(tf.int64, shape = [])
## Prep the data
# Let's trying building
buildDataFromIris()
# Reading in the data
data = genfromtxt('cs-training.csv',delimiter=',')  # Training data
test_data = genfromtxt('cs-testing.csv',delimiter=',')  # Test data

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






#renorm:
x_test=x_test*np.pi/2 #+ np.pi/4
x_train=x_train*np.pi/2 #+ np.pi/4

checker = y_truth2
checker2 = guess
error = tf.nn.softmax_cross_entropy_with_logits(labels = y_truth, logits = model)  #reduce_sum(tf.multiply(model,y_truth)) # - tf.reduce_sum(tf.multiply(model[0], y_truth))
correct_prediction = tf.equal(guess, y_truth2) #tf.equal(np.argmax(model), np.argmax(y_truth))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.int32))

train_step =  tf.train.GradientDescentOptimizer(learning_rate = 0.03).minimize(error) #tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(error)
#train_step = tf.train.MomentumOptimizer(7e-3,0.9).minimize(error)

sess.run(tf.global_variables_initializer())


iters = 9000
bsize = 1

train_accuracy = 0
for i in range(iters):
    
    batch = get_batch(bsize)
    train_step.run(feed_dict={qubits[0]: batch[0][:,0], 
                              qubits[1]: batch[0][:,1], 
                              qubits[2]: batch[0][:,2], 
                              qubits[3]: batch[0][:,3],
                              y_truth: batch[1],
                              y_truth2: batch[2]})
    if i % 10 == 0:
        train_accuracy = get_train_accuracy()
        print('step %d, training accuracy %g' % (i, train_accuracy))
    if i % 50 == 0:
        test_accuracy = get_test_accuracy()
        print('step %d, test accuracy %g' % (i, test_accuracy))


"""
    if i % 50 == 0 :
        print(get_thing())
        print(get_thing2())
        """


thetas_np=np.zeros(len(thetas))
for i in range(len(thetas)):    
    thetas_np[i]=thetas[i][0].eval()

print(thetas_np)

#sess.close()