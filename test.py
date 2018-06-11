import numpy as np
from functions_for_data import shuffle_data
from numpy import genfromtxt
import csv
import random


data = np.load('ae_test.npy')

threes_test = data[data[:,8] == 2,:]

fives_test  = data[data[:,8] == 4,:]



data2 = np.load('ae_training.npy')


threes_train = data2[data2[:,8] == 2,:]

fives_train  = data2[data2[:,8] == 4,:]

def make_n(arr,n):
    for i in range(arr.shape[0]):
        arr[:,arr.shape[1]-1] = n
    return arr

threes_train = make_n(threes_train,0)
three_test = make_n(threes_test, 0)

fives_train = make_n(fives_train,1)
fives_test = make_n(fives_test, 1)


np.random.shuffle(threes_train)
np.random.shuffle(threes_test)

np.random.shuffle(fives_test)
np.random.shuffle(fives_train)

tf_test = np.vstack((threes_test[:50,:], fives_test[:50,:]))
tf_train = np.vstack((threes_train[:200,:], fives_train[:200,:]))



data3 = np.ones((tf_test.shape[0],tf_test.shape[1]))
data3[:,1:] = tf_test[:,:-1]
data3[:,0] = tf_test[:,8]


data4 = np.ones((tf_train.shape[0], tf_train.shape[1]))
data4[:,1:] = tf_train[:,:-1]
data4[:,0] = tf_train[:,8]




np.savetxt('24_train.csv',data4, fmt='%10.5f', delimiter =',',newline='\n')
np.savetxt('24_test.csv',data3, fmt='%10.5f', delimiter =',',newline='\n')
