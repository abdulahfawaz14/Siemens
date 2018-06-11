import numpy as np
from random import shuffle
import random
from functions_for_data import shuffle_data

samples = 50
total = 4*samples*2

A = np.zeros((total,10))
a = np.ones(3)
a = np.triu(a)
for i in range(4*samples):
    a=np.rot90(a)
    b = np.random.rand()
    A[i,0] = 0
    A[i,1:] =  np.reshape(a,(1,9))*b
    
c = np.array([[1,1,0],[1,1,0],[1,1,0]])

for i in range(4*samples):
    c=np.rot90(c)
    d = np.random.rand()
    A[i+ 4*samples,0] = 1
    A[i + 4*samples,1:] =  np.reshape(c,(1,9))*d
    
np.savetxt('directions_train.csv',A, fmt='%10.5f', delimiter =',',newline='\n')


samples = 10
total = 4*samples*2

A = np.zeros((total,10))
a = np.ones(3)
a = np.triu(a)
for i in range(4*samples):
    a=np.rot90(a)
    b = np.random.rand()
    A[i,0] = 0
    A[i,1:] =  np.reshape(a,(1,9))*b
    
c = np.array([[1,1,0],[1,1,0],[1,1,0]])

for i in range(4*samples):
    c=np.rot90(c)
    d = np.random.rand()
    A[i+ 4*samples,0] = 1
    A[i + 4*samples,1:] =  np.reshape(c,(1,9))*d

np.savetxt('directions_test.csv',A, fmt='%10.5f', delimiter =',',newline='\n')