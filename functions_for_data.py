import math
from random import shuffle


def shuffle_data(data,test_size):
    l=[x for x in range(data.shape[0])]
    shuffle(l)
    data_test=data[l[:test_size],:]
    data_train=data[l[test_size:],:]
    return data_train,data_test

def min_max_norm(data,new_min=-1*math.pi,new_max=math.pi):
    for i in range(data.shape[1]):
        old_min=min(data[:,i])
        old_max=max(data[:,i])
        data[:,i]=((data[:,i]-old_min)/(old_max-old_min)) *(new_max-new_min)+new_min
    return data