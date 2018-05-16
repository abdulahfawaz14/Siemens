import numpy as np
from functions_for_data import shuffle_data
import random
import scipy
from scipy.special import comb
import itertools


number_per_class = 20

total = 9

data_set = []


def connected_or_not(ones):
    L = len(ones)
    new_ones = list(ones)
    connected = [new_ones[0]]
    del new_ones[0]
    #check vertically
    while len(new_ones)!= 0: # so long as we're adding or have stuff to add
        s = len(connected)
        for a in connected:
            for i in new_ones:
                if abs((i-1)%3 -(a-1)%3) == 1 and abs((i-1)//3 -(a-1)//3) == 0:
                    connected.append(i)
                    new_ones.remove(i)

                elif abs((i-1)%3 -(a-1)%3) == 0 and abs((i-1)//3 -(a-1)//3) == 1:
                    connected.append(i)
                    new_ones.remove(i)
        if len(connected) == s:
            return 0
            break
    if len(connected) == L:
        return 1
    else:
        return 0



SET = list(range(1,total+1))
for i in list(range(2,total+1)):
    for k in range(int(comb(total,i))):
        a = random.sample(set(SET),i)
        a.sort()
        c=connected_or_not(a)
        a.append(c)

        if a not in data_set:
            data_set.append(a)
np.save('continuous_indexes_and_labels', data_set)