from utilities import full_tensor
from qiskit.tools.qi.qi import partial_trace
import math
import random
import numpy as np

def initialise_params(no_params):
    params=np.zeros(no_params)
    for i in range(no_params):
        params[i]=(random.random())
    params=(2*(params)-1)*math.pi
    return params

def get_batches(data,number_batches):    
    new_data = np.zeros(shape=(number_batches,data.shape[1]))
    L = data.shape[0]
    A = random.sample(list(range(L)),number_batches)
    for i in range(number_batches):
        new_data[i,:]=data[A[i],:]
    return new_data

def initial_encode(input_set, ancilla):
    """encodes a list (already normalised between -pi and pi)"""
    b=[]
    for j in range(len(input_set)):
        b.append(np.array([math.cos(input_set[j]),math.sin(input_set[j])]))
    for k in range(ancilla):
        b.append(np.array([1,0]))
    psi=full_tensor(b)
    return psi

def initial_encode2(input_set, ancilla):
    """encodes a list (already normalised between -pi and pi)"""
    b=[]
    b.append(np.array([math.cos(input_set[0]),math.sin(input_set[0])]))
    for k in range(ancilla):
        b.append(np.array([1,0]))
    for j in range(len(input_set)-1):
        b.append(np.array([math.cos(input_set[j+1]),math.sin(input_set[j+1])]))
    psi=full_tensor(b)
    return psi


def keep_last(psi,total):
    dm=np.kron(np.transpose(np.conjugate(psi)),psi)
    dm=np.reshape(dm,(2**total,2**total))
    dm=partial_trace(dm,list(range(total-1)))
    return dm

def keep_n(psi,total,chosen_qubit):
    dm=np.kron(np.transpose(np.conjugate(psi)),psi)
    dm=np.reshape(dm,(2**total,2**total))
    LIST=list(range(total-1))
    del LIST[chosen_qubit]
    dm=partial_trace(dm,LIST)
    return dm

def prob_zero(dm):
    if dm[0,0]<0:
        print('negative probability')
    return dm[0,0]

def eval_cost(prob_zero,label,rounding):
    if rounding==1:
        prob_zero=round(prob_zero)
    answer=0.5*(((1-prob_zero)-label)**2)
    return answer