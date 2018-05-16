
# coding: utf-8

# # Start. Initialise modules and hyper parameters

# In[1]:


get_ipython().system('jupyter nbconvert --to script Homebrew Quantum Simulator .ipynb')


# In[2]:


import numpy as np
from matplotlib import pyplot as plt
import math
import qutip
import pandas as pd
from qiskit.tools.qi.qi import partial_trace
import scipy
from random import shuffle
import random


# # Import Data & First Look

# In[83]:


iris = pd.read_csv('Iris.csv', header=None)


# In[84]:


iris=np.array(iris)


# In[85]:


#numericalise the labels for graphing
iris[:,4][iris[:,4] == 'setosa']=1
iris[:,4][iris[:,4] == 'versicolor']=2
iris[:,4][iris[:,4] == 'virginica']=3


# In[86]:


def shuffle_data(data,test_size):
    l=[x for x in range(data.shape[0])]
    shuffle(l)
    data_test=data[l[:test_size],:]
    data_train=data[l[test_size:],:]
    return data_train,data_test

def normalise(data):
    l=data.shape[1]
    for i in range(l-1):
        mn=np.min(data[:,i])
        data[:,i]=data[:,i]-mn #add the lowest value
        mx=np.max(data[:,i])
        data[:,i]=(data[:,i]*2/mx) -1
        data[:,i]=data[:,i]*math.pi
    return data

def min_max_norm(data,new_min=-1*math.pi,new_max=math.pi):
    for i in range(data.shape[1]):
        old_min=min(data[:,i])
        old_max=max(data[:,i])
        data[:,i]=((data[:,i]-old_min)/(old_max-old_min)) *(new_max-new_min)+new_min
    return data


# In[87]:


iris[:,:4]=min_max_norm(iris[:,:4], 0,0.5*math.pi)


# In[89]:


easy_iris=iris[:66,:]


# In[90]:


easy_iris[32:,4]=0


# In[91]:


easy_iris_train, easy_iris_test=shuffle_data(easy_iris,10)


# In[92]:


plt.scatter(easy_iris_train[:,1], easy_iris_train[:,0], c=easy_iris_train[:,4])
plt.show()


# In[94]:


easy_iris[:,4]


# In[95]:


iris_hard=iris[33:103,:]
iris_hard.shape
iris_hard[33:,4]=1
iris_hard_train,iris_hard_test=shuffle_data(iris_hard,8)


# In[96]:


iris_hard[:,4]


# ### Normalise the input values between -pi and pi

# In[59]:


iris_hard_train,iris_hard_test=shuffle_data(iris_hard,8)


# In[60]:


iris_hard_test[:,4]


# ### Toolset Data

# In[17]:


tools = pd.read_csv('annotation.csv', header=None)


# In[18]:


tools=np.array(tools)


# In[19]:


tools=tools[1:,1:]


# In[20]:


tools=np.array([[float(y) for y in x] for x in tools])


# In[29]:


tools[:,:2]=min_max_norm(tools[:,:2], 0,0.5*math.pi)


# In[30]:


plt.scatter(tools[:,1], tools[:,0], c=tools[:,2])
plt.show()


# In[21]:


def shuffle_data(data,test_size):
    l=[x for x in range(data.shape[0])]
    shuffle(l)
    data_test=data[l[:test_size],:]
    data_train=data[l[test_size:],:]
    return data_train,data_test

def normalise(data):
    l=data.shape[1]
    for i in range(l-1):
        mn=np.min(data[:,i])
        data[:,i]=data[:,i]-mn #add the lowest value
        mx=np.max(data[:,i])
        data[:,i]=(data[:,i]*2/mx) -1
        data[:,i]=data[:,i]*math.pi
    return data

def min_max_norm(data,new_min=-1*math.pi,new_max=math.pi):
    for i in range(data.shape[1]):
        old_min=min(data[:,i])
        old_max=max(data[:,i])
        data[:,i]=(((data[:,i]-old_min)/(old_max-old_min)) *(new_max-new_min))+new_min
    return data


# In[1034]:


tools_train


# # Create Quantum Circuit 

# ## First begin with Tensor Network tutorial with np.einsum

# ### Creating Tensors

# In[218]:


# rank = number of indexes specified in size
# dimension of index = size of each dimension given 

b=np.random.rand(2,2,4,4) # rank 4 tensor with index dimensions 2,2,4,4

c=np.random.rand(4,4,2,2,3) # rank 5 tensor with index dimensions 4,4,2,2,3


# ### Contracting Tensors with einsum

# In[219]:


np.einsum('ijkl,klnmo->ijnmo',b,c) 
# specify precisely the summed indexes and the arrangement of the output indexes via einsum


# ### Reshaping tensors 

# In[220]:


#can reshape tensor into matrix but must multiply index dimensions correctly to get correct matrix size
c2=np.reshape(c, (16,12))
#cruicially, can reshape back to original too
c3=np.reshape(c, (4,4,2,2,3))
print(c3-c)


# In[115]:


np.load('ae_x_test.npy').shape


# In[114]:


np.load('ae_y_test.npy')


# ### Projecting a vector/input into a tensor (same as contracting except one is a tensor)

# In[82]:


iris[:,4]


# ## End of Tensor Network Tutorial -----

# # Coding MPS's

# # Useful Functions
# 

# In[74]:


def full_tensor(sequence):
    """tensor product of a sequence in the form of a list of arrays""" 
    for k in range(len(sequence)-1):
        if k==0:
            a=sequence[0]
        a=np.array(np.kron(a,sequence[k+1]), dtype=complex)
    return a

def encode(input_set):
    "encodes (normalised between -1 and 1) input data into appropriate qubit vectors. "
    dm=np.zeros([input_set.shape[0],2**input_set.shape[1]], dtype=complex)
    for i in range(input_set.shape[0]): # training set size
        b=[]
        for j in range(input_set.shape[1]): # size of input
            b.append(np.array([math.cos(input_set[i,j]*0.5),math.sin(input_set[i,j]*0.5)]))
        dm[i,:]=full_tensor(b)
    dm=np.kron(np.transpose(dm),dm)
    return dm

def encode_list(input_set,ancilla):
    """encodes a list (already normalised between -pi and pi)"""
    b=[]
    for j in range(len(input_set)):
        b.append(np.array([math.cos(input_set[j]*0.5),math.sin(input_set[j]*0.5)]))
    for k in range(ancilla):
        b.append(np.array([1,0]))
    dm=full_tensor(b)
    dm=np.reshape(np.kron(np.transpose(dm),dm),(2**(len(input_set)+ancilla),2**(len(input_set)+ancilla)))
    return dm


def blockshaped(arr, nrows, ncols):
    #taken from online. this splits arrays into blocks of given size as described below. will need for partial tracing
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def remove_last(dm, env_size=1):
    "traces out the last qubit from a dm"
    if dm.shape[0]!= dm.shape[1]:
        print('Error. Array must be a square matrix')
    if dm.shape[0]==2**env_size:
        print('Error. Environment size is the same size as the matrix you wanna trace')
    if dm.shape[0]>2**env_size:
        blocks=blockshaped(dm,2**env_size,2**env_size)
        no_blocks=len(blocks)
        new_dm=np.zeros(no_blocks, dtype=complex)
        for j in range(no_blocks):
            new_dm[j]=np.trace(blocks[j])
        new_dm=np.reshape(new_dm,(int(math.sqrt(no_blocks)),int(math.sqrt(no_blocks))))
    return new_dm

def tracing_out(dm,tracing_out):
    l=[]
    for i in range(tracing_out):
        l.append(i) #l contains list of indexes to trace out
    new_dm=partial_trace(dm,l)
    return new_dm

def measure_zero(dm):
    "measures a qubit and outputs the probability of getting a zero"
    if dm.shape[0]!=2:
        print("your density matrix isnt 2x2")
    a=np.array([[1,0],[0,0]])
    prob=np.trace(np.matmul(dm,a))
    return prob
    
def initialise(no_inputs, ancilla, bond_dimension,identity=0):
    """generates the right number of random unitaries of the right size"""
    total=no_inputs+ancilla
    operator_list=[]
    temp_operator_list=[]
    for i in range(no_inputs):
        if identity==1:
            operator_list.append(np.identity(bond_dimension).astype(complex))
        elif identity==2:
            tt=qutip.rand_unitary(bond_dimension).full()
            operator_list.append(np.dot(tt,tt*1J))
        else:
            operator_list.append(qutip.rand_unitary(bond_dimension).full().astype(complex))
    
    return operator_list

def evaluate_circuit(inputs,operator_list, trace):
    for i in range(len(operator_list)):
        diff=int(math.log2(inputs.shape[0])-math.log2(operator_list[i].shape[0]))
        if diff>0:
            inputs=np.matmul(np.kron(operator_list[i],np.identity(2**diff)), inputs)
            inputs=np.matmul(inputs,np.matrix(np.kron(operator_list[i],np.identity(2**diff))).getH())
        else:
            inputs=np.matmul(operator_list[i], inputs)
            inputs=np.matmul(inputs,np.matrix(operator_list[i]).getH())
        if trace==1:
            inputs=partial_trace(inputs,[0])
    return inputs

def extract_final(dm):
    L=dm.shape[0]
    if L>2:
        dm=tracing_out(dm,int(math.log2(L)-1))
    return dm

def cost_calc(label,answer):
    answer=(1-np.real(answer))
    cost=abs(label-answer)
    return cost
def cost_calc2(label,answer):
    answer=(1-np.real(answer))
    cost=abs(label-np.round(answer))
    return cost

def eval_cost_function(operator_list,test_data,rounding):
    cost=np.zeros(test_data.shape[0])
    for k in range(len(test_data)):
        input_values=encode_list([test_data[k,0],test_data[k,1]],ancilla)
        label=test_data[k,2]
        final_dm1=evaluate_circuit(input_values,operator_list, trace=1)
        final_dm2=extract_final(final_dm1)
        answer=measure_zero(final_dm2)
        if answer>1 or answer<0:
            print("error, weird probability")
            print(answer)
            print(final_dm2)
            print('dm2 trace is', np.trace(final_dm2))
            print(final_dm1)
            print('dm1 trace is', np.trace(final_dm2))
        if rounding==0:
            cost[k]=cost_calc(label,answer)
        if rounding==1:
            cost[k]=cost_calc2(label,answer)
    
    total_cost=sum(cost)
    return total_cost

"""def test(operator_list):
    for k in range(len(test_label)):
        input_values=encode_list([test_iris0[k],test_iris1[k]],ancilla)
        label=test_label[k]
        final_dm1=evaluate_circuit(input_values,operator_list, trace=1)
        final_dm2=extract_final(final_dm1)
        answer=ure_zero(final_dm2)

        cost[k]=cost_calc(label,answer)
    total_cost=sum(cost)
    return total_cost"""
def regularise(mat,power):
    "just zeros tiny numbers"
    np.imag(mat)[abs(np.imag(mat))<10**(-power)]=0
    np.real(mat)[abs(np.real(mat))<10**(-power)]=0
    return mat

def ry(theta,qubit,total):
    ry=np.array([[math.cos(theta/2), -1*math.sin(theta/2)],[math.sin(theta/2),math.cos(theta/2)]])
    ry=np.kron(np.identity(2**qubit),ry)
    ry=np.kron(ry,np.identity(2**(total-qubit-1)))
    return ry   
        
def initialise_params(no_params):
    params=np.zeros(no_params)
    for i in range(no_params):
        params[i]=(random.random())
    params=(2*(params)-1)*math.pi
    return params

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


def cnot(ctrl,target,total):
    if ctrl<target:
        cnot=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    if target<ctrl:
        cnot=np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    for k in range(min(ctrl,target)):
        cnot=np.kron(np.identity(2),cnot)
    for k in range(total-max(ctrl,target)-1):
        cnot=np.kron(cnot,np.identity(2))
    return cnot

def keep_last(psi,total):
    dm=np.kron(np.transpose(psi),psi)
    dm=np.reshape(dm,(2**total,2**total))
    dm=partial_trace(dm,list(range(total-1)))
    return dm

def keep_n(psi,total,chosen_qubit):
    dm=np.kron(np.transpose(psi),psi)
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

def evaluate_MPS(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total+j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS_plotter(params, training_data, ancilla,rounding):
    guesses=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total+j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        guesses[i]=1-round(zero_prob)
    return guesses

def evaluate_4TTN(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)
    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_4TTN_plotter(params, training_data, ancilla,rounding):
    guesses=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        guesses[i]=1-round(zero_prob)
    return guesses
    return total_cost

def evaluate_4TTN_closed(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)
    total_cost=np.sum(answers)#/training_data.shape[0]
    return total_cost


# In[39]:


def evaluate_MPS_double(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        #psi=initial_encode(training_data[i,:L],ancilla)
        norm1=(training_data[i,0]**2+training_data[i,1]**2)**0.5
        norm2=(training_data[i,2]**2+training_data[i,3]**2)**0.5
        psi=np.kron(np.array([training_data[i,0]/norm1, training_data[i,1]/norm1]),np.array([training_data[i,2]/norm2, training_data[i,3]/norm2]))
        psi=np.reshape(psi,(2**total,1))
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        psi=np.matmul(cnot(0,1,total),psi)
        for j in range(total):
            psi=np.matmul(ry(params[total+j],j,total),psi)
        psi=np.matmul(cnot(1,0,total),psi)
        
        psi=np.matmul(ry(params[4],0,total),psi)
        dm=np.kron(np.transpose(psi),psi)
        dm=np.reshape(dm,(2**total,2**total))
        dm=partial_trace(dm,[1])
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(dm)
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,4],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS_double2(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        #psi=initial_encode(training_data[i,:L],ancilla)
        norm1=(training_data[i,0]**2+training_data[i,1]**2)**0.5
        norm2=(training_data[i,2]**2+training_data[i,3]**2)**0.5
        psi=np.kron(np.array([training_data[i,0]/norm1, training_data[i,1]/norm1]),np.array([training_data[i,2]/norm2, training_data[i,3]/norm2]))
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(1):
            psi=np.matmul(ry(params[0],0,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        psi=np.matmul(cnot(0,1,total),psi)
        for j in range(1):
            psi=np.matmul(ry(params[1],1,total),psi)
        psi=np.matmul(cnot(1,0,total),psi)
        
        psi=np.matmul(ry(params[2],0,total),psi)
        dm=np.kron(np.transpose(psi),psi)
        dm=np.reshape(dm,(2**total,2**total))
        dm=partial_trace(dm,[1])
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(dm)
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,4],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS2(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total+j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost


# In[24]:


xtrain=np.load("ae_x_train.npy")
xtest=np.load("ae_x_test.npy")
ytrain=np.load("ae_y_train.npy")
ytest=np.load("ae_y_test.npy")


# In[28]:


easy_iris[:,0]


# In[45]:


xtest2=np.reshape(xtest,(10000,8))


# In[49]:


ae_test=np.zeros(shape=(10000,9))


# In[50]:


ae_test[:,:8]=xtest2
ae_test[:,8]=ytest


# In[53]:


ae_training=np.load('ae_training.npy.npz')


# In[57]:


np.save('ae_test',ae_test)


# In[64]:


ae_training[ae_training[:,8]==0][:100,:]
ae_training[ae_training[:,8]!=0]


# In[76]:


learning_zero=np.zeros(shape=(400,9))
learning_zero[:100,:]=ae_training[ae_training[:,8]==0][:100,:]
learning_zero[100:,:]=ae_training[ae_training[:,8]!=0][:300,:]
learning_zero[100:,8]=1


# In[92]:


np.save('learning_zero_train',learning_zero)


# In[60]:


the_zeros=ae_training[:,ae_training[:,8]==0


# In[79]:


easy_iris[:,4]


# In[82]:


learning_zero[:,:8]=min_max_norm(learning_zero[:,:8], 0,0.5*math.pi)


# In[112]:


rounding=0
no_qubits=4
ancilla=0
total=no_qubits+ancilla
print(total)
data=iris_hard
data1,data2=shuffle_data(data,8)
params=initialise_params(2*(total)-1)
function=evaluate_MPS2
params=scipy.optimize.minimize(function,params,args=(data1,ancilla,0), method='CG')['x']
print(function(params,data1,ancilla,rounding=1))
function(params,data2,ancilla,rounding=1)


# In[113]:


params/math.pi


# In[114]:


data2[:,4]


# In[52]:


easy_iris_test[:,4]


# In[115]:


np.save('NEW_MPS_hardiris_good', params)


# In[107]:


params=np.load('NEW_MPS_hardiris_params.npy')
function(params,data2,ancilla,rounding=1)


# In[ ]:





# In[91]:


params


# In[30]:


learning_zero=np.load('learning_zero_train.npy')


# In[ ]:





# In[93]:


np.save('8MPS_identifyingzero',params)


# In[68]:


training_data=easy_iris
for i in range(1):
        norm1=(training_data[i,0]**2+training_data[i,1]**2)**0.5
        print(norm1)
        norm2=(training_data[i,2]**2+training_data[i,3]**2)**0.5
        print(norm2)
        psi=np.kron(np.array([training_data[i,0]/norm1, training_data[i,1]/norm1]),np.array([training_data[i,2]/norm2, training_data[i,3]/norm2]))
        for j in range(total):
                    psi=np.matmul(ry(params[j],j,total),psi)
                    """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        print(psi)
        psi=np.matmul(cnot(0,1,total),psi)  
        print(psi)
        psi=np.matmul(cnot(1,0,total),psi)
        print(np.matmul(ry(params[0],0,2),psi))


# In[666]:


easy_iris[:,3]


# In[701]:


np.savez('4TTN_iris_hard_params',params)


# In[707]:


fake_params=[3.64129925,1.04422998, 2.99327683, 4.18602991,1.3215133,2.78138208,4.68090534]
real_params=[]
for i in fake_params:
    real_params.append(i*2)
print(real_params)
test_data=iris


# In[713]:


evaluate_4TTN(params,iris_hard,ancilla,rounding=1)


# In[698]:


data=iris_hard_train


# In[699]:


plt.scatter(data[:,1], data[:,0], c=evaluate_4TTN_plotter(params,data,ancilla,rounding=1))
plt.show()


# In[700]:


plt.scatter(data[:,1], data[:,0], c=data[:,4])
plt.show()


# In[712]:





# In[478]:


plt.scatter(iris[:,1], iris[:,0], c=iris[:,4])
plt.show()


# In[381]:


np.savez('4q_0a_hardiris4',params)


# In[331]:


pyplot


# In[544]:


"""THIS CIRCUIT IS MPS STYLE ONLY"""
total_cost=0
no_qubits=4
ancilla=0
total=ancilla+no_qubits
no_params=(2*total)-1
training_data=easy_iris#_train #remember, labels are data[:,4] here
test_data=iris_hard_test
params=initialise_params(no_params)
cost=np.zeros(training_data.shape[0])
total_cost=evaluate_MPS(params,training_data,ancilla,rounding=0)
print(params)
for p in range(no_params):
    for k in reversed(range(no_params)):
        b=math.pi
        s=-1*math.pi
        n=100
        for z in range(2):
            cost_list=np.zeros(n)

            new_params=np.copy(params)
            candidate_list=np.zeros(n)
            for l in range(n):
                candidate_list[l]=s+((b*(1-l))/n)
                new_params[k]=candidate_list[l]
                cost_list[l]=evaluate_MPS(new_params,training_data,ancilla,rounding=0)
            winner=np.argmin(cost_list)
            if winner==0:
                neighbour=1
            elif winner==n-1:
                neighbour=winner-1
            else:
                if cost_list[winner+1]<cost_list[winner-1]:
                    neighbour=winner+1
                if cost_list[winner-1]<cost_list[winner+1]:
                    neighbour=winner-1

            total_cost=np.min(cost_list)
            params[k]=candidate_list[winner]
            s=np.min([candidate_list[winner],candidate_list[neighbour]])
            b=np.max([candidate_list[winner],candidate_list[neighbour]])
            """if z%2==0:
                iris_hard_train, iris_hard_train=shuffle_data(iris_hard,8)"""
            """delta=1
            for z in range(10):
                up_params=np.copy(params)
                up_params[k]=up_params[k]+np.random.rand()*delta
                up_cost=evaluate_MPS(training_data,up_params,ancilla,rounding=0)
                diff_up=(up_cost-total_cost)
                down_params=np.copy(params)
                down_params[k]=down_params[k]-np.random.rand()*delta
                down_cost=evaluate_MPS(training_data,down_params,ancilla,rounding=0)
                diff_down=(down_cost-total_cost)
                if diff_up<0 and diff_down<0:
                    if diff_up<diff_down:
                        params[k]=up_params[k]
                        new_cost=up_cost
                    elif diff_down<diff_up:
                        params[k]=down_params[k]
                        new_cost=down_cost
                elif diff_up<0 and diff_down>0:
                    params[k]=up_params[k]
                    new_cost=up_cost
                elif diff_up>0 and diff_down<0:
                    params[k]=down_params[k]
                    new_cost=down_cost
                if abs(new_cost-total_cost)<0.1:
                    #print('hey')
                    delta=delta*0.99
                    #print(delta)
                total_cost=new_cost"""
        print('cost is', total_cost)
          #  print('deta is', delta)
        print('rounded cost is',evaluate_MPS(params,training_data,ancilla,rounding=1))
        print('params are', params)


# In[531]:


"""THIS CIRCUIT IS MPS STYLE ONLY"""
total_cost=0
no_qubits=4
ancilla=0
total=ancilla+no_qubits
no_params=(2*total)-1
training_data=easy_iris_train #remember, labels are data[:,4] here
test_data=easy_iris_test
params=initialise_params(no_params)
cost=np.zeros(training_data.shape[0])
total_cost=evaluate_MPS(params,training_data,ancilla,rounding=0)
print(params)
mu=1.5
delta=0.0000000000001
for m in range(2):
    for k in range(no_params):
        for z in range(200):
            new_params=np.copy(params)
            new_params[k]=new_params[k]+delta
            new_cost=evaluate_MPS(new_params,training_data,ancilla,rounding=0)
            diff=new_cost-total_cost
            if abs(diff)>0.1:
                params[k]=params[k]+delta
            if abs(diff)<0.1:
                params[k]=params[k]+delta*mu
        
            total_cost=new_cost
            #print(total_cost)
    print('cost is', total_cost)
      #  print('deta is', delta)
    #print('rounded cost is',evaluate_MPS(params,training_data,ancilla,rounding=1))
    #print('params are', params)


# In[ ]:





# In[518]:


params


# In[541]:


np.savez('4q_MPS_circuit_iris_hard_params_6_good', params)


# In[519]:


print('cost is', total_cost)
print('rounded cost is',evaluate_MPS(params,iris_hard,ancilla,rounding=1))
print('params are', params)
print(params/math.pi)


# In[ ]:





# In[252]:


print('rounded cost is',evaluate_MPS(test_data,params,ancilla,rounding=1))
print('params are', params)


# In[484]:


params/math.pi


# In[201]:


b=np.random.rand(10)
np.argmin(b)


# In[290]:


"""Nelder Mead Style"""

"""THIS CIRCUIT IS MPS STYLE ONLY"""
total_cost=0
no_qubits=4
ancilla=0
total=ancilla+no_qubits
no_params=(2*total)-1
training_data=iris_hard_train #remember, labels are data[:,4] here
test_data=iris_hard_test
alpha=1
beta=0.5
gamma=2
delta=0.5
points=np.zeros((no_params,no_params+1))
point_costs=np.zeros(no_params+1)
for i in range(no_params+1):
    params[i]=initialise_params(no_params)
    point_costs[i]=evaluate_MPS(training_data,params[i],ancilla,rounding=0)

best=np.min(point_costs)
points=points[:,np.argsort(point_costs)
centroid
        
    print('cost is', total_cost)
      #  print('deta is', delta)
    #print('rounded cost is',evaluate_MPS(training_data,params,ancilla,rounding=1))
   # print('params are', params)


# In[60]:


cnot(1,0,2)


# In[26]:


np.savez('2_8_operators_halfhalftraintest_iris.npz',operator_list)


# In[ ]:


qutip.Qobj(encode_list(small_iris_train[:,1],2))


# In[29]:


qutip.Qobj(operator_list[0])


# In[18]:


print("Final Cost with rounding is " ,eval_cost_function(operator_list,data,rounding=1))
print("Test Result without rounding is ", eval_cost_function(operator_list,test_data,rounding=0))
print("Test Result with rounding is ", eval_cost_function(operator_list,test_data,rounding=1))


# In[90]:


eval_cost_function(operator_list,data,rounding=1)


# In[305]:


scipy.optimize.minimize(evaluate_MPS(params,training_data,ancilla,rounding=0),params, method='CG')


# In[307]:





# In[300]:


params


# In[168]:


N=np.random.rand(4,4)
def trace_last(dm,tracing_out):
    for i in range(tracing_out):
        L=int(dm.shape[0]*0.5)
        a=np.trace(N[0:L,0:L])
        b=np.trace(N[0:L,L:2*L])
        c=np.trace(N[L:2*L,0:L])
        d=np.trace(N[L:2*l,L:2*L])
        l=[a,b,c,d]
        l=np.asarray(l)
        dm=np.reshape(l,(2,2))
    return dm


# In[171]:


N


# In[184]:


a=np.trace(N[0:2,0:2])
b=np.trace(N[0:2,2:4])
c=np.trace(N[2:4,0:2])
d=np.trace(N[2:4,2:4])
l=[a,b,c,d]
l=np.asarray(l)
print(l)
np.reshape(l,(2,2))


# In[43]:


small_iris[30:45,:]


# In[300]:


# function(normalise) that normalises data appropriately XXX
# function(create_dm) produces initial rho XXXXX
# function(get rid of) ditches the first qubit XXXXX
# function(measure) measures the last qubit (super easy) XXXX
# function(elongate) that tensors a unitary with appropriate sized identity matrix XXXX
# function(initialise) that initialises right number (and size) of unitaries XXXX
# process (optimise) that optimises these unitaries according to a cost function
# process (evaluate) that evaluates the circuit.
z=np.random.rand(5,1)
np.kron(np.transpose(z),z)


# In[659]:


print(small_iris[0:10,1].shape)
print(test_iris1.shape)


# # TO DO LIST:
# 
# ## - check the existing quantum architectures that we are using
# 
# ## - read on "proper" evolution with unitarity constraints
# 
# ## - code the circuit
# 
# ## - try a completely adaptive implementation
# 
# ## - consider a purely quantum optimisation

# In[34]:


A=qutip.rand_unitary(4).full()
A


# In[35]:


q,r=np.linalg.qr(A)


# In[36]:


np.imag(r)[abs(np.imag(r))<0.000001]=0
np.real(r)[abs(np.real(r))<0.000001]=0
r


# In[37]:


q


# In[56]:


for k in range(0):
    print(k)


# In[ ]:




