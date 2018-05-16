
# coding: utf-8

# In[1]:


import qiskit
from qiskit import QuantumProgram
import pandas as pd
import numpy as np
import math
import random
from random import shuffle
import projectq
import scipy
from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure, Ry, CNOT


# ## Normalise Data

# In[89]:


params=np.load('NEW_MPS_easyiris_params2.npy')


# In[90]:


print(params)


# In[48]:


iris = pd.read_csv('Iris.csv', header=None)
iris=np.array(iris)
#numericalise the labels for graphing
iris[:,4][iris[:,4] == 'setosa']=1
iris[:,4][iris[:,4] == 'versicolor']=2
iris[:,4][iris[:,4] == 'virginica']=3

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


iris[:,:4]=min_max_norm(iris[:,:4], 0,0.5*math.pi)
easy_iris=iris[:66,:]
easy_iris[32:,4]=0
easy_iris_train, easy_iris_test=shuffle_data(easy_iris,8)


# In[49]:


easy_iris_test[:,4]


# In[100]:


iris_hard=iris[33:103,:]
iris_hard.shape
iris_hard[33:,4]=1
iris_hard_train,iris_hard_test=shuffle_data(iris_hard,)
print(iris_hard[:,4])
data2=iris_hard_test


# In[12]:


d1=np.load('learning_zero_train.npy')
d1


# In[12]:


learning_zero=d1
data1,data2=shuffle_data(learning_zero,50)


# In[96]:


def MPS(params, data, shots,rounding):
    answers=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x=np.zeros(shots)
        for k in range(shots):
            eng = MainEngine()  # create a default compiler (the back-end is a simulator)

            q1 = eng.allocate_qubit()  # allocate 1 qubit
            q2 = eng.allocate_qubit()
            q3 = eng.allocate_qubit()
            q4 = eng.allocate_qubit()

            Ry(data[i,0]*2) | q1  
            Ry(data[i,1]*2) | q2  
            Ry(data[i,2]*2) | q3  
            Ry(data[i,3]*2) | q4

            Ry(params[0]) | q1  
            Ry(params[1]) | q2  
            Ry(params[2]) | q3  
            Ry(params[3]) | q4  
            
            CNOT | (q1,q2)
            Ry(params[4]) | q2
            
            CNOT | (q2,q3)
            Ry(params[5]) | q3
            
            CNOT | (q3,q4)
            Ry(params[6]) | q4
            
            Measure | q4  # measure the qubit
            Measure | q1
            Measure | q2
            Measure | q3
            eng.flush()

            x[k]=int(q4)
            #print(x)
        #print(x)
        idx=x==0
        num_zeros=shots-sum(x)
        prop_zeros=num_zeros/shots
        #print(prop_zeros)
        if rounding==1:
            answers[i]=1-round(prop_zeros)
            answers[i]=abs(answers[i]-data[i,4])
        if rounding ==0:
            prop_ones=1-prop_zeros
            answers[i]=abs(prop_ones-data[i,4])
    print(np.sum(answers))
    print(answers)
    return(np.sum(answers))


# In[57]:


def MPS_8(params, data, shots,rounding):
    answers=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x=np.zeros(shots)
        for k in range(shots):
            eng = MainEngine()  # create a default compiler (the back-end is a simulator)

            q1 = eng.allocate_qubit()  # allocate 1 qubit
            q2 = eng.allocate_qubit()
            q3 = eng.allocate_qubit()
            q4 = eng.allocate_qubit()
            q5 = eng.allocate_qubit()  # allocate 1 qubit
            q6 = eng.allocate_qubit()
            q7 = eng.allocate_qubit()
            q8 = eng.allocate_qubit()

            Ry(data[i,0]*2) | q1  
            Ry(data[i,1]*2) | q2  
            Ry(data[i,2]*2) | q3  
            Ry(data[i,3]*2) | q4
            Ry(data[i,4]*2) | q5  
            Ry(data[i,5]*2) | q6  
            Ry(data[i,6]*2) | q7  
            Ry(data[i,7]*2) | q8


            Ry(params[0]) | q1  
            Ry(params[1]) | q2  
            Ry(params[2]) | q3  
            Ry(params[3]) | q4 
            Ry(params[4]) | q5  
            Ry(params[5]) | q6  
            Ry(params[6]) | q7  
            Ry(params[7]) | q8 
            
            
            CNOT | (q1,q2)
            Ry(params[8]) | q2
            CNOT | (q2,q3)
            Ry(params[9]) | q3
            CNOT | (q3,q4)
            Ry(params[10]) | q4
            CNOT | (q4,q5)
            Ry(params[11]) | q5
            CNOT | (q5,q6)
            Ry(params[12]) | q6
            CNOT | (q6,q7)
            Ry(params[13]) | q7
            CNOT | (q7,q8)
            Ry(params[14]) | q8
            
            
            Measure | q4  # measure the qubit
            Measure | q1
            Measure | q2
            Measure | q3
            Measure | q5
            Measure | q6
            Measure | q7
            Measure | q8
    
            x[k]=int(q8)
            eng.flush()
        #print(x)
        idx=x==0
        num_zeros=sum(idx)
        #print(num_zeros)
        prop_zeros=num_zeros/shots
        if rounding==1:
            answers[i]=round(prop_zeros)
            answers[i]=abs(answers[i]-data[i,8])
        if rounding ==0:
            prop_ones=1-prop_zeros
            answers[i]=abs(prop_ones-data[i,8])
    print(np.sum(answers)/data.shape[0])
    return(np.sum(answers)/data.shape[0])


# In[84]:


import projectq
from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure, Ry, CNOT  # import the operations we want to perform (Hadamard and measurement)

shots=1024    


 
x=np.zeros(shots)

function=MPS
params=scipy.optimize.minimize(function,params,args=(easy_iris_train,1024,0), method='CG')['x']
print(MPS(params,easy_iris_test,shots=1024,rounding=0))


# In[110]:


params=np.load('NEW_MPS_hardiris_good.npy')


# In[114]:


iris_hard_test[:,4]


# In[108]:


params


# In[115]:


print(MPS(params,iris_hard_train,shots=1024,rounding=1))


# In[95]:


easy_iris_test[:,4]


# In[44]:


easy_iris_test[:,4]


# In[24]:


params2=np.load('4MPS_hard_Yonly_good.npz')
params2=params2['arr_0']
shots=1024
print(MPS(params2,iris_hard_train,shots,rounding=1))


# In[69]:


from projectq.backends import CircuitDrawer
drawing_engine = CircuitDrawer()
eng = MainEngine(drawing_engine)  # create a default compiler (the back-end is a simulator)

q1 = eng.allocate_qubit()  # allocate 1 qubit
q2 = eng.allocate_qubit()
q3 = eng.allocate_qubit()
q4 = eng.allocate_qubit()

Ry(2) | q1  
Ry(2) | q2  
Ry(2) | q3  
Ry(2) | q4

Ry(1) | q1  
Ry(1) | q2  
Ry(1) | q3  
Ry(1) | q4  

CNOT | (q1,q2)
Ry(params[4]) | q2

CNOT | (q2,q3)
Ry(params[5]) | q3

CNOT | (q3,q4)
Ry(params[6]) | q4

Measure | q4  # measure the qubit
Measure | q1
Measure | q2
Measure | q3
eng.flush()

print(drawing_engine.get_latex())


# In[35]:


x=np.random.randint(0,2,size=(10,1))
print(x)
idx=x==0
print(idx)
print(sum(idx)/len(x))


# In[79]:


from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure  # import the operations we want to perform (Hadamard and measurement)
for k in range(10):
    eng = MainEngine()  # create a default compiler (the back-end is a simulator)

    qubit = eng.allocate_qubit()  # allocate 1 qubit
    H | qubit  # apply a Hadamard gate
    Measure | qubit  # measure the qubit  # flush all gates (and execute measurements)
    print("Measured {}".format(int(qubit)))  # output measurement result


# In[3]:


def min_max_norm(data,new_min=-1*math.pi,new_max=math.pi):
    for i in range(data.shape[1]):
        old_min=min(data[:,i])
        old_max=max(data[:,i])
        data[:,i]=((data[:,i]-old_min)/(old_max-old_min)) *(new_max-new_min)+new_min
    return data


# ## Import Data

# In[4]:


#import iris
iris = pd.read_csv('Iris.csv', header=None)
iris=np.array(iris)

#numericalize 
iris[:,4][iris[:,4] == 'setosa']=1
iris[:,4][iris[:,4] == 'versicolor']=2
iris[:,4][iris[:,4] == 'virginica']=3


# In[5]:


#normalise the non-label bit between -2 and 2
min_max_norm(iris[:,:4], 0,0.5*math.pi)
#print(iris)


# In[7]:


iris[:64,4]


# ## Create template

# ### first see if i can repeat what they did

# ### Break the data down into encodings

# In[6]:


def initialise_params(no_params):
    params=np.zeros(no_params)
    for i in range(no_params):
        params[i]=(random.random())
    params=(2*(params)-1)*math.pi
    return params

def shuffle_data(data,test_size):
    l=[x for x in range(data.shape[0])]
    shuffle(l)
    data_test=data[l[:test_size],:]
    data_train=data[l[test_size:],:]
    return data_train,data_test


# In[80]:


initialise_params(10)


# In[8]:


data_values=iris[:66,:4]
labels=iris[:66,4]-1

qp=QuantumProgram()
no_qubits=4
qr=qp.create_quantum_register('qr',no_qubits)
cr = qp.create_classical_register('cr', 1)
qc = qp.create_circuit('test', [qr], [cr])
params=initialise_params(7)
params=[1,1.5,2,2.5,3,3.5,4]
single_cost=np.zeros(len(labels))
#for k in range(len(labels)):
def evaluate_cost(data_values, labels, params):
    for k in range(len(labels)):
        for j in range(data_values.shape[1]):
            qc.ry((2*data_values[k,j]),qr[j])

        for j in range(no_qubits):
            qc.ry(params[j],qr[j])

        qc.cx(qr[0],qr[1])
        qc.cx(qr[3],qr[2])
        qc.ry(params[no_qubits],qr[1])
        qc.ry(params[no_qubits+1],qr[2])
        qc.cx(qr[1],qr[2])
        qc.ry(params[no_qubits+2],qr[2])
        qc.measure(qr[2],cr[0])
        result=qp.execute('test')
    #    print(result.get_counts('test'))
        prob_zero=result[0]['data']['counts']['0']/(result[0]['data']['counts']['1']+result[0]['data']['counts']['0'])

        single_cost[k]=abs((1-prob_zero)-labels[k])
        if single_cost[k]<0:
            print('oops')
            print(prob_zero)
            print(labels[k])
        #print(single_cost[k])
    total_cost=np.sum(single_cost)
    print(total_cost)
    return total_cost
print('begin')
total_cost=evaluate_cost(data_values, labels, params)
"""repetitions=10
while total_cost>0.005:
    for l in range(len(params)):
        new_params=np.copy(params)
        new_params[l]=new_params[l]+0.01
        diff=(evaluate_cost(data_values,labels,new_params)-total_cost)/total_cost
        params[l]=params[l]+(diff*params[l])
        total_cost=evaluate_cost(data_values, labels,params)
    print(total_cost)"""
print(total_cost)
print(params)


# In[10]:


from projectq.backends import CircuitDrawer


# In[ ]:


def evaluate_cost(data_values, labels, params):
    for k in range(len(labels)):
        for j in range(data_values.shape[1]):
            qc.ry((2*data_values[k,j]),qr[j])

        for j in range(no_qubits):
            qc.ry(params[j],qr[j])

        qc.cx(qr[0],qr[1])
        qc.cx(qr[3],qr[2])
        qc.ry(params[no_qubits],qr[1])
        qc.ry(params[no_qubits+1],qr[2])
        qc.cx(qr[1],qr[2])
        qc.ry(params[no_qubits+2],qr[2])
        qc.measure(qr[2],cr[0])
        result=qp.execute('test')
    #    print(result.get_counts('test'))
        prob_zero=result[0]['data']['counts']['0']/(result[0]['data']['counts']['1']+result[0]['data']['counts']['0'])

        single_cost[k]=(labels[k]*prob_zero)+((1-labels[k])*(1-prob_zero))
        if single_cost[k]<0:
            print('oops')
            print(prob_zero)
            print(labels[k])
        #print(single_cost[k])
    total_cost=np.sum(single_cost)/len(single_cost)
    print(total_cost)
    return total_cost


# In[104]:


result[0]['data']['counts']['0']/result[0]['data']['counts']['1']


# In[120]:


params


# In[122]:


new_params=np.copy(params)
print(new_params)
new_params[0]=1
print(new_params)
print(params)


# In[5]:


qp = QuantumProgram()

qr = qp.create_quantum_register('qr', 2)
cr = qp.create_classical_register('cr', 2)
qc = qp.create_circuit('Bell', [qr], [cr])

coupling_map = {1:[0,2], 2:[3], 3:[4, 14], 5:[4], 6:[5,7,11], 7:[10], 8:[7],9:[8, 10], 11:[10], 12:[5, 11, 13], 13:[4, 14], 15:[0, 2, 14]}

qc.h(qr.ry[0])
qc.h(qr[0])
qc.cx(qr[0], qr[1])
qc.measure(qr[0], cr[0])
qc.measure(qr[1], cr[1])

#result = qp.execute('Bell')
#print(result.get_counts('Bell'))


# In[10]:


coupling_map = {1:[0,2], 2:[3], 3:[4, 14], 5:[4], 6:[5,7,11], 7:[10], 8:[7],9:[8, 10], 11:[10], 12:[5, 11, 13], 13:[4, 14], 15:[0, 2, 14]}
coupling_map[3]


# In[ ]:


# Import the QISKit SDK


# Create a Quantum Register with 2 qubits
q = QuantumRegister(2)
# Create a Classical Register with 2 bits.
c = ClassicalRegister(2)
# Create a Quantum Circuit
qc = QuantumCircuit(q, c)

# Add a H gate on qubit 0, putting this qubit in superposition.
qc.h(q[0])
# Add a CX (CNOT) gate on control qubit 0 and target qubit 1, putting
# the qubits in a Bell state.
qc.cx(q[0], q[1])
# Add a Measure gate to see the state.
qc.measure(q, c)
print("Local backends: ", available_backends({'local': True}))

# Compile and run the Quantum circuit on a simulator backend
sim_result = execute(qc, 'local_qasm_simulator')
print("simulation: ", sim_result)
print(sim_result.get_counts(qc))


# In[5]:


import cuncsd_sq as csd


# In[ ]:




