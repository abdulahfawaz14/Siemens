import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import math
from scipy.stats import ortho_group
from utilities import full_tensor
import scipy 


no_qubits = 8

def _z():
    return np.array([[1,0],[0,-1]])
                   

def qubits_different():
    A = (np.identity(4) - np.kron(_z(),_z())) / 2
    "return 1 when they are different"
    return np.reshape(A,(2,2,2,2))

def double_different():
    
    "TAKES AB,CD. returns 1 if A =/= B OR, FAILING THAT, IF C=/=D"
    D  = np.reshape(qubits_different(),(4,4))
    
    E =np.kron(D,np.identity(4))+ np.kron((np.identity(4)-D),D)
    
    return np.reshape(E,(2,2,2,2,2,2,2,2))
    

def qubits_same():
    A = (np.identity(4) + np.kron(_z(),_z())) / 2
    "return 1 when they are different"
    return np.reshape(A,(2,2,2,2))

def is_a_greater():
    "returns 1 if A>B. requires third qubit to be in state ZERO |0>"
    A = (np.identity(8) + np.kron(np.kron(np.identity(2), _z()) , _z() )) / 2
    B = (np.identity(8) - np.kron(np.kron(_z(), _z()), np.identity(2))) / 2
    C = np.matmul(A,B)
    return np.reshape(C,(2,2,2,2,2,2))

def new_is_a_greater():
    A = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]])
    A = np.reshape(A,(2,2,2,2))
    return A

def double_greater():
        
    "TAKES AB,CD. returns 1 if A > B OR, FAILING THAT, IF C>D"
    D  = np.reshape(new_is_a_greater(),(4,4))
    
    E =np.kron(D,np.identity(4))+ np.kron((np.identity(4)-D),D)
    
    return np.reshape(E,(2,2,2,2,2,2,2,2))
    

def double_greater_long():
    
    "TAKES AB0,CD0. returns 1 if A > B OR, FAILING THAT, IF C>D"
    D  = np.reshape(is_a_greater(),(8,8))
    
    E =np.kron(D,np.identity(8))+ np.kron((np.identity(8)-D),D)
    
    return np.reshape(E,(2,2,2,2,2,2,2,2,2,2,2,2))

    
def double_greater_exp(theta):
    A = double_greater()
    A = np.reshape(double_greater(),(16,16))
    
    return np.reshape(np.expm1(1J*theta*A),(2,2,2,2,2,2,2,2))

def double_different_exp(theta):
    A = double_different()
    A = np.reshape(double_different(),(16,16))
    
    return np.reshape(np.expm1(1J*theta*A),(2,2,2,2,2,2,2,2))


ONE = np.array([0,1])
ZERO = np.array([1,0])
PLUS = np.array([1,1])/math.sqrt(2)

"""
Z = np.einsum('a,b,c,d,abcdefgh, e, f, g,h->',ZERO, ONE, ONE, 
              ZERO,  double_different(), ZERO, ONE, ONE, ZERO)
print(Z)
"""



def U_checks(theta):
    return np.einsum('aebfAEBF,cidjCIDJ,gkhlGKHL -> abcdefghijklABCDEFGHIJKL'
                     ,double_greater_exp(theta) ,double_greater_exp(theta),double_greater_exp(theta))

def _B(no_qubits):
    b=[]
    for i in range(no_qubits):
        b.append(np.array([[0,1],[1,0]]))
    return full_tensor(b)

def _B_exp(theta,no_qubits=4):
    A =_B(no_qubits)
    return np.expm1(1J*theta*A)
# Parameters
    

    
p = 2

betas = np.random.rand(p)
lambdas = np.random.rand(p)


#make all contribs 

#each contrib is U(C)U(B)U(C)U(B)|s_mini> U(B)U(C)U(B)U(C)

def initial_state(no_qubits):
    a =[]
    for i in range(no_qubits):
        a.append(np.array([1,1]))
    A = full_tensor(a)
    A = A/math.sqrt(2**no_qubits)
    return A

def _B_exp(theta,no_qubits=4):
    A =_B(no_qubits)
    A =  scipy.linalg.expm(1J*theta*A)
    A = np.conj(np.transpose(A))
    return np.reshape(A,(2,2,2,2,2,2,2,2))


def _B_exp_Perp(theta,no_qubits=4):
    A =_B(no_qubits)
    A = scipy.linalg.expm(1J*theta*A)
    return np.reshape(A,(2,2,2,2,2,2,2,2))

def double_greater_exp(theta):
    A = double_greater()
    A = np.reshape(double_greater(),(16,16))
    
    return np.reshape(scipy.linalg.expm(1J*theta*A),(2,2,2,2,2,2,2,2))

def double_different_exp(theta):
    A = double_different()
    A = np.reshape(double_different(),(16,16))
    
    return np.reshape(scipy.linalg.expm(1J*theta*A),(2,2,2,2,2,2,2,2))

def double_greater_exp_Perp(theta):
    A = double_greater()
    A = np.reshape(double_greater(),(16,16))
    A = np.conj(np.transpose(A))
    return np.reshape(scipy.linalg.expm(1J*theta*A),(2,2,2,2,2,2,2,2))

def double_different_exp_Perp(theta):
    A = double_different()
    A = np.reshape(double_different(),(16,16))
    A = np.conj(np.transpose(A))
    return np.reshape(scipy.linalg.expm(1J*theta*A),(2,2,2,2,2,2,2,2))


def greater_contrib(gamma, beta):
    x = double_greater()
    
    for i in range(len(gamma)): #this is p
        x = np.matmul(x,_B_exp(beta[i]))
        x = np.matmul(_B_exp_Perp(beta[i]), x)
        
        x = np.matmul(x,double_greater_exp(gamma[i]))
        x = np.matmul(double_greater_exp_Perp(gamma[i]), x)
        
    return np.reshape(x,(2,2,2,2,2,2,2,2))

def different_contrib(gamma, beta):
    x = double_different()

    for i in range(len(gamma)): #this is p
        x = np.matmul(x,_B_exp(beta[i]))
        x = np.matmul(_B_exp_Perp(beta[i]), x)
        
        x = np.matmul(x,double_different_exp(gamma[i]))
        x = np.matmul(double_different_exp_Perp(gamma[i]), x)
        
    return np.reshape(x,(2,2,2,2,2,2,2,2))


s = initial_state(4)
s = np.reshape(s,(4**2,1))

s_dagger = np.transpose(s)


def c1(params):
    c1 = np.matmul(s_dagger,np.matmul(np.reshape(different_contrib(params[:1],params[1:]),(16,16)),s))
    c1= np.abs(c1)**2
    
    return c1[0][0]


def c2(params):
    c2 = np.matmul(s_dagger,np.matmul(np.reshape(greater_contrib(params[:2],params[:2]),(16,16)),s))
    c2 = np.abs(c2)**2
    
    return c2[0][0]


def c1_rev(params):
    c1 = np.matmul(s_dagger,np.matmul(np.reshape(different_contrib(params[:1],params[1:]),(16,16)),s))
    c1= np.abs(c1)**2
    c1 = 1 - c1[0][0]
    return c1


def c2_rev(params):
    c2 = np.matmul(s_dagger,np.matmul(np.reshape(greater_contrib(params[:2],params[:2]),(16,16)),s))
    c2 = np.abs(c2)**2
    
    return 1-c2[0][0]

gamma = [0,0]
beta = [0,0]
params = [1.3,1.2]
params2 = [0,0]

function = c1_rev

params = scipy.optimize.minimize( c1_rev,x0=params,args=(), method='CG')['x']
    
print(c1_rev(params))
print(c1(params))

function = c2_rev


params2 = scipy.optimize.minimize( function, params2, method='CG')['x']
print(1-c2(params2))

def big_c(params):
    return c1_rev(params) + c2_rev(params)

function = big_c

params2 = scipy.optimize.minimize( function, params2, method='CG')['x']
print(1-c1(params2))
print(1-c2(params2))

# checks = params2
# diffs = params1

check_0 = double_greater_exp(params2[0])
check_1 = double_greater_exp(params2[1])

diff_0 = double_different_exp(params[0])
diff_1= double_different_exp(params[1])



"""

IDEA IS THAT WE DO NOT NEED ALL QUBITS AT ONCE. 

TOTAL F IS CONT1 + CONT 2 + CONT3 + CONT 4 ETC I.E FROM EACH BLOCK

CALCULATE CONTRIBUTION FROM EACH FOR GIVEN GAMMA AND BETA IS EASY

OPTIMISING SHOULD THEREFORE ALSO BE EASY

HA

DOUBT IT







"""



