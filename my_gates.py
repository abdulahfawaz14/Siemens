import numpy as np
import math
from math import e

def rx(theta, qubit, total):
    rx=np.array([[math.cos(theta/2), -1J*math.sin(theta/2)],
                 [1J*math.sin(theta/2),math.cos(theta/2)]])
    rx=np.kron(np.identity(2**qubit),rx)
    rx=np.kron(rx,np.identity(2**(total-qubit-1)))
    return rx


def ry(theta, qubit, total):
    ry=np.array([[math.cos(theta/2), -1*math.sin(theta/2)],
                 [math.sin(theta/2),math.cos(theta/2)]])
    ry=np.kron(np.identity(2**qubit),ry)
    ry=np.kron(ry,np.identity(2**(total-qubit-1)))
    return ry   

def cnot(ctrl, target, total):
    if ctrl < target:
        cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        
    if target < ctrl:
        cnot = np.array([[0,1,0,0],[1,0,0,0],[0,0,1,0],[0,0,0,1]])
    for k in range(min(ctrl,target)):
        cnot=np.kron(np.identity(2),cnot)
    for k in range(total-max(ctrl,target)-1):
        cnot=np.kron(cnot,np.identity(2))
    return cnot

def swap(q1, q2, total):
    swap= np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])

    for k in range(min(q1,q2)):
        swap=np.kron(np.identity(2),swap)
    for k in range(total-max(q1,q2)-1):
        swap=np.kron(swap,np.identity(2))
    return swap

def two_Q(theta1, theta2, q1, direction, total):
    """Theta1 is on the higher one"""
    """Q1 is the higher one"""
    """ higher one literally means above in circuit diagram"""
    
    r1 = ry(theta1,q1,total)
    r2 = ry(theta2,q1+1,total)
    two_Q = np.matmul(r1,r2)
    
    if direction == 'down':
        c1 = cnot(q1,q1+1,total)
        c2 = cnot(q1+1,q1,total)
    if direction == 'up':
        c1 = cnot(q1+1,q1,total)
        c2 = cnot(q1,q1+1,total)
        
    two_Q = np.matmul(c1,two_Q)
    
    two_Q = np.matmul(c2, two_Q)
    return two_Q

def rz(phi,qubit,total):
    rz=np.array([[1, 0],[0,e**(1j*phi)]])
    rz=np.kron(np.identity(2**qubit),rz)
    rz=np.kron(rz,np.identity(2**(total-qubit-1)))
    return rz   
        
def three_Q(theta1, theta2, theta3, theta4, theta5, theta6, q1, total):
    r1= ry(theta1,q1,total)
    r2 = ry(theta2,q1+1,total)
    r3 = ry(theta3,q1+2,total)
    r = np.matmul(r1,r2)
    r = np.matmul(r,r3)
    
    c = cnot(q1,q1+1,total)
    
    three_Q = np.matmul(c,r)
    
    c = ry(theta4, q1+1, total)
        
    three_Q = np.matmul(c, three_Q)
    
    c = cnot(q1+1, q1+2, total)
    
    three_Q = np.matmul(c, three_Q)
    
    c = swap(q1+1, q1+2,total)
    
    three_Q = np.matmul(c, three_Q)
    
    r1 = ry(theta5,q1,total)
    r2 = ry(theta6,q1+1,total)
    r = np.matmul(r1, r2)
    
    
    three_Q = np.matmul(r, three_Q)
    
    c = cnot(q1,q1+1,total)
    
    three_Q = np.matmul(c,r)
    
    c = swap(q1+1,q1+2,total)
    
    three_Q = np.matmul(c, three_Q)   
    
    return three_Q

def two_Q_2(theta1, theta2, q1, q2, total):
    
    """theta 1 acts on q1 """     

    r1 = ry(theta1,q1,total)
    r2 = ry(theta2,q2,total)
    two_Q_2 = np.matmul(r1,r2)
    k = 0
    while abs(q1-q2)>1:

        two_Q_2 = np.matmul(two_Q_2,swap(q1, q1+(np.sign(q2-q1)*+1), total))
        
        q2 = q1+(np.sign(q2-q1)*+1)
        k = k+1
    
    c1 = cnot(q1,q2,total)
    
    two_Q_2 = np.matmul(c1, two_Q_2)

    for  i in range(k):
        two_Q_2 = np.matmul(two_Q_2,swap(q1, q1-(np.sign(q2-q1)*-1), total))
        q2 = q1-(np.sign(q2-q1)*-1)
        
    return two_Q_2


