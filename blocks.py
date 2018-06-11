import numpy as np
import math
from functions_for_data import shuffle_data
import random
import tensorflow as tf

def _single_qubit_rotation(theta):
    return tf.stack([(tf.cos(theta/2), -tf.sin(theta/2)),
                     (tf.sin(theta/2), tf.cos(theta/2))], axis=0)
    

def _single_qubit_rotation_X(theta):
    return tf.stack([(tf.complex(tf.cos(theta/2),0.0), tf.complex(0.0,-1*tf.sin(theta/2))),
                     (tf.complex(0.0,-1*tf.sin(theta/2)), tf.complex(tf.cos(theta/2),0.0))], axis=0)    
    
    
def _single_qubit_rotation_phase(theta):
    return tf.constant(np.array([[np.cos(theta/2), -1J*np.sin(theta/2)], [-1J*np.sin(theta/2), np.cos(theta/2)]]).astype('complex64'))



    
def rz(theta):
    return tf.stack(([(1, 0),
                     (0, tf.complex(tf.cos(theta),tf.sin(theta)))]), axis=0)

def _theta(name=None,identity = False):

    value = 0.0 if identity else np.random.uniform(low=0.0, high=np.pi)

    if name is not None:
        return tf.Variable(value, name=name)
    else:
        return tf.Variable(value)

def ry_tf(theta):
     return tf.stack([(tf.cos(theta/2), -tf.sin(theta/2)),
                     (tf.sin(theta/2), tf.cos(theta/2))], axis=0)
def _had():
     return tf.cast(tf.stack([(1/tf.sqrt(2.0), 1/tf.sqrt(2.0)),
                     (1/tf.sqrt(2.0), -1/tf.sqrt(2.0))], axis=0), tf.complex64)
    
def _X():
    gate = np.array([[1, 1], [1J, -1J]])/2
    return tf.constant(gate.astype('complex64'))
  
def _X_Perp():
    gate = np.array([[1, 1], [1J, -1J]])/2
    _x = tf.constant(gate.astype('complex64'))
    
    return tf.conj(tf.transpose(_x))
    
def _cnot():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).astype('float32'))

def _cnot_phase():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]).astype('complex64'))


def _cz():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]).astype('float32'))

def _cnotR():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]).astype('float32'))
    

def unitary_block(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    cn = _cnot()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef->abef", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])


def unitary_block_phase(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = tf.cast(_single_qubit_rotation(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation(theta2), dtype=tf.complex64)
    cn = _cnot_phase()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef->abef", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def unitary_block_X_phase(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = tf.cast(_single_qubit_rotation_X(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation_X(theta2), dtype=tf.complex64)
    cn = _cnot_phase()
    h = _had()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("Aa,Bb,ac,bd,cdef,eE,fF->ABEF", u1, u2,h,h, cnReshaped,h,h)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def unitary_block_X_phase_simple(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = tf.cast(_single_qubit_rotation_X(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation_X(theta2), dtype=tf.complex64)
    cn = _cnot_phase()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef->abef", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def unitary_block_X_phase2(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_2")
  #  theta4 = _theta(name + "_theta_2")

    u1 = tf.cast(_single_qubit_rotation_X(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation_X(theta2), dtype=tf.complex64)
    u3 = tf.cast(_single_qubit_rotation_X(theta1), dtype=tf.complex64)
    cn = _cnot_phase()
    h = _had()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("Aa,Bb,ac,bd,cdef,eE,fF,EgFh,gG->ABGh", u1, u2,h,h, cnReshaped,u3, h,cnReshaped,h)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3])

def unitary_block_ancilla(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
  #  theta4 = _theta(name + "_theta_2")

    u1 = tf.cast(_single_qubit_rotation_X(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation_X(theta2), dtype=tf.complex64)
    cn = _cnot_phase()
    h = _had()
    _x=_X()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("abcd,ce,ef,dg,fghi->abhi", cnReshaped, _x, u1,u2,cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def unitary_block_phase_Rotations(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = rz(theta1)
    u2 = rz(theta2)
    cn = _cnot_phase()
    had = _had()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,ce,df,efgh,gi,hj->abij", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.conj(tf.transpose(tf.reshape(uBlock, (4, 4)))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def unitary_block_triple(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_3")
    theta4 = _theta(name + "_theta_4")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    u3 = _single_qubit_rotation(theta3)
    u4 = _single_qubit_rotation(theta4)
    cn = _cnot()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ab,cd,ef,bdgh,hi,ifjk->acegjk", u1,u2,u3,cnReshaped, u4, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (8, 8))), (2, 2, 2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3, theta4])


def unitary_block_triple2(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_3")
    theta4 = _theta(name + "_theta_4")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    u3 = _single_qubit_rotation(theta3)
    u4 = _single_qubit_rotation(theta4)
    cn = _cnot()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ab,cd,bdef,jk,fl,klmn->ajcemn", u1,u2,cnReshaped,u3, u4, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (8, 8))), (2, 2, 2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3, theta4])


def unitary_blockR(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    cn = _cnotR()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef->abef", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])

def top_unitary_block(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_3")
    theta4 = _theta(name + "_theta_4")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    u3 = _single_qubit_rotation(theta3)
    u4 = _single_qubit_rotation(theta4)

    cn = _cnot()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef,eg,fh->abgh", u1, u2, cnReshaped, u3, u4)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3, theta4])


def top_unitary_block_phase(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_3")
    theta4 = _theta(name + "_theta_4")
    u1 = tf.cast(_single_qubit_rotation(theta1), dtype=tf.complex64)
    u2 = tf.cast(_single_qubit_rotation(theta2), dtype=tf.complex64)
    u3 = tf.cast(_single_qubit_rotation(theta3), dtype=tf.complex64)
    u4 = tf.cast(_single_qubit_rotation(theta4), dtype=tf.complex64)
    cn = _cnot_phase()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef,eg,fh->abgh", u1, u2, cnReshaped, u3, u4)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4)),conjugate = False), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3, theta4])


def circuit_template():
    
    circuit_block=tf.reshape(tf.constant(np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1]]).astype('float32')), (2,2,2,2,2,2))
    return circuit_block

def c_theta(name):
    theta = _theta(name)
    c_block = tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, np.cos(theta/2), -np.sin(theta/2)], [0, 0, np.sin(theta/2), np.cos(theta/2)]]).astype('float32'))
    
    c_blockReshaped = tf.reshape(c_block, (2, 2, 2, 2))  
    cBlockPerp = tf.reshape(tf.transpose(tf.reshape(c_block, (4, 4))), (2, 2, 2, 2)) 
    return (c_blockReshaped, c_blockPerp, [theta])
   
   
def top_unitary_blockR(name):
    theta1 = _theta(name + "_theta_1")
    theta2 = _theta(name + "_theta_2")
    theta3 = _theta(name + "_theta_3")
    theta4 = _theta(name + "_theta_4")
    u1 = _single_qubit_rotation(theta1)
    u2 = _single_qubit_rotation(theta2)
    u3 = _single_qubit_rotation(theta3)
    u4 = _single_qubit_rotation(theta4)

    cn = _cnotR()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef,eg,fh->abgh", u1, u2, cnReshaped, u3, u4)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2, theta3, theta4])

def ttn_4_online():
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])

    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_blockR('1b')
    u2a, u2aPerp, tu2a = top_unitary_block('2a')

    return (
        tf.einsum("a,b,c,d,abef,cdgh,fgij,jk,iklm,elno,mhpq,n,o,p,q->", q1, q2, q3, q4, u1a, u1b, u2a, POVMX, u2aPerp,
                  u1aPerp, u1bPerp, q1, q2, q3, q4),
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )
        
def swap_test():
    control_swap = tf.constant(np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]].astype('float32')))
    control_swap_reshaped  = tf.reshape(control_swap,(2,2,2,2,2,2))
    hadamard = tf.constant(np.array([[1,1],[1,-1]]))
    
    swap_test_mat = tf.einsum("ab, bcdefg,eh->ecdhfg", hadamard,control_swap_reshaped,hadamard)
    return swap_test_mat

def ttn_4_online2():
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])

    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_blockR('1b')
    u2a, u2aPerp, tu2a = top_unitary_block('2a')
    X = tf.einsum("a,b,c,d,abef,cdgh,fgij,jk->k", q1, q2, q3, q4, u1a, u1b, u2a, POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )        
        
def peps_8_online():
    
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])
    q5 = tf.placeholder(tf.float32, shape=[2])
    q6 = tf.placeholder(tf.float32, shape=[2])
    q7 = tf.placeholder(tf.float32, shape=[2])
    q8 = tf.placeholder(tf.float32, shape=[2])
    q9 = tf.placeholder(tf.float32, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_block('1b')
    u1c, u1cPerp, tu1c = unitary_block('1c')
    u1d, u1dPerp, tu1d = unitary_block('1d')
    u1e, u1ePerp, tu1e = unitary_block('1e')
    u1f, u1fPerp, tu1f = unitary_block('1f')
    u1g, u1gPerp, tu1g = unitary_block('1g')
    u1h, u1hPerp, tu1h = unitary_block('1h')
    u1i, u1iPerp, tu1i = unitary_block('1i')
    u1j, u1jPerp, tu1j = unitary_block('1j')
    u1k, u1kPerp, tu1k = unitary_block('1k')
    u2a, u2aPerp, tu2a = top_unitary_block('2a')

    X = tf.einsum("a,b,c,d,e,f,g,h,i,abjk,kclm,jdno,lepq,mfrs,oqtu,usvw,tgxy,vhzA,wiBC,yADE,ECFG,FH->H", q1, q2, q3, q4, q5, q6, q7, q8, q9, u1a, u1b, u1c, u1d, u1e, u1f, 
                  u1g, u1h, u1i, u1j, u1k, u2a, 
                  POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))) ,
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, tu2a]
    )        

def peps_8_online2():
    
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])
    q5 = tf.placeholder(tf.float32, shape=[2])
    q6 = tf.placeholder(tf.float32, shape=[2])
    q7 = tf.placeholder(tf.float32, shape=[2])
    q8 = tf.placeholder(tf.float32, shape=[2])
    q9 = tf.placeholder(tf.float32, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_blockR('1b')
    u1c, u1cPerp, tu1c = unitary_block('1c')
    u1d, u1dPerp, tu1d = unitary_block('1d')
    u1e, u1ePerp, tu1e = unitary_blockR('1e')
    u1f, u1fPerp, tu1f = unitary_block('1f')
    u1g, u1gPerp, tu1g = unitary_block('1g')
    u1h, u1hPerp, tu1h = unitary_blockR('1h')
    u1i, u1iPerp, tu1i = unitary_blockR('1i')
    u1j, u1jPerp, tu1j = unitary_blockR('1j')
    u1k, u1kPerp, tu1k = unitary_block('1k')
    u2a, u2aPerp, tu2a = top_unitary_blockR('2a')

    X = tf.einsum("a,b,c,d,e,f,g,h,i,abjk,kclm,leno,jdpq,qgrs,rotu,mfvw,wixy,uxzA,hyBC,sBDE,zEFG,GH->H", q1, q2, q3, q4, q5, q6, q7, q8, q9, u1a, u1b, u1c, u1d, u1e, u1f, 
                  u1g, u1h, u1i, u1j, u1k, u2a, 
                  POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))) ,
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, tu2a]
    )     



def mps_4_online():
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_block('1b')
    u2a, u2aPerp, tu2a = top_unitary_block('2a')

    return (
        tf.einsum("a,b,c,d,abef,fcgh,hdij,jk,iklm,glno,enpq,p,q,o,m->", 
                  q1, q2, q3, q4, u1a, u1b, u2a, POVMX, u2aPerp,
                  u1aPerp, u1bPerp, q1, q2, q3, q4),
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )     

def mps_2_online_mixed():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])



    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_X_phase('1b')
    u2a, u2aPerp, tu2a = unitary_block_phase('2a')

    X = tf.einsum("a,b,abcd,cdef,efgh,hi->i", 
                  q1, q2, u1a, u1b, u2a, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X[0])))
    return (Y,
        [q1, q2],
        [tu1a, tu1b, tu2a]
    )    

def mps_5_online_mixed():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    cn = _cnot_phase()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_X_phase_simple('1b')
    u1c, u1cPerp, tu1c = unitary_block_phase('1c')
    u1d, u1dPerp, tu1d = unitary_block_X_phase_simple('1d')
    u1e, u1ePerp, tu1e = unitary_block_phase('1e')
    u1f, u1fPerp, tu1f = unitary_block_X_phase_simple('1f')
    u1g, u1gPerp, tu1g = unitary_block_phase('1g')
    u1h, u1hPerp, tu1h = unitary_block_X_phase_simple('1h')
    
    
    u1i, u1iPerp, tu1i = unitary_block_X_phase_simple('1i')
    #u1j,u1jPerp, tu1i = unitary_block_phase()
    #u2a, u2aPerp, tu2a = unitary_block_phase('2a')
    
    
    
    X = tf.einsum("a,b,c,d,e,A,abfg,gchi,idjk,kelm,mABC,fhno,ojpq,qlrs,sBtu,uCvw,wx->x", 
                  q1, q2, q3, q4, q5, ancilla, u1a, u1c, u1e, u1g, cnReshaped, u1b, u1d, u1f, u1h, u1i, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X)))
    return (Y,
        [q1, q2, q3, q4, q5],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i]
    )    
   
 
    
    
def mps_5_online_mixed_A():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    ancilla = tf.constant((np.array([1,0]).astype('complex64')))

    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_X_phase('1b')
    u1c, u1cPerp, tu1c = unitary_block_phase('1c')
    u1d, u1dPerp, tu1d = unitary_block_X_phase('1d')
    u1e, u1ePerp, tu1e = unitary_block_phase('1e')
    u1f, u1fPerp, tu1f = unitary_block_X_phase('1f')
    u1g, u1gPerp, tu1g = unitary_block_phase('1g')
    u1h, u1hPerp, tu1h = unitary_block_X_phase('1h')
    
    u2a, u2aPerp, tu2a = unitary_block_phase('2a')
    uxa,uxaPerp,tuxa = unitary_block_ancilla('3a')
    
    X = tf.einsum("a,b,c,d,e,Z,abfg,fghi,icjk,jklm,mdno,nopq,qers,rstu,tuvw,wZxy,yz->z", 
                  q1, q2, q3, q4, q5,q6, u1a, u1b, u1c, u1d, u1e, u1f, u1g, u1h, u2a, uxa, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X)))
    return (Y,
        [q1, q2, q3, q4, q5, q6],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu2a, tuxa]
    )        
    
def mps_5_online_mixed2():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_X_phase2('1b')
    u1c, u1cPerp, tu1c = unitary_block_phase('1c')
    u1d, u1dPerp, tu1d = unitary_block_X_phase2('1d')
    u1e, u1ePerp, tu1e = unitary_block_phase('1e')
    u1f, u1fPerp, tu1f = unitary_block_X_phase2('1f')
    u1g, u1gPerp, tu1g = unitary_block_phase('1g')
    u1h, u1hPerp, tu1h = unitary_block_X_phase2('1h')
    
    u2a, u2aPerp, tu2a = unitary_block_phase('2a')

    X = tf.einsum("a,b,c,d,e,abfg,fghi,icjk,jklm,mdno,nopq,qers,rstu,tuvw,wx->x", 
                  q1, q2, q3, q4, q5, u1a, u1b, u1c, u1d, u1e, u1f, u1g, u1h, u2a, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X)))
    return (Y,
        [q1, q2, q3, q4, q5],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu2a]
    )    
    
    
def mps_5_online_mixed_A():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_X_phase('1b')
    u1c, u1cPerp, tu1c = unitary_block_phase('1c')
    u1d, u1dPerp, tu1d = unitary_block_X_phase('1d')
    u1e, u1ePerp, tu1e = unitary_block_phase('1e')
    u1f, u1fPerp, tu1f = unitary_block_X_phase('1f')
    u1g, u1gPerp, tu1g = unitary_block_phase('1g')
    u1h, u1hPerp, tu1h = unitary_block_X_phase('1h')
    
    u2a, u2aPerp, tu2a = unitary_block_phase('2a')
    uxa,uxaPerp,tuxa = unitary_block_ancilla('3a')
    
    X = tf.einsum("a,b,c,d,e,Z,abfg,fghi,icjk,jklm,mdno,nopq,qers,rstu,tuvw,wZxy,yz->z", 
                  q1, q2, q3, q4, q5,q6, u1a, u1b, u1c, u1d, u1e, u1f, u1g, u1h, u2a, uxa, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X)))
    return (Y,
        [q1, q2, q3, q4, q5, q6],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu2a, tuxa]
    ) 
        
def mps_4_online_phase2():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_phase('1b')
    u2a, u2aPerp, tu2a = top_unitary_block_phase('2a')

    X = tf.einsum("A,B,C,D,Aa,Bb,Cc,Dd,abef,fcgh,hdij,jk->k", 
                  q1, q2, q3, q4, _x, _x, _x, _x, u1a, u1b, u2a, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X[0])))
    return (Y,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )     

def mps_4_online_phase():
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    _x_P = _X_Perp()
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_phase('1b')
    u2a, u2aPerp, tu2a = top_unitary_block_phase('2a')

    X = tf.einsum("A,B,C,D,Aa,Bb,Cc,Dd,abef,fcgh,hdij,jk,iklm,glno,enpq,pP,qQ,oO,mM,P,Q,O,M->", 
                  q1, q2, q3, q4, _x,_x,_x,_x, u1a, u1b, u2a, POVMX, u2aPerp,
                  u1aPerp, u1bPerp,_x_P, _x_P, _x_P, _x_P, tf.conj(q1), tf.conj(q2), 
                  tf.conj(q3), tf.conj(q4))
    
    X = tf.cast(X, dtype=tf.float32)
    return (X,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )     


def mps_9_online():
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])
    q5 = tf.placeholder(tf.float32, shape=[2])
    q6 = tf.placeholder(tf.float32, shape=[2])
    q7 = tf.placeholder(tf.float32, shape=[2])
    q8 = tf.placeholder(tf.float32, shape=[2])
    q9 = tf.placeholder(tf.float32, shape=[2])

    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_block('1b')
    u1c, u1cPerp, tu1c = unitary_block('1c')
    u1d, u1dPerp, tu1d = unitary_block('1d')
    u1e, u1ePerp, tu1e = unitary_block('1e')
    u1f, u1fPerp, tu1f = unitary_block('1f')
    u1g, u1gPerp, tu1g = unitary_block('1g')
    u2a, u2aPerp, tu2a = top_unitary_blockR('2a')

    X = tf.einsum("a,b,c,d,e,f,g,h,i,abjk,kclm,mdno,oepq,qfrs,sgtu,uhvw,wixy,yz->z", q1, q2, q3, q4, q5, q6, q7, q8, q9, u1a, u1b, u1c, u1d, u1e, u1f, 
                  u1g, u2a, 
                  POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))) ,
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu2a]
    )     
    
def peps_8_online_full():
    
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])
    q5 = tf.placeholder(tf.float32, shape=[2])
    q6 = tf.placeholder(tf.float32, shape=[2])
    q7 = tf.placeholder(tf.float32, shape=[2])
    q8 = tf.placeholder(tf.float32, shape=[2])
    q9 = tf.placeholder(tf.float32, shape=[2])

    ancilla = tf.constant(np.array([1, 0]).astype('float32'))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_block('1b')
    u1c, u1cPerp, tu1c = unitary_block('1c')
    u1d, u1dPerp, tu1d = unitary_block('1d')
    u1e, u1ePerp, tu1e = unitary_block_triple('1e')
    u1f, u1fPerp, tu1f = unitary_block_triple('1f')
    u1g, u1gPerp, tu1g = unitary_block_triple('1g')
    u1h, u1hPerp, tu1h = unitary_block_triple('1h')
   

    X = tf.einsum("Z,a,b,c,d,e,A,B,C,D,abfg,gchi,idjk,kelm,hAlnop,fBnqrs,sCjtuv,vDmwxy,yz->z", 
                  ancilla, q1, q2, q3, q4, q5, q6, q7, q8, q9, u1a, u1b, u1c, u1d, u1e, u1f, 
                  u1g, u1h, POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))) ,
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h]
    )     
    
def peps_8_online_full_a():
    
    q1 = tf.placeholder(tf.float32, shape=[2])
    q2 = tf.placeholder(tf.float32, shape=[2])
    q3 = tf.placeholder(tf.float32, shape=[2])
    q4 = tf.placeholder(tf.float32, shape=[2])
    q5 = tf.placeholder(tf.float32, shape=[2])
    q6 = tf.placeholder(tf.float32, shape=[2])
    q7 = tf.placeholder(tf.float32, shape=[2])
    q8 = tf.placeholder(tf.float32, shape=[2])
    q9 = tf.placeholder(tf.float32, shape=[2])

    ancilla = tf.constant(np.array([1, 0]).astype('float32'))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('float32'))
    u1a, u1aPerp, tu1a = unitary_block('1a')
    u1b, u1bPerp, tu1b = unitary_block('1b')
    u1c, u1cPerp, tu1c = unitary_block('1c')
    u1d, u1dPerp, tu1d = unitary_block('1d')
    u1i, u1iPerp, tu1i = unitary_block('1i')
    
    u1e, u1ePerp, tu1e = unitary_block_triple('1e')
    u1f, u1fPerp, tu1f = unitary_block_triple('1f')
    u1g, u1gPerp, tu1g = unitary_block_triple('1g')
    u1h, u1hPerp, tu1h = unitary_block_triple('1h')
   

    X = tf.einsum("a,b,c,d,e,f,g,h,i,j,abkl,lcmn,ndop,pgqr,mqfstu,ktevwx,xhyz,uziABC,rCjDEF,FG->G", 
                  ancilla, q1, q2, q3, q4, q5, q6, q7, q8, q9, u1a, u1b, u1c, u1d, u1e, u1f, 
                  u1i, u1g, u1h, POVMX)
    return (
        tf.reduce_sum(tf.square(tf.abs(X))) ,
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i]
    ) 

    
    
def measurement_basis(name):
    theta1 = _theta(name + "_theta_1")
    u1 = _single_qubit_rotation(theta1)
    cn = _cnot()
    cnReshaped = tf.reshape(cn, (2, 2, 2, 2))
    uBlock = tf.einsum("ac,bd,cdef->abef", u1, u2, cnReshaped)
    # uBlock = tf.Variable(np.random.rand(2,2,2,2).astype('float32'))
    uBlockPerp = tf.reshape(tf.transpose(tf.reshape(uBlock, (4, 4))), (2, 2, 2, 2))
    return (uBlock, uBlockPerp, [theta1, theta2])
 

def _single_qubit_rotation(theta):
    return tf.stack([(tf.cos(theta/2), -tf.sin(theta/2)),
                     (tf.sin(theta/2), tf.cos(theta/2))], axis=0)
        

def crazy():
    t1 = tf.placeholder(tf.float32, shape=[1])
    t2 = tf.placeholder(tf.float32, shape=[1])
    t3 = tf.placeholder(tf.float32, shape=[1])
    t4 = tf.placeholder(tf.float32, shape=[1])


    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    _x = _X()
    cz = _cz()
    u1a, u1aPerp, tu1a = unitary_block_phase('1a')
    u1b, u1bPerp, tu1b = unitary_block_phase('1b')
    u2a, u2aPerp, tu2a = top_unitary_block_phase('2a')

    X = tf.einsum("A,B,C,D,Aa,Bb,Cc,Dd,abef,fcgh,hdij,jk->k", 
                  q1, q2, q3, q4, _x, _x, _x, _x, u1a, u1b, u2a, POVMX)
    Y = tf.reduce_sum(tf.square(tf.abs(X[0])))
    return (Y,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu2a]
    )     





