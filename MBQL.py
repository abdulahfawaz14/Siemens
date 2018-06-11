import numpy as np
import math
from functions_for_data import shuffle_data
import random
import tensorflow as tf
from utilities import full_tensor
import scipy

""" 
IMPORT REFERENCE
Have measurement M

and wavefunction x (y = reshaped)

Wavefunction:
z = np.einsum('ab,ac->b', y,M)

or np.einsum ('abcdef....etc, aX -> bcdef...etc', y M)
print(z)


WAVEFUNCTION IS : 
    print(z/np.linalg.norm(z))  
    
PROBABILITY OF RESULT M:
    BEFORE NORMALISING Z!
    np.einsum('a,a->',z,z)


"""




def _theta(name=None,identity = False):

    value = 0.0 if identity else np.random.uniform(low=0.0, high=np.pi)

    if name is not None:
        return tf.Variable(value, name=name)
    else:
        return tf.Variable(value)
"""
init = tf.contrib.layers.xavier_initializer()


def _theta(name=None,identity = False):

    value = 0.0 if identity else np.random.uniform(low=0.0, high=np.pi)

    if name is not None:
        return tf.Variable(init((1, 1)), name=name)
    else:
        return tf.Variable(init((1, 1)))
"""

    
def measurement_M(name):
    theta1 = _theta(name + "_theta_1")
    
    u1 = tf.stack([(1, tf.complex(tf.cos(theta1), 1*tf.sin(theta1))),
                 (1 ,  -1*tf.complex(tf.cos(theta1), 1*tf.sin(theta1)))], axis=0) 
    return(u1,[theta1])
    

def _plus():
    return tf.constant((np.array([1,1])/math.sqrt(2)).astype('complex64'))

def _cz():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]).astype('complex64'))

def _cp():
    return tf.constant(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1J]]).astype('complex64'))


def create_line(no_qubits):
    plus = _plus() 
    l = []       
    init = np.array([1,1])/math.sqrt(2)
    for i in range(no_qubits):
        l.append(init)
    A = full_tensor(l)
    return A



def line_4():
    # 1D cluster state of 4 qubits + 1 ancilla in a line.
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    
    graph_state = tf.einsum('a,b,c,d,s,abef,fcgh,hdij,jskl->egikl', q1,q2,q3,q4, plus,
                            czReshaped,czReshaped,czReshaped, czReshaped)

    graph_state = graph_state/tf.norm(graph_state)
    
    x = tf.einsum('egikl,eX->gikl',graph_state,u1a)
    x = x/tf.norm(x)
    
    x = tf.einsum('gikl,gY->ikl',x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('ikl,iZ->kl',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('kl,kN->l',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('l,lM->M',x,u1e)
    
    X = tf.einsum('M,MK->K',x,POVMX)
    
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu1c, tu1d, tu1e]
    )       

def square_9():
    # 2D cluster state of 9 qubits + 8 ancilla in a 2D structure: 8 qubits, (1 qubit +7 ancilla) i.e (8 x 2)
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])
    q9 = tf.placeholder(tf.complex64, shape=[2])
    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')
    u1n, tu1n = measurement_M('1n')
    u1o, tu1o = measurement_M('1o')
    u1p, tu1p = measurement_M('1p')


    graph_state_1 = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,abqr,rcst,tduv,vewx,xfyz,zgAB,BhCD,DiEF,FjGH,HKIJ,JlKL,LmMN,NnOP,PoQR,RpST,TqUV->VsuwyACEGIKMOQSU'
                              ,q1,q2,q3,q4,q5,q6,q7,q8,q9,plus,plus,plus,plus,plus,plus,plus, 
                              czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped)
    

    graph_state = tf.einsum('abcdefghijklmnop,boqr,cnst,dmuv,elwx,fkyz,gjAB->aqsuwyAhiBzxvtrp', 
                            graph_state_1, czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped)
    
    x = tf.einsum('abcdefghijklmnop,aA->bcdefghijklmnop',graph_state,u1a)
    x = x/tf.norm(x)
    
    x = tf.einsum('bcdefghijklmnop,bB->cdefghijklmnop',x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmnop,cC->defghijklmnop',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmnop,dD->efghijklmnop',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmnop,eE->fghijklmnop',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmnop,fF->ghijklmnop',x,u1f)
    x = x/tf.norm(x)
    
    x = tf.einsum('ghijklmnop,gG->hijklmnop',x,u1g)
    x = x/tf.norm(x)

    x = tf.einsum('hijklmnop,hH->ijklmnop',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmnop,iI->jklmnop',x,u1i)
    x = x/tf.norm(x)

    x = tf.einsum('jklmnop,jJ->klmnop',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmnop,kK->lmnop',x,u1k)
    x = x/tf.norm(x)

    x = tf.einsum('lmnop,lL->mnop',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mnop,mM->nop',x,u1m)
    x = x/tf.norm(x)    
    
    x = tf.einsum('nop,nN->op',x,u1n)
    x = x/tf.norm(x)
    
    x = tf.einsum('op,oO->p',x,u1o)
    x = x/tf.norm(x)    
    
        
    X = tf.einsum('p,pP->P',x,POVMX)
    
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, 
         tu1l, tu1m, tu1n, tu1o, tu1p]
    )       
    


def square_8():
    # 2D cluster state of 9 qubits + 8 ancilla in a 2D structure: 8 qubits, (1 qubit +7 ancilla) i.e (8 x 2)
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])
    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')
    u1n, tu1n = measurement_M('1n')
    u1o, tu1o = measurement_M('1o')
    u1p, tu1p = measurement_M('1p')


    graph_state_1 = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,abqr,rcst,tduv,vewx,xfyz,zgAB,BhCD,DiEF,FjGH,HKIJ,JlKL,LmMN,NnOP,PoQR,RpST,TqUV->VsuwyACEGIKMOQSU'
                              ,q1,q2,q3,q4,q5,q6,q7,q8,plus,plus,plus,plus,plus,plus,plus,plus, 
                              czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped)
    

    graph_state = tf.einsum('abcdefghijklmnop,boqr,cnst,dmuv,elwx,fkyz,gjAB->aqsuwyAhiBzxvtrp', 
                            graph_state_1, czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped)
    
    x = tf.einsum('abcdefghijklmnop,aA->bcdefghijklmnop',graph_state,u1a)
    x = x/tf.norm(x)
    
    x = tf.einsum('bcdefghijklmnop,bB->cdefghijklmnop',x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmnop,cC->defghijklmnop',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmnop,dD->efghijklmnop',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmnop,eE->fghijklmnop',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmnop,fF->ghijklmnop',x,u1f)
    x = x/tf.norm(x)
    
    x = tf.einsum('ghijklmnop,gG->hijklmnop',x,u1g)
    x = x/tf.norm(x)

    x = tf.einsum('hijklmnop,hH->ijklmnop',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmnop,iI->jklmnop',x,u1i)
    x = x/tf.norm(x)

    x = tf.einsum('jklmnop,jJ->klmnop',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmnop,kK->lmnop',x,u1k)
    x = x/tf.norm(x)

    x = tf.einsum('lmnop,lL->mnop',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mnop,mM->nop',x,u1m)
    x = x/tf.norm(x)    
    
    x = tf.einsum('nop,nN->op',x,u1n)
    x = x/tf.norm(x)
    
    x = tf.einsum('op,oO->p',x,u1o)
    x = x/tf.norm(x)    
    
        
    X = tf.einsum('p,pP->P',x,POVMX)
    
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4, q5, q6, q7, q8],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, 
         tu1l, tu1m, tu1n, tu1o, tu1p]
    )       
    
def square_8_2():
    # 2D cluster state of 9 qubits + 8 ancilla in a 2D structure: 8 qubits, (1 qubit +7 ancilla) i.e (8 x 2)
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])
    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')
    u1n, tu1n = measurement_M('1n')
    u1o, tu1o = measurement_M('1o')
    u1p, tu1p = measurement_M('1p')


    graph_state_1 = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,abqr,rcst,tduv,vewx,xfyz,zgAB,BhCD,DiEF,FjGH,HKIJ,JlKL,LmMN,NnOP,PoQR,RpST,TqUV->VsuwyACEGIKMOQSU'
                              ,q1,plus, q3, plus, q5, plus, q7, plus, q8, plus, q6,plus, q4,plus, q2,plus, 
                              czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,
                              czReshaped,czReshaped)
    

    graph_state = tf.einsum('abcdefghijklmnop,boqr,cnst,dmuv,elwx,fkyz,gjAB->aqsuwyAhiBzxvtrp', 
                            graph_state_1, czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped)
    
    x = tf.einsum('abcdefghijklmnop,aA->bcdefghijklmnop',graph_state,u1a)
    x = x/tf.norm(x)
    
    x = tf.einsum('bcdefghijklmnop,bB->cdefghijklmnop',x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmnop,cC->defghijklmnop',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmnop,dD->efghijklmnop',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmnop,eE->fghijklmnop',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmnop,fF->ghijklmnop',x,u1f)
    x = x/tf.norm(x)
    
    x = tf.einsum('ghijklmnop,gG->hijklmnop',x,u1g)
    x = x/tf.norm(x)

    x = tf.einsum('hijklmnop,hH->ijklmnop',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmnop,iI->jklmnop',x,u1i)
    x = x/tf.norm(x)

    x = tf.einsum('jklmnop,jJ->klmnop',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmnop,kK->lmnop',x,u1k)
    x = x/tf.norm(x)

    x = tf.einsum('lmnop,lL->mnop',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mnop,mM->nop',x,u1m)
    x = x/tf.norm(x)    
    
    x = tf.einsum('nop,nN->op',x,u1n)
    x = x/tf.norm(x)
    
    x = tf.einsum('op,oO->p',x,u1o)
    x = x/tf.norm(x)    
    
        
    X = tf.einsum('p,pP->P',x,POVMX)
    
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4, q5, q6, q7, q8],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, 
         tu1l, tu1m, tu1n, tu1o, tu1p]
    )           
    
def square_4_z():
    # 2D cluster state of 4 qubits + 4 ancilla in a 2D structure: 4 qubits, (4 ancilla) i.e (4 x 2).
    #The middle two ancilla are readout
    
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')



    graph_state = tf.einsum('a,b,c,d,e,f,g,h,abij,jckl,ldmn,neop,pfqr,rgst,thuv,viwx,kuyz,msAB->xyAoqBzw',
                              q1, q2, q3, q4, plus, plus, plus, plus, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped )
    

    
    x = tf.einsum('abcdefgh,aA->bcdefgh',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefgh,bB->cdefgh', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefgh,cC->defgh',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defgh,dD->efgh',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efgh,eE->fgh',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fgh,hH->fg',x,u1f)
    x = x/tf.norm(x)

    x = tf.einsum('fg,fF->g',x,u1g)
    x = x/tf.norm(x)
    
    X = tf.einsum('g,gG->G', x, POVMX)

    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g]
    ) 

def square_4():
    # 2D cluster state of 4 qubits + 4 ancilla in a 2D structure: 4 qubits, (4 ancilla) i.e (4 x 2).
    #The middle two ancilla are readout
    
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')



    graph_state = tf.einsum('a,b,c,d,e,f,g,h,abij,jckl,ldmn,neop,pfqr,rgst,thuv,viwx,kuyz,msAB->xyAoqBzw',
                              q1, q2, q3, q4, plus, plus, plus, plus, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped )
    

    
    x = tf.einsum('abcdefgh,aA->bcdefgh',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefgh,bB->cdefgh', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefgh,cC->defgh',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defgh,dD->efgh',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efgh,eE->fgh',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fgh,hH->fg',x,u1f)
    x = x/tf.norm(x)
    
    #x = tf.cast(tf.reshape(x,(1,4)),tf.float32)
    x = tf.multiply(x,tf.conj(x))    
    x = tf.cast(x, tf.float32)     
    x = tf.reshape(x,(1,4))
    y = tf.argmax(x[0])
    x = x[0]
    return (
        x, y,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f]
    )       

def nn_4():
    # NN style 3 layer cluster state
    # EACH LAYER COUNTED AND MEASURED FROM TOP TO BOTTOM
    # layer 1 is all four qubits
    # layer 2 is three ancilla. each ancilla connected to TWO previous qubits 
    # layer 3 is the two output qubits. each outqubit connected to TWO of hte previous layers' qubits
    # REPEAT: the qubits are measured from top to bottom layer by layer
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')



    graph_state = tf.einsum('a,b,c,d,e,f,g,h,i,abjk,kclm,mdno,jepq,lqrs,nftu,ugvw,svxy,xhzA,yABC,BiDE,wEFG,CGHI,tFJK,rDLM,oKNO->pLJNzMOHI',
                              q1, q2, q3, q4, plus, plus, plus, plus, plus,
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped)
    

    
    x = tf.einsum('abcdefghi,aA->bcdefghi',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefghi,bB->cdefghi', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghi,cC->defghi',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghi,dD->efghi',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghi,eE->fghi',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghi,fF->ghi',x,u1f)
    x = x/tf.norm(x)

    x = tf.einsum('ghi,gG->hi',x,u1g)
    x = x/tf.norm(x)
    
    #x = tf.cast(tf.reshape(x,(1,4)),tf.float32)
    x = tf.multiply(x,tf.conj(x))    
    x = tf.cast(x, tf.float32)     
    x = tf.reshape(x,(1,4))
    y = tf.argmax(x[0])
    x = x[0]
    return (
        x, y,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g]
    )       


def nn_4_1():
    # 2D cluster state of 4 qubits + 4 ancilla in a 2D structure: 4 qubits, (4 ancilla) i.e (4 x 2).
    #The middle two ancilla are readout
    
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')



    graph_state = tf.einsum('a,b,c,d,e,f,g,h,i,Z,abjk,kclm,mdno,jepq,lqrs,nftu,ugvw,svxy,xhzA,yABC,BiDE,wEFG,CGHI,tFJK,rDLM,oKNO,HZPQ,IQRS->pLJNzMOPRS',
                              q1, q2, q3, q4, plus, plus, plus, plus, plus, plus,
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, czReshaped, czReshaped)
    

    
    x = tf.einsum('abcdefghij,aA->bcdefghij',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefghij,bB->cdefghij', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghij,cC->defghij',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghij,dD->efghij',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghij,eE->fghij',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghij,fF->ghij',x,u1f)
    x = x/tf.norm(x)

    x = tf.einsum('ghij,gG->hij',x,u1g)
    x = x/tf.norm(x)
    
    x = tf.einsum('hij,jJ->hi',x,u1h)
    x = x/tf.norm(x)
    
    #x = tf.cast(tf.reshape(x,(1,4)),tf.float32)
    x = tf.multiply(x,tf.conj(x))    
    x = tf.cast(x, tf.float32)     
    x = tf.reshape(x,(1,4))
    y = tf.argmax(x[0])
    x = x[0]
    return (
        x, y,
        [q1, q2, q3, q4],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h]
    )       
def nn_8():
    #neural network style: three layers. first has 8 qubits, second has four, third has one or two output (last is output)
    #of middle layer, top is connected to qubits (1-3), middle is to (3-5), third is to (5-7) and fourth to (6-8)
    # middle connects to final two as first two to first then last two to last
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])    
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')

    graph_state_1 = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,abop,pcqr,qrst,teuv,vfwx,xgyz,zhAB,ijCD,DkEF,FlGH,mnIJ->oqsuwyABCEGHIJ',
                              q1, q2, q3, q4, q5, q6, q7, q8, plus, plus, plus, plus, plus, plus, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped)
    
    graph_state = tf.einsum('abcdefghijklmn,aiop,bpqr,crst,sjuv,dvwx,exyz,yzAB,AkCD,fDEF,ElGH,gFIJ,IHKL,hLMN,JmOP,OPRS,RQTU->oquwCGKMtBTNSU',
                              graph_state_1,
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped)
    
    x = tf.einsum('abcdefghijklmn,aA->bcdefghijklmn',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefghijklmn,bB->cdefghijklmn', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmn,cC->defghijklmn',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmn,dD->efghijklmn',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmn,eE->fghijklmn',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmn,fF->ghijklmn',x,u1f)
    x = x/tf.norm(x)

    x = tf.einsum('ghijklmn,gG->hijklmn',x,u1g)
    x = x/tf.norm(x)
    
    x = tf.einsum('hijklmn,hH->ijklmn',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmn,iI->jklmn',x,u1i)
    x = x/tf.norm(x)
    
    x = tf.einsum('jklmn,jJ->klmn',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmn,kK->lmn',x,u1k)
    x = x/tf.norm(x)
    
    x = tf.einsum('lmn,lL->mn',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mn,mM->n',x,u1m)
    x = x/tf.norm(x)
    
    
    #x = tf.cast(tf.reshape(x,(1,4)),tf.float32)
    X = tf.einsum('n,nN->N', x, POVMX)
    
    #X = tf.reduce_sum(tf.square(tf.abs(x)))
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4,q5, q6, q7, q8],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, tu1l, tu1m]
    )       

def ttn_8():
    #neural network style: three layers. first has 8 qubits, second has four, third has one or two output (last is output)
    #of middle layer, top is connected to qubits (1-3), middle is to (3-5), third is to (5-7) and fourth to (6-8)
    # middle connects to final two as first two to first then last two to last
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])    
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])

    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')
    u1n, tu1n = measurement_M('1n')

    graph_state = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,aipq,bqrs,cjtu,duvw,ekxy,fyzA,glBC,hCDE,smFG,wGHI,AnJK,EKLM,IoNO,MOPQ->prtvxzBDFHJLNPQ',
                              q1, q2, q3, q4, q5, q6, q7, q8, plus, plus, plus, plus, plus, plus, plus,
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped, 
                              czReshaped, czReshaped, czReshaped, czReshaped,
                              czReshaped, czReshaped)
    
    
    x = tf.einsum('abcdefghijklmno,aA->bcdefghijklmno',graph_state,u1a)
    x = x/tf.norm(x)

    x = tf.einsum('bcdefghijklmno,bB->cdefghijklmno', x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmno,cC->defghijklmno',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmno,dD->efghijklmno',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmno,eE->fghijklmno',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmno,fF->ghijklmno',x,u1f)
    x = x/tf.norm(x)

    x = tf.einsum('ghijklmno,gG->hijklmno',x,u1g)
    x = x/tf.norm(x)
    
    x = tf.einsum('hijklmno,hH->ijklmno',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmno,iI->jklmno',x,u1i)
    x = x/tf.norm(x)
    
    x = tf.einsum('jklmno,jJ->klmno',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmno,kK->lmno',x,u1k)
    x = x/tf.norm(x)
    
    x = tf.einsum('lmno,lL->mno',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mno,mM->no',x,u1m)
    x = x/tf.norm(x)
    
    x = tf.einsum('no,nN->o',x,u1n)
    x = x/tf.norm(x)
    
    
    #x = tf.cast(tf.reshape(x,(1,4)),tf.float32)
    X = tf.einsum('o,oO->O', x, POVMX)
    
    #X = tf.reduce_sum(tf.square(tf.abs(x)))
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4,q5, q6, q7, q8],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, tu1l, tu1m, tu1n]
    )       

def nn_image_9():
    # 2D cluster state of 9 qubits + 8 ancilla in a 2D structure: 8 qubits, (1 qubit +7 ancilla) i.e (8 x 2)
    # M implies we are measuring in the state 1, e ^ i theta and expecting a positive result i.e 1
    # can be viewed in circuit based model as CZs everywhere then measurements and if = 1 's 
    # NB input is e^-ix, e^ix
    
    
    q1 = tf.placeholder(tf.complex64, shape=[2])
    q2 = tf.placeholder(tf.complex64, shape=[2])
    q3 = tf.placeholder(tf.complex64, shape=[2])
    q4 = tf.placeholder(tf.complex64, shape=[2])
    q5 = tf.placeholder(tf.complex64, shape=[2])
    q6 = tf.placeholder(tf.complex64, shape=[2])
    q7 = tf.placeholder(tf.complex64, shape=[2])
    q8 = tf.placeholder(tf.complex64, shape=[2])
    q9 = tf.placeholder(tf.complex64, shape=[2])
    
    cz = _cz()
    czReshaped = tf.reshape(cz,(2,2,2,2))
    
    POVMX = tf.constant(np.array([[1, 0], [0, 0]]).astype('complex64'))
    plus = tf.constant((np.array([1, 1])/math.sqrt(2)).astype('complex64'))
    
    #plus = _plus()
    u1a, tu1a = measurement_M('1a')
    u1b, tu1b = measurement_M('1b')
    u1c, tu1c = measurement_M('1c')
    u1d, tu1d = measurement_M('1d')
    u1e, tu1e = measurement_M('1e')
    u1f, tu1f = measurement_M('1f')
    u1g, tu1g = measurement_M('1g')
    u1h, tu1h = measurement_M('1h')
    u1i, tu1i = measurement_M('1i')
    u1j, tu1j = measurement_M('1j')
    u1k, tu1k = measurement_M('1k')
    u1l, tu1l = measurement_M('1l')
    u1m, tu1m = measurement_M('1m')



    graph_state_1 = tf.einsum('a,b,c,d,e,f,g,h,i,j,k,l,m,n,ajop,bpqr,drst,etuv,qkwx,cxyz,uzAB,fBCD,slEF,AFGH,gHIJ,hJKL,GmMN,CNOP,KPQR,iRST->owyEMOIQSvDLTn',
                              q1, q2, q3, q4, q5, q6, q7, q8, q9, plus, plus, plus, plus, plus,
                              czReshaped, czReshaped, czReshaped, czReshaped,
                              czReshaped, czReshaped, czReshaped, czReshaped,
                              czReshaped, czReshaped, czReshaped, czReshaped,
                              czReshaped, czReshaped, czReshaped, czReshaped)
    

    graph_state = tf.einsum('abcdefghijklmn,jkop,olqr,pmst,rtuv,qnwx,sxyz,uzAB,vBCD->abcdefghiwyACD', 
                            graph_state_1, czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped,czReshaped)
    
    x = tf.einsum('abcdefghijklmn,aA->bcdefghijklmn',graph_state,u1a)
    x = x/tf.norm(x)
    
    x = tf.einsum('bcdefghijklmn,bB->cdefghijklmn',x,u1b)
    x = x/tf.norm(x)
    
    x = tf.einsum('cdefghijklmn,cC->defghijklmn',x,u1c)
    x = x/tf.norm(x)
    
    x = tf.einsum('defghijklmn,dD->efghijklmn',x,u1d)
    x = x/tf.norm(x)
    
    x = tf.einsum('efghijklmn,eE->fghijklmn',x,u1e)
    x = x/tf.norm(x)

    x = tf.einsum('fghijklmn,fF->ghijklmn',x,u1f)
    x = x/tf.norm(x)
    
    x = tf.einsum('ghijklmn,gG->hijklmn',x,u1g)
    x = x/tf.norm(x)

    x = tf.einsum('hijklmn,hH->ijklmn',x,u1h)
    x = x/tf.norm(x)
    
    x = tf.einsum('ijklmn,iI->jklmn',x,u1i)
    x = x/tf.norm(x)

    x = tf.einsum('jklmn,jJ->klmn',x,u1j)
    x = x/tf.norm(x)
    
    x = tf.einsum('klmn,kK->lmn',x,u1k)
    x = x/tf.norm(x)

    x = tf.einsum('lmn,lL->mn',x,u1l)
    x = x/tf.norm(x)
    
    x = tf.einsum('mn,mM->n',x,u1m)
    x = x/tf.norm(x)    
    
    X = tf.einsum('n,nN->N',x,POVMX)

    
    
    return (
        tf.reduce_sum(tf.square(tf.abs(X))),
        [q1, q2, q3, q4, q5, q6, q7, q8, q9],
        [tu1a, tu1b, tu1c, tu1d, tu1e, tu1f, tu1g, tu1h, tu1i, tu1j, tu1k, 
         tu1l, tu1m]
    )       
    


def _single_qubit_rotation(theta):
    return tf.stack([(tf.cos(theta/2), -tf.sin(theta/2)),
                     (tf.sin(theta/2), tf.cos(theta/2))], axis=0)
        