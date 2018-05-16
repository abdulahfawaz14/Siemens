import numpy as np


def full_tensor(sequence):
    """tensor product of a sequence in the form of a list of arrays""" 
    for k in range(len(sequence)-1):
        if k==0:
            a=sequence[0]
        a=np.array(np.kron(a,sequence[k+1]), dtype=complex)
    return a
