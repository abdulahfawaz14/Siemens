import numpy as np
import math
from functions_for_data import shuffle_data
import random

a = np.array([0,1])

b = np.array([1,0])
B = np.array([1,1])/(2**0.5)
c = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
c=c.reshape(2,2,2,2)


"""C = np.einsum('i,j,ijno',B,a,c)
C= np.einsum('ij->j',C)

print(C.shape)
print(C)"""

C = np.einsum('ijkl, lmno', c, c)
print(C.shape)

D = np.einsum('i,j,k,ijklmn',a,a,b,C)
print(D.shape)
D = np.einsum('ijk->k',D)
print(D)


print(np.einsum('a,b,c, abde, ecfg ->g',a,b,B,c,c))