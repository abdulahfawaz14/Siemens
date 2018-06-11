import numpy as np
import scipy
import math
a= np.random.rand(4)
a = a/scipy.linalg.norm(a)

print(a)
M = np.array([[1,1],[1,-1]])/math.sqrt(2)
W = np.array([[1,0],[0,0]])
U = np.array([[0,0],[0,1]])
a = np.reshape(a, (2,2))

c = np.einsum('ab,bc -> ac', a,M)
#print(c)
d = np.einsum('ab,ac -> bc', a,M)

#print(d)

e = np.einsum('ab-> b', d)


f = np.einsum('ab,bc->a',a,W)
print(f)


g = np.einsum('ab,bc->a',a,U)
A = np.reshape(a,(4,1))
print(g)
P = np.einsum('ij, ij->ij',a,a)
print(P)

""" Given wavefunction a

unnormalised remaining wavevector is np.einsum( a, U/W-> trace out whichever e.g first,, a,U/W )

probability of measurement U/W is 


"""


#e = e/scipy.linalg.norm(e)