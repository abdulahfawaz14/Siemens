import numpy as np
import random

x = np.random.rand(4,1)

x = x/ np.linalg.norm(x)

print(x)
y = np.reshape(x,(2,2))
M  = np.array([[1,0],[0,0]])


z = np.einsum('ab,ac->b', y,M)
print(z)

A = np.array([x[0],x[1]])
C = A/np.linalg.norm(A)

print(C)
B = np.einsum('a,a->', z,z)
print(z/np.linalg.norm(z))

print(B)
print((x[0])**2+(x[1])**2)
