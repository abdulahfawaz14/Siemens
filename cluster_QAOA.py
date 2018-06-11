import numpy

import math

from utilities import full_tensor


plus = np.array([1,1])/math.sqrt(2)

state = full_tensor([plus,plus,plus])
up = np.array([1,0])
down = np.array([0,1])
M = np.array([[1,0],[0,0]])
Z = np.array([[1,0],[0,-1]])
cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cnot_Re = np.reshape(cnot, (2,2,2,2))
op = full_tensor([Z,Z]) 
op_2 = (np.identity(4) - op)/2 # imposes they are different

op_2_Re = np.reshape(op_2,(2,2,2,2))

op_Re = np.reshape(op, (2,2,2,2))

op_3 = (np.identity(4)+op)/2
#print(op_3)
op_3_Re = np.reshape(op_3, (2,2,2,2))
small_state = full_tensor([plus,plus])
state_Re = np.reshape(state, (2,2,2))



new_state = full_tensor([plus,plus,up])
new_state_Re = np.reshape(new_state, (2,2,2))

final_state = np.matmul(np.kron(op, np.identity(2)),new_state)

final_state_2 = np.reshape(final_state, (2,2,2))
final_state_2 = np.einsum('abc,acAC,bBCD->ABD', final_state_2, cnot_Re,cnot_Re)
print(np.reshape(final_state_2,(1,8)))

final_state_3 = np.einsum('abc,cC->ab', final_state_2, M)
print(final_state_3)


"""
final_state = np.einsum('abc,abAB,BcEC->AEC', state_Re, op_Re, op_Re)

final_state_2 = np.reshape(final_state,(1,8))

print(final_state_2)
"""
