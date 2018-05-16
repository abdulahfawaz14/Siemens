from circuit_utilities import initialise_params, initial_encode, initial_encode2, keep_last, keep_n, prob_zero, eval_cost, partial_trace
from my_gates import ry, cnot, two_Q, swap, rz, rx, three_Q

import numpy as np


def evaluate_MPS(params, training_data, ancilla, total, rounding):
    """ UNITARIES ON ALL FOLLOWED BY CASCADE OF CNOT(a,b) and unitary on b """
    
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total+j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS_a(params, training_data, ancilla, total, rounding):
    """ UNITARIES ON ALL FOLLOWED BY CASCADE OF CNOT(a,b) and unitary on b """
    
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))

        for j in range(total-ancilla-1):
            psi=np.matmul(three_Q(params[6*j], params[6*j + 1], params[6*j + 2],
                                  params[6*j + 3 ], params[6*j + 4],
                                  params[6*j + 5], j,total),psi)

        psi=np.matmul(ry(params[6*(total-ancilla-1)],total-1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost



def evaluate_MPS_plotter(params, training_data, ancilla, total, rounding):
    guesses=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total+j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        guesses[i]=1-round(zero_prob)
    return guesses

def evaluate_4TTN(params, training_data, ancilla,total, rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)
    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_4TTN_plotter(params, training_data, ancilla,rounding):
    guesses=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        guesses[i]=1-round(zero_prob)
    return guesses
    return total_cost

def evaluate_4TTN_closed(params, training_data, ancilla,rounding):
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=initial_encode(training_data[i,:L],ancilla)
        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(ry(params[j],j,total),psi)
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        
        j=0
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total],j+1,total),psi)
        j=3
        psi=np.matmul(cnot(j,j-1,total),psi)
        psi=np.matmul(ry(params[total+1],j-1,total),psi)
        j=1
        psi=np.matmul(cnot(j,j+1,total),psi)
        psi=np.matmul(ry(params[total+2],j+1,total),psi)        
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_n(psi,total,2))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)
    total_cost=np.sum(answers)#/training_data.shape[0]
    return total_cost

def PEPS9(params, training_data, ancilla, rounding):
    """Requires 27 parameters. """

    
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """Stage 1: The top left five block"""
        """Encoding 4-3-2-1 + ancilla"""
        psi = np.real(initial_encode(np.flipud(training_data[i,:4]),0))
        psi = np.kron(psi,np.array([1,0]))
        psi = np.reshape(psi,(2**5,1))
        total = 5

        for j in range(3):
            psi = np.matmul(two_Q(params[2*j],params[(2*j)+1],3-j, 'down',
                                total),psi)
            
        psi = np.matmul(swap(0,1,total),psi)
        
        for j in range(2):
            psi = np.matmul(two_Q(params[2*j+6], params[2*j+7],1+j, 
                                'down', total), psi)
        """NB: Adding the bottom five qubits now"""
        psi = np.kron(psi,initial_encode(training_data[i,4:9],0))
        
        total = 10
        psi = np.reshape(psi,(2**total,1))
        
        """Qubits 1 and 5 (on sheet) require a swap and swap back"""
        psi = np.matmul(swap(3,4,total),psi)
        
        psi = np.matmul(two_Q( params[10], params[11], 4, 'down',
                                total),psi)
        
        psi = np.matmul(swap(3,4,total),psi)
        
        psi = np.matmul(two_Q( params[12], params[13], 5, 'up',
                        total),psi)
               
        psi = np.matmul(swap(4,5,total),psi)        
        psi = np.matmul(swap(5,6,total),psi)
        psi = np.matmul(swap(4,5,total),psi)
        
        psi = np.matmul(two_Q( params[14], params[15], 5, 'up',
                        total),psi)        

        psi = np.matmul(swap(5,6,total),psi)
        
        psi = np.matmul(two_Q(params[16],params[17],6,'down',total),psi)        
        
        psi = np.matmul(swap(2,3,total),psi)
        psi = np.matmul(swap(3,4,total),psi)
        psi = np.matmul(swap(4,5,total),psi)        
        psi = np.matmul(swap(5,6,total),psi)
        
        psi = np.matmul(two_Q(params[18],params[19],6,'down',total),psi)        

        psi = np.matmul(two_Q(params[20],params[21],7,'down',total),psi)        

        psi = np.matmul(swap(1,2,total),psi)
        psi = np.matmul(swap(2,3,total),psi)
        psi = np.matmul(swap(3,4,total),psi)
        psi = np.matmul(swap(4,5,total),psi)        
        psi = np.matmul(swap(5,6,total),psi)        
        psi = np.matmul(swap(6,7,total),psi)        
        
        psi = np.matmul(two_Q(params[22],params[23],7,'down',total),psi)        
        
        psi = np.matmul(two_Q(params[24],params[25],8,'down',total),psi)        

        psi = np.matmul(ry(params[26],9,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob = prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i] = eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost = np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS_yz(params, training_data, ancilla, total, rounding):
    
    """ AS MPS BUT INITIAL UNITARIES ARE ALL RZ,RY,RZ FORM """
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    lst=list(range(total-1))
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))

        """Stage 1: Unitaries on all of the qubits, ancilla or not"""
        for j in range(total):
            psi=np.matmul(rz(params[j],j,total),psi)            
        for j in range(total):
            psi=np.matmul(ry(params[j + total],j,total),psi)  
        for j in range(total):
            psi=np.matmul(rz(params[j + (total*2)],j,total),psi)  
            
            """Stage 2: CNOT plus unitary for N-1 times (cascading)"""
        for j in range(total-1):
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(ry(params[total * 3 +j],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        dm=np.kron(np.transpose(np.conjugate(psi)),psi)
        dm=np.reshape(dm,(2**total,2**total))
        dm=partial_trace(dm,lst)
       # print(dm)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(dm)
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost


def evaluate_MPS_double(params, training_data, ancilla, total, rounding):
    """ LIKE MPS BUT DOUBLE THE CNOTS IN EACH CASCADE"""    
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))
        
        for j in range(total-1):
            psi = np.matmul(two_Q(params[4*j], params[4*j + 1], j, 'down', total), psi)
            psi = np.matmul(two_Q(params[4*j + 2], params[4*j + 3], j, 'down', total), psi)
        psi=np.matmul(ry(params[4*total - 4],total-1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost

def evaluate_MPS_xyz(params, training_data, ancilla, total, rounding):
    """ LIKE MPS BUT EACH SINGLE UNITARY IS A DIFFERENT TYPE"""    
    answers=np.zeros(training_data.shape[0])
    L=training_data.shape[1]-1
    for i in range(training_data.shape[0]):
        """FOR EACH DATA POINT"""
        """Lets Encode the data elements first"""
        
        psi=np.real(initial_encode2(training_data[i,:L],ancilla))
        psi=np.reshape(psi,(2**total,1))
        
        for j in range(total-1):
            psi=np.matmul(ry(params[3*j],j,total),psi)
            psi=np.matmul(ry(params[3*j + 1],j+1,total),psi)
            psi=np.matmul(cnot(j,j+1,total),psi)
            psi=np.matmul(rz(params[3*j + 2],j+1,total),psi)
        """Stage 3: Trace and Measure"""
        zero_prob=prob_zero(keep_last(psi,total))
        """Stage 4: Calculate Cost"""
        answers[i]=eval_cost(zero_prob,training_data[i,L],rounding)

    total_cost=np.sum(answers)/training_data.shape[0]
    return total_cost
