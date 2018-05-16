from circuit_utilities import initialise_params, initial_encode, initial_encode2, keep_last,get_batches, keep_n, prob_zero, eval_cost, partial_trace
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

def evaluate_MPS_batches(params, data, ancilla, total, number_batches, rounding):
    """ UNITARIES ON ALL FOLLOWED BY CASCADE OF CNOT(a,b) and unitary on b """
    training_data = get_batches(data,number_batches)
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
