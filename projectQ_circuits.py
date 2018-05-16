from projectq import MainEngine  # import the main compiler engine
from projectq.ops import H, Measure, Ry, CNOT

def MPS(params, data, shots,rounding):
    answers=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x=np.zeros(shots)
        for k in range(shots):
            eng = MainEngine()  # create a default compiler (the back-end is a simulator)

            q1 = eng.allocate_qubit()  # allocate 1 qubit
            q2 = eng.allocate_qubit()
            q3 = eng.allocate_qubit()
            q4 = eng.allocate_qubit()

            Ry(data[i,0]*2) | q1  
            Ry(data[i,1]*2) | q2  
            Ry(data[i,2]*2) | q3  
            Ry(data[i,3]*2) | q4

            Ry(params[0]) | q1  
            Ry(params[1]) | q2  
            Ry(params[2]) | q3  
            Ry(params[3]) | q4  
            
            CNOT | (q1,q2)
            Ry(params[4]) | q2
            
            CNOT | (q2,q3)
            Ry(params[5]) | q3
            
            CNOT | (q3,q4)
            Ry(params[6]) | q4
            
            Measure | q4  # measure the qubit
            Measure | q1
            Measure | q2
            Measure | q3
            eng.flush()

            x[k]=int(q4)
            #print(x)
        #print(x)
        idx=x==0
        num_zeros=shots-sum(x)
        prop_zeros=num_zeros/shots
        #print(prop_zeros)
        if rounding==1:
            answers[i]=1-round(prop_zeros)
            answers[i]=abs(answers[i]-data[i,4])
        if rounding ==0:
            prop_ones=1-prop_zeros
            answers[i]=abs(prop_ones-data[i,4])
    print(np.sum(answers))
    print(answers)
    return(np.sum(answers))

def MPS_8(params, data, shots,rounding):
    answers=np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        x=np.zeros(shots)
        for k in range(shots):
            eng = MainEngine()  # create a default compiler (the back-end is a simulator)

            q1 = eng.allocate_qubit()  # allocate 1 qubit
            q2 = eng.allocate_qubit()
            q3 = eng.allocate_qubit()
            q4 = eng.allocate_qubit()
            q5 = eng.allocate_qubit()  # allocate 1 qubit
            q6 = eng.allocate_qubit()
            q7 = eng.allocate_qubit()
            q8 = eng.allocate_qubit()

            Ry(data[i,0]*2) | q1  
            Ry(data[i,1]*2) | q2  
            Ry(data[i,2]*2) | q3  
            Ry(data[i,3]*2) | q4
            Ry(data[i,4]*2) | q5  
            Ry(data[i,5]*2) | q6  
            Ry(data[i,6]*2) | q7  
            Ry(data[i,7]*2) | q8


            Ry(params[0]) | q1  
            Ry(params[1]) | q2  
            Ry(params[2]) | q3  
            Ry(params[3]) | q4 
            Ry(params[4]) | q5  
            Ry(params[5]) | q6  
            Ry(params[6]) | q7  
            Ry(params[7]) | q8 
            
            
            CNOT | (q1,q2)
            Ry(params[8]) | q2
            CNOT | (q2,q3)
            Ry(params[9]) | q3
            CNOT | (q3,q4)
            Ry(params[10]) | q4
            CNOT | (q4,q5)
            Ry(params[11]) | q5
            CNOT | (q5,q6)
            Ry(params[12]) | q6
            CNOT | (q6,q7)
            Ry(params[13]) | q7
            CNOT | (q7,q8)
            Ry(params[14]) | q8
            
            
            Measure | q4  # measure the qubit
            Measure | q1
            Measure | q2
            Measure | q3
            Measure | q5
            Measure | q6
            Measure | q7
            Measure | q8
    
            x[k]=int(q8)
            eng.flush()
        #print(x)
        idx=x==0
        
        num_zeros=sum(idx)
        #print(num_zeros)
        prop_zeros=num_zeros/shots
        if rounding==1:
            answers[i]=1-round(prop_zeros)
            answers[i]=abs(answers[i]-data[i,8])
        if rounding ==0:
            prop_ones=1-prop_zeros
            answers[i]=abs(prop_ones-data[i,8])
                
    print(np.sum(answers)/data.shape[0])
    return(np.sum(answers)/data.shape[0])