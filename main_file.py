import numpy as np
import pandas as pd
from mps_circuits import evaluate_MPS, evaluate_MPS_yz, evaluate_MPS_double, evaluate_MPS_batches, evaluate_MPS_xyz, evaluate_MPS_a
from other_circuits import PEPS9, evaluate_4TTN, PEPS_2, PEPS_3
from my_gates import two_Q_2

import scipy 
from functions_for_data import shuffle_data
from circuit_utilities import initialise_params

iris_hard = np.load('iris_hard.npy')
iris_hard [:,4] = iris_hard[:,4] - 1

easy_iris = np.load('easy_iris.npy')
bars_n_stripes = np.load('bars_n_stripes_dataset.npy')

training_zeros = np.load('learning_zero_train.npy')
continuous_data = np.load('continuous_data.npy')

no_qubits = 9
ancilla = 0
number_batches = 30
total = no_qubits + ancilla

data = continuous_data
data[:,3:6]=np.flip(data[:,3:6],1)
#data1, data2 = shuffle_data(iris_hard, 8)
data1,data2 = shuffle_data(data,80)
data3,data4 = shuffle_data(data1,20)

params = initialise_params(2*total -1)

function = evaluate_MPS
params = scipy.optimize.minimize( function, params,args = (data2,ancilla, total, 0), method='CG')['x']

proportion_wrong = (function(params,data2,ancilla,total, rounding = 1) * 2)
data_size = data2.shape[0]

percent_right = (1- proportion_wrong) * 100

print('Accuracy is ', percent_right, 'percent')

print('done')