# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 15:43:54 2018

@author: z003x5wa
"""
import numpy as np


Y = np.array([[0,-1J],[1J,0]])
X = np.array([[0,1],[1,0]])
print(np.kron(Y,Y) + np.kron(X,X))