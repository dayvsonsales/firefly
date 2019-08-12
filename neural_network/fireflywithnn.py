#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:38:52 2019

@author: dayvsonsales
"""

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from firefly import FireflyAlgorithm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

def set_coefs_return_predicts(result):
    decision_variable_values = result['best_decision_variable_values']
    mlp.coefs_ = [np.array([[ decision_variable_values[0], decision_variable_values[1], decision_variable_values[2], decision_variable_values[3] ],
                          [decision_variable_values[4], decision_variable_values[5], decision_variable_values[6], decision_variable_values[7]]]), np.array([[ decision_variable_values[8]],
                          [ decision_variable_values[9]],
                          [decision_variable_values[10]],
                          [decision_variable_values[11]]])]
    
    return mlp.predict(X)

def get_accuracy(result, y):
    predictions = set_coefs_return_predicts(result)
    acc = float(classification_report(y, predictions, output_dict=True)['accuracy'])
    
    return acc
    

class XORFunctionWrapper():
    
    def __init__(self, mlp, X, y, accuracy):
        self.mlp = mlp
        self.X = X
        self.y = y
        self.accuracy = accuracy
    
    def maximum_decision_variable_values(self):
        return [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    def minimum_decision_variable_values(self):
        return [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    
    def set_accuracy(self, acc):
        self.accuracy.append(acc)
        return acc

    def objective_function_value(self, decision_variable_values):
        self.mlp.coefs_ = [np.array([[ decision_variable_values[0], decision_variable_values[1], decision_variable_values[2], decision_variable_values[3] ],
                          [decision_variable_values[4], decision_variable_values[5], decision_variable_values[6], decision_variable_values[7]]]), np.array([[ decision_variable_values[8]],
                          [ decision_variable_values[9]],
                          [decision_variable_values[10]],
                          [decision_variable_values[11]]])]
        predictions = self.mlp.predict(self.X)
        
        acc = float(classification_report(self.y, predictions, output_dict=True)['accuracy'])
        
        return acc  

    def initial_decision_variable_value_estimates(self):
        pass


data = pd.read_csv('/Users/dayvsonsales/trab-icomp/dataset_xor2.csv')

feature_cols = ['x1', 'x2']

X = data[feature_cols]
y = data.result

number_of_variables = 12
number_of_fireflies = 100
max_generation = 75
randomization_parameter_alpha = 0.2
absorption_coefficient_gamma = 1.0

results = []
max_accuracy = []
max_r = 0
for i in range(12):
    mlp = MLPClassifier(activation='tanh', hidden_layer_sizes=(4, ), max_iter=1)
    mlp.fit(X,y)
    
    xor_function = XORFunctionWrapper(mlp, X, y, [])
    
    firefly_algorithm = FireflyAlgorithm(xor_function, number_of_variables)
    result = firefly_algorithm.search(number_of_fireflies = number_of_fireflies, max_generation = max_generation, 
                                              randomization_parameter_alpha = randomization_parameter_alpha, absorption_coefficient_gamma = absorption_coefficient_gamma)
    print(result)
    results.append(result)
    
    if(result['best_objective_function_value'] > max_r):
        max_r = result['best_objective_function_value']
        max_accuracy = xor_function.accuracy
    
    
plt.plot(max_accuracy)
plt.show(block=True)

#plt.subplot(5, 2, 1)
#plt.plot(y)
#c = 2
#epoca = 1
#for r in results:
#    plt.subplot(5, 2, c)
#    c = c + 1
    
#plt.show(block=True)
