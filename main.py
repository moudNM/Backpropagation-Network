'''
Student Name: Nur Muhammad Bin Khameed
SRN: 160269044
CO3311 Neural Networks CW2

Main Class

Instructions:
Please install these libraries to run this file:
os, sys, xlrd, openpyxl, numpy, pandas, matplotlib

There are 4 terminating conditions for training of the network.
1) Weight change falls below some threshold
2) Sum of square of last 4 errors fall below some threshold
3) Terminates when all data matches target label
4) Based on number of epochs

Fields taken in are number of clusters, learning rate, training data set, testing data set, termination condition and
termination condition value, respectively.
'''

import backpropagation as b

# weight change termination
n = b.Network(3, 0.1, 'TrainingDataSet.xlsx', 'TestingDataSet.xlsx', 'weight_change', 0.005)

# sum of square of last 4 errors termination
# n = b.Network(2, 0.1, 'TrainingDataSet.xlsx', 'TestingDataSet.xlsx', 'error_sum', 0.001)

# 24 matches termination
# n = b.Network(4, 0.1, 'TrainingDataSet.xlsx', 'TestingDataSet.xlsx', 'no_of_matches', 24)

# number of epochs termination
# n = b.Network(4, 0.1, 'TrainingDataSet.xlsx', 'TestingDataSet.xlsx', 'no_of_epochs', 100)

n.execute_network()
