'''
Student Name: Nur Muhammad Bin Khameed
SRN: 160269044
CO3311 Neural Networks CW2

Main Class

Instructions:
Please install these libraries to run this file:
os, sys, xlrd, openpyxl, numpy, pandas, matplotlib
'''

import os.path
import sys
import xlrd
import openpyxl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Network:
    def __init__(self, no_of_clusters, learning_rate, training_dataset, testing_dataset,
                 termination_condition='weight_change', termination_condition_value=0):

        # training dataset
        self.training_dataset = pd.DataFrame(columns=['x', 'y', 'z', '2-cluster', '3-cluster', '4-cluster'])
        # testing dataset
        self.testing_dataset = pd.DataFrame(columns=['x', 'y', 'z', '2-cluster', '3-cluster', '4-cluster'])

        self.epoch_counter = pd.DataFrame(
            columns=['epoch_number', 'iteration_number'
                                     'x', 'y', 'z', 'b1', 'bw1', 'wx1', 'wy1', 'wz1', 'net1', 'a1',
                     'b2', 'bw2', 'wx2', 'wy2', 'wz2', 'net2', 'a2',
                     'b3', 'bw3', 'wx3', 'wy3', 'net3', 'a3',
                     'predicted', 'target', 'target_val', 'unit1_error', 'unit2_error', 'unit3_error',
                     'sum_of_square_errors', 'iter_weight_change', 'epoch_weight_change', 'no_of_matches'])

        self.no_of_clusters = no_of_clusters
        self.learning_rate = learning_rate
        self.iteration_number = 0
        self.current_epoch = pd.DataFrame(
            columns=['epoch_number', 'iteration_number',
                     'x', 'y', 'z', 'b1', 'bw1', 'wx1', 'wy1', 'wz1', 'net1', 'a1',
                     'b2', 'bw2', 'wx2', 'wy2', 'wz2', 'net2', 'a2',
                     'b3', 'bw3', 'wx3', 'wy3', 'net3', 'a3',
                     'predicted', 'target', 'target_val', 'unit1_error', 'unit2_error', 'unit3_error',
                     'sum_of_square_errors', 'iter_weight_change', 'epoch_weight_change', 'no_of_matches'])

        # checks
        self.check_fails = False

        # check if training dataset file exists has correct extension
        if isinstance(training_dataset, str) and training_dataset.endswith('.xls') or training_dataset.endswith(
                '.xlsx'):

            if (os.path.isfile(training_dataset)):

                # check if headers of data correct
                if set(['x', 'y', 'z', '2-cluster', '3-cluster', '4-cluster']).issubset(self.training_dataset.columns):
                    self.training_dataset = pd.read_excel(training_dataset)

                else:
                    self.check_fails = True
                    print('One or more columns x,y,z,2-cluster,3-cluster,4-cluster are missing')

            else:
                self.check_fails = True
                print('Dataset file does not exist.')
        else:
            self.check_fails = True
            print('Wrong file format for dataset.')

        # check if testing dataset file exists has correct extension
        if isinstance(testing_dataset, str) and testing_dataset.endswith('.xls') or testing_dataset.endswith('.xlsx'):

            if (os.path.isfile(testing_dataset)):

                # check if headers of data correct
                if set(['x', 'y', 'z']).issubset(
                        self.testing_dataset.columns):

                    self.testing_dataset = pd.read_excel(testing_dataset)

                else:
                    self.check_fails = True
                    print('One or more columns x,y,z,2-cluster,3-cluster,4-cluster are missing')

            else:
                self.check_fails = True
                print('Dataset file does not exist.')
        else:
            self.check_fails = True
            print('Wrong file format for dataset.')

        # set termination mode
        self.termination_condition = termination_condition
        self.termination_condition_value = termination_condition_value

        if (termination_condition == 'weight_change' or termination_condition == 'error_sum' or
                termination_condition == 'no_of_matches' or termination_condition == 'no_of_epochs'):
            print()

        else:
            self.check_fails = True
            print('Wrong terminating condition.')

        # exit program if parameters incorrect
        if (self.check_fails):
            exit()

        self.epoch_number = 0

        # units
        self.unit1 = []
        self.unit2 = []
        self.unit3 = []
        self.setup_units()

    def new_epoch(self):

        self.epoch_number += 1
        print('epoch number:', self.epoch_number)

        # write dataset into current_epoch
        for index, row in self.training_dataset.iterrows():

            self.iteration_number += 1
            self.current_epoch.at[self.iteration_number - 1, 'epoch_number'] = self.epoch_number
            self.current_epoch.at[self.iteration_number - 1, 'iteration_number'] = self.iteration_number
            self.current_epoch.at[self.iteration_number - 1, 'x'] = row[0]
            self.current_epoch.at[self.iteration_number - 1, 'y'] = row[1]
            self.current_epoch.at[self.iteration_number - 1, 'z'] = row[2]

            if (self.no_of_clusters == 2):
                self.current_epoch.at[self.iteration_number - 1, 'target'] = row[3]
                if (self.current_epoch.at[self.iteration_number - 1, 'target'] == 1):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.25
                else:
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.75
            elif (self.no_of_clusters == 3):
                self.current_epoch.at[self.iteration_number - 1, 'target'] = row[4]
                if (self.current_epoch.at[self.iteration_number - 1, 'target'] == 1):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.25
                elif (self.current_epoch.at[self.iteration_number - 1, 'target'] == 2):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.50
                else:
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.75
            else:
                self.current_epoch.at[self.iteration_number - 1, 'target'] = row[5]
                if (self.current_epoch.at[self.iteration_number - 1, 'target'] == 1):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.20
                elif (self.current_epoch.at[self.iteration_number - 1, 'target'] == 2):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.40
                elif (self.current_epoch.at[self.iteration_number - 1, 'target'] == 3):
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.60
                else:
                    self.current_epoch.at[self.iteration_number - 1, 'target_val'] = 0.80

            # write unit 1,2,3 to current row
            # write unit1
            if self.iteration_number == 1:
                self.current_epoch.at[index, 'b1'] = self.unit1[0]
                self.current_epoch.at[index, 'bw1'] = self.unit1[1]
                self.current_epoch.at[index, 'wx1'] = self.unit1[2]
                self.current_epoch.at[index, 'wy1'] = self.unit1[3]
                self.current_epoch.at[index, 'wz1'] = self.unit1[4]

                # write unit2
                self.current_epoch.at[index, 'b2'] = self.unit2[0]
                self.current_epoch.at[index, 'bw2'] = self.unit2[1]
                self.current_epoch.at[index, 'wx2'] = self.unit2[2]
                self.current_epoch.at[index, 'wy2'] = self.unit2[3]
                self.current_epoch.at[index, 'wz2'] = self.unit2[4]

                # write unit3
                self.current_epoch.at[index, 'b3'] = self.unit3[0]
                self.current_epoch.at[index, 'bw3'] = self.unit3[1]
                self.current_epoch.at[index, 'wx3'] = self.unit3[2]
                self.current_epoch.at[index, 'wy3'] = self.unit3[3]

            # unit1 net and activation
            self.current_epoch.at[self.iteration_number - 1, 'net1'] = (
                    (self.current_epoch.at[self.iteration_number - 1, 'b1'] * self.current_epoch.at[
                        self.iteration_number - 1, 'bw1'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'x'] * self.current_epoch.at[
                self.iteration_number - 1, 'wx1'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'y'] * self.current_epoch.at[
                self.iteration_number - 1, 'wy1'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'z'] * self.current_epoch.at[
                self.iteration_number - 1, 'wz1'])
            )
            self.current_epoch.at[self.iteration_number - 1, 'a1'] = 1 / (
                    1 + np.power(np.e, -self.current_epoch.at[self.iteration_number - 1, 'net1']))

            # unit2 net and activation
            self.current_epoch.at[self.iteration_number - 1, 'net2'] = (
                    (self.current_epoch.at[self.iteration_number - 1, 'b2'] * self.current_epoch.at[
                        self.iteration_number - 1, 'bw2'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'x'] * self.current_epoch.at[
                self.iteration_number - 1, 'wx2'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'y'] * self.current_epoch.at[
                self.iteration_number - 1, 'wy2'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'z'] * self.current_epoch.at[
                self.iteration_number - 1, 'wz2'])
            )
            self.current_epoch.at[self.iteration_number - 1, 'a2'] = 1 / (
                    1 + np.power(np.e, -self.current_epoch.at[self.iteration_number - 1, 'net2']))

            # unit3 net and activation
            self.current_epoch.at[self.iteration_number - 1, 'net3'] = (
                    (self.current_epoch.at[self.iteration_number - 1, 'b3'] * self.current_epoch.at[
                        self.iteration_number - 1, 'bw3'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'a1'] * self.current_epoch.at[
                self.iteration_number - 1, 'wx3'])
                    + (self.current_epoch.at[self.iteration_number - 1, 'a2'] * self.current_epoch.at[
                self.iteration_number - 1, 'wy3'])
            )
            self.current_epoch.at[self.iteration_number - 1, 'a3'] = 1 / (
                    1 + np.power(np.e, -self.current_epoch.at[self.iteration_number - 1, 'net3']))

            # 2 clusters -> 0.25, 0.75 target
            if self.no_of_clusters == 2:
                if self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.5:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 1
                else:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 2
            # 3 clusters -> 0.25,0.5,0.75
            elif self.no_of_clusters == 3:
                if self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.375:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 1
                elif (self.current_epoch.at[self.iteration_number - 1, 'a3'] >= 0.375) and (
                        self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.625):
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 2
                else:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 3

            # 4 clusters -> 0.2, 0.4, 0.6, 0.8

            elif self.no_of_clusters == 4:
                if self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.3:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 1

                elif (self.current_epoch.at[self.iteration_number - 1, 'a3'] >= 0.3) and (
                        self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.5):
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 2

                elif (self.current_epoch.at[self.iteration_number - 1, 'a3'] >= 0.5) and (
                        self.current_epoch.at[self.iteration_number - 1, 'a3'] < 0.7):
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 3

                else:
                    self.current_epoch.at[self.iteration_number - 1, 'predicted'] = 4



            # if ((self.current_epoch.at[self.iteration_number - 1, 'a3'] >= 0.5)):
            #     print(self.current_epoch.at[self.iteration_number - 1, 'iteration_number'])

            # error calculation
            # unit 3 error
            self.current_epoch.at[self.iteration_number - 1, 'unit3_error'] = (self.current_epoch.at[
                                                                                   self.iteration_number - 1, 'target_val'] -
                                                                               self.current_epoch.at[
                                                                                   self.iteration_number - 1, 'a3']) * (
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'a3']) * (
                                                                                      1 - self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'a3'])

            # unit 1 and 2 error
            self.current_epoch.at[self.iteration_number - 1, 'unit1_error'] = self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'unit3_error'] * \
                                                                              self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'a1'] * (
                                                                                      1 - self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'a1']) * \
                                                                              self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'wx3']

            self.current_epoch.at[self.iteration_number - 1, 'unit2_error'] = self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'unit3_error'] * \
                                                                              self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'a2'] * (
                                                                                      1 - self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'a2']) * \
                                                                              self.current_epoch.at[
                                                                                  self.iteration_number - 1, 'wy3']

            # calculate sum of last 4 square errors
            if (self.iteration_number >= 4):
                self.current_epoch.at[self.iteration_number - 1, 'sum_of_square_errors'] = np.power(
                    self.current_epoch.at[self.iteration_number - 1 - 3, 'unit3_error'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1 - 2, 'unit3_error'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1 - 1, 'unit3_error'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'unit3_error'], 2)

            else:
                self.current_epoch.at[self.iteration_number - 1, 'sum_of_square_errors'] = None

            # calculate iteration weight change
            if (self.iteration_number >= 2):
                self.current_epoch.at[self.iteration_number - 1, 'iter_weight_change'] = np.sqrt(np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'bw1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wx1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wy1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wz1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wz1'], 2) \
                                                                                                 + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'bw2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wx2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wy2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wz2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wz2'], 2) \
                                                                                                 + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'bw3'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wx3'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 1, 'wy3'], 2)

                                                                                                 )

            else:
                self.current_epoch.at[self.iteration_number - 1, 'iter_weight_change'] = None

            # calculate epoch weight change and number of matches
            if (self.iteration_number % 24 == 0):

                self.current_epoch.at[self.iteration_number - 1, 'epoch_weight_change'] = np.sqrt(np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'bw1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wx1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wy1'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wz1'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wz1'], 2) \
                                                                                                  + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'bw2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wx2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wy2'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wz2'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wz2'], 2) \
                                                                                                  + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'bw3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'bw3'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wx3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wx3'], 2) + np.power(
                    self.current_epoch.at[self.iteration_number - 1, 'wy3'] - self.current_epoch.at[
                        self.iteration_number - 1 - 23, 'wy3'], 2)

                                                                                                  )

                no_of_matches = 0
                for i in range(0, 24):
                    if (self.current_epoch.at[self.iteration_number - 1 - i, 'predicted'] == self.current_epoch.at[
                        self.iteration_number - 1 - i, 'target']):
                        no_of_matches += 1
                self.current_epoch.at[self.iteration_number - 1, 'no_of_matches'] = no_of_matches

            # update weights
            # update unit1
            self.current_epoch.at[self.iteration_number - 1 + 1, 'b1'] = self.current_epoch.at[
                self.iteration_number - 1, 'b1']
            self.current_epoch.at[self.iteration_number - 1 + 1, 'bw1'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'bw1'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit1_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'b1'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wx1'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wx1'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit1_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'x'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wy1'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wy1'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit1_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'y'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wz1'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wz1'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit1_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'z'])

            # update unit2
            self.current_epoch.at[self.iteration_number - 1 + 1, 'b2'] = self.current_epoch.at[
                self.iteration_number - 1, 'b2']
            self.current_epoch.at[self.iteration_number - 1 + 1, 'bw2'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'bw2'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit2_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'b2'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wx2'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wx2'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit2_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'x'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wy2'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wy2'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit2_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'y'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wz2'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wz2'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit2_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'z'])

            # update unit3
            self.current_epoch.at[self.iteration_number - 1 + 1, 'b3'] = self.current_epoch.at[
                self.iteration_number - 1, 'b3']
            self.current_epoch.at[self.iteration_number - 1 + 1, 'bw3'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'bw3'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit3_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'b3'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wx3'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wx3'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit3_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'x'])
            self.current_epoch.at[self.iteration_number - 1 + 1, 'wy3'] = self.current_epoch.at[
                                                                              self.iteration_number - 1, 'wy3'] + (
                                                                                  self.learning_rate *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'unit3_error'] *
                                                                                  self.current_epoch.at[
                                                                                      self.iteration_number - 1, 'y'])

    def testing(self):

        self.testing_epoch = pd.DataFrame(columns=['row_number',
                                                   'x', 'y', 'z', 'b1', 'bw1', 'wx1', 'wy1', 'wz1', 'net1', 'a1',
                                                   'b2', 'bw2', 'wx2', 'wy2', 'wz2', 'net2', 'a2',
                                                   'b3', 'bw3', 'wx3', 'wy3', 'net3', 'a3',
                                                   'predicted'])

        # write dataset into current_epoch
        for index, row in self.testing_dataset.iterrows():

            self.testing_epoch.at[index, 'row_number'] = index + 1
            self.testing_epoch.at[index, 'x'] = row[0]
            self.testing_epoch.at[index, 'y'] = row[1]
            self.testing_epoch.at[index, 'z'] = row[2]

            # write unit 1,2,3 to current row
            # write unit1

            self.testing_epoch.at[index, 'b1'] = self.unit1[0]
            self.testing_epoch.at[index, 'bw1'] = self.unit1[1]
            self.testing_epoch.at[index, 'wx1'] = self.unit1[2]
            self.testing_epoch.at[index, 'wy1'] = self.unit1[3]
            self.testing_epoch.at[index, 'wz1'] = self.unit1[4]

            # write unit2
            self.testing_epoch.at[index, 'b2'] = self.unit2[0]
            self.testing_epoch.at[index, 'bw2'] = self.unit2[1]
            self.testing_epoch.at[index, 'wx2'] = self.unit2[2]
            self.testing_epoch.at[index, 'wy2'] = self.unit2[3]
            self.testing_epoch.at[index, 'wz2'] = self.unit2[4]

            # write unit3
            self.testing_epoch.at[index, 'b3'] = self.unit3[0]
            self.testing_epoch.at[index, 'bw3'] = self.unit3[1]
            self.testing_epoch.at[index, 'wx3'] = self.unit3[2]
            self.testing_epoch.at[index, 'wy3'] = self.unit3[3]

            # unit1 net and activation
            self.testing_epoch.at[index, 'net1'] = (
                    (self.testing_epoch.at[index, 'b1'] * self.testing_epoch.at[
                        index, 'bw1'])
                    + (self.testing_epoch.at[index, 'x'] * self.testing_epoch.at[
                index, 'wx1'])
                    + (self.testing_epoch.at[index, 'y'] * self.testing_epoch.at[
                index, 'wy1'])
                    + (self.testing_epoch.at[index, 'z'] * self.testing_epoch.at[
                index, 'wz1'])
            )
            self.testing_epoch.at[index, 'a1'] = 1 / (
                    1 + np.power(np.e, -self.testing_epoch.at[index, 'net1']))

            # unit2 net and activation
            self.testing_epoch.at[index, 'net2'] = (
                    (self.testing_epoch.at[index, 'b2'] * self.testing_epoch.at[
                        index, 'bw2'])
                    + (self.testing_epoch.at[index, 'x'] * self.testing_epoch.at[
                index, 'wx2'])
                    + (self.testing_epoch.at[index, 'y'] * self.testing_epoch.at[
                index, 'wy2'])
                    + (self.testing_epoch.at[index, 'z'] * self.testing_epoch.at[
                index, 'wz2'])
            )
            self.testing_epoch.at[index, 'a2'] = 1 / (
                    1 + np.power(np.e, -self.testing_epoch.at[index, 'net2']))

            # unit3 net and activation

            self.testing_epoch.at[index, 'net3'] = (
                    (self.testing_epoch.at[index, 'b3'] * self.testing_epoch.at[
                        index, 'bw3'])
                    + (self.testing_epoch.at[index, 'a1'] * self.testing_epoch.at[
                index, 'wx3'])
                    + (self.testing_epoch.at[index, 'a2'] * self.testing_epoch.at[
                index, 'wy3'])
            )

            self.testing_epoch.at[index, 'a3'] = 1 / (
                    1 + np.power(np.e, -self.testing_epoch.at[index, 'net3']))

            # 2 clusters -> 0.25, 0.75 target
            if self.no_of_clusters == 2:
                if self.testing_epoch.at[index, 'a3'] < 0.5:
                    self.testing_epoch.at[index, 'predicted'] = 1
                else:
                    self.testing_epoch.at[index, 'predicted'] = 2
            # 3 clusters -> 0.25,0.5,0.75
            elif self.no_of_clusters == 3:
                if self.testing_epoch.at[index, 'a3'] < 0.375:
                    self.testing_epoch.at[index, 'predicted'] = 1
                elif (self.testing_epoch.at[index, 'a3'] >= 0.375) and (
                        self.testing_epoch.at[index, 'a3'] < 0.625):
                    self.testing_epoch.at[index, 'predicted'] = 2
                else:
                    self.testing_epoch.at[index, 'predicted'] = 3
            # 4 clusters -> 0.2, 0.4, 0.6, 0.8
            elif self.no_of_clusters == 4:
                if self.testing_epoch.at[index, 'a3'] < 0.3:
                    self.testing_epoch.at[index, 'predicted'] = 1
                elif (self.testing_epoch.at[index, 'a3'] >= 0.3) and (
                        self.testing_epoch.at[index, 'a3'] < 0.5):
                    self.testing_epoch.at[index, 'predicted'] = 2
                elif (self.testing_epoch.at[index, 'a3'] >= 0.5) and (
                        self.testing_epoch.at[index, 'a3'] < 0.7):
                    self.testing_epoch.at[index, 'predicted'] = 3
                else:
                    self.testing_epoch.at[index, 'predicted'] = 4

    # get range of values for x,y,z
    def get_range(self, dataset):
        min_x = 0
        min_y = 0
        min_z = 0

        max_x = 0
        max_y = 0
        max_z = 0

        for index, row in dataset.iterrows():
            x = row[0]
            y = row[1]
            z = row[2]

            if x < min_x:
                min_x = x

            if y < min_y:
                min_y = y

            if z < min_z:
                min_z = z

            if x > max_x:
                max_x = x

            if y > max_y:
                max_y = y

            if z > max_z:
                max_z = z

        return min_x, max_x, min_y, max_y, min_z, max_z

    # get random x,y,z from the range of values in the dataset
    def random_vector(self, min_x, max_x, min_y, max_y, min_z, max_z):
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        z = np.random.uniform(min_z, max_z)
        return [x, y, z]

    # set up all units
    def setup_units(self):
        # get range of x,y,z from table data
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_range(self.training_dataset)

        u = self.random_vector(min_x, max_x, min_y, max_y, min_z, max_z)
        self.unit1 = [1, 1] + u

        u = self.random_vector(min_x, max_x, min_y, max_y, min_z, max_z)
        self.unit2 = [1, 1] + u

        u = self.random_vector(min_x, max_x, min_y, max_y, min_z, max_z)
        self.unit3 = [1, 1] + [u[0], u[1]]

    # check weight change
    def check_weight(self):
        terminate = True
        if self.current_epoch.at[self.iteration_number - 1, 'epoch_weight_change'] > self.termination_condition_value:
            terminate = False
        return terminate

    # check last 4 sum of errors
    def check_sum(self):
        terminate = True
        if self.current_epoch.at[self.iteration_number - 1, 'sum_of_square_errors'] > self.termination_condition_value:
            terminate = False
        return terminate

    # check number of matches
    def check_matches(self):
        terminate = self.current_epoch.at[self.iteration_number - 1, 'no_of_matches'] == 24
        return terminate

    def plot_graph(self):
        # plot graphs and save it
        fig = plt.figure()
        # plots = []
        plot = fig.add_subplot(111)

        x_points = []
        y_points = []


        for index, row in self.current_epoch.iterrows():
            if not np.isnan(row['epoch_weight_change']):
                x_points.append(row['epoch_number'])
                y_points.append(row['epoch_weight_change'])

        p = plot.plot(x_points, y_points, linestyle='-', marker='.', markersize=1)

        plot.set_xlabel('Epoch Number')
        plot.set_ylabel('Weight Change')
        plot.set_title('Backpropagation Network with ' + str(self.no_of_clusters) + ' clusters')

        graph_file_name = 'Epoch_Weight_Change_Figure.png'
        plt.savefig(graph_file_name)


    # execute network
    def execute_network(self):

        # training the network
        terminate = False
        # terminate when weight change == value
        if (self.termination_condition == 'weight_change'):
            while (not terminate):
                self.new_epoch()
                terminate = self.check_weight()

        # terminate when error sum == value
        elif (self.termination_condition == 'error_sum'):
            while (not terminate):
                self.new_epoch()
                terminate = self.check_sum()

        # terminate when matches == 24
        elif (self.termination_condition == 'no_of_matches'):
            while (not terminate):
                self.new_epoch()
                terminate = self.check_matches()

        # terminate when
        else:
            start = 0
            end = self.termination_condition_value

            while (start < end):
                self.new_epoch()
                start += 1

        # save output tables
        self.write_to_file('Training_Result.xlsx', 'current_epoch')

        self.plot_graph()


        # classifying data set
        self.unit1 = [self.current_epoch.at[self.iteration_number, 'b1'],
                      self.current_epoch.at[self.iteration_number, 'bw1'],
                      self.current_epoch.at[self.iteration_number, 'wx1'],
                      self.current_epoch.at[self.iteration_number, 'wy1'],
                      self.current_epoch.at[self.iteration_number, 'wz1']]
        self.unit2 = [self.current_epoch.at[self.iteration_number, 'b2'],
                      self.current_epoch.at[self.iteration_number, 'bw2'],
                      self.current_epoch.at[self.iteration_number, 'wx2'],
                      self.current_epoch.at[self.iteration_number, 'wy2'],
                      self.current_epoch.at[self.iteration_number, 'wz2']]
        self.unit3 = [self.current_epoch.at[self.iteration_number, 'b3'],
                      self.current_epoch.at[self.iteration_number, 'bw3'],
                      self.current_epoch.at[self.iteration_number, 'wx3'],
                      self.current_epoch.at[self.iteration_number, 'wy3']]

        self.testing()
        self.write_to_file('Testing_Result.xlsx', 'testing_epoch')

    def write_to_file(self, output_file_name, dataframe):
        writer = pd.ExcelWriter(output_file_name)

        if (dataframe == 'current_epoch'):
            self.current_epoch.to_excel(writer, 'Sheet1', index=False)

        elif (dataframe == 'testing_epoch'):
            self.testing_epoch.to_excel(writer, 'Sheet1', index=False)
        writer.save()
