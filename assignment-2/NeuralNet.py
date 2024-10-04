#####################################################################################################################
#   Assignment 2: Neural Network Analysis
#   This is a starter code in Python 3.6 for a neural network.
#   You need to have numpy and pandas installed before running this code.
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it
#       in the README file.
#
#####################################################################################################################


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo 
import tensorflow as tf
import keras
from keras import layers


class NeuralNet:
    def __init__(self, dataFile, header=True):
        # CHANGE: change to fetch data using ucimlrepo
        self.raw_input = fetch_ucirepo(id=2)
        
    # TODO: Write code for pre-processing the dataset, which would include
    # standardization, normalization,
    #   categorical to numerical, etc
    def preprocess(self):
        self.processed_data = self.raw_input
        
        # Standardization
        self.processed_data = (self.processed_data - self.processed_data.mean()) / self.processed_data.std()
        
        # Normalization
        self.processed_data = (self.processed_data - self.processed_data.min()) / (self.processed_data.max() - self.processed_data.min())
        
        # Drop rows with missing values
        self.processed_data = self.processed_data.dropna()
        
        # Categorical to numerical
        self.processed_data = pd.get_dummies(self.processed_data)
        
        return 0

    # TODO: Train and evaluate models for all combinations of parameters
    # specified in the init method. We would like to obtain following outputs:
    #   1. Training Accuracy and Error (Loss) for every model
    #   2. Test Accuracy and Error (Loss) for every model
    #   3. History Curve (Plot of Accuracy against training steps) for all
    #       the models in a single plot. The plot should be color coded i.e.
    #       different color for each model

    def train_evaluate(self):
        ncols = len(self.processed_data.columns)
        nrows = len(self.processed_data.index)
        X = self.processed_data.iloc[:, 0:(ncols - 1)]
        y = self.processed_data.iloc[:, (ncols-1)]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y)

        # Below are the hyperparameters that you need to use for model evaluation
        # You can assume any fixed number of neurons for each hidden layer. 
        
        # CHANGE: change logistic to sigmoid 
        activations = ['sigmoid', 'tanh', 'relu']
        learning_rate = [0.01, 0.1]
        max_iterations = [100, 200] # also known as epochs
        num_hidden_layers = [2, 3]
        
        # Create the neural network and be sure to keep track of the performance
        #   metrics
        
        
        # Plot the model history for each model in a single plot
        # model history is a plot of accuracy vs number of epochs
        # you may want to create a large sized plot to show multiple lines
        # in a same figure.

        return 0




if __name__ == "__main__":
    neural_network = NeuralNet("train.csv") # put in path to your file
    neural_network.preprocess()
    neural_network.train_evaluate()
