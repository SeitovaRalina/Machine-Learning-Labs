import numpy as np
import pandas as pd
from scipy.special import expit
from optimizers import RMSPropOptimizer
import random
from typing import Callable

from sklearn.model_selection import train_test_split


def categorical_cross_entropy_loss(y_true, y_pred, derivative=False):
    if derivative:
        return y_pred - y_true
    else:
        num_examples = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / num_examples
        return loss
def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, derivative=False) -> np.ndarray:
    if derivative:
        return y_pred - y_true
    else:
        res = y_true * np.log(y_pred)
        return np.where(res == -np.inf, 0, res)
def mean_squared_error(y_true, y_pred, derivative=False):
    if derivative:
        return -2 * np.mean(y_true - y_pred)
    else:
        return np.mean((y_true - y_pred) ** 2)
def sigmoid(x, derivative=False):
    if derivative:
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        return expit(x)
def tanh(x, derivative=False):
    if derivative:
        return 1 - np.tanh(x) ** 2
    else:
        return np.tanh(x)
def relu(x, derivative=False):
    if derivative:
        return (x > 0).astype(float)
    else:
        return np.maximum(0, x)

class Layer:
    def __init__(self, neurons, activation_function):
        self.neurons = neurons
        self.activation_function = activation_function

class MLP:
    def __init__(self, num_input, num_hidden, lossfunc, optimizator):
        self.input_size = num_input
        self.lossfunc = lossfunc
        self.hidden_sizes = [x.neurons for x in num_hidden]
        self.hidden_sizes = self.hidden_sizes[:-1]
        self.output_size = num_hidden[-1].neurons
        self.num_layers = len(num_hidden)
        self.weights = []
        self.optimizator = optimizator
        self.biases = []
        input_dim = self.input_size

        for hidden_dim in self.hidden_sizes:
            weight = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
            self.weights.append(weight)
            bias = np.zeros(hidden_dim)
            self.biases.append(bias)
            input_dim = hidden_dim

        weight = np.random.randn(input_dim, self.output_size) / np.sqrt(input_dim)
        self.weights.append(weight)
        bias = np.zeros(self.output_size)
        self.biases.append(bias)

        self.activation_functions = [x.activation_function for x in num_hidden]
        self.activation_functions.insert(0, sigmoid)
    def forward(self, X):
        activations = [X]
        for i in range(self.num_layers):
            linear = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation = self.activation_functions[i](linear)
            activations.append(activation)
        return activations

    def backward(self, X, y, activations):
        grads = []
        num_examples = X.shape[0]

        dE_dh = self.lossfunc(y, activations[-1], derivative=True)
        for i in range(self.num_layers - 1, -1, -1):
            dE_dt = dE_dh * self.activation_functions[i](activations[i + 1], derivative=True)

            dE_dx = np.dot(activations[i].T, dE_dt) / num_examples

            dE_dh = np.dot(dE_dt, self.weights[i].T)

            grads.insert(0, dE_dx)

        return grads

    def train(self, X, y, learning_rate=0.1, batch_size=32, num_epochs=10):
        for epoch in range(num_epochs):
            indices = np.random.permutation(len(X))
            X_shuffle = X[indices]
            y_shuffle = y[indices]
        for i in range(0, len(X), batch_size):
            X_batch = X_shuffle[i:i + batch_size]
            y_batch = y_shuffle[i:i + batch_size]

            activations = self.forward(X_batch)
            y_pred = activations[-1]

            loss = self.lossfunc(y_batch, y_pred, derivative=False)

            backprop_grads = self.backward(X_batch, y_batch, activations)
            self.grads = backprop_grads

            opt = self.optimizator(num_layers=self.num_layers, weights=self.weights, grads=self.grads)
            self.weights = opt.optimize()
        # print(f"Epoch {epoch+1}/{num_epochs}, loss={loss:.4f}")

    def predict(self, X):
        activations = self.forward(X)
        y_pred = activations[-1]
        return y_pred

mlp = MLP(num_input = 12,num_hidden=[Layer(neurons=6,activation_function= sigmoid),
                                     Layer(neurons=5,activation_function= sigmoid),
                                     Layer(neurons=1,activation_function= sigmoid)],
                                     lossfunc=mean_squared_error,optimizator = RMSPropOptimizer)


