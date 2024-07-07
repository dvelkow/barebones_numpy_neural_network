import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        self.error_hidden = np.dot(self.delta2, self.W2.T)
        self.delta1 = self.error_hidden * self.sigmoid_derivative(self.a1)

        self.W2 += np.dot(self.a1.T, self.delta2)
        self.b2 += np.sum(self.delta2, axis=0, keepdims=True)
        self.W1 += np.dot(X.T, self.delta1)
        self.b1 += np.sum(self.delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for _ in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

    def predict(self, X):
        return self.forward(X)
