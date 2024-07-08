import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output):
        self.error = y - output
        self.delta2 = self.error * self.sigmoid_derivative(output)
        self.error_hidden = np.dot(self.delta2, self.W2.T)
        self.delta1 = self.error_hidden * self.relu_derivative(self.a1)

        self.W2 += self.learning_rate * np.dot(self.a1.T, self.delta2)
        self.b2 += self.learning_rate * np.sum(self.delta2, axis=0, keepdims=True)
        self.W1 += self.learning_rate * np.dot(X.T, self.delta1)
        self.b1 += self.learning_rate * np.sum(self.delta1, axis=0, keepdims=True)

    def train(self, X, y, epochs, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]  # Use full batch gradient descent if batch size is not specified

        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - self.forward(X)))
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

    def save_model(self, file_path):
        np.savez(file_path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)

    def load_model(self, file_path):
        npzfile = np.load(file_path)
        self.W1 = npzfile['W1']
        self.b1 = npzfile['b1']
        self.W2 = npzfile['W2']
        self.b2 = npzfile['b2']

# Example:
if __name__ == "__main__":
    # dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # initialization
    nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1)

    # training
    nn.train(X, y, epochs=10000, batch_size=4)

    # making predictions
    predictions = nn.predict(X)
    print("Predictions:")
    print(predictions)

    nn.save_model("model.npz")

    nn.load_model("model.npz")
