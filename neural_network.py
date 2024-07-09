import numpy as np

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, regularization=0.01):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.weights = [np.random.randn(y, x) * np.sqrt(2. / x) 
                        for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in layer_sizes[1:]]
        self.activations = []
        self.loss_history = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i].T) + self.biases[i].T
            a = self.relu(z)
            self.activations.append(a)
        z_final = np.dot(self.activations[-1], self.weights[-1].T) + self.biases[-1].T
        self.activations.append(self.sigmoid(z_final))
        return self.activations[-1]

    def backward(self, X, y):
        m = X.shape[0]
        delta = self.activations[-1] - y
        dW = [np.dot(delta.T, self.activations[-2]) / m + self.regularization * self.weights[-1]]
        db = [np.sum(delta, axis=0, keepdims=True).T / m]

        for l in range(len(self.weights) - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1]) * self.relu_derivative(self.activations[l+1])
            dW.insert(0, np.dot(delta.T, self.activations[l]) / m + self.regularization * self.weights[l])
            db.insert(0, np.sum(delta, axis=0, keepdims=True).T / m)

        return dW, db

    def update_parameters(self, dW, db):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW[i]
            self.biases[i] -= self.learning_rate * db[i]

    def train(self, X, y, epochs, batch_size, validation_data=None, visualizer=None):
        if visualizer:
            visualizer.initialize()
        
        for epoch in range(epochs):
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.forward(X_batch)
                dW, db = self.backward(X_batch, y_batch)
                self.update_parameters(dW, db)
            
            train_loss = self.calculate_loss(X, y)
            self.loss_history.append(train_loss)
            
            if validation_data:
                val_loss = self.calculate_loss(validation_data[0], validation_data[1])
                if visualizer:
                    visualizer.update(epoch, epochs, train_loss, val_loss)
                elif epoch % 100 == 0:
                    print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if visualizer:
                    visualizer.update(epoch, epochs, train_loss)
                elif epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {train_loss:.6f}')

    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        return np.mean(np.square(y - predictions)) + self.regularization * sum(np.sum(w**2) for w in self.weights) / 2

    def predict(self, X):
        return self.forward(X)