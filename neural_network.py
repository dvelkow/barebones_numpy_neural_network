import numpy as np
from blessed import Terminal
import time
import matplotlib.pyplot as plt
import random

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

    def train(self, X, y, epochs, batch_size, validation_data=None, term=None):
        if term:
            self.initialize_visual(term)
        
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
                if term:
                    self.update_visual(term, epoch, epochs, train_loss, val_loss)
                elif epoch % 100 == 0:
                    print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if term:
                    self.update_visual(term, epoch, epochs, train_loss)
                elif epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {train_loss:.6f}')

    def calculate_loss(self, X, y):
        predictions = self.forward(X)
        return np.mean(np.square(y - predictions)) + self.regularization * sum(np.sum(w**2) for w in self.weights) / 2

    def predict(self, X):
        return self.forward(X)

    def initialize_visual(self, term):
        print(term.clear)
        print(term.black_on_khaki(term.center('Neural Network Training')))
        print('\n' * 2)
        print(term.move_y(term.height // 2 - 10) + term.center('Network Architecture'))
        
        layer_names = ['Input', 'Hidden', 'Hidden', 'Output']
        colors = [term.olivedrab1, term.dodgerblue1, term.dodgerblue1, term.orangered1]
        for i, (name, size, color) in enumerate(zip(layer_names, self.layer_sizes, colors)):
            y_pos = term.height // 2 - 5 + i * 3
            print(term.move_xy(term.width // 2 - 10, y_pos) + color(f"{name} Layer: {size} nodes"))
        
        print(term.move_xy(0, term.height - 2) + "Press Ctrl+C to stop training")

    def update_visual(self, term, epoch, total_epochs, train_loss, val_loss=None):
        progress = int((epoch + 1) / total_epochs * 50)
        
        with term.location(0, 3):
            print(term.blue(f"Epoch: {epoch + 1}/{total_epochs}"))
            print(term.green(f"Train Loss: {train_loss:.6f}"))
            if val_loss is not None:
                print(term.yellow(f"Validation Loss: {val_loss:.6f}"))
            print(term.red('Progress: [') + term.goldenrod(('=' * progress).ljust(50)) + term.red(']'))
        
        # Animate network activity
        for i, size in enumerate(self.layer_sizes):
            y_pos = term.height // 2 - 5 + i * 3
            x_pos = term.width // 2 + 15
            active_node = random.randint(0, size - 1)
            with term.location(x_pos, y_pos):
                nodes = '○' * size
                nodes = nodes[:active_node] + '●' + nodes[active_node+1:]
                print(term.khaki(nodes))
        
        time.sleep(0.05)

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history)
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

def main():
    term = Terminal()
    
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Initialization
    nn = NeuralNetwork(layer_sizes=[2, 4, 4, 1], learning_rate=0.1, regularization=0.01)
    
    try:
        with term.fullscreen(), term.hidden_cursor():
            # Training
            nn.train(X, y, epochs=5000, batch_size=4, term=term)
    except KeyboardInterrupt:
        pass
    
    # Making predictions
    predictions = nn.predict(X)
    
    term.clear()
    print(term.black_on_khaki(term.center('Training Complete')))
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(term.bold(f"Input: {X[i]}") + term.move_right(5) + 
              term.green(f"Predicted: {pred[0]:.4f}") + term.move_right(5) + 
              term.blue(f"Actual: {y[i][0]}"))
    
    # Plot loss history
    nn.plot_loss()

if __name__ == "__main__":
    main()