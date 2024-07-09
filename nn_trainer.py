import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from visualizer import Visualizer

def plot_loss(loss_history):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def main():
    # XOR Dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Initialization
    layer_sizes = [2, 4, 4, 1]
    nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.1, regularization=0.01)
    visualizer = Visualizer(layer_sizes)
    
    try:
        with visualizer.term.fullscreen(), visualizer.term.hidden_cursor():
            # Training
            nn.train(X, y, epochs=5000, batch_size=4, visualizer=visualizer)
    except KeyboardInterrupt:
        pass
    
    # Making predictions
    predictions = nn.predict(X)
    
    visualizer.term.clear()
    print(visualizer.term.black_on_khaki(visualizer.term.center('Training Complete')))
    print("\nPredictions:")
    for i, pred in enumerate(predictions):
        print(visualizer.term.bold(f"Input: {X[i]}") + visualizer.term.move_right(5) + 
              visualizer.term.green(f"Predicted: {pred[0]:.4f}") + visualizer.term.move_right(5) + 
              visualizer.term.blue(f"Actual: {y[i][0]}"))
    
    # Plot loss history
    plot_loss(nn.loss_history)

if __name__ == "__main__":
    main()