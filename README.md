# Project Overview

This is a from-scratch implementation of a neural network using NumPy for core functionality, with Blessed for terminal visualization and Matplotlib for post-training graphing. The project includes:
- A flexible neural network that can handle multiple hidden layers
- Real-time visualization of the training process in your terminal
- Implementation of backpropagation, mini-batch gradient descent, and regularization




![image](https://github.com/dvelkow/barebones_numpy_neural_network/assets/71397644/8c6e728b-4dd9-43fc-9db0-374d69a1b441)

![image](https://github.com/dvelkow/barebones_numpy_neural_network/assets/71397644/cedff2c1-0c35-4d67-9b90-95a50c73b3c7)

## Features

- **Customizable Architecture**: You can change the amount of hidden layers depending on your needs.
- **Real-time Training Visualization**: You see it update Real Time on the terminal
- **XOR Problem Solver**: The network is demonstrating solving the XOR problem, which is a classic benchmark for neural networks.
- **Loss History Plotting**: Through a graph made with the help of Matplotlib after the training finishes you can see how your network improved over time
- **Flexible Input**: Can be adapted to solve various binary classification problems.

## How to Use

1. Clone this repo:
   ```
   git clone https://github.com/your-username/barebones-numpy-neural-network.git
   ```

2. Make sure you have the requirements:
   ```
   pip install numpy matplotlib blessed
   ```

3. Run the main script:
   ```
   python3 nn_trainer.py
   ```

## Customization

Modify the `main()` function in nn_trainer.py if you want to experiment with different network configurations, you are free to:

Change `layer_sizes` to adjust the network architecture
Modify `learning_rate` and 'regularization' parameters
Adjust `epochs` and `batch_size` for different training dynamics

Example:
```
layer_sizes = [2, 8, 8, 1]  # Two hidden layers with 8 neurons each
nn = NeuralNetwork(layer_sizes=layer_sizes, learning_rate=0.05, regularization=0.01) #decreased the learning_rate
nn.train(X, y, epochs=10000, batch_size=4, visualizer=visualizer) #changed the epochs (might cause overfitting)
```

## What I Learned in the process

Building this helped me understand:
- How Backpropagation actually works, because before that I had experience with it only through libraries and never dived in the processes
- How Forward Propagation work, more specifically how data flows through a neural network and how predictions are made
- The impact of different activation functions on network performance and training dynamics
- The process of handling non-linear problems through XOR

## Future Ideas

- Add more activation functions
- Maybe tackle some complex datasets with it
- Extend the network to handle multi-class classification problems
