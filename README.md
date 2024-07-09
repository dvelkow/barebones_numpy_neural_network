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
   python neural_network.py
   ```

## Customization

The `main()` function is basically the customizable part, by changing the values there one could solve different problems.

## What I Learned in the process

Building this helped me understand:
- How backpropagation actually works, because before that I had experience with it only through libraries and never dived in the processe
- The impact of different activation functions on network performance and training dynamics
- Why we need activation functions
- The impact of different activation functions on network performance and training dynamics
The importance of proper weight initialization in preventing issues like vanishing/exploding gradients
How regularization techniques help in reducing overfitting and improving generalization
The trade-offs between batch size, learning rate, and convergence speed in gradient descent

## Future Ideas

- Add more activation functions
- Maybe tackle some complex dataset with it
