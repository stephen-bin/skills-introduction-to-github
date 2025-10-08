import numpy as np


class NeuralNetwork:
    """
    A simple feedforward neural network implementation.
    
    This neural network supports:
    - Customizable layer sizes
    - Forward propagation
    - Backpropagation with gradient descent
    - Sigmoid activation function
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with random weights and biases.
        
        Args:
            layer_sizes (list): List of integers representing the number of neurons in each layer.
                               Example: [2, 3, 1] creates a network with 2 input neurons, 
                               3 hidden neurons, and 1 output neuron.
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases with random values
        self.weights = []
        self.biases = []
        
        for i in range(self.num_layers - 1):
            # He initialization for better gradient flow
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Derivative of sigmoid function."""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Perform forward propagation through the network.
        
        Args:
            X (numpy.ndarray): Input data of shape (num_samples, input_size)
        
        Returns:
            numpy.ndarray: Output of the network
        """
        self.layer_outputs = [X]
        
        for i in range(self.num_layers - 1):
            z = np.dot(self.layer_outputs[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            self.layer_outputs.append(a)
        
        return self.layer_outputs[-1]
    
    def backward(self, X, y, learning_rate):
        """
        Perform backpropagation and update weights and biases.
        
        Args:
            X (numpy.ndarray): Input data
            y (numpy.ndarray): Target labels
            learning_rate (float): Learning rate for gradient descent
        """
        m = X.shape[0]
        
        # Calculate output layer error
        deltas = [self.layer_outputs[-1] - y]
        
        # Backpropagate the error
        for i in range(self.num_layers - 2, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.sigmoid_derivative(self.layer_outputs[i])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(self.num_layers - 1):
            self.weights[i] -= learning_rate * np.dot(self.layer_outputs[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(deltas[i], axis=0, keepdims=True) / m
    
    def train(self, X, y, epochs, learning_rate=0.1, verbose=True):
        """
        Train the neural network.
        
        Args:
            X (numpy.ndarray): Training data
            y (numpy.ndarray): Training labels
            epochs (int): Number of training epochs
            learning_rate (float): Learning rate for gradient descent
            verbose (bool): Whether to print training progress
        
        Returns:
            list: History of losses during training
        """
        losses = []
        
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((y - output) ** 2)
            losses.append(loss)
            
            # Backward propagation
            self.backward(X, y, learning_rate)
            
            if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (numpy.ndarray): Input data
        
        Returns:
            numpy.ndarray: Predictions
        """
        return self.forward(X)
