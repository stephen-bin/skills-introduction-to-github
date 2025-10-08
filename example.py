"""
Example usage of the Neural Network implementation.

This example demonstrates training a neural network on the XOR problem,
a classic non-linearly separable problem that requires at least one hidden layer.
"""

import numpy as np
from neural_network import NeuralNetwork


def main():
    print("=" * 60)
    print("Neural Network Example: XOR Problem")
    print("=" * 60)
    
    # XOR problem data
    # Inputs: [x1, x2]
    # Output: x1 XOR x2
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    print("\nTraining Data (XOR):")
    print("Inputs:")
    print(X)
    print("\nExpected Outputs:")
    print(y.T)
    
    # Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
    print("\n" + "-" * 60)
    print("Creating Neural Network: [2, 4, 1]")
    print("  - Input layer: 2 neurons")
    print("  - Hidden layer: 4 neurons")
    print("  - Output layer: 1 neuron")
    print("-" * 60)
    
    nn = NeuralNetwork([2, 4, 1])
    
    # Train the network
    print("\nTraining the network...")
    losses = nn.train(X, y, epochs=5000, learning_rate=0.5, verbose=True)
    
    # Test the network
    print("\n" + "=" * 60)
    print("Testing the trained network:")
    print("=" * 60)
    
    predictions = nn.predict(X)
    
    print("\nInput\t\tExpected\tPredicted\tRounded")
    print("-" * 60)
    for i in range(len(X)):
        predicted = predictions[i][0]
        rounded = round(predicted)
        expected = y[i][0]
        print(f"{X[i]}\t{expected}\t\t{predicted:.6f}\t{rounded}")
    
    # Calculate accuracy
    rounded_predictions = np.round(predictions)
    accuracy = np.mean(rounded_predictions == y) * 100
    print("\n" + "=" * 60)
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    # Additional example: Simple AND gate
    print("\n\n" + "=" * 60)
    print("Bonus Example: AND Gate")
    print("=" * 60)
    
    X_and = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    
    y_and = np.array([[0],
                      [0],
                      [0],
                      [1]])
    
    print("\nTraining Data (AND):")
    print("Inputs:", X_and.tolist())
    print("Expected:", y_and.T.tolist()[0])
    
    nn_and = NeuralNetwork([2, 3, 1])
    print("\nTraining AND gate network...")
    nn_and.train(X_and, y_and, epochs=2000, learning_rate=0.5, verbose=False)
    
    predictions_and = nn_and.predict(X_and)
    print("\nResults:")
    print("Input\t\tExpected\tPredicted\tRounded")
    print("-" * 60)
    for i in range(len(X_and)):
        predicted = predictions_and[i][0]
        rounded = round(predicted)
        expected = y_and[i][0]
        print(f"{X_and[i]}\t{expected}\t\t{predicted:.6f}\t{rounded}")
    
    accuracy_and = np.mean(np.round(predictions_and) == y_and) * 100
    print(f"\nAccuracy: {accuracy_and:.2f}%")


if __name__ == "__main__":
    main()
