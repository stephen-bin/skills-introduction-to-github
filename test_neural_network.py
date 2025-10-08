"""
Tests for the Neural Network implementation.
"""

import numpy as np
from neural_network import NeuralNetwork


def test_initialization():
    """Test that the neural network initializes correctly."""
    nn = NeuralNetwork([2, 3, 1])
    
    assert nn.num_layers == 3, "Number of layers should be 3"
    assert len(nn.weights) == 2, "Should have 2 weight matrices"
    assert len(nn.biases) == 2, "Should have 2 bias vectors"
    assert nn.weights[0].shape == (2, 3), "First weight matrix should be 2x3"
    assert nn.weights[1].shape == (3, 1), "Second weight matrix should be 3x1"
    print("✓ Initialization test passed")


def test_forward_propagation():
    """Test forward propagation."""
    nn = NeuralNetwork([2, 2, 1])
    X = np.array([[0, 0], [1, 1]])
    
    output = nn.forward(X)
    
    assert output.shape == (2, 1), "Output shape should be (2, 1)"
    assert np.all((output >= 0) & (output <= 1)), "Output should be between 0 and 1"
    print("✓ Forward propagation test passed")


def test_sigmoid():
    """Test sigmoid activation function."""
    nn = NeuralNetwork([2, 1])
    
    # Test sigmoid
    assert abs(nn.sigmoid(0) - 0.5) < 0.01, "sigmoid(0) should be ~0.5"
    assert nn.sigmoid(10) > 0.99, "sigmoid(10) should be close to 1"
    assert nn.sigmoid(-10) < 0.01, "sigmoid(-10) should be close to 0"
    
    # Test derivative
    s = nn.sigmoid(0)
    assert abs(nn.sigmoid_derivative(s) - 0.25) < 0.01, "sigmoid_derivative should work"
    print("✓ Sigmoid function test passed")


def test_training():
    """Test that training reduces loss."""
    # Simple linearly separable problem
    X = np.array([[0, 0], [1, 1]])
    y = np.array([[0], [1]])
    
    nn = NeuralNetwork([2, 2, 1])
    
    # Get initial loss
    initial_output = nn.forward(X)
    initial_loss = np.mean((y - initial_output) ** 2)
    
    # Train
    losses = nn.train(X, y, epochs=500, learning_rate=0.5, verbose=False)
    
    # Get final loss
    final_loss = losses[-1]
    
    assert final_loss < initial_loss, "Training should reduce loss"
    assert final_loss < 0.1, "Final loss should be reasonably small"
    print("✓ Training test passed")


def test_xor_learning():
    """Test that the network can learn XOR."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    nn = NeuralNetwork([2, 4, 1])
    nn.train(X, y, epochs=3000, learning_rate=0.5, verbose=False)
    
    predictions = nn.predict(X)
    rounded = np.round(predictions)
    
    accuracy = np.mean(rounded == y)
    assert accuracy >= 0.9, f"XOR accuracy should be at least 90%, got {accuracy * 100}%"
    print(f"✓ XOR learning test passed (accuracy: {accuracy * 100}%)")


def test_predict():
    """Test prediction function."""
    nn = NeuralNetwork([2, 2, 1])
    X = np.array([[0.5, 0.5]])
    
    predictions = nn.predict(X)
    
    assert predictions.shape == (1, 1), "Prediction shape should match"
    assert 0 <= predictions[0][0] <= 1, "Prediction should be between 0 and 1"
    print("✓ Prediction test passed")


def run_all_tests():
    """Run all tests."""
    print("Running Neural Network Tests...")
    print("=" * 60)
    
    test_initialization()
    test_forward_propagation()
    test_sigmoid()
    test_training()
    test_xor_learning()
    test_predict()
    
    print("=" * 60)
    print("All tests passed! ✓")


if __name__ == "__main__":
    run_all_tests()
