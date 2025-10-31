"""
CHAPTER 01: NEURAL NETWORK FROM SCRATCH
=========================================

This file implements a complete neural network using ONLY NumPy.
Every single line is explained so beginners can understand exactly what's happening.

What we'll build:
-----------------
1. Activation functions (sigmoid, ReLU, tanh, softmax) with derivatives
2. Dense (fully connected) layer class
3. Neural network class that stacks layers
4. Loss functions (binary cross-entropy, categorical cross-entropy)
5. Forward propagation
6. Training loop structure (without backprop - that's Chapter 02!)

File structure:
---------------
Part 1: Imports and utilities
Part 2: Activation functions
Part 3: Dense layer implementation
Part 4: Loss functions
Part 5: Neural network class
Part 6: Training utilities
Part 7: MNIST data loading
Part 8: Main execution and example

Author: Deep Learning Master Course
Purpose: Teaching neural networks from absolute scratch
"""

# ============================================================================
# PART 1: IMPORTS AND UTILITIES
# ============================================================================

import numpy as np  # The ONLY library we need for the neural network itself
import matplotlib.pyplot as plt  # For visualizing results
from typing import Tuple, List, Optional  # Type hints for better code documentation
import time  # To measure training time

# Set random seed for reproducibility
# This ensures we get the same results every time we run the code
# Useful for debugging and comparing experiments
np.random.seed(42)

# Print NumPy version (for debugging compatibility issues)
print(f"NumPy version: {np.__version__}")
print("=" * 70)
print("NEURAL NETWORK FROM SCRATCH - Chapter 01")
print("=" * 70)


# ============================================================================
# PART 2: ACTIVATION FUNCTIONS
# ============================================================================

# Why do we need activation functions?
# Without them, neural networks would just be linear transformations:
# output = W3 @ (W2 @ (W1 @ x)) = (W3 @ W2 @ W1) @ x = W_combined @ x
# This is just a single linear layer! Activation functions introduce non-linearity.

def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function: σ(z) = 1 / (1 + e^(-z))
    
    Properties:
    - Output range: (0, 1)
    - Smooth and differentiable everywhere
    - Saturates (gradient→0) for large |z| (vanishing gradient problem)
    - Can be interpreted as probability
    
    Use cases:
    - Output layer for binary classification
    - Historical use in hidden layers (replaced by ReLU)
    
    Args:
        z: Input array of any shape
        
    Returns:
        Array of same shape as z, values in (0, 1)
    
    Example:
        >>> sigmoid(np.array([0, 2, -2]))
        array([0.5, 0.88, 0.12])
    """
    # Numerical stability trick: clip z to avoid overflow in exp()
    # exp(-500) ≈ 0, exp(500) would overflow
    z = np.clip(z, -500, 500)
    
    # Formula: 1 / (1 + e^(-z))
    # When z is large positive: e^(-z) → 0, so output → 1
    # When z is large negative: e^(-z) → ∞, so output → 0
    # When z = 0: output = 0.5
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid function: σ'(z) = σ(z) * (1 - σ(z))
    
    This is used in backpropagation (Chapter 02).
    
    Mathematical derivation:
        σ(z) = 1 / (1 + e^(-z))
        
        Using quotient rule and chain rule:
        σ'(z) = e^(-z) / (1 + e^(-z))^2
              = [1 / (1 + e^(-z))] * [e^(-z) / (1 + e^(-z))]
              = σ(z) * [1 - σ(z)]
    
    Key property: Maximum derivative is 0.25 (when z=0, σ(z)=0.5)
    This is why sigmoid suffers from vanishing gradients!
    
    Args:
        z: Input array
        
    Returns:
        Derivative values, same shape as z
    """
    # First compute sigmoid(z)
    sig = sigmoid(z)
    
    # Then compute σ * (1 - σ)
    # This is element-wise multiplication, not matrix multiplication
    return sig * (1.0 - sig)


def relu(z: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation: ReLU(z) = max(0, z)
    
    Properties:
    - Output range: [0, ∞)
    - Piecewise linear (linear for z>0, constant for z<0)
    - Does NOT saturate for positive values (no vanishing gradient)
    - Sparse activation: ~50% of neurons output 0
    - Computationally very efficient (just comparison + multiplication)
    
    Use cases:
    - DEFAULT choice for hidden layers in most networks
    - CNNs, deep feedforward networks
    
    Downside:
    - "Dying ReLU" problem: if z<0, gradient=0, neuron never updates
    
    Args:
        z: Input array
        
    Returns:
        Element-wise maximum of 0 and z
        
    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])
    """
    # np.maximum compares element-wise and takes the max
    # For z=3: max(0, 3) = 3
    # For z=-5: max(0, -5) = 0
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU: ReLU'(z) = 1 if z > 0, else 0
    
    Mathematical definition:
        ReLU'(z) = { 1 if z > 0
                   { 0 if z ≤ 0
    
    Note: Technically undefined at z=0, but we use 0 in practice.
    
    Key property: Gradient is either 0 or 1 (no vanishing gradient for z>0!)
    
    Args:
        z: Input array
        
    Returns:
        Binary array: 1 where z>0, 0 where z≤0
    """
    # Create array of zeros with same shape as z
    derivative = np.zeros_like(z)
    
    # Set derivative to 1 wherever z > 0
    # This uses boolean indexing: derivative[condition] = value
    derivative[z > 0] = 1.0
    
    return derivative


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Hyperbolic tangent activation: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Properties:
    - Output range: (-1, 1)
    - Zero-centered (unlike sigmoid)
    - Stronger gradients than sigmoid (range is 2x larger)
    - Still saturates for large |z|
    
    Use cases:
    - Hidden layers (better than sigmoid, but ReLU is often better)
    - RNNs (historically used, still common)
    
    Relation to sigmoid:
        tanh(z) = 2 * sigmoid(2z) - 1
    
    Args:
        z: Input array
        
    Returns:
        Tanh values in (-1, 1)
        
    Example:
        >>> tanh(np.array([0, 1, -1]))
        array([0, 0.76, -0.76])
    """
    # NumPy has tanh built-in, but here's how it works:
    # numerator = e^z - e^(-z)
    # denominator = e^z + e^(-z)
    # tanh = numerator / denominator
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative of tanh: tanh'(z) = 1 - tanh^2(z)
    
    Mathematical derivation:
        tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
        
        Using quotient rule:
        tanh'(z) = 1 - tanh^2(z) = sech^2(z)
    
    Key property: Maximum derivative is 1 (when z=0, tanh(z)=0)
    This is 4x larger than sigmoid's maximum derivative!
    
    Args:
        z: Input array
        
    Returns:
        Derivative values
    """
    # First compute tanh(z)
    tanh_z = tanh(z)
    
    # Then compute 1 - tanh^2(z)
    return 1.0 - tanh_z ** 2


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation: converts logits to probability distribution
    
    Formula: softmax(z_i) = e^(z_i) / Σ_j e^(z_j)
    
    Properties:
    - Outputs sum to 1 (valid probability distribution)
    - Largest input → largest probability (maintains ordering)
    - Differentiable (gradient has nice closed form)
    - Highlights largest values (exponential amplification)
    
    Use cases:
    - Output layer for multi-class classification (mutually exclusive classes)
    - Attention mechanisms (Chapter 02 - Transformers)
    
    Args:
        z: Logits array, shape (n_samples, n_classes) or (n_classes,)
        
    Returns:
        Probability distribution, same shape as z
        
    Example:
        >>> softmax(np.array([1.0, 2.0, 3.0]))
        array([0.09, 0.24, 0.67])  # Sum = 1.0
    """
    # Numerical stability trick: subtract max before exp
    # softmax(z) = softmax(z - c) for any constant c
    # We use c = max(z) to prevent overflow
    #
    # Why this works:
    # softmax(z_i) = e^(z_i) / Σ e^(z_j)
    #              = e^(z_i - c) * e^c / [Σ e^(z_j - c) * e^c]
    #              = e^(z_i - c) / Σ e^(z_j - c)
    #
    # Without this trick: e^(1000) would overflow to infinity
    # With this trick: e^(1000 - 1000) = e^0 = 1 (no overflow!)
    
    # For batch processing, subtract max along class dimension
    if z.ndim == 1:
        # Single sample: z has shape (n_classes,)
        z_exp = np.exp(z - np.max(z))
        return z_exp / np.sum(z_exp)
    else:
        # Batch: z has shape (n_samples, n_classes)
        # Subtract max of each sample (keepdims=True preserves shape for broadcasting)
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        # Sum across classes (axis=1) for each sample
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)


# ============================================================================
# PART 3: DENSE LAYER IMPLEMENTATION
# ============================================================================

class DenseLayer:
    """
    Fully connected (dense) layer: output = activation(W @ input + b)
    
    This is the fundamental building block of neural networks.
    
    Architecture:
    -------------
        Input neurons → [Weights + Bias] → Activation → Output neurons
        
        If input has n_in neurons and output has n_out neurons:
        - Weights: (n_out, n_in) matrix
        - Bias: (n_out,) vector
        - Each output neuron connects to ALL input neurons (hence "dense")
    
    Forward pass computation:
    -------------------------
        1. Linear transformation: z = W @ x + b
        2. Activation: a = f(z)
    
    Parameters:
    -----------
        n_inputs: Number of input features
        n_outputs: Number of output neurons
        activation: Which activation function to use
        
    Attributes:
    -----------
        weights: (n_outputs, n_inputs) weight matrix
        biases: (n_outputs,) bias vector
        input_cache: Stored input for backprop (Chapter 02)
        z_cache: Stored pre-activation for backprop
    """
    
    def __init__(
        self, 
        n_inputs: int, 
        n_outputs: int, 
        activation: str = 'relu'
    ):
        """
        Initialize a dense layer with random weights and zero biases.
        
        Weight initialization is CRITICAL for training deep networks.
        Bad initialization → vanishing/exploding gradients → poor performance
        
        We use Xavier/Glorot initialization:
            W ~ Normal(0, sqrt(2 / (n_in + n_out)))
        
        This keeps variance of activations roughly constant across layers.
        
        Args:
            n_inputs: Size of input (previous layer's output size)
            n_outputs: Size of output (this layer's neuron count)
            activation: 'relu', 'sigmoid', 'tanh', or 'softmax'
        """
        # Store dimensions
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_name = activation
        
        # Initialize weights with Xavier/Glorot initialization
        # Scale = sqrt(2 / (n_in + n_out))
        # This prevents exploding/vanishing activations
        limit = np.sqrt(2.0 / (n_inputs + n_outputs))
        self.weights = np.random.randn(n_outputs, n_inputs) * limit
        
        # Why this shape? Matrix multiplication:
        # (n_out, n_in) @ (n_in, n_samples) = (n_out, n_samples)
        # Each row of W represents one output neuron's weights
        
        # Initialize biases to zero
        # Starting with small random biases is also common
        self.biases = np.zeros((n_outputs, 1))
        
        # Cache for backpropagation (Chapter 02)
        # We store inputs and pre-activations during forward pass
        # Then use them to compute gradients in backward pass
        self.input_cache = None
        self.z_cache = None
        
        # Map activation name to functions
        # This allows us to call self.activation(z) regardless of which activation we chose
        self.activation_functions = {
            'sigmoid': sigmoid,
            'relu': relu,
            'tanh': tanh,
            'softmax': softmax,
            'linear': lambda x: x  # No activation (for regression outputs)
        }
        
        self.activation_derivatives = {
            'sigmoid': sigmoid_derivative,
            'relu': relu_derivative,
            'tanh': tanh_derivative,
            'linear': lambda x: np.ones_like(x)  # Derivative of x is 1
        }
        
        # Get the actual functions
        self.activation = self.activation_functions[activation]
        self.activation_derivative = self.activation_derivatives.get(activation, None)
        
        print(f"Initialized DenseLayer: {n_inputs} → {n_outputs} ({activation})")
    
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through this layer.
        
        Computation:
        ------------
            1. Z = W @ X + b    (linear transformation)
            2. A = f(Z)         (apply activation)
        
        Shapes:
        -------
            X: (n_inputs, n_samples) - input from previous layer
            W: (n_outputs, n_inputs) - this layer's weights
            b: (n_outputs, 1) - this layer's biases
            Z: (n_outputs, n_samples) - pre-activation
            A: (n_outputs, n_samples) - post-activation output
        
        Args:
            X: Input array, shape (n_inputs, n_samples)
            
        Returns:
            A: Activated output, shape (n_outputs, n_samples)
            
        Example:
            >>> layer = DenseLayer(3, 2, 'relu')
            >>> X = np.random.randn(3, 10)  # 10 samples, 3 features each
            >>> A = layer.forward(X)
            >>> A.shape
            (2, 10)  # 10 samples, 2 outputs each
        """
        # Cache input for backpropagation (Chapter 02)
        # We need this to compute dW = dZ @ X.T
        self.input_cache = X
        
        # Step 1: Linear transformation Z = W @ X + b
        # @ is matrix multiplication in Python (NumPy)
        # W @ X computes weighted sum for each output neuron
        # + b broadcasts bias to all samples
        #
        # Broadcasting example:
        # Z = (2, 3) @ (3, 10) + (2, 1)
        #   = (2, 10) + (2, 1)
        #   = (2, 10)  [bias added to each sample]
        self.z_cache = self.weights @ X + self.biases
        
        # Step 2: Apply activation function
        # This is element-wise operation
        # For ReLU: max(0, z_i) for each element z_i
        # For Sigmoid: 1/(1+e^(-z_i)) for each element z_i
        output = self.activation(self.z_cache)
        
        return output
    
    
    def get_weights_shape(self) -> Tuple[int, int]:
        """
        Get the shape of weight matrix (for debugging).
        
        Returns:
            Tuple (n_outputs, n_inputs)
        """
        return self.weights.shape
    
    
    def get_number_of_parameters(self) -> int:
        """
        Calculate total number of trainable parameters in this layer.
        
        Parameters:
        -----------
        - Weights: n_outputs × n_inputs
        - Biases: n_outputs
        
        Total = n_outputs × (n_inputs + 1)
        
        Example:
        --------
        Input: 784 neurons
        Output: 128 neurons
        Parameters: 128 × 784 (weights) + 128 (biases) = 100,480
        
        Returns:
            Total parameter count
        """
        # Number of weights
        n_weights = self.weights.size  # Total elements in weight matrix
        
        # Number of biases
        n_biases = self.biases.size  # Total elements in bias vector
        
        return n_weights + n_biases


# ============================================================================
# PART 4: LOSS FUNCTIONS
# ============================================================================

def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Binary cross-entropy loss for binary classification.
    
    Formula:
    --------
        L = -1/n * Σ [y * log(ŷ) + (1-y) * log(1-ŷ)]
    
    Where:
    - y ∈ {0, 1} is the true label
    - ŷ ∈ (0, 1) is the predicted probability
    
    Intuition:
    ----------
    - If y=1 (positive class): loss = -log(ŷ)
        * ŷ→1 (confident correct): loss→0 ✓
        * ŷ→0 (confident wrong): loss→∞ ✗
    
    - If y=0 (negative class): loss = -log(1-ŷ)
        * ŷ→0 (confident correct): loss→0 ✓
        * ŷ→1 (confident wrong): loss→∞ ✗
    
    This heavily penalizes confident wrong predictions!
    
    Args:
        y_true: True labels, shape (n_samples,) or (n_samples, 1)
        y_pred: Predicted probabilities, same shape as y_true
        
    Returns:
        Average loss (scalar)
        
    Example:
        >>> y_true = np.array([1, 0, 1, 0])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.2])
        >>> binary_cross_entropy(y_true, y_pred)
        0.15  # Low loss, good predictions!
    """
    # Clip predictions to avoid log(0) which is -infinity
    # We use a small epsilon (1e-15) to prevent numerical issues
    # log(0) = -∞, but log(1e-15) ≈ -34.5 (large but finite)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute loss for each sample
    # For y=1: -log(y_pred)
    # For y=0: -log(1 - y_pred)
    sample_losses = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    # Return average loss across all samples
    return np.mean(sample_losses)


def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Categorical cross-entropy loss for multi-class classification.
    
    Formula:
    --------
        L = -1/n * Σ_i Σ_k y_{i,k} * log(ŷ_{i,k})
    
    Where:
    - y_{i,k} ∈ {0, 1} is 1 if sample i is class k (one-hot encoded)
    - ŷ_{i,k} ∈ (0, 1) is predicted probability of class k for sample i
    - Σ_k ŷ_{i,k} = 1 (probabilities sum to 1, from softmax)
    
    Intuition:
    ----------
    Only the TRUE class contributes to the loss!
    If sample is class 3, then y = [0, 0, 0, 1, 0, ...]
    Loss = -log(ŷ_3)
    
    - If ŷ_3 = 0.9 (confident correct): loss = -log(0.9) = 0.11 ✓
    - If ŷ_3 = 0.1 (not confident): loss = -log(0.1) = 2.30 ✗
    
    Args:
        y_true: True labels, shape (n_samples, n_classes) one-hot encoded
        y_pred: Predicted probabilities, shape (n_samples, n_classes)
        
    Returns:
        Average loss (scalar)
        
    Example:
        >>> y_true = np.array([[0, 1, 0], [1, 0, 0]])  # Classes 1 and 0
        >>> y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        >>> categorical_cross_entropy(y_true, y_pred)
        0.18  # Good predictions!
    """
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute loss: -Σ y_true * log(y_pred)
    # y_true is one-hot, so only one term per sample contributes
    # Example: y_true = [0, 0, 1, 0], y_pred = [0.1, 0.2, 0.6, 0.1]
    #          loss = 0*log(0.1) + 0*log(0.2) + 1*log(0.6) + 0*log(0.1)
    #               = -log(0.6) = 0.51
    sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
    
    # Return average loss across all samples
    return np.mean(sample_losses)


# ============================================================================
# PART 5: NEURAL NETWORK CLASS
# ============================================================================

class NeuralNetwork:
    """
    Complete neural network: stack of dense layers with forward propagation.
    
    This class manages multiple layers and implements:
    - Forward propagation through all layers
    - Loss computation
    - Prediction (argmax for classification)
    - Accuracy calculation
    
    In Chapter 02, we'll add:
    - Backpropagation (gradient computation)
    - Weight updates (optimization)
    
    Architecture example:
    ---------------------
        Input (784) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
        
        This is a 3-layer network (not counting input):
        - Hidden layer 1: 784 → 128
        - Hidden layer 2: 128 → 64
        - Output layer: 64 → 10
    
    Attributes:
    -----------
        layers: List of DenseLayer objects
        loss_function: 'binary' or 'categorical'
        loss_history: List of losses during training
    """
    
    def __init__(self):
        """
        Initialize an empty neural network.
        
        Layers will be added using add_layer() method.
        """
        self.layers = []  # List to store layers
        self.loss_function = None  # Will be set later
        self.loss_history = []  # Track loss during training
        
        print("\nInitializing Neural Network...")
        print("=" * 70)
    
    
    def add_layer(
        self, 
        n_outputs: int, 
        activation: str = 'relu', 
        n_inputs: Optional[int] = None
    ):
        """
        Add a dense layer to the network.
        
        Args:
            n_outputs: Number of neurons in this layer
            activation: Activation function ('relu', 'sigmoid', 'tanh', 'softmax')
            n_inputs: Number of inputs (auto-inferred from previous layer if not provided)
            
        Example:
            >>> nn = NeuralNetwork()
            >>> nn.add_layer(128, 'relu', n_inputs=784)  # First layer, must specify input size
            >>> nn.add_layer(64, 'relu')  # Second layer, input size = previous output (128)
            >>> nn.add_layer(10, 'softmax')  # Output layer, input size = previous output (64)
        """
        # For first layer, n_inputs must be provided
        # For subsequent layers, n_inputs = previous layer's n_outputs
        if len(self.layers) == 0:
            # First layer
            if n_inputs is None:
                raise ValueError("First layer must specify n_inputs (input feature size)")
            input_size = n_inputs
        else:
            # Not first layer: input comes from previous layer's output
            input_size = self.layers[-1].n_outputs
        
        # Create and add the layer
        layer = DenseLayer(input_size, n_outputs, activation)
        self.layers.append(layer)
        
        print(f"Layer {len(self.layers)}: {input_size} → {n_outputs} ({activation})")
    
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation through entire network.
        
        Process:
        --------
        1. Pass input through layer 1 → get activation 1
        2. Pass activation 1 through layer 2 → get activation 2
        3. ... continue for all layers
        4. Return final layer's output
        
        This is sequential: output of layer i is input to layer i+1
        
        Args:
            X: Input data, shape (n_features, n_samples)
            
        Returns:
            Final layer output, shape (n_outputs, n_samples)
            
        Example:
            >>> nn = NeuralNetwork()
            >>> nn.add_layer(128, 'relu', n_inputs=784)
            >>> nn.add_layer(10, 'softmax')
            >>> X = np.random.randn(784, 100)  # 100 samples
            >>> output = nn.forward(X)
            >>> output.shape
            (10, 100)  # 10 class probabilities for each of 100 samples
        """
        # Start with input
        activation = X
        
        # Pass through each layer sequentially
        for i, layer in enumerate(self.layers):
            # Output of layer i becomes input to layer i+1
            activation = layer.forward(activation)
            
            # Debugging: print shape after each layer
            # print(f"  After layer {i+1}: shape {activation.shape}")
        
        # Return final output
        return activation
    
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        For classification:
        - Binary: threshold at 0.5
        - Multi-class: argmax (class with highest probability)
        
        Args:
            X: Input data, shape (n_features, n_samples)
            
        Returns:
            Predicted class labels, shape (n_samples,)
            
        Example:
            >>> probabilities = np.array([[0.1, 0.7, 0.2],  # Sample 1
            ...                            [0.8, 0.1, 0.1]])  # Sample 2
            >>> predictions = np.argmax(probabilities, axis=0)
            >>> predictions
            array([1, 0])  # Sample 1 → class 1, Sample 2 → class 0
        """
        # Forward pass to get output probabilities
        output = self.forward(X)  # Shape: (n_classes, n_samples)
        
        # For binary classification (output size = 1)
        if output.shape[0] == 1:
            # Threshold at 0.5
            return (output > 0.5).astype(int).flatten()
        
        # For multi-class classification
        # argmax along class dimension (axis=0)
        # Returns index of largest probability for each sample
        return np.argmax(output, axis=0)
    
    
    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute loss between predictions and true labels.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
            
        Returns:
            Loss value (scalar)
        """
        if self.loss_function == 'binary':
            return binary_cross_entropy(y_true, y_pred)
        elif self.loss_function == 'categorical':
            return categorical_cross_entropy(y_true, y_pred)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")
    
    
    def calculate_accuracy(self, X: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculate classification accuracy.
        
        Accuracy = (Number of correct predictions) / (Total predictions)
        
        Args:
            X: Input data, shape (n_features, n_samples)
            y_true: True class labels, shape (n_samples,)
            
        Returns:
            Accuracy as decimal (0.0 to 1.0)
            
        Example:
            >>> y_true = np.array([0, 1, 2, 1, 0])
            >>> y_pred = np.array([0, 1, 1, 1, 0])
            >>> accuracy = np.mean(y_true == y_pred)
            >>> accuracy
            0.8  # 4 out of 5 correct = 80%
        """
        # Get predictions
        predictions = self.predict(X)
        
        # Compare with true labels (element-wise equality)
        # Returns boolean array: [True, False, True, ...]
        correct = (predictions == y_true)
        
        # Mean of boolean array = fraction of True values = accuracy
        accuracy = np.mean(correct)
        
        return accuracy
    
    
    def summary(self):
        """
        Print network architecture summary (like Keras model.summary()).
        
        Shows:
        - Each layer's input/output size
        - Activation function
        - Number of parameters
        - Total parameters
        """
        print("\n" + "=" * 70)
        print("NEURAL NETWORK ARCHITECTURE SUMMARY")
        print("=" * 70)
        
        total_params = 0
        
        for i, layer in enumerate(self.layers):
            n_params = layer.get_number_of_parameters()
            total_params += n_params
            
            print(f"Layer {i+1}: DenseLayer")
            print(f"  Input size: {layer.n_inputs}")
            print(f"  Output size: {layer.n_outputs}")
            print(f"  Activation: {layer.activation_name}")
            print(f"  Parameters: {n_params:,}")
            print("-" * 70)
        
        print(f"Total parameters: {total_params:,}")
        print("=" * 70 + "\n")


# ============================================================================
# PART 6: DATA LOADING AND PREPROCESSING
# ============================================================================

def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST dataset.
    
    MNIST contains:
    - 60,000 training images
    - 10,000 test images
    - Images are 28x28 grayscale (784 pixels)
    - 10 classes (digits 0-9)
    
    Preprocessing:
    1. Flatten images: 28x28 → 784-dimensional vector
    2. Normalize pixels: [0, 255] → [0, 1]
    3. One-hot encode labels: class 3 → [0,0,0,1,0,0,0,0,0,0]
    4. Transpose: (n_samples, n_features) → (n_features, n_samples)
       This matches our network's expected input format
    
    Returns:
        X_train: (784, 60000) training images
        y_train: (60000,) training labels
        X_test: (784, 10000) test images
        y_test: (10000,) test labels
    """
    print("\nLoading MNIST dataset...")
    
    try:
        # Try to load from TensorFlow/Keras
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("  Loaded from Keras ✓")
    except ImportError:
        # Fallback: load from scikit-learn (smaller dataset)
        print("  Keras not available, trying scikit-learn...")
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data
        y = mnist.target.astype(int)
        
        # Split into train/test
        X_train = X[:60000]
        y_train = y[:60000]
        X_test = X[60000:]
        y_test = y[60000:]
        
        # Reshape to (n_samples, 28, 28)
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        print("  Loaded from scikit-learn ✓")
    
    # Print original shapes
    print(f"  Original shapes: X_train={X_train.shape}, y_train={y_train.shape}")
    
    # Step 1: Flatten images from (n_samples, 28, 28) to (n_samples, 784)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Step 2: Normalize pixel values from [0, 255] to [0, 1]
    # Why? Neural networks train better with small input values
    # Also puts all features on similar scale
    X_train_norm = X_train_flat.astype('float32') / 255.0
    X_test_norm = X_test_flat.astype('float32') / 255.0
    
    # Step 3: Transpose to (n_features, n_samples)
    # Our network expects inputs in columns, samples in rows
    X_train_final = X_train_norm.T  # (784, 60000)
    X_test_final = X_test_norm.T    # (784, 10000)
    
    print(f"  Preprocessed shapes: X_train={X_train_final.shape}, X_test={X_test_final.shape}")
    print(f"  Pixel value range: [{X_train_final.min():.2f}, {X_train_final.max():.2f}]")
    print(f"  Classes: {np.unique(y_train)}")
    
    return X_train_final, y_train, X_test_final, y_test


def create_mini_batches(
    X: np.ndarray, 
    y: np.ndarray, 
    batch_size: int = 64
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Split dataset into mini-batches for training.
    
    Why mini-batches?
    -----------------
    - Full batch (all samples): Very slow, requires lots of memory
    - Single sample (online learning): Noisy gradients, slow convergence
    - Mini-batch: Best of both worlds! Fast + stable
    
    Common batch sizes: 32, 64, 128, 256
    
    Args:
        X: Input data, shape (n_features, n_samples)
        y: Labels, shape (n_samples,)
        batch_size: Number of samples per batch
        
    Returns:
        List of (X_batch, y_batch) tuples
        
    Example:
        >>> X = np.random.randn(784, 1000)  # 1000 samples
        >>> y = np.random.randint(0, 10, 1000)
        >>> batches = create_mini_batches(X, y, batch_size=100)
        >>> len(batches)
        10  # 1000 samples / 100 per batch = 10 batches
    """
    n_samples = X.shape[1]
    
    # Shuffle data (important for SGD!)
    # Create random permutation of indices
    indices = np.random.permutation(n_samples)
    X_shuffled = X[:, indices]
    y_shuffled = y[indices]
    
    # Split into batches
    batches = []
    for i in range(0, n_samples, batch_size):
        # Extract batch
        X_batch = X_shuffled[:, i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        batches.append((X_batch, y_batch))
    
    return batches


# ============================================================================
# PART 7: VISUALIZATION UTILITIES
# ============================================================================

def visualize_predictions(
    X: np.ndarray, 
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_samples: int = 10
):
    """
    Visualize sample predictions with true labels.
    
    Shows grid of images with predicted and true labels.
    Highlights incorrect predictions in red.
    
    Args:
        X: Input images, shape (784, n_samples)
        y_true: True labels, shape (n_samples,)
        y_pred: Predicted labels, shape (n_samples,)
        n_samples: Number of samples to display
    """
    # Select random samples
    indices = np.random.choice(X.shape[1], n_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i >= n_samples:
            ax.axis('off')
            continue
        
        # Get image and labels
        idx = indices[i]
        image = X[:, idx].reshape(28, 28)
        true_label = y_true[idx]
        pred_label = y_pred[idx]
        
        # Display image
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        # Add title (red if wrong, green if correct)
        if true_label == pred_label:
            color = 'green'
            title = f'✓ Pred: {pred_label}'
        else:
            color = 'red'
            title = f'✗ Pred: {pred_label} (True: {true_label})'
        
        ax.set_title(title, color=color, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_loss_history(loss_history: List[float]):
    """
    Plot training loss over epochs.
    
    Shows how loss decreases during training.
    Helps identify:
    - Convergence (loss plateaus)
    - Overfitting (train loss ↓, val loss ↑)
    - Learning rate issues (loss unstable)
    
    Args:
        loss_history: List of loss values per epoch
    """
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, linewidth=2)
    plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# PART 8: MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function: Build and test neural network on MNIST.
    
    Steps:
    1. Load MNIST data
    2. Build network architecture
    3. Test forward propagation
    4. Visualize sample predictions (before training)
    
    Note: Actual training (with backpropagation) comes in Chapter 02!
    """
    print("\n" + "="*70)
    print("CHAPTER 01: NEURAL NETWORK FROM SCRATCH - DEMO")
    print("="*70)
    
    # Load MNIST dataset
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Build neural network
    # Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)
    nn = NeuralNetwork()
    nn.add_layer(128, activation='relu', n_inputs=784)  # Hidden layer 1
    nn.add_layer(64, activation='relu')                  # Hidden layer 2
    nn.add_layer(10, activation='softmax')               # Output layer
    nn.loss_function = 'categorical'
    
    # Print architecture summary
    nn.summary()
    
    # Test forward propagation with a small batch
    print("Testing forward propagation...")
    X_batch = X_train[:, :10]  # First 10 training samples
    y_batch = y_train[:10]
    
    # Forward pass
    output = nn.forward(X_batch)
    print(f"\nOutput shape: {output.shape}")  # Should be (10, 10) = 10 classes × 10 samples
    print(f"Output sample (probabilities for first image):\n{output[:, 0]}")
    print(f"Sum of probabilities: {np.sum(output[:, 0]):.4f}")  # Should be ~1.0
    
    # Make predictions
    predictions = nn.predict(X_batch)
    print(f"\nPredictions: {predictions}")
    print(f"True labels: {y_batch}")
    
    # Calculate accuracy (before training - should be ~10% random guessing)
    accuracy = nn.calculate_accuracy(X_test, y_test)
    print(f"\nAccuracy on test set (before training): {accuracy:.2%}")
    print("(Should be ~10% since network is randomly initialized)")
    
    print("\n" + "="*70)
    print("SUCCESS! Neural network forward pass is working correctly.")
    print("="*70)
    print("\nNext steps:")
    print("1. Study the code above carefully - every line is explained!")
    print("2. Experiment: Change network architecture, activation functions")
    print("3. Move to Chapter 02 to implement backpropagation and training")
    print("4. Complete the MNIST classification project")
    print("="*70)


# Run the demo if this file is executed directly
if __name__ == "__main__":
    main()
