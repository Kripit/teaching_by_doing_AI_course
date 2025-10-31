"""
Chapter 04: Regularization Techniques from Scratch
===================================================

This file implements all major regularization techniques in pure NumPy:
1. L1 and L2 regularization (weight decay)
2. Dropout (inverted dropout with train/test modes)
3. Batch Normalization (with running statistics)
4. Early Stopping
5. Complete regularized neural network

Each component has extensive comments explaining:
- What the code does
- Why it works
- Mathematical intuition
- Implementation details

Author: Teaching by Doing AI Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import copy

# Set random seed for reproducibility
# This ensures that random operations (dropout masks, weight initialization)
# produce the same results every time you run the code
np.random.seed(42)


# ============================================================================
# ACTIVATION FUNCTIONS
# ============================================================================

def relu(x: np.ndarray) -> np.ndarray:
    """
    ReLU (Rectified Linear Unit) activation function.
    
    Formula: f(x) = max(0, x)
    - If x > 0: output = x (identity)
    - If x ≤ 0: output = 0 (cut off negative values)
    
    Why ReLU?
    - Simple to compute
    - No vanishing gradient problem (gradient is 1 for x > 0)
    - Introduces non-linearity
    - Most popular activation in deep learning
    
    Args:
        x: Input array of any shape
        
    Returns:
        Output array with same shape as x, with negative values zeroed
        
    Example:
        >>> relu(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 2])
    """
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Derivative of ReLU activation.
    
    Formula: f'(x) = {1 if x > 0, else 0}
    
    This is used during backpropagation:
    - If input was positive: gradient flows through (multiply by 1)
    - If input was negative: gradient is blocked (multiply by 0)
    
    Args:
        x: Input array (same x that was passed to relu during forward pass)
        
    Returns:
        Binary array (0s and 1s) indicating where gradient should flow
        
    Example:
        >>> relu_derivative(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 1])  # Gradient only flows through positive values
    """
    return (x > 0).astype(float)


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax activation for multi-class classification.
    
    Formula: softmax(x)_i = exp(x_i) / sum(exp(x_j)) for all j
    
    Properties:
    - Outputs are probabilities (sum to 1)
    - All outputs in range (0, 1)
    - Used in output layer for classification
    
    Numerical stability trick:
    - Subtract max(x) before exponential to prevent overflow
    - Since exp(x - max) / sum(exp(x - max)) = exp(x) / sum(exp(x))
    - This doesn't change the result but prevents exp(large_number) = inf
    
    Args:
        x: Input array of shape (batch_size, num_classes)
        
    Returns:
        Probability distribution of shape (batch_size, num_classes)
        Each row sums to 1
        
    Example:
        >>> softmax(np.array([[2.0, 1.0, 0.1]]))
        array([[0.659, 0.242, 0.099]])  # Probabilities sum to 1
    """
    # Subtract max for numerical stability (prevents overflow)
    # Broadcasting: max computed per row, subtracted from each element in that row
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    
    # Divide by sum to get probabilities
    # keepdims=True ensures result is (batch_size, 1) for proper broadcasting
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Categorical cross-entropy loss for multi-class classification.
    
    Formula: L = -1/m * sum(sum(y_true * log(y_pred)))
    
    Where:
    - m = batch size
    - y_true = one-hot encoded labels (shape: batch_size × num_classes)
    - y_pred = predicted probabilities from softmax (shape: batch_size × num_classes)
    
    Why cross-entropy?
    - Measures "surprise" or "information content"
    - Large loss when confident wrong prediction
    - Small loss when confident correct prediction
    - Convex function (easier to optimize)
    
    Numerical stability:
    - Add epsilon (1e-15) to prevent log(0) = -inf
    - Clip predictions to [epsilon, 1-epsilon] range
    
    Args:
        y_pred: Predicted probabilities, shape (batch_size, num_classes)
        y_true: True labels (one-hot), shape (batch_size, num_classes)
        
    Returns:
        Scalar loss value (average over batch)
        
    Example:
        >>> y_true = np.array([[0, 1, 0], [1, 0, 0]])  # 2 samples, 3 classes
        >>> y_pred = np.array([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]])
        >>> cross_entropy_loss(y_pred, y_true)
        0.223  # Low loss (good predictions)
    """
    # Number of samples in batch
    m = y_true.shape[0]
    
    # Clip predictions to prevent log(0)
    # If y_pred = 0 → log(0) = -inf (BAD!)
    # If y_pred = 1 → log(1) = 0 (OK)
    # Epsilon = 1e-15 is tiny enough to not affect results
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Compute cross-entropy
    # Only the true class contributes to loss (y_true masks out others)
    # Example: if y_true = [0, 1, 0], only y_pred[1] contributes
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
    
    return loss


# ============================================================================
# DENSE LAYER WITH L1/L2 REGULARIZATION
# ============================================================================

class DenseLayer:
    """
    Fully connected (dense) layer with L1 and L2 regularization support.
    
    This layer implements:
    - Forward pass: z = x @ W + b
    - Backward pass: compute gradients with regularization
    - L1 regularization: adds λ * |W| to loss
    - L2 regularization: adds λ * W² to loss
    
    Attributes:
        input_size: Number of input features
        output_size: Number of output neurons
        l1_lambda: L1 regularization strength (default: 0, no L1)
        l2_lambda: L2 regularization strength (default: 0, no L2)
        W: Weight matrix, shape (input_size, output_size)
        b: Bias vector, shape (1, output_size)
        
    Mathematical Details:
        Forward: z = x @ W + b
        Where:
        - x: input activations (batch_size, input_size)
        - W: weights (input_size, output_size)
        - b: bias (1, output_size)
        - z: pre-activation output (batch_size, output_size)
        
        Backward (chain rule):
        - dL/dW = (1/m) * x^T @ dz + regularization_gradient
        - dL/db = (1/m) * sum(dz)
        - dL/dx = dz @ W^T
        
        Regularization gradients:
        - L1: dL/dW += l1_lambda * sign(W)
        - L2: dL/dW += l2_lambda * W
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0
    ):
        """
        Initialize dense layer with random weights.
        
        Weight initialization: He initialization for ReLU
        - Weights ~ N(0, sqrt(2/input_size))
        - This prevents vanishing/exploding activations
        - Works well with ReLU activation
        
        Bias initialization: zeros
        - Biases can start at zero without issues
        - They will be adjusted during training
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons in this layer
            l1_lambda: L1 regularization strength (0 = no L1)
            l2_lambda: L2 regularization strength (0 = no L2)
        """
        self.input_size = input_size
        self.output_size = output_size
        self.l1_lambda = l1_lambda  # Lasso regularization
        self.l2_lambda = l2_lambda  # Ridge regularization
        
        # He initialization: std = sqrt(2 / input_size)
        # This is optimal for ReLU activation
        # Why? Maintains variance of activations across layers
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        
        # Bias initialization: zeros
        # Shape: (1, output_size) for easy broadcasting
        self.b = np.zeros((1, output_size))
        
        # Cache for backpropagation
        # Stored during forward pass, used during backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute z = x @ W + b
        
        Steps:
        1. Matrix multiply input by weights
        2. Add bias (broadcasted across batch)
        3. Store input for backward pass
        
        Matrix dimensions:
        - x: (batch_size, input_size)
        - W: (input_size, output_size)
        - x @ W: (batch_size, output_size)
        - b: (1, output_size) → broadcasted to (batch_size, output_size)
        - z: (batch_size, output_size)
        
        Args:
            x: Input activations, shape (batch_size, input_size)
            
        Returns:
            z: Pre-activation output, shape (batch_size, output_size)
            
        Example:
            >>> layer = DenseLayer(3, 2)
            >>> x = np.array([[1, 2, 3], [4, 5, 6]])  # 2 samples, 3 features
            >>> z = layer.forward(x)
            >>> z.shape
            (2, 2)  # 2 samples, 2 neurons
        """
        # Store input for backward pass
        # We need this to compute gradient: dL/dW = x^T @ dz
        self.cache['x'] = x
        
        # Compute pre-activation: z = x @ W + b
        # @ is matrix multiplication operator in NumPy
        # + broadcasts bias across batch dimension
        z = x @ self.W + self.b
        
        return z
    
    def backward(self, dz: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass: compute gradients and update weights.
        
        This implements:
        1. Gradient of loss w.r.t. weights: dW = (1/m) * x^T @ dz
        2. Gradient of loss w.r.t. bias: db = (1/m) * sum(dz)
        3. Gradient to pass back: dx = dz @ W^T
        4. Add regularization gradients if enabled
        5. Update weights using gradient descent
        
        Regularization:
        - L1: dW += l1_lambda * sign(W)
          - sign(W) = +1 if W > 0, -1 if W < 0, 0 if W = 0
          - This shrinks weights by constant amount
          - Drives small weights to exactly zero
          
        - L2: dW += l2_lambda * W
          - Gradient is proportional to weight magnitude
          - Large weights get penalized more
          - Weights decay exponentially
        
        Args:
            dz: Gradient from next layer, shape (batch_size, output_size)
            learning_rate: Step size for weight update
            
        Returns:
            dx: Gradient to propagate back, shape (batch_size, input_size)
            
        Mathematical Details:
            dL/dW = (1/m) * x^T @ dz + regularization
            dL/db = (1/m) * sum(dz, axis=0)
            dL/dx = dz @ W^T
            
        Example gradient flow:
            Forward: x → [W, b] → z → activation → loss
            Backward: dx ← [dW, db] ← dz ← d(activation) ← dL
        """
        # Retrieve input from forward pass
        x = self.cache['x']
        
        # Batch size (number of samples)
        m = x.shape[0]
        
        # ===== Compute weight gradient =====
        # dL/dW = (1/m) * x^T @ dz
        # Matrix dimensions:
        # - x^T: (input_size, batch_size)
        # - dz: (batch_size, output_size)
        # - dW: (input_size, output_size) ✓ Same shape as W
        dW = (1 / m) * (x.T @ dz)
        
        # Add L1 regularization gradient if enabled
        if self.l1_lambda > 0:
            # L1 gradient: l1_lambda * sign(W)
            # sign(W) returns: +1 for positive, -1 for negative, 0 for zero
            # This creates sparse weights (many exactly zero)
            dW += self.l1_lambda * np.sign(self.W)
        
        # Add L2 regularization gradient if enabled
        if self.l2_lambda > 0:
            # L2 gradient: l2_lambda * W
            # This shrinks weights proportional to their magnitude
            # Large weights get penalized more than small weights
            dW += self.l2_lambda * self.W
        
        # ===== Compute bias gradient =====
        # dL/db = (1/m) * sum(dz, axis=0)
        # Sum over batch dimension (axis=0)
        # keepdims keeps shape as (1, output_size) instead of (output_size,)
        db = (1 / m) * np.sum(dz, axis=0, keepdims=True)
        
        # ===== Compute gradient to previous layer =====
        # dL/dx = dz @ W^T
        # This gradient gets passed to the previous layer
        # Matrix dimensions:
        # - dz: (batch_size, output_size)
        # - W^T: (output_size, input_size)
        # - dx: (batch_size, input_size) ✓ Same shape as x
        dx = dz @ self.W.T
        
        # ===== Update weights and biases =====
        # Gradient descent: W := W - learning_rate * dW
        # We move in the opposite direction of the gradient
        # This minimizes the loss function
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        # Return gradient to propagate back to previous layer
        return dx
    
    def get_l1_loss(self) -> float:
        """
        Compute L1 regularization loss: λ * sum(|W|)
        
        This is added to the total loss during training.
        L1 encourages sparsity (many weights become exactly 0).
        
        Returns:
            L1 loss contribution (scalar)
            
        Example:
            >>> layer = DenseLayer(3, 2, l1_lambda=0.01)
            >>> layer.W = np.array([[1, -2], [3, -4], [5, -6]])
            >>> layer.get_l1_loss()
            0.21  # 0.01 * (1+2+3+4+5+6) = 0.01 * 21 = 0.21
        """
        if self.l1_lambda > 0:
            # Sum of absolute values of all weights
            # np.abs() computes element-wise absolute value
            # np.sum() sums all elements into a scalar
            return self.l1_lambda * np.sum(np.abs(self.W))
        return 0.0
    
    def get_l2_loss(self) -> float:
        """
        Compute L2 regularization loss: (λ/2) * sum(W²)
        
        This is added to the total loss during training.
        L2 encourages small weights (weights close to but not exactly 0).
        
        The factor of 1/2 is conventional:
        - Makes derivative cleaner: d/dW (W²/2) = W
        - Can be absorbed into λ (doesn't change optimal solution)
        
        Returns:
            L2 loss contribution (scalar)
            
        Example:
            >>> layer = DenseLayer(3, 2, l2_lambda=0.01)
            >>> layer.W = np.array([[1, 2], [3, 4], [5, 6]])
            >>> layer.get_l2_loss()
            0.455  # 0.01 * 0.5 * (1+4+9+16+25+36) = 0.005 * 91 = 0.455
        """
        if self.l2_lambda > 0:
            # Sum of squared weights, multiplied by λ/2
            # self.W ** 2 computes element-wise square
            # np.sum() sums all squared elements
            return 0.5 * self.l2_lambda * np.sum(self.W ** 2)
        return 0.0


# ============================================================================
# DROPOUT LAYER
# ============================================================================

class Dropout:
    """
    Dropout regularization layer (inverted dropout implementation).
    
    During training:
    - Randomly set fraction p of neurons to zero
    - Scale remaining neurons by 1/(1-p)
    - Each forward pass uses different random mask
    
    During testing:
    - Use all neurons (no dropout)
    - No scaling needed (due to inverted dropout)
    
    Why dropout works:
    1. Ensemble effect: Training many "sub-networks" simultaneously
    2. Prevents co-adaptation: Forces each neuron to be useful independently
    3. Acts like noise: Adds randomness that helps generalization
    
    Inverted dropout (what we implement):
    - Scale during training by 1/(1-p)
    - No scaling needed at test time
    - Simpler test-time code (important for deployment!)
    
    Alternative (original dropout):
    - No scaling during training
    - Scale by (1-p) at test time
    - More complex test code
    
    Attributes:
        p: Dropout rate (fraction of neurons to drop)
        training: Boolean flag (True = training mode, False = test mode)
        mask: Binary mask (stored during forward, used during backward)
    """
    
    def __init__(self, p: float = 0.5):
        """
        Initialize dropout layer.
        
        Args:
            p: Dropout rate (probability of dropping a neuron)
               - p = 0.5: Drop 50% of neurons (most common)
               - p = 0.3: Drop 30% (conservative)
               - p = 0.7: Drop 70% (aggressive)
               
        Typical values by layer:
        - Input layer: p = 0.0 - 0.2 (don't drop too much raw data)
        - Hidden layers: p = 0.3 - 0.5 (standard)
        - Large hidden layers: p = 0.5 - 0.7 (more dropout for more neurons)
        - Output layer: p = 0.0 (NEVER drop outputs!)
        """
        self.p = p  # Dropout rate
        self.training = True  # Start in training mode
        self.mask = None  # Will store binary dropout mask
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with dropout.
        
        Training mode:
        1. Generate random binary mask
        2. Apply mask (multiply element-wise)
        3. Scale by 1/(1-p) to maintain expected value
        4. Store mask for backward pass
        
        Test mode:
        - Return input unchanged (use all neurons)
        
        Why scale by 1/(1-p)?
        - Without scaling:
          - Training: E[output] = (1-p) * input (some neurons dropped)
          - Test: E[output] = input (all neurons active)
          - MISMATCH! Network behaves differently!
        - With scaling:
          - Training: E[output] = (1-p) * input * 1/(1-p) = input ✓
          - Test: E[output] = input ✓
          - CONSISTENT! Network behaves similarly!
        
        Args:
            x: Input activations, shape (batch_size, num_features)
            
        Returns:
            Output activations with dropout applied (training) or unchanged (test)
            
        Example (training mode, p=0.5):
            >>> dropout = Dropout(p=0.5)
            >>> x = np.array([[1.0, 2.0, 3.0, 4.0]])
            >>> out = dropout.forward(x)
            >>> out
            array([[2.0, 0.0, 6.0, 0.0]])  # Dropped 50%, scaled 2x
        """
        if not self.training:
            # Test mode: no dropout, return input as-is
            return x
        
        # Training mode: apply dropout
        
        # Generate random binary mask
        # np.random.rand() generates uniform random in [0, 1)
        # Comparison (< p) gives True/False, converted to 1/0
        # Result: 1 with probability (1-p), 0 with probability p
        self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
        
        # Example with p=0.5:
        # - np.random.rand(2, 3) might give [[0.3, 0.7, 0.2], [0.9, 0.1, 0.6]]
        # - (rand > 0.5) gives [[False, True, False], [True, False, True]]
        # - .astype(float) gives [[0, 1, 0], [1, 0, 1]]
        # - About 50% are 0 (dropped), 50% are 1 (kept)
        
        # Apply mask and scale by 1/(1-p)
        # Element-wise multiplication drops neurons
        # Division scales remaining neurons to maintain expected value
        return (x * self.mask) / (1 - self.p)
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Backward pass for dropout.
        
        Gradient only flows through neurons that weren't dropped!
        - If neuron was kept (mask=1): gradient flows through
        - If neuron was dropped (mask=0): gradient is blocked
        
        We apply the SAME mask and scaling as forward pass.
        This is mathematically correct due to chain rule.
        
        Args:
            dout: Gradient from next layer, shape (batch_size, num_features)
            
        Returns:
            Gradient to propagate back, same shape as dout
            
        Example:
            Forward:  x=[1,2,3,4] → mask=[1,0,1,0] → out=[2,0,6,0]
            Backward: dout=[0.5,0.3,0.2,0.1] → mask=[1,0,1,0] → dx=[1,0,0.4,0]
        """
        if not self.training:
            # Test mode: no dropout was applied, gradient passes through unchanged
            return dout
        
        # Training mode: apply same mask and scaling
        # This implements: d(dropout(x))/dx = mask / (1-p)
        return (dout * self.mask) / (1 - self.p)
    
    def train(self):
        """
        Set layer to training mode.
        
        Call this before training loop:
        - Enables dropout (neurons randomly dropped)
        - Makes network stochastic (different output each forward pass)
        """
        self.training = True
    
    def eval(self):
        """
        Set layer to evaluation mode.
        
        Call this before testing/validation:
        - Disables dropout (all neurons active)
        - Makes network deterministic (same output for same input)
        """
        self.training = False


# ============================================================================
# BATCH NORMALIZATION LAYER
# ============================================================================

class BatchNormalization:
    """
    Batch Normalization layer.
    
    This layer normalizes activations to have:
    - Mean ≈ 0
    - Variance ≈ 1
    
    Then applies learnable scale (γ) and shift (β) parameters.
    
    Algorithm (per mini-batch):
    1. Compute batch mean: μ = (1/m) * sum(x)
    2. Compute batch variance: σ² = (1/m) * sum((x-μ)²)
    3. Normalize: x_hat = (x - μ) / sqrt(σ² + ε)
    4. Scale and shift: y = γ * x_hat + β
    
    Why it works:
    - Reduces internal covariate shift (stabilizes input distributions)
    - Acts as regularization (batch statistics add noise)
    - Allows higher learning rates (gradients more stable)
    - Reduces sensitivity to initialization
    
    Training vs Testing:
    - Training: Use batch statistics (computed from current batch)
    - Testing: Use running statistics (accumulated during training)
    
    Attributes:
        num_features: Number of features (input dimensions)
        gamma: Scale parameter, shape (1, num_features), learnable
        beta: Shift parameter, shape (1, num_features), learnable
        running_mean: Running average of batch means, for test time
        running_var: Running average of batch variances, for test time
        momentum: EMA momentum for running statistics (default 0.9)
        epsilon: Small constant for numerical stability (default 1e-5)
    """
    
    def __init__(
        self,
        num_features: int,
        momentum: float = 0.9,
        epsilon: float = 1e-5
    ):
        """
        Initialize batch normalization layer.
        
        Args:
            num_features: Number of input features (dimension to normalize)
            momentum: Momentum for running statistics (typically 0.9 or 0.99)
            epsilon: Small constant to prevent division by zero
        """
        self.num_features = num_features
        self.momentum = momentum
        self.epsilon = epsilon
        
        # Learnable parameters (initialized per paper recommendations)
        # gamma = 1: No scaling initially
        # beta = 0: No shift initially
        # Network can learn to adjust these if needed
        self.gamma = np.ones((1, num_features))  # Scale parameter
        self.beta = np.zeros((1, num_features))  # Shift parameter
        
        # Running statistics (for test time)
        # Updated during training with exponential moving average
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Training mode flag
        self.training = True
        
        # Cache for backward pass
        self.cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through batch normalization.
        
        Training mode:
        1. Compute batch mean and variance
        2. Normalize using batch statistics
        3. Update running statistics
        4. Apply scale and shift
        
        Test mode:
        1. Normalize using running statistics
        2. Apply scale and shift
        
        Args:
            x: Input activations, shape (batch_size, num_features)
            
        Returns:
            Normalized activations, same shape as input
            
        Example:
            >>> bn = BatchNormalization(num_features=3)
            >>> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            >>> out = bn.forward(x)
            >>> out.shape
            (3, 3)
        """
        if self.training:
            # ===== Training mode: use batch statistics =====
            
            # Step 1: Compute batch mean
            # Mean over batch dimension (axis=0)
            # Shape: (1, num_features) - one mean per feature
            batch_mean = np.mean(x, axis=0, keepdims=True)
            
            # Step 2: Compute batch variance
            # Variance = E[(x - μ)²]
            # keepdims keeps shape as (1, num_features)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            # Step 3: Normalize
            # x_hat = (x - μ) / sqrt(σ² + ε)
            # Epsilon prevents division by zero
            x_normalized = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Step 4: Update running statistics (exponential moving average)
            # running_mean = momentum * running_mean + (1-momentum) * batch_mean
            # This accumulates statistics across all training batches
            self.running_mean = (
                self.momentum * self.running_mean +
                (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var +
                (1 - self.momentum) * batch_var
            )
            
            # Store values for backward pass
            # We need these to compute gradients
            self.cache['x'] = x
            self.cache['x_normalized'] = x_normalized
            self.cache['batch_mean'] = batch_mean
            self.cache['batch_var'] = batch_var
            
        else:
            # ===== Test mode: use running statistics =====
            
            # Normalize using accumulated statistics from training
            # These are more stable than single batch statistics
            x_normalized = (
                (x - self.running_mean) /
                np.sqrt(self.running_var + self.epsilon)
            )
        
        # Step 5: Scale and shift (both training and test)
        # y = γ * x_hat + β
        # γ and β are learnable parameters
        # Network can "undo" normalization if needed
        out = self.gamma * x_normalized + self.beta
        
        return out
    
    def backward(self, dout: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass for batch normalization.
        
        This is mathematically complex! We need gradients for:
        1. Input (to propagate back): dL/dx
        2. Scale parameter: dL/dγ
        3. Shift parameter: dL/dβ
        
        The gradient computation involves chain rule through:
        - Scale/shift operation
        - Normalization operation
        - Mean and variance computation
        
        Simplified formulas (see paper for full derivation):
        - dL/dγ = sum(dout * x_normalized)
        - dL/dβ = sum(dout)
        - dL/dx = complex expression involving all of the above
        
        Args:
            dout: Gradient from next layer, shape (batch_size, num_features)
            learning_rate: Step size for parameter updates
            
        Returns:
            dx: Gradient to previous layer, shape (batch_size, num_features)
        """
        # Retrieve cached values from forward pass
        x = self.cache['x']
        x_normalized = self.cache['x_normalized']
        batch_mean = self.cache['batch_mean']
        batch_var = self.cache['batch_var']
        
        # Batch size
        m = x.shape[0]
        
        # ===== Compute parameter gradients =====
        
        # Gradient w.r.t. gamma (scale parameter)
        # dL/dγ = sum(dout * x_hat)
        # Sum over batch dimension (all samples contribute)
        dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
        
        # Gradient w.r.t. beta (shift parameter)
        # dL/dβ = sum(dout)
        # Sum over batch dimension
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        # ===== Compute input gradient (complex!) =====
        
        # This involves chain rule through normalization
        # Full derivation is complex, but here's the intuition:
        # - Changes in x affect: normalized value, mean, and variance
        # - All three affect the output
        # - We need to account for all paths (chain rule)
        
        # Step 1: Gradient w.r.t. normalized x
        # dL/d(x_hat) = dout * gamma
        dx_normalized = dout * self.gamma
        
        # Step 2: Gradient through normalization formula
        # This is the complex part!
        # x_normalized = (x - mean) / sqrt(var + eps)
        
        # Standard deviation
        std = np.sqrt(batch_var + self.epsilon)
        
        # Gradient formula (derived using chain rule):
        # dx = (1/(m*std)) * (m*dx_hat - sum(dx_hat) - x_hat*sum(dx_hat*x_hat))
        dx = (1.0 / m) * (1.0 / std) * (
            m * dx_normalized -
            np.sum(dx_normalized, axis=0, keepdims=True) -
            x_normalized * np.sum(dx_normalized * x_normalized, axis=0, keepdims=True)
        )
        
        # ===== Update parameters =====
        
        # Gradient descent on gamma and beta
        self.gamma -= learning_rate * dgamma
        self.beta -= learning_rate * dbeta
        
        return dx
    
    def train(self):
        """Set layer to training mode (use batch statistics)."""
        self.training = True
    
    def eval(self):
        """Set layer to evaluation mode (use running statistics)."""
        self.training = False


# ============================================================================
# REGULARIZED NEURAL NETWORK
# ============================================================================

class RegularizedNeuralNetwork:
    """
    Complete neural network with regularization techniques.
    
    This network combines:
    - Dense layers with L1/L2 regularization
    - Dropout layers
    - Batch Normalization layers
    - Early stopping logic
    
    Architecture (example):
        Input (784)
        ↓
        Dense(784 → 256) + L2
        ↓
        BatchNorm(256)
        ↓
        ReLU
        ↓
        Dropout(0.5)
        ↓
        Dense(256 → 128) + L2
        ↓
        BatchNorm(128)
        ↓
        ReLU
        ↓
        Dropout(0.5)
        ↓
        Dense(128 → 10) + L2
        ↓
        Softmax
    
    Attributes:
        layers: List of all layers in the network
        layer_configs: Configuration for each layer
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True
    ):
        """
        Initialize regularized neural network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            l1_lambda: L1 regularization strength (0 = no L1)
            l2_lambda: L2 regularization strength (0 = no L2)
            dropout_rate: Dropout rate (0 = no dropout)
            use_batch_norm: Whether to use batch normalization
            
        Example:
            >>> net = RegularizedNeuralNetwork(
            ...     layer_sizes=[784, 256, 128, 10],
            ...     l2_lambda=0.001,
            ...     dropout_rate=0.5,
            ...     use_batch_norm=True
            ... )
        """
        self.layer_sizes = layer_sizes
        self.layers = []
        self.layer_configs = []
        
        # Build network architecture
        for i in range(len(layer_sizes) - 1):
            is_output_layer = (i == len(layer_sizes) - 2)
            
            # Dense layer with regularization
            dense = DenseLayer(
                layer_sizes[i],
                layer_sizes[i + 1],
                l1_lambda=0.0 if is_output_layer else l1_lambda,
                l2_lambda=0.0 if is_output_layer else l2_lambda
            )
            self.layers.append(dense)
            self.layer_configs.append('dense')
            
            # Don't add batch norm or dropout after output layer
            if not is_output_layer:
                # Batch normalization (optional)
                if use_batch_norm:
                    bn = BatchNormalization(layer_sizes[i + 1])
                    self.layers.append(bn)
                    self.layer_configs.append('batch_norm')
                
                # ReLU activation
                self.layer_configs.append('relu')
                
                # Dropout (optional)
                if dropout_rate > 0:
                    dropout = Dropout(dropout_rate)
                    self.layers.append(dropout)
                    self.layer_configs.append('dropout')
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the entire network.
        
        Sequentially applies all layers:
        1. Dense layer
        2. Batch norm (if enabled)
        3. ReLU activation
        4. Dropout (if enabled)
        5. Repeat for each layer
        6. Final dense layer + softmax
        
        Args:
            x: Input data, shape (batch_size, input_size)
            
        Returns:
            Predictions (probabilities), shape (batch_size, num_classes)
        """
        layer_idx = 0
        
        for i, config in enumerate(self.layer_configs):
            if config == 'dense':
                # Pass through dense layer
                x = self.layers[layer_idx].forward(x)
                layer_idx += 1
                
            elif config == 'batch_norm':
                # Pass through batch normalization
                x = self.layers[layer_idx].forward(x)
                layer_idx += 1
                
            elif config == 'relu':
                # Apply ReLU activation (not a layer object)
                x = relu(x)
                
            elif config == 'dropout':
                # Pass through dropout layer
                x = self.layers[layer_idx].forward(x)
                layer_idx += 1
        
        # Apply softmax to final output
        x = softmax(x)
        
        return x
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray, learning_rate: float):
        """
        Backward pass through entire network.
        
        Computes gradients and updates all parameters.
        Handles the special case of softmax + cross-entropy.
        
        Args:
            y_pred: Predicted probabilities, shape (batch_size, num_classes)
            y_true: True labels (one-hot), shape (batch_size, num_classes)
            learning_rate: Step size for weight updates
        """
        # Initial gradient (softmax + cross-entropy combined)
        # Gradient = predicted - true (remarkably simple!)
        dout = y_pred - y_true
        
        # Backpropagate through layers in reverse order
        layer_idx = len(self.layers) - 1
        
        for config in reversed(self.layer_configs):
            if config == 'dense':
                # Backprop through dense layer
                dout = self.layers[layer_idx].backward(dout, learning_rate)
                layer_idx -= 1
                
            elif config == 'batch_norm':
                # Backprop through batch norm
                dout = self.layers[layer_idx].backward(dout, learning_rate)
                layer_idx -= 1
                
            elif config == 'relu':
                # Backprop through ReLU (just apply derivative)
                dout = dout * relu_derivative(dout)
                
            elif config == 'dropout':
                # Backprop through dropout
                dout = self.layers[layer_idx].backward(dout)
                layer_idx -= 1
    
    def compute_loss(
        self,
        x: np.ndarray,
        y: np.ndarray,
        include_regularization: bool = True
    ) -> float:
        """
        Compute total loss (data loss + regularization loss).
        
        Args:
            x: Input data
            y: True labels (one-hot encoded)
            include_regularization: Whether to include L1/L2 penalties
            
        Returns:
            Total loss (scalar)
        """
        # Forward pass to get predictions
        y_pred = self.forward(x)
        
        # Data loss (cross-entropy)
        data_loss = cross_entropy_loss(y_pred, y)
        
        if not include_regularization:
            return data_loss
        
        # Add regularization losses from all dense layers
        reg_loss = 0.0
        for layer in self.layers:
            if isinstance(layer, DenseLayer):
                reg_loss += layer.get_l1_loss()
                reg_loss += layer.get_l2_loss()
        
        return data_loss + reg_loss
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Make predictions (class labels).
        
        Args:
            x: Input data, shape (batch_size, input_size)
            
        Returns:
            Predicted class labels, shape (batch_size,)
        """
        # Forward pass to get probabilities
        probs = self.forward(x)
        
        # Return class with highest probability
        return np.argmax(probs, axis=1)
    
    def accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute classification accuracy.
        
        Args:
            x: Input data
            y: True labels (one-hot encoded)
            
        Returns:
            Accuracy (fraction of correct predictions)
        """
        # Get predictions
        preds = self.predict(x)
        
        # Convert one-hot to class labels
        true_labels = np.argmax(y, axis=1)
        
        # Compute accuracy
        return np.mean(preds == true_labels)
    
    def train_mode(self):
        """Set all layers to training mode."""
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()
    
    def eval_mode(self):
        """Set all layers to evaluation mode."""
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 10,
        verbose: bool = True
    ) -> Dict:
        """
        Train the network with early stopping.
        
        Args:
            X_train: Training data
            y_train: Training labels (one-hot)
            X_val: Validation data
            y_val: Validation labels (one-hot)
            epochs: Maximum number of epochs
            batch_size: Mini-batch size
            learning_rate: Step size for gradient descent
            early_stopping_patience: Stop after this many epochs without improvement
            verbose: Print progress
            
        Returns:
            Training history dictionary
        """
        n_samples = X_train.shape[0]
        n_batches = n_samples // batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        for epoch in range(epochs):
            # Set to training mode
            self.train_mode()
            
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]
            
            # Train on mini-batches
            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                
                X_batch = X_train_shuffled[start:end]
                y_batch = y_train_shuffled[start:end]
                
                # Forward and backward pass
                y_pred = self.forward(X_batch)
                self.backward(y_pred, y_batch, learning_rate)
            
            # Evaluate on training and validation sets
            self.eval_mode()
            
            train_loss = self.compute_loss(X_train, y_train)
            val_loss = self.compute_loss(X_val, y_val, include_regularization=False)
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best weights
                best_weights = copy.deepcopy([layer for layer in self.layers])
            else:
                patience_counter += 1
            
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                # Restore best weights
                self.layers = best_weights
                break
        
        return self.history


# ============================================================================
# DEMO: COMPARING REGULARIZATION TECHNIQUES
# ============================================================================

def demo_regularization():
    """
    Demonstrate different regularization techniques.
    
    Creates synthetic data with clear overfitting potential,
    then trains models with different regularization strategies.
    """
    print("=" * 70)
    print("REGULARIZATION TECHNIQUES DEMONSTRATION")
    print("=" * 70)
    
    # Create synthetic dataset (XOR-like problem)
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Generate data
    X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
    
    # One-hot encode labels
    y_one_hot = np.eye(2)[y]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Create validation set from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"\nDataset sizes:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Model 1: No regularization (baseline - will overfit!)
    print("\n" + "=" * 70)
    print("Model 1: NO REGULARIZATION (Baseline)")
    print("=" * 70)
    
    model_baseline = RegularizedNeuralNetwork(
        layer_sizes=[2, 64, 64, 2],
        l1_lambda=0.0,
        l2_lambda=0.0,
        dropout_rate=0.0,
        use_batch_norm=False
    )
    
    model_baseline.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=20,
        verbose=False
    )
    
    print(f"Final Results:")
    print(f"  Train Acc: {model_baseline.accuracy(X_train, y_train):.4f}")
    print(f"  Val Acc: {model_baseline.accuracy(X_val, y_val):.4f}")
    print(f"  Test Acc: {model_baseline.accuracy(X_test, y_test):.4f}")
    
    # Model 2: L2 regularization only
    print("\n" + "=" * 70)
    print("Model 2: L2 REGULARIZATION (Weight Decay)")
    print("=" * 70)
    
    model_l2 = RegularizedNeuralNetwork(
        layer_sizes=[2, 64, 64, 2],
        l1_lambda=0.0,
        l2_lambda=0.01,
        dropout_rate=0.0,
        use_batch_norm=False
    )
    
    model_l2.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=20,
        verbose=False
    )
    
    print(f"Final Results:")
    print(f"  Train Acc: {model_l2.accuracy(X_train, y_train):.4f}")
    print(f"  Val Acc: {model_l2.accuracy(X_val, y_val):.4f}")
    print(f"  Test Acc: {model_l2.accuracy(X_test, y_test):.4f}")
    
    # Model 3: Dropout only
    print("\n" + "=" * 70)
    print("Model 3: DROPOUT REGULARIZATION")
    print("=" * 70)
    
    model_dropout = RegularizedNeuralNetwork(
        layer_sizes=[2, 64, 64, 2],
        l1_lambda=0.0,
        l2_lambda=0.0,
        dropout_rate=0.5,
        use_batch_norm=False
    )
    
    model_dropout.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=20,
        verbose=False
    )
    
    print(f"Final Results:")
    print(f"  Train Acc: {model_dropout.accuracy(X_train, y_train):.4f}")
    print(f"  Val Acc: {model_dropout.accuracy(X_val, y_val):.4f}")
    print(f"  Test Acc: {model_dropout.accuracy(X_test, y_test):.4f}")
    
    # Model 4: Batch Normalization only
    print("\n" + "=" * 70)
    print("Model 4: BATCH NORMALIZATION")
    print("=" * 70)
    
    model_batchnorm = RegularizedNeuralNetwork(
        layer_sizes=[2, 64, 64, 2],
        l1_lambda=0.0,
        l2_lambda=0.0,
        dropout_rate=0.0,
        use_batch_norm=True
    )
    
    model_batchnorm.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=20,
        verbose=False
    )
    
    print(f"Final Results:")
    print(f"  Train Acc: {model_batchnorm.accuracy(X_train, y_train):.4f}")
    print(f"  Val Acc: {model_batchnorm.accuracy(X_val, y_val):.4f}")
    print(f"  Test Acc: {model_batchnorm.accuracy(X_test, y_test):.4f}")
    
    # Model 5: ALL regularization techniques combined
    print("\n" + "=" * 70)
    print("Model 5: ALL TECHNIQUES COMBINED")
    print("=" * 70)
    
    model_full = RegularizedNeuralNetwork(
        layer_sizes=[2, 64, 64, 2],
        l1_lambda=0.0,
        l2_lambda=0.01,
        dropout_rate=0.3,
        use_batch_norm=True
    )
    
    model_full.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=20,
        verbose=False
    )
    
    print(f"Final Results:")
    print(f"  Train Acc: {model_full.accuracy(X_train, y_train):.4f}")
    print(f"  Val Acc: {model_full.accuracy(X_val, y_val):.4f}")
    print(f"  Test Acc: {model_full.accuracy(X_test, y_test):.4f}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nKey Observations:")
    print("1. Baseline model likely overfits (high train, lower val/test)")
    print("2. L2 regularization reduces overfitting via weight decay")
    print("3. Dropout adds noise that helps generalization")
    print("4. Batch Normalization stabilizes training and acts as regularizer")
    print("5. Combining techniques often works best!")
    print("\nBest practice: Start with batch norm + L2, add dropout if needed")
    print("=" * 70)


if __name__ == "__main__":
    print(__doc__)
    demo_regularization()
