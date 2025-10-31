"""
CHAPTER 03: OPTIMIZATION ALGORITHMS FROM SCRATCH
=================================================

This file implements all major optimization algorithms used in deep learning:
- Vanilla Stochastic Gradient Descent (SGD)
- SGD with Momentum
- RMSprop (Root Mean Square Propagation)
- Adam (Adaptive Moment Estimation)
- Learning rate schedules

We'll train neural networks with each optimizer and compare their performance!

File structure:
---------------
Part 1: Imports and setup
Part 2: Base Optimizer class
Part 3: SGD (vanilla and with momentum)
Part 4: RMSprop
Part 5: Adam
Part 6: Learning rate schedules
Part 7: Neural network with optimizer support
Part 8: Training and comparison
Part 9: Main execution

Author: Deep Learning Master Course
Purpose: Understanding optimization algorithms from first principles
"""

# ============================================================================
# PART 1: IMPORTS AND SETUP
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Callable
import time
from collections import defaultdict
import copy

# Set random seed for reproducibility
np.random.seed(42)

print("=" * 70)
print("OPTIMIZATION ALGORITHMS FROM SCRATCH - Chapter 03")
print("=" * 70)
print("\nImplementing:")
print("✓ SGD (Stochastic Gradient Descent)")
print("✓ SGD with Momentum")
print("✓ RMSprop")
print("✓ Adam (Adaptive Moment Estimation)")
print("✓ Learning rate schedules")
print("=" * 70 + "\n")


# ============================================================================
# PART 2: BASE OPTIMIZER CLASS
# ============================================================================

class Optimizer:
    """
    Base class for all optimizers.
    
    Each optimizer must implement:
        - __init__: Initialize hyperparameters and state
        - update: Update parameters using gradients
        - get_config: Return optimizer configuration
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Initial learning rate
        """
        # Base learning rate
        # This can be modified by learning rate schedules
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        
        # Training step counter
        # Used for learning rate schedules and bias correction
        self.t = 0
    
    def update(self, param: np.ndarray, grad: np.ndarray, param_name: str = '') -> np.ndarray:
        """
        Update parameter using gradient.
        
        Args:
            param: Parameter array to update
            grad: Gradient of loss w.r.t. parameter
            param_name: Name/identifier for this parameter (for state tracking)
            
        Returns:
            Updated parameter array
        """
        raise NotImplementedError("Subclass must implement update()")
    
    def get_config(self) -> Dict:
        """Return optimizer configuration"""
        return {'learning_rate': self.learning_rate}
    
    def reset(self):
        """Reset optimizer state (for new training run)"""
        self.t = 0


# ============================================================================
# PART 3: STOCHASTIC GRADIENT DESCENT
# ============================================================================

class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer.
    
    Update rule:
        θ_{t+1} = θ_t - η * ∇L(θ_t)
    
    This is the simplest optimizer: just follow the negative gradient!
    """
    
    def __init__(self, learning_rate: float = 0.01):
        """
        Initialize SGD optimizer.
        
        Args:
            learning_rate: Step size for parameter updates
        """
        super().__init__(learning_rate)
        print(f"Initialized SGD optimizer with lr={learning_rate}")
    
    def update(self, param: np.ndarray, grad: np.ndarray, param_name: str = '') -> np.ndarray:
        """
        Update parameter using vanilla gradient descent.
        
        Formula:
            param_new = param_old - learning_rate * gradient
        
        Args:
            param: Current parameter values
            grad: Gradient of loss w.r.t. parameter
            param_name: Parameter identifier (unused in vanilla SGD)
            
        Returns:
            Updated parameter
        """
        # Increment step counter
        self.t += 1
        
        # Simple gradient descent update
        # This is the most basic optimizer!
        # Just move in the direction opposite to the gradient
        param_updated = param - self.learning_rate * grad
        
        return param_updated


class MomentumSGD(Optimizer):
    """
    SGD with Momentum optimizer.
    
    Update rule:
        v_t = β * v_{t-1} + (1-β) * ∇L(θ_t)
        θ_{t+1} = θ_t - η * v_t
    
    Momentum accumulates gradients over time, which:
        - Accelerates progress in consistent directions
        - Dampens oscillations in ravines
        - Helps escape local minima
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        """
        Initialize SGD with momentum.
        
        Args:
            learning_rate: Step size for updates
            momentum: Momentum coefficient β (typically 0.9)
                     - 0.0 = no momentum (vanilla SGD)
                     - 0.9 = standard momentum
                     - 0.99 = high momentum (slow to change direction)
        """
        super().__init__(learning_rate)
        
        # Momentum coefficient
        # This controls how much of the previous velocity to keep
        self.momentum = momentum
        
        # Velocity dictionary: stores v_t for each parameter
        # velocity = exponentially weighted average of gradients
        # This is our "memory" of past gradients
        self.velocity = {}
        
        print(f"Initialized Momentum SGD with lr={learning_rate}, momentum={momentum}")
    
    def update(self, param: np.ndarray, grad: np.ndarray, param_name: str = '') -> np.ndarray:
        """
        Update parameter using momentum.
        
        Formula:
            v_t = β * v_{t-1} + (1-β) * grad
            param_new = param_old - η * v_t
        
        The velocity v_t is an exponentially-weighted moving average:
            v_t = (1-β) * [grad_t + β*grad_{t-1} + β²*grad_{t-2} + ...]
        
        This means recent gradients have more influence than old ones.
        
        Args:
            param: Current parameter values
            grad: Gradient of loss w.r.t. parameter
            param_name: Parameter identifier (for velocity tracking)
            
        Returns:
            Updated parameter
        """
        # Increment step counter
        self.t += 1
        
        # Initialize velocity for this parameter if first time seeing it
        if param_name not in self.velocity:
            # Start with zero velocity (no initial motion)
            self.velocity[param_name] = np.zeros_like(param)
        
        # Update velocity: v_t = β * v_{t-1} + (1-β) * grad
        # 
        # Interpretation:
        #   - β * v_{t-1}: Keep some of the previous velocity (momentum effect)
        #   - (1-β) * grad: Add some of the current gradient (new information)
        #
        # Example with β=0.9:
        #   v_t = 0.9 * v_{t-1} + 0.1 * grad
        #   This means 90% old velocity + 10% new gradient
        self.velocity[param_name] = (
            self.momentum * self.velocity[param_name] + 
            (1 - self.momentum) * grad
        )
        
        # Update parameter using velocity
        # Note: We move by velocity, not raw gradient!
        param_updated = param - self.learning_rate * self.velocity[param_name]
        
        return param_updated
    
    def get_config(self) -> Dict:
        """Return optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'momentum': self.momentum
        }


# ============================================================================
# PART 4: RMSPROP
# ============================================================================

class RMSprop(Optimizer):
    """
    RMSprop (Root Mean Square Propagation) optimizer.
    
    Update rule:
        s_t = β * s_{t-1} + (1-β) * (∇L(θ_t))²
        θ_{t+1} = θ_t - (η / √(s_t + ε)) * ∇L(θ_t)
    
    RMSprop adapts the learning rate for each parameter:
        - Parameters with large gradients get smaller effective learning rates
        - Parameters with small gradients get larger effective learning rates
    
    This helps with:
        - Different scales of parameters
        - Ravines (steep in some directions, flat in others)
        - Preventing gradient explosion
    """
    
    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        """
        Initialize RMSprop optimizer.
        
        Args:
            learning_rate: Base learning rate η
            beta: Decay rate for squared gradient moving average (typically 0.9 or 0.99)
            epsilon: Small constant for numerical stability (prevents division by zero)
        """
        super().__init__(learning_rate)
        
        # Decay rate for squared gradients
        # Higher β = longer memory (smoother adaptation)
        self.beta = beta
        
        # Small constant to prevent division by zero
        # If s_t = 0, we'd divide by zero without this
        self.epsilon = epsilon
        
        # Squared gradient moving average: stores s_t for each parameter
        # This tracks the "typical magnitude" of gradients
        self.squared_grad = {}
        
        print(f"Initialized RMSprop with lr={learning_rate}, beta={beta}, epsilon={epsilon}")
    
    def update(self, param: np.ndarray, grad: np.ndarray, param_name: str = '') -> np.ndarray:
        """
        Update parameter using RMSprop.
        
        Formula:
            s_t = β * s_{t-1} + (1-β) * grad²
            param_new = param_old - (η / √(s_t + ε)) * grad
        
        The term √(s_t) is approximately the RMS (root mean square) of recent gradients.
        Dividing by this normalizes gradients to similar scales.
        
        Args:
            param: Current parameter values
            grad: Gradient of loss w.r.t. parameter
            param_name: Parameter identifier (for squared gradient tracking)
            
        Returns:
            Updated parameter
        """
        # Increment step counter
        self.t += 1
        
        # Initialize squared gradient accumulator for this parameter
        if param_name not in self.squared_grad:
            # Start with zeros (no history yet)
            self.squared_grad[param_name] = np.zeros_like(param)
        
        # Update squared gradient moving average: s_t = β * s_{t-1} + (1-β) * grad²
        # 
        # Note: grad² means element-wise squaring (not matrix multiplication!)
        # This tracks the magnitude of gradients over time
        #
        # Interpretation:
        #   - If gradients are consistently large: s_t becomes large
        #   - If gradients are consistently small: s_t becomes small
        #   - s_t adapts to the scale of each individual parameter
        self.squared_grad[param_name] = (
            self.beta * self.squared_grad[param_name] + 
            (1 - self.beta) * grad**2
        )
        
        # Compute adaptive learning rate for each parameter
        # 
        # Formula: η / √(s_t + ε)
        # 
        # Effect:
        #   - Large s_t (large typical gradients) → small effective LR → gentle updates
        #   - Small s_t (small typical gradients) → large effective LR → larger updates
        #
        # This automatically balances learning rates across parameters!
        adaptive_lr = self.learning_rate / (np.sqrt(self.squared_grad[param_name]) + self.epsilon)
        
        # Update parameter
        # Each element of param gets its own adaptive learning rate!
        param_updated = param - adaptive_lr * grad
        
        return param_updated
    
    def get_config(self) -> Dict:
        """Return optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'beta': self.beta,
            'epsilon': self.epsilon
        }


# ============================================================================
# PART 5: ADAM (ADAPTIVE MOMENT ESTIMATION)
# ============================================================================

class Adam(Optimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    
    Update rule:
        m_t = β₁ * m_{t-1} + (1-β₁) * ∇L(θ_t)           [momentum]
        v_t = β₂ * v_{t-1} + (1-β₂) * (∇L(θ_t))²        [RMSprop]
        m̂_t = m_t / (1 - β₁^t)                          [bias correction]
        v̂_t = v_t / (1 - β₂^t)                          [bias correction]
        θ_{t+1} = θ_t - (η / √(v̂_t + ε)) * m̂_t
    
    Adam = Momentum + RMSprop + Bias Correction
    
    This is the MOST POPULAR optimizer in deep learning!
    """
    
    def __init__(
        self, 
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8
    ):
        """
        Initialize Adam optimizer.
        
        Args:
            learning_rate: Learning rate η (default 0.001 from paper)
            beta1: Decay rate for first moment (momentum) (default 0.9)
            beta2: Decay rate for second moment (RMSprop) (default 0.999)
            epsilon: Small constant for numerical stability (default 1e-8)
        
        These default values work well 90% of the time!
        """
        super().__init__(learning_rate)
        
        # Momentum coefficient
        # Controls how much past gradients influence the update direction
        self.beta1 = beta1
        
        # RMSprop coefficient
        # Controls how much past squared gradients influence the adaptive LR
        self.beta2 = beta2
        
        # Numerical stability constant
        self.epsilon = epsilon
        
        # First moment (momentum): m_t
        # This is like velocity in momentum SGD
        # Tracks the exponentially-weighted mean of gradients
        self.m = {}
        
        # Second moment (RMSprop): v_t
        # This is like squared gradient in RMSprop
        # Tracks the exponentially-weighted mean of squared gradients
        self.v = {}
        
        print(f"Initialized Adam with lr={learning_rate}, beta1={beta1}, beta2={beta2}")
    
    def update(self, param: np.ndarray, grad: np.ndarray, param_name: str = '') -> np.ndarray:
        """
        Update parameter using Adam optimizer.
        
        Full algorithm:
            1. Compute biased first moment:  m_t = β₁*m_{t-1} + (1-β₁)*grad
            2. Compute biased second moment: v_t = β₂*v_{t-1} + (1-β₂)*grad²
            3. Correct first moment bias:    m̂_t = m_t / (1 - β₁^t)
            4. Correct second moment bias:   v̂_t = v_t / (1 - β₂^t)
            5. Update parameter:             param -= (η/√(v̂_t+ε)) * m̂_t
        
        Args:
            param: Current parameter values
            grad: Gradient of loss w.r.t. parameter
            param_name: Parameter identifier (for moment tracking)
            
        Returns:
            Updated parameter
        """
        # Increment step counter (used for bias correction)
        self.t += 1
        
        # Initialize moments for this parameter if first time
        if param_name not in self.m:
            self.m[param_name] = np.zeros_like(param)
            self.v[param_name] = np.zeros_like(param)
        
        # ===== Step 1: Update biased first moment (momentum) =====
        # m_t = β₁ * m_{t-1} + (1-β₁) * grad
        # 
        # This is exponentially-weighted moving average of gradients
        # Similar to momentum in Momentum SGD
        # Gives us the "direction" to move
        self.m[param_name] = (
            self.beta1 * self.m[param_name] + 
            (1 - self.beta1) * grad
        )
        
        # ===== Step 2: Update biased second moment (RMSprop) =====
        # v_t = β₂ * v_{t-1} + (1-β₂) * grad²
        # 
        # This is exponentially-weighted moving average of squared gradients
        # Similar to RMSprop
        # Tells us how much to adapt the learning rate
        self.v[param_name] = (
            self.beta2 * self.v[param_name] + 
            (1 - self.beta2) * grad**2
        )
        
        # ===== Step 3: Bias correction for first moment =====
        # m̂_t = m_t / (1 - β₁^t)
        # 
        # Why needed?
        #   - We initialize m_0 = 0
        #   - First update: m_1 = (1-β₁)*grad (much smaller than grad!)
        #   - This creates a bias toward zero in early iterations
        #   - Dividing by (1-β₁^t) corrects this bias
        # 
        # Example with β₁=0.9:
        #   t=1:  (1-0.9¹) = 0.1    → strong correction
        #   t=10: (1-0.9¹⁰) ≈ 0.65  → moderate correction
        #   t→∞:  (1-0.9^∞) = 1     → no correction needed
        m_hat = self.m[param_name] / (1 - self.beta1**self.t)
        
        # ===== Step 4: Bias correction for second moment =====
        # v̂_t = v_t / (1 - β₂^t)
        # 
        # Same reasoning as first moment
        # Even more important because β₂=0.999 is closer to 1
        v_hat = self.v[param_name] / (1 - self.beta2**self.t)
        
        # ===== Step 5: Update parameter =====
        # θ_{t+1} = θ_t - (η / √(v̂_t + ε)) * m̂_t
        # 
        # Breakdown:
        #   - m̂_t: Bias-corrected momentum (direction to move)
        #   - √(v̂_t): RMS of gradients (scale normalization)
        #   - η / √(v̂_t + ε): Adaptive learning rate per parameter
        #   - ε: Prevents division by zero
        # 
        # This combines:
        #   ✓ Momentum (acceleration in consistent directions)
        #   ✓ RMSprop (adaptive learning rate per parameter)
        #   ✓ Bias correction (accurate estimates from the start)
        param_updated = param - (self.learning_rate / (np.sqrt(v_hat) + self.epsilon)) * m_hat
        
        return param_updated
    
    def get_config(self) -> Dict:
        """Return optimizer configuration"""
        return {
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'epsilon': self.epsilon
        }


# ============================================================================
# PART 6: LEARNING RATE SCHEDULES
# ============================================================================

class LearningRateSchedule:
    """Base class for learning rate schedules"""
    
    def __call__(self, epoch: int) -> float:
        """Return learning rate for given epoch"""
        raise NotImplementedError


class StepDecay(LearningRateSchedule):
    """
    Step decay: reduce LR by factor every N epochs.
    
    lr = initial_lr * decay_rate^(floor(epoch / step_size))
    """
    
    def __init__(self, initial_lr: float, decay_rate: float = 0.5, step_size: int = 10):
        """
        Args:
            initial_lr: Starting learning rate
            decay_rate: Factor to multiply LR by (e.g., 0.5 = halve LR)
            step_size: Number of epochs between decays
        """
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.step_size = step_size
    
    def __call__(self, epoch: int) -> float:
        """Compute learning rate for current epoch"""
        # Number of decays that have occurred
        n_decays = epoch // self.step_size
        
        # Apply decay: lr = initial_lr * decay_rate^n_decays
        return self.initial_lr * (self.decay_rate ** n_decays)


class ExponentialDecay(LearningRateSchedule):
    """
    Exponential decay: smooth exponential decrease.
    
    lr = initial_lr * decay_rate^epoch
    """
    
    def __init__(self, initial_lr: float, decay_rate: float = 0.96):
        """
        Args:
            initial_lr: Starting learning rate
            decay_rate: Decay factor per epoch (0.95-0.99 typical)
        """
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def __call__(self, epoch: int) -> float:
        """Compute learning rate for current epoch"""
        return self.initial_lr * (self.decay_rate ** epoch)


class CosineAnnealing(LearningRateSchedule):
    """
    Cosine annealing: smooth decrease following cosine curve.
    
    lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + cos(π * epoch / total_epochs))
    
    Popular in computer vision (ResNet, etc.)
    """
    
    def __init__(self, initial_lr: float, min_lr: float, total_epochs: int):
        """
        Args:
            initial_lr: Starting (maximum) learning rate
            min_lr: Minimum learning rate to decay to
            total_epochs: Total number of training epochs
        """
        self.max_lr = initial_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
    
    def __call__(self, epoch: int) -> float:
        """Compute learning rate for current epoch"""
        # Cosine annealing formula
        # Starts at max_lr, smoothly decreases to min_lr
        progress = epoch / self.total_epochs
        lr = self.min_lr + (self.max_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        return lr


# ============================================================================
# PART 7: NEURAL NETWORK WITH OPTIMIZER SUPPORT
# ============================================================================

class DenseLayer:
    """Dense layer compatible with all optimizers"""
    
    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'relu'):
        """Initialize layer with He initialization"""
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation_name = activation
        
        # He initialization for weights
        self.W = np.random.randn(n_outputs, n_inputs) * np.sqrt(2.0 / n_inputs)
        self.b = np.zeros((n_outputs, 1))
        
        # Cache for backprop
        self.cache = {}
    
    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """Forward pass"""
        self.cache['A_prev'] = A_prev
        Z = self.W @ A_prev + self.b
        self.cache['Z'] = Z
        
        # Apply activation
        if self.activation_name == 'relu':
            A = np.maximum(0, Z)
        elif self.activation_name == 'sigmoid':
            A = 1.0 / (1.0 + np.exp(-np.clip(Z, -500, 500)))
        elif self.activation_name == 'softmax':
            Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
            exp_Z = np.exp(Z_shifted)
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        else:
            A = Z
        
        self.cache['A'] = A
        return A
    
    def backward(self, dA: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass - returns gradients but does NOT update parameters.
        
        Returns:
            dW, db, dA_prev (gradients)
        """
        A_prev = self.cache['A_prev']
        Z = self.cache['Z']
        m = A_prev.shape[1]
        
        # Compute dZ based on activation
        if self.activation_name == 'relu':
            dZ = dA * (Z > 0)
        elif self.activation_name == 'sigmoid' or self.activation_name == 'softmax':
            dZ = dA  # Assumes loss already computed gradient
        else:
            dZ = dA
        
        # Compute gradients
        dW = (1.0 / m) * (dZ @ A_prev.T)
        db = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = self.W.T @ dZ
        
        return dW, db, dA_prev


class NeuralNetworkOptimizer:
    """Neural network that can use any optimizer"""
    
    def __init__(self, optimizer: Optimizer):
        """
        Initialize network with optimizer.
        
        Args:
            optimizer: Optimizer instance (SGD, Momentum, RMSprop, Adam)
        """
        self.layers = []
        self.optimizer = optimizer
        self.training_history = defaultdict(list)
    
    def add_layer(self, n_outputs: int, activation: str = 'relu', n_inputs: Optional[int] = None):
        """Add a layer to the network"""
        if len(self.layers) == 0 and n_inputs is None:
            raise ValueError("First layer needs n_inputs")
        
        input_size = n_inputs if n_inputs else self.layers[-1].n_outputs
        layer = DenseLayer(input_size, n_outputs, activation)
        self.layers.append(layer)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through all layers"""
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A
    
    def backward(self, Y_pred: np.ndarray, Y_true: np.ndarray):
        """
        Backward pass and parameter update using optimizer.
        
        This is where the optimizer gets used!
        """
        # Compute initial gradient (softmax + cross-entropy)
        dA = Y_pred - Y_true
        
        # Backpropagate through layers
        for idx, layer in enumerate(reversed(self.layers)):
            # Compute gradients for this layer
            dW, db, dA_prev = layer.backward(dA)
            
            # Update parameters using optimizer
            # Each optimizer implements its own update logic
            layer_name = f"layer_{len(self.layers) - idx - 1}"
            layer.W = self.optimizer.update(layer.W, dW, f"{layer_name}_W")
            layer.b = self.optimizer.update(layer.b, db, f"{layer_name}_b")
            
            # Propagate gradient to previous layer
            dA = dA_prev
    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        y_train_labels: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        y_val_labels: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr_schedule: Optional[LearningRateSchedule] = None,
        verbose: bool = True
    ):
        """Train network with optimizer"""
        n_samples = X_train.shape[1]
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            # Update learning rate if schedule provided
            if lr_schedule is not None:
                self.optimizer.learning_rate = lr_schedule(epoch)
            
            epoch_loss = 0.0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            Y_shuffled = Y_train[:, indices]
            
            # Mini-batch training
            for batch_idx in range(n_batches):
                start = batch_idx * batch_size
                end = start + batch_size
                X_batch = X_shuffled[:, start:end]
                Y_batch = Y_shuffled[:, start:end]
                
                # Forward
                Y_pred = self.forward(X_batch)
                
                # Loss
                Y_pred_clipped = np.clip(Y_pred, 1e-15, 1 - 1e-15)
                batch_loss = -np.sum(Y_batch * np.log(Y_pred_clipped)) / batch_size
                epoch_loss += batch_loss
                
                # Backward (this uses the optimizer!)
                self.backward(Y_pred, Y_batch)
            
            # Track metrics
            avg_loss = epoch_loss / n_batches
            train_acc = self.calculate_accuracy(X_train, y_train_labels)
            val_acc = self.calculate_accuracy(X_val, y_val_labels)
            
            self.training_history['loss'].append(avg_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['lr'].append(self.optimizer.learning_rate)
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1:2d}: Loss={avg_loss:.4f}, "
                      f"Train={train_acc:.2%}, Val={val_acc:.2%}, "
                      f"LR={self.optimizer.learning_rate:.6f}")
    
    def calculate_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate accuracy"""
        Y_pred = self.forward(X)
        predictions = np.argmax(Y_pred, axis=0)
        return np.mean(predictions == y)


# ============================================================================
# PART 8: TRAINING AND COMPARISON
# ============================================================================

def load_mnist():
    """Load MNIST dataset"""
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data
        y = mnist.target.astype(int)
        X_train, y_train = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
    
    # Preprocess
    X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
    X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
    Y_train = np.eye(10)[y_train].T
    Y_test = np.eye(10)[y_test].T
    
    return X_train, Y_train, y_train, X_test, Y_test, y_test


def compare_optimizers():
    """Compare all optimizers on MNIST"""
    print("\n" + "=" * 70)
    print("COMPARING OPTIMIZERS ON MNIST")
    print("=" * 70 + "\n")
    
    # Load data
    X_train, Y_train, y_train, X_test, Y_test, y_test = load_mnist()
    print(f"Loaded MNIST: {X_train.shape[1]} train, {X_test.shape[1]} test\n")
    
    # Use smaller subset for faster comparison
    X_train = X_train[:, :10000]
    Y_train = Y_train[:, :10000]
    y_train = y_train[:10000]
    
    # Define optimizers to compare
    optimizers = [
        ('SGD', SGD(learning_rate=0.1)),
        ('Momentum', MomentumSGD(learning_rate=0.1, momentum=0.9)),
        ('RMSprop', RMSprop(learning_rate=0.001)),
        ('Adam', Adam(learning_rate=0.001)),
    ]
    
    results = {}
    
    # Train with each optimizer
    for name, optimizer in optimizers:
        print(f"\nTraining with {name}...")
        print("-" * 70)
        
        # Create network
        nn = NeuralNetworkOptimizer(optimizer)
        nn.add_layer(128, 'relu', n_inputs=784)
        nn.add_layer(64, 'relu')
        nn.add_layer(10, 'softmax')
        
        # Train
        nn.train(
            X_train, Y_train, y_train,
            X_test, Y_test, y_test,
            epochs=20,
            batch_size=128,
            verbose=True
        )
        
        results[name] = nn.training_history
        print()
    
    # Plot comparison
    plot_optimizer_comparison(results)
    
    return results


def plot_optimizer_comparison(results: Dict):
    """Plot comparison of all optimizers"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    ax = axes[0, 0]
    for name, history in results.items():
        ax.plot(history['loss'], label=name, linewidth=2)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Train accuracy
    ax = axes[0, 1]
    for name, history in results.items():
        ax.plot(history['train_acc'], label=name, linewidth=2)
    ax.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Val accuracy
    ax = axes[1, 0]
    for name, history in results.items():
        ax.plot(history['val_acc'], label=name, linewidth=2)
    ax.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning rate
    ax = axes[1, 1]
    for name, history in results.items():
        ax.plot(history['lr'], label=name, linewidth=2)
    ax.set_title('Learning Rate', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimizer_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PART 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("OPTIMIZATION ALGORITHMS - COMPLETE IMPLEMENTATION")
    print("=" * 70)
    
    # Compare all optimizers
    results = compare_optimizers()
    
    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    for name, history in results.items():
        final_val_acc = history['val_acc'][-1]
        print(f"{name:12s}: Val Accuracy = {final_val_acc:.2%}")
    
    print("\n" + "=" * 70)
    print("COMPLETE! You've implemented all major optimizers!")
    print("=" * 70)
    print("\nKey Takeaways:")
    print("✓ Adam converges fastest (typically best choice)")
    print("✓ Momentum helps vanilla SGD significantly")
    print("✓ RMSprop adapts learning rates per parameter")
    print("✓ All optimizers benefit from good learning rate tuning")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
