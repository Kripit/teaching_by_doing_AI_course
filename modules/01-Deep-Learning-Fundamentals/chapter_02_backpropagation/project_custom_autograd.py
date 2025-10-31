"""
PROJECT: BUILD YOUR OWN AUTOMATIC DIFFERENTIATION ENGINE
=========================================================

This project implements a micrograd-style autograd system from scratch.
You'll build an automatic differentiation engine that can:
- Track computational graphs
- Compute gradients automatically using backpropagation
- Train neural networks without writing manual backward passes

This is how PyTorch and TensorFlow work under the hood!

What you'll learn:
------------------
1. How automatic differentiation works
2. Computational graph representation
3. Topological sorting for backward pass
4. Building complex operations from primitives
5. Training neural networks with your autograd engine

File structure:
---------------
Part 1: Value class (core autograd engine)
Part 2: Basic operations (+, -, *, /, **)
Part 3: Activation functions (tanh, relu, sigmoid)
Part 4: Neural network building blocks
Part 5: Training on toy datasets
Part 6: Comparison with NumPy implementation

Author: Deep Learning Master Course
Purpose: Understanding automatic differentiation from first principles
"""

# ============================================================================
# PART 1: VALUE CLASS - THE CORE AUTOGRAD ENGINE
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Tuple, List, Set, Callable
import math

class Value:
    """
    A Value object represents a scalar value in a computational graph.
    
    It stores:
        - data: The actual numerical value (forward pass)
        - grad: The gradient (backward pass)
        - _backward: Function to compute local gradients
        - _prev: Parent nodes in the computational graph
        - _op: Operation that created this value
    
    This is the CORE of automatic differentiation!
    """
    
    def __init__(self, data: float, _children: Tuple = (), _op: str = ''):
        """
        Initialize a Value node in the computational graph.
        
        Args:
            data: The numerical value (scalar)
            _children: Tuple of parent Value objects
            _op: String describing the operation that created this node
        
        Example:
            a = Value(2.0)           # Leaf node
            b = Value(3.0)           # Leaf node  
            c = a * b                # c = Value(6.0, _children=(a, b), _op='*')
        """
        # The actual numerical value
        # This is what gets computed in the forward pass
        self.data = data
        
        # The gradient of the loss with respect to this value
        # Initially 0, gets filled in during backward pass
        # This represents: dL/d(this_value)
        self.grad = 0.0
        
        # Function to compute local gradients and propagate to parents
        # This is the KEY to automatic differentiation
        # Each operation defines its own _backward function
        self._backward = lambda: None
        
        # Set of parent nodes (inputs to the operation that created this node)
        # This builds the computational graph
        # Example: if c = a + b, then c._prev = {a, b}
        self._prev = set(_children)
        
        # String describing the operation (for visualization/debugging)
        # Examples: '+', '*', 'tanh', 'relu'
        self._op = _op
    
    def __repr__(self) -> str:
        """String representation for debugging"""
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
    
    # ========================================================================
    # PART 2: ARITHMETIC OPERATIONS
    # ========================================================================
    
    def __add__(self, other: Union['Value', float]) -> 'Value':
        """
        Addition: self + other
        
        Forward pass:
            out = self + other
        
        Backward pass (chain rule):
            dL/d(self) = dL/d(out) * d(out)/d(self) = dL/d(out) * 1
            dL/d(other) = dL/d(out) * d(out)/d(other) = dL/d(out) * 1
        
        The derivative of addition is 1 for both inputs!
        """
        # Convert other to Value if it's a float
        # This allows: Value(2.0) + 3.0
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass: compute the sum
        out = Value(self.data + other.data, (self, other), '+')
        
        # Define backward pass
        # This function will be called during backpropagation
        def _backward():
            # Chain rule: gradient flows equally to both inputs
            # d(a + b)/da = 1, so grad_a = grad_out * 1
            # d(a + b)/db = 1, so grad_b = grad_out * 1
            #
            # += because a value might be used multiple times
            # (gradient accumulation)
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        
        out._backward = _backward
        
        return out
    
    def __mul__(self, other: Union['Value', float]) -> 'Value':
        """
        Multiplication: self * other
        
        Forward pass:
            out = self * other
        
        Backward pass (chain rule):
            dL/d(self) = dL/d(out) * d(out)/d(self) = dL/d(out) * other
            dL/d(other) = dL/d(out) * d(out)/d(other) = dL/d(out) * self
        
        Derivative of multiplication: d(a*b)/da = b, d(a*b)/db = a
        """
        other = other if isinstance(other, Value) else Value(other)
        
        # Forward pass
        out = Value(self.data * other.data, (self, other), '*')
        
        # Define backward pass
        def _backward():
            # d(a * b)/da = b
            self.grad += other.data * out.grad
            # d(a * b)/db = a
            other.grad += self.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def __pow__(self, other: Union[int, float]) -> 'Value':
        """
        Power: self ** other
        
        Forward pass:
            out = self ** other
        
        Backward pass:
            d(x^n)/dx = n * x^(n-1)
            dL/d(self) = dL/d(out) * other * self^(other-1)
        """
        # Only support scalar exponents (not Value exponents)
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        
        # Forward pass
        out = Value(self.data ** other, (self,), f'**{other}')
        
        # Define backward pass
        def _backward():
            # Power rule: d(x^n)/dx = n * x^(n-1)
            self.grad += other * (self.data ** (other - 1)) * out.grad
        
        out._backward = _backward
        
        return out
    
    def __neg__(self) -> 'Value':
        """Negation: -self"""
        # -x = -1 * x
        return self * -1
    
    def __sub__(self, other: Union['Value', float]) -> 'Value':
        """Subtraction: self - other"""
        # a - b = a + (-b)
        return self + (-other)
    
    def __truediv__(self, other: Union['Value', float]) -> 'Value':
        """Division: self / other"""
        # a / b = a * b^(-1)
        return self * (other ** -1)
    
    # Reverse operations (for when Value is on the right side)
    def __radd__(self, other: Union[float, int]) -> 'Value':
        """Reverse addition: other + self"""
        return self + other
    
    def __rmul__(self, other: Union[float, int]) -> 'Value':
        """Reverse multiplication: other * self"""
        return self * other
    
    def __rsub__(self, other: Union[float, int]) -> 'Value':
        """Reverse subtraction: other - self"""
        return Value(other) - self
    
    def __rtruediv__(self, other: Union[float, int]) -> 'Value':
        """Reverse division: other / self"""
        return Value(other) / self
    
    # ========================================================================
    # PART 3: ACTIVATION FUNCTIONS
    # ========================================================================
    
    def tanh(self) -> 'Value':
        """
        Hyperbolic tangent activation.
        
        Forward:
            tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
        
        Backward:
            d(tanh(x))/dx = 1 - tanh^2(x)
        """
        # Forward pass
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)  # More numerically stable
        out = Value(t, (self,), 'tanh')
        
        # Define backward pass
        def _backward():
            # Derivative: 1 - tanh^2(x)
            self.grad += (1 - t**2) * out.grad
        
        out._backward = _backward
        
        return out
    
    def relu(self) -> 'Value':
        """
        ReLU activation: f(x) = max(0, x)
        
        Forward:
            relu(x) = max(0, x)
        
        Backward:
            d(relu(x))/dx = 1 if x > 0, else 0
        """
        # Forward pass
        out = Value(max(0, self.data), (self,), 'relu')
        
        # Define backward pass
        def _backward():
            # Gradient flows through only if input was positive
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward
        
        return out
    
    def sigmoid(self) -> 'Value':
        """
        Sigmoid activation: f(x) = 1 / (1 + e^(-x))
        
        Forward:
            sigmoid(x) = 1 / (1 + e^(-x))
        
        Backward:
            d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        """
        # Forward pass
        sig = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(sig, (self,), 'sigmoid')
        
        # Define backward pass
        def _backward():
            # Derivative: sigmoid * (1 - sigmoid)
            self.grad += sig * (1 - sig) * out.grad
        
        out._backward = _backward
        
        return out
    
    def exp(self) -> 'Value':
        """
        Exponential: e^x
        
        Backward:
            d(e^x)/dx = e^x
        """
        # Forward pass
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        
        # Define backward pass
        def _backward():
            # Derivative of e^x is e^x
            self.grad += out.data * out.grad
        
        out._backward = _backward
        
        return out
    
    def log(self) -> 'Value':
        """
        Natural logarithm: ln(x)
        
        Backward:
            d(ln(x))/dx = 1/x
        """
        # Forward pass
        out = Value(math.log(self.data), (self,), 'log')
        
        # Define backward pass
        def _backward():
            # Derivative: 1/x
            self.grad += (1.0 / self.data) * out.grad
        
        out._backward = _backward
        
        return out
    
    # ========================================================================
    # PART 4: BACKPROPAGATION
    # ========================================================================
    
    def backward(self):
        """
        Perform backpropagation to compute gradients.
        
        Algorithm:
            1. Build topological order of computational graph
            2. Initialize output gradient to 1.0 (dL/dL = 1)
            3. Visit nodes in reverse topological order
            4. Call each node's _backward() function
        
        This is AUTOMATIC differentiation!
        """
        # Step 1: Build topological order using DFS
        # We need to visit nodes in reverse order of computation
        topo = []
        visited = set()
        
        def build_topo(v):
            """
            Depth-first search to build topological order.
            
            Why topological sort?
            - We need to compute gradients in reverse order
            - A node's gradient depends on all its children's gradients
            - Topological order ensures children are processed first
            """
            if v not in visited:
                visited.add(v)
                # Recursively visit parents first
                for child in v._prev:
                    build_topo(child)
                # Add this node after all parents are added
                topo.append(v)
        
        build_topo(self)
        
        # Step 2: Initialize gradient of output to 1
        # This represents: dL/dL = 1 (loss with respect to itself)
        self.grad = 1.0
        
        # Step 3: Backpropagate through computational graph
        # Traverse in REVERSE topological order
        for node in reversed(topo):
            # Call the _backward function defined for this operation
            # This computes local gradients and propagates to parents
            node._backward()
    
    def zero_grad(self):
        """
        Reset gradients to zero.
        
        This must be called before each backward pass!
        Otherwise gradients accumulate (which is sometimes useful).
        """
        self.grad = 0.0


# ============================================================================
# PART 5: NEURAL NETWORK BUILDING BLOCKS
# ============================================================================

class Neuron:
    """
    A single neuron with multiple inputs.
    
    Computation:
        y = activation(w1*x1 + w2*x2 + ... + wn*xn + b)
    
    Where:
        - w_i are weights (parameters to be learned)
        - x_i are inputs
        - b is bias
        - activation is a non-linear function (tanh, relu, etc.)
    """
    
    def __init__(self, n_inputs: int, activation: str = 'tanh'):
        """
        Initialize neuron with random weights and bias.
        
        Args:
            n_inputs: Number of input connections
            activation: Activation function ('tanh', 'relu', 'sigmoid', 'linear')
        """
        # Initialize weights randomly in range [-1, 1]
        # Each weight is a Value object (so gradients can be computed)
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        
        # Initialize bias to 0
        # Bias is also a Value object
        self.bias = Value(0.0)
        
        # Store activation function
        self.activation = activation
    
    def __call__(self, x: List[Value]) -> Value:
        """
        Forward pass through neuron.
        
        Args:
            x: List of input Values
            
        Returns:
            Output Value after activation
        """
        # Compute weighted sum: w1*x1 + w2*x2 + ... + wn*xn + b
        # This builds a computational graph automatically!
        act = sum((wi * xi for wi, xi in zip(self.weights, x)), self.bias)
        
        # Apply activation function
        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        else:  # linear
            return act
    
    def parameters(self) -> List[Value]:
        """Return all parameters (weights + bias)"""
        return self.weights + [self.bias]


class Layer:
    """
    A layer of neurons.
    
    A layer with n_inputs and n_outputs has:
        - n_outputs neurons
        - Each neuron has n_inputs connections
        - Total parameters: n_outputs * (n_inputs + 1)
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, activation: str = 'tanh'):
        """
        Initialize layer with multiple neurons.
        
        Args:
            n_inputs: Number of inputs to each neuron
            n_outputs: Number of neurons in this layer
            activation: Activation function for all neurons
        """
        # Create n_outputs neurons, each with n_inputs connections
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_outputs)]
    
    def __call__(self, x: List[Value]) -> List[Value]:
        """
        Forward pass through layer.
        
        Args:
            x: List of input Values
            
        Returns:
            List of output Values (one per neuron)
        """
        # Each neuron processes the same input
        # Outputs from all neurons form the layer's output
        outs = [neuron(x) for neuron in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self) -> List[Value]:
        """Return all parameters from all neurons"""
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """
    Multi-Layer Perceptron (feedforward neural network).
    
    Architecture:
        Input → Layer1 → Layer2 → ... → LayerN → Output
    """
    
    def __init__(self, n_inputs: int, layer_sizes: List[int], activations: List[str] = None):
        """
        Initialize MLP with multiple layers.
        
        Args:
            n_inputs: Number of input features
            layer_sizes: List of neurons in each layer
            activations: List of activation functions (one per layer)
        
        Example:
            # 2 inputs, hidden layer with 4 neurons, output layer with 1 neuron
            mlp = MLP(2, [4, 1], ['tanh', 'linear'])
        """
        # Default activations: tanh for hidden layers, linear for output
        if activations is None:
            activations = ['tanh'] * (len(layer_sizes) - 1) + ['linear']
        
        # Build layers
        # Layer i connects outputs from layer i-1 to layer i
        sz = [n_inputs] + layer_sizes
        self.layers = [Layer(sz[i], sz[i+1], activations[i]) 
                      for i in range(len(layer_sizes))]
    
    def __call__(self, x: List[Value]) -> Union[Value, List[Value]]:
        """
        Forward pass through entire network.
        
        Args:
            x: List of input Values
            
        Returns:
            Output Value(s) from final layer
        """
        # Propagate through each layer sequentially
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self) -> List[Value]:
        """Return all parameters from all layers"""
        return [p for layer in self.layers for p in layer.parameters()]
    
    def zero_grad(self):
        """Reset all parameter gradients to zero"""
        for p in self.parameters():
            p.grad = 0.0


# ============================================================================
# PART 6: TRAINING ON TOY DATASETS
# ============================================================================

def generate_circle_dataset(n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a toy dataset: points inside/outside a circle.
    
    - Class 0: Points inside circle (radius < 0.5)
    - Class 1: Points outside circle (radius > 0.5)
    
    This is a non-linearly separable problem!
    """
    X = np.random.randn(n_samples, 2)
    y = (np.linalg.norm(X, axis=1) > 0.5).astype(int)
    return X, y


def train_mlp_on_circles():
    """Train MLP on circle dataset using custom autograd engine"""
    print("\n" + "=" * 70)
    print("TRAINING MLP WITH CUSTOM AUTOGRAD ENGINE")
    print("=" * 70)
    print("\nDataset: Circle classification (non-linearly separable)")
    print("Architecture: 2 → 16 → 16 → 1")
    print("=" * 70 + "\n")
    
    # Generate dataset
    X_train, y_train = generate_circle_dataset(100)
    print(f"Generated {len(X_train)} training samples")
    
    # Build MLP: 2 inputs → 16 hidden → 16 hidden → 1 output
    model = MLP(2, [16, 16, 1], ['tanh', 'tanh', 'sigmoid'])
    print(f"Model has {len(model.parameters())} parameters\n")
    
    # Training loop
    learning_rate = 0.1
    epochs = 100
    
    print(f"{'Epoch':<8} {'Loss':<12} {'Accuracy':<12}")
    print("-" * 35)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Forward pass for all samples
        total_loss = Value(0.0)
        correct = 0
        
        for i in range(len(X_train)):
            # Convert input to Value objects
            x = [Value(X_train[i, 0]), Value(X_train[i, 1])]
            y_true = y_train[i]
            
            # Forward pass
            y_pred = model(x)
            
            # Binary cross-entropy loss
            # Loss = -[y*log(p) + (1-y)*log(1-p)]
            if y_true == 1:
                loss = -y_pred.log()
            else:
                loss = -(Value(1.0) - y_pred).log()
            
            total_loss = total_loss + loss
            
            # Accuracy
            pred_class = 1 if y_pred.data > 0.5 else 0
            if pred_class == y_true:
                correct += 1
        
        # Average loss
        avg_loss = total_loss * (1.0 / len(X_train))
        accuracy = correct / len(X_train)
        
        losses.append(avg_loss.data)
        accuracies.append(accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"{epoch+1:<8} {avg_loss.data:<12.4f} {accuracy:<12.2%}")
        
        # Backward pass
        model.zero_grad()
        avg_loss.backward()
        
        # Update parameters
        for p in model.parameters():
            p.data -= learning_rate * p.grad
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final Loss: {losses[-1]:.4f}")
    print(f"Final Accuracy: {accuracies[-1]:.2%}")
    print("=" * 70 + "\n")
    
    # Visualize results
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Dataset
    ax1.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1], 
               c='blue', label='Class 0', alpha=0.6)
    ax1.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1], 
               c='red', label='Class 1', alpha=0.6)
    ax1.set_title('Training Data', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss curve
    ax2.plot(losses, 'b-', linewidth=2)
    ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax3.plot(accuracies, 'g-', linewidth=2)
    ax3.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Decision boundary
    plot_decision_boundary(model, X_train, y_train)


def plot_decision_boundary(model: MLP, X: np.ndarray, y: np.ndarray):
    """Visualize decision boundary learned by model"""
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in mesh
    Z = []
    for i in range(len(xx.ravel())):
        x_val = [Value(xx.ravel()[i]), Value(yy.ravel()[i])]
        pred = model(x_val)
        Z.append(pred.data)
    Z = np.array(Z).reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
    plt.colorbar(label='Prediction')
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', edgecolors='k', label='Class 0', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', edgecolors='k', label='Class 1', s=50)
    plt.title('Decision Boundary (Custom Autograd)', fontsize=16, fontweight='bold')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================================================================
# PART 7: COMPARISON WITH NUMPY IMPLEMENTATION
# ============================================================================

def compare_with_numpy():
    """
    Compare autograd engine with manual NumPy backprop.
    
    This demonstrates that our autograd engine produces the same results!
    """
    print("\n" + "=" * 70)
    print("COMPARING AUTOGRAD WITH MANUAL NUMPY BACKPROP")
    print("=" * 70 + "\n")
    
    # Simple computation: y = (x1*w1 + x2*w2)**2
    x1_val, x2_val = 2.0, 3.0
    w1_val, w2_val = 0.5, -0.3
    
    # ==== Method 1: Our autograd engine ====
    print("Method 1: Custom Autograd Engine")
    x1 = Value(x1_val)
    x2 = Value(x2_val)
    w1 = Value(w1_val)
    w2 = Value(w2_val)
    
    # Forward
    a = x1 * w1
    b = x2 * w2
    c = a + b
    y = c ** 2
    
    print(f"  Forward: y = {y.data:.4f}")
    
    # Backward
    y.backward()
    print(f"  Gradients: dw1={w1.grad:.4f}, dw2={w2.grad:.4f}")
    
    # ==== Method 2: Manual NumPy backprop ====
    print("\nMethod 2: Manual NumPy Backpropagation")
    
    # Forward
    a_np = x1_val * w1_val
    b_np = x2_val * w2_val
    c_np = a_np + b_np
    y_np = c_np ** 2
    
    print(f"  Forward: y = {y_np:.4f}")
    
    # Backward (manual chain rule)
    dy_dc = 2 * c_np          # d(c^2)/dc = 2c
    dc_da = 1.0               # d(a+b)/da = 1
    dc_db = 1.0               # d(a+b)/db = 1
    da_dw1 = x1_val           # d(x1*w1)/dw1 = x1
    db_dw2 = x2_val           # d(x2*w2)/dw2 = x2
    
    dw1_np = dy_dc * dc_da * da_dw1
    dw2_np = dy_dc * dc_db * db_dw2
    
    print(f"  Gradients: dw1={dw1_np:.4f}, dw2={dw2_np:.4f}")
    
    # ==== Comparison ====
    print("\n" + "-" * 70)
    print("Results Match:", 
          f"dw1 diff={abs(w1.grad - dw1_np):.2e}, "
          f"dw2 diff={abs(w2.grad - dw2_np):.2e}")
    print("=" * 70 + "\n")


# ============================================================================
# PART 8: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution"""
    print("\n" + "=" * 70)
    print("PROJECT: CUSTOM AUTOMATIC DIFFERENTIATION ENGINE")
    print("=" * 70)
    print("\nYou've built a micrograd-style autograd system!")
    print("\nCapabilities:")
    print("  ✓ Automatic gradient computation")
    print("  ✓ Computational graph tracking")
    print("  ✓ Support for arithmetic operations")
    print("  ✓ Activation functions (tanh, relu, sigmoid)")
    print("  ✓ Neural network building blocks")
    print("  ✓ Training on real datasets")
    print("=" * 70)
    
    # Demo 1: Simple gradient computation
    print("\n" + "=" * 70)
    print("DEMO 1: SIMPLE GRADIENT COMPUTATION")
    print("=" * 70)
    a = Value(2.0)
    b = Value(3.0)
    c = a * b + b**2
    c.backward()
    print(f"\na = {a.data}, b = {b.data}")
    print(f"c = a * b + b^2 = {c.data}")
    print(f"dc/da = {a.grad:.4f} (expected: b = {b.data:.4f})")
    print(f"dc/db = {b.grad:.4f} (expected: a + 2*b = {a.data + 2*b.data:.4f})")
    
    # Demo 2: Neuron
    print("\n" + "=" * 70)
    print("DEMO 2: SINGLE NEURON")
    print("=" * 70)
    neuron = Neuron(2, activation='tanh')
    x = [Value(1.0), Value(-2.0)]
    y = neuron(x)
    print(f"\nInput: x = [1.0, -2.0]")
    print(f"Output: y = {y.data:.4f}")
    y.backward()
    print(f"Weight gradients: {[w.grad for w in neuron.weights]}")
    
    # Demo 3: Comparison with NumPy
    compare_with_numpy()
    
    # Demo 4: Train on circle dataset
    train_mlp_on_circles()
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nWhat you've learned:")
    print("  ✓ How automatic differentiation works")
    print("  ✓ Computational graph representation")
    print("  ✓ Implementing backpropagation automatically")
    print("  ✓ Building neural networks from scratch")
    print("  ✓ Training with custom autograd engine")
    print("\nThis is the SAME principle used by PyTorch and TensorFlow!")
    print("You now understand how modern deep learning frameworks work!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
