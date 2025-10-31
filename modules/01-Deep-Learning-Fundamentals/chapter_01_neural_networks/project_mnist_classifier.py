"""
PROJECT: MNIST HANDWRITTEN DIGIT CLASSIFIER
============================================

In this project, you'll build a complete neural network to classify handwritten digits.

Goal: Achieve 95%+ accuracy on MNIST test set

What you'll implement:
----------------------
1. Load and preprocess MNIST dataset
2. Build a multi-layer neural network
3. Implement forward propagation
4. Train the network (simplified - full backprop in Chapter 02)
5. Evaluate performance
6. Visualize results
7. Analyze errors

This project reinforces everything from Chapter 01:
- Neural network architecture design
- Activation functions
- Forward propagation
- Loss computation
- Data preprocessing
- Model evaluation

Note: We use simplified "gradient updates" here. Chapter 02 implements proper backpropagation.

Author: Deep Learning Master Course
Level: Beginner
Estimated Time: 2-3 hours
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List
import time

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PROJECT: MNIST HANDWRITTEN DIGIT CLASSIFIER")
print("=" * 70)
print("\nObjective: Build a neural network to recognize handwritten digits")
print("Target Accuracy: 95%+ on test set")
print("=" * 70 + "\n")


# ============================================================================
# PART 1: LOAD AND EXPLORE DATA
# ============================================================================

def load_mnist():
    """
    Load MNIST dataset and perform initial exploration.
    
    Returns:
        X_train: Training images (784, 60000)
        y_train: Training labels (60000,)
        X_test: Test images (784, 10000)
        y_test: Test labels (10000,)
    """
    print("STEP 1: Loading MNIST Dataset")
    print("-" * 70)
    
    try:
        # Try loading from Keras (most common)
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("✓ Loaded from TensorFlow/Keras")
    except:
        # Fallback: scikit-learn
        from sklearn.datasets import fetch_openml
        print("  TensorFlow not available, loading from scikit-learn...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data
        y = mnist.target.astype(int)
        X_train, y_train = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]
        X_train = X_train.reshape(-1, 28, 28)
        X_test = X_test.reshape(-1, 28, 28)
        print("✓ Loaded from scikit-learn")
    
    # Print dataset information
    print(f"\nDataset Information:")
    print(f"  Training samples: {X_train.shape[0]:,}")
    print(f"  Test samples: {X_test.shape[0]:,}")
    print(f"  Image shape: {X_train.shape[1]} × {X_train.shape[2]}")
    print(f"  Pixel value range: [{X_train.min()}, {X_train.max()}]")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    print(f"  Classes: {np.unique(y_train)}")
    
    # Show class distribution
    print(f"\nClass Distribution (Training Set):")
    for digit in range(10):
        count = np.sum(y_train == digit)
        percentage = (count / len(y_train)) * 100
        print(f"  Digit {digit}: {count:,} samples ({percentage:.1f}%)")
    
    return X_train, y_train, X_test, y_test


def visualize_samples(X, y, n_samples=20):
    """
    Visualize random samples from dataset.
    
    Args:
        X: Images (n_samples, 28, 28)
        y: Labels (n_samples,)
        n_samples: Number of samples to display
    """
    print(f"\nVisualizing {n_samples} random samples...")
    
    # Select random indices
    indices = np.random.choice(len(X), n_samples, replace=False)
    
    # Create figure
    fig, axes = plt.subplots(4, 5, figsize=(12, 10))
    fig.suptitle('MNIST Dataset Samples', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        ax.imshow(X[idx], cmap='gray')
        ax.set_title(f'Label: {y[idx]}', fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 2: DATA PREPROCESSING
# ============================================================================

def preprocess_data(X_train, y_train, X_test, y_test):
    """
    Preprocess MNIST data for neural network.
    
    Steps:
    1. Flatten images: (28, 28) → (784,)
    2. Normalize pixels: [0, 255] → [0, 1]
    3. Transpose: (n_samples, 784) → (784, n_samples)
    4. One-hot encode labels (for categorical cross-entropy)
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Preprocessed data ready for neural network
    """
    print("\nSTEP 2: Preprocessing Data")
    print("-" * 70)
    
    # Step 1: Flatten images
    print("  Flattening images (28×28 → 784)...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Step 2: Normalize to [0, 1]
    print("  Normalizing pixel values ([0,255] → [0,1])...")
    X_train_norm = X_train_flat.astype('float32') / 255.0
    X_test_norm = X_test_flat.astype('float32') / 255.0
    
    # Step 3: Transpose for network input
    print("  Transposing (samples, features) → (features, samples)...")
    X_train_T = X_train_norm.T  # (784, 60000)
    X_test_T = X_test_norm.T    # (784, 10000)
    
    # Step 4: One-hot encode labels
    print("  One-hot encoding labels...")
    y_train_onehot = one_hot_encode(y_train, n_classes=10)
    y_test_onehot = one_hot_encode(y_test, n_classes=10)
    
    print(f"\nPreprocessed Shapes:")
    print(f"  X_train: {X_train_T.shape}  (features × samples)")
    print(f"  y_train: {y_train_onehot.shape}  (samples × classes)")
    print(f"  X_test: {X_test_T.shape}")
    print(f"  y_test: {y_test_onehot.shape}")
    
    return X_train_T, y_train_onehot, X_test_T, y_test_onehot


def one_hot_encode(y, n_classes=10):
    """
    Convert class labels to one-hot encoding.
    
    Example:
        y = [3, 0, 1]
        one_hot = [[0,0,0,1,0,0,0,0,0,0],
                   [1,0,0,0,0,0,0,0,0,0],
                   [0,1,0,0,0,0,0,0,0,0]]
    
    Args:
        y: Class labels (n_samples,)
        n_classes: Number of classes
        
    Returns:
        One-hot encoded labels (n_samples, n_classes)
    """
    n_samples = len(y)
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), y] = 1
    return one_hot


# ============================================================================
# PART 3: NEURAL NETWORK (FROM CHAPTER 01)
# ============================================================================

# Copy activation functions from chapter_01
def sigmoid(z):
    """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    """ReLU activation: max(0, z)"""
    return np.maximum(0, z)

def softmax(z):
    """Softmax activation for multi-class classification"""
    if z.ndim == 1:
        z_exp = np.exp(z - np.max(z))
        return z_exp / np.sum(z_exp)
    else:
        z_exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return z_exp / np.sum(z_exp, axis=1, keepdims=True)


class DenseLayer:
    """Fully connected layer with forward propagation"""
    
    def __init__(self, n_inputs, n_outputs, activation='relu'):
        # Xavier initialization
        limit = np.sqrt(2.0 / (n_inputs + n_outputs))
        self.weights = np.random.randn(n_outputs, n_inputs) * limit
        self.biases = np.zeros((n_outputs, 1))
        
        # Activation functions
        self.activation_name = activation
        if activation == 'relu':
            self.activation = relu
        elif activation == 'sigmoid':
            self.activation = sigmoid
        elif activation == 'softmax':
            self.activation = softmax
        else:
            self.activation = lambda x: x  # Linear
        
        # Cache for backprop
        self.input_cache = None
        self.z_cache = None
    
    def forward(self, X):
        """Forward pass: A = activation(W @ X + b)"""
        self.input_cache = X
        self.z_cache = self.weights @ X + self.biases
        return self.activation(self.z_cache)


class NeuralNetwork:
    """Multi-layer neural network"""
    
    def __init__(self):
        self.layers = []
        self.loss_history = []
        self.accuracy_history = []
    
    def add_layer(self, n_outputs, activation='relu', n_inputs=None):
        """Add a dense layer to the network"""
        if len(self.layers) == 0 and n_inputs is None:
            raise ValueError("First layer must specify n_inputs")
        
        input_size = n_inputs if n_inputs else self.layers[-1].weights.shape[0]
        layer = DenseLayer(input_size, n_outputs, activation)
        self.layers.append(layer)
    
    def forward(self, X):
        """Forward propagation through all layers"""
        activation = X
        for layer in self.layers:
            activation = layer.forward(activation)
        return activation
    
    def predict(self, X):
        """Make predictions (class labels)"""
        output = self.forward(X)
        return np.argmax(output, axis=0)
    
    def calculate_accuracy(self, X, y_true):
        """Calculate classification accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y_true)


# ============================================================================
# PART 4: TRAINING (SIMPLIFIED - FULL BACKPROP IN CHAPTER 02)
# ============================================================================

def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical cross-entropy loss.
    
    Args:
        y_true: One-hot encoded labels (n_samples, n_classes)
        y_pred: Predicted probabilities (n_classes, n_samples)
        
    Returns:
        Average loss (scalar)
    """
    epsilon = 1e-15
    y_pred_T = y_pred.T  # Transpose to (n_samples, n_classes)
    y_pred_clipped = np.clip(y_pred_T, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_true.shape[0]
    return loss


def train_simple(network, X_train, y_train, X_val, y_val_labels, epochs=10, batch_size=128, learning_rate=0.01):
    """
    Simplified training loop (without proper backpropagation).
    
    Note: This is a placeholder! Chapter 02 implements proper gradient descent.
    Here we use random weight perturbations to demonstrate the training process.
    
    Args:
        network: Neural network to train
        X_train: Training images
        y_train: Training labels (one-hot)
        X_val: Validation images
        y_val_labels: Validation labels (class indices)
        epochs: Number of training epochs
        batch_size: Mini-batch size
        learning_rate: Learning rate for updates
    """
    print("\nSTEP 3: Training Network (Simplified)")
    print("-" * 70)
    print("NOTE: This uses simplified training. Chapter 02 implements proper backpropagation!")
    print()
    
    n_samples = X_train.shape[1]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        
        # Shuffle data
        indices = np.random.permutation(n_samples)
        X_shuffled = X_train[:, indices]
        y_shuffled = y_train[indices]
        
        # Mini-batch training
        for batch_idx in range(n_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            X_batch = X_shuffled[:, start_idx:end_idx]
            y_batch = y_shuffled[start_idx:end_idx]
            
            # Forward pass
            predictions = network.forward(X_batch)
            
            # Compute loss
            batch_loss = categorical_cross_entropy(y_batch, predictions)
            epoch_loss += batch_loss
            
            # Simplified weight updates (placeholder for Chapter 02)
            # In reality, we'd compute gradients via backpropagation
            # For now, add small random perturbations to demonstrate improvement
            for layer in network.layers:
                layer.weights -= learning_rate * 0.001 * np.random.randn(*layer.weights.shape)
                layer.biases -= learning_rate * 0.001 * np.random.randn(*layer.biases.shape)
        
        # Average loss for epoch
        avg_loss = epoch_loss / n_batches
        network.loss_history.append(avg_loss)
        
        # Validation accuracy
        val_accuracy = network.calculate_accuracy(X_val, y_val_labels)
        network.accuracy_history.append(val_accuracy)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Loss: {avg_loss:.4f} - "
              f"Val Acc: {val_accuracy:.2%} - "
              f"Time: {epoch_time:.1f}s")
    
    print("\nTraining Complete!")


# ============================================================================
# PART 5: EVALUATION AND VISUALIZATION
# ============================================================================

def evaluate_model(network, X_test, y_test, y_test_labels):
    """
    Comprehensive model evaluation.
    
    Args:
        network: Trained network
        X_test: Test images
        y_test: Test labels (one-hot)
        y_test_labels: Test labels (class indices)
    """
    print("\nSTEP 4: Model Evaluation")
    print("-" * 70)
    
    # Overall accuracy
    test_accuracy = network.calculate_accuracy(X_test, y_test_labels)
    print(f"\nTest Set Accuracy: {test_accuracy:.2%}")
    
    # Loss
    predictions = network.forward(X_test)
    test_loss = categorical_cross_entropy(y_test, predictions)
    print(f"Test Set Loss: {test_loss:.4f}")
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    y_pred = network.predict(X_test)
    for digit in range(10):
        mask = (y_test_labels == digit)
        if np.sum(mask) > 0:
            class_accuracy = np.mean(y_pred[mask] == y_test_labels[mask])
            print(f"  Digit {digit}: {class_accuracy:.2%} ({np.sum(mask)} samples)")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    confusion = compute_confusion_matrix(y_test_labels, y_pred, n_classes=10)
    print_confusion_matrix(confusion)
    
    return test_accuracy, y_pred


def compute_confusion_matrix(y_true, y_pred, n_classes=10):
    """
    Compute confusion matrix.
    
    confusion[i, j] = number of samples with true label i predicted as j
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        n_classes: Number of classes
        
    Returns:
        Confusion matrix (n_classes, n_classes)
    """
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    for i in range(len(y_true)):
        confusion[y_true[i], y_pred[i]] += 1
    return confusion


def print_confusion_matrix(confusion):
    """Print confusion matrix in readable format"""
    print("\n       Predicted →")
    print("     ", end="")
    for i in range(10):
        print(f"{i:5}", end="")
    print()
    print("True")
    print(" ↓")
    for i in range(10):
        print(f" {i}  ", end="")
        for j in range(10):
            print(f"{confusion[i, j]:5}", end="")
        print()


def plot_training_history(network):
    """Plot loss and accuracy curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(network.loss_history, linewidth=2, color='red')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(network.accuracy_history, linewidth=2, color='green')
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def visualize_predictions(X_test, y_test_labels, y_pred, n_samples=20):
    """Visualize predictions with true labels"""
    indices = np.random.choice(X_test.shape[1], n_samples, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        idx = indices[i]
        image = X_test[:, idx].reshape(28, 28)
        true_label = y_test_labels[idx]
        pred_label = y_pred[idx]
        
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        
        if true_label == pred_label:
            color = 'green'
            title = f'✓ Predicted: {pred_label}'
        else:
            color = 'red'
            title = f'✗ Pred: {pred_label} (True: {true_label})'
        
        ax.set_title(title, color=color, fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def analyze_errors(X_test, y_test_labels, y_pred):
    """Analyze misclassified examples"""
    print("\nError Analysis")
    print("-" * 70)
    
    # Find misclassified samples
    errors = (y_pred != y_test_labels)
    error_indices = np.where(errors)[0]
    n_errors = len(error_indices)
    
    print(f"Total errors: {n_errors} / {len(y_test_labels)} ({n_errors/len(y_test_labels)*100:.2f}%)")
    
    if n_errors == 0:
        print("Perfect classification! No errors to analyze.")
        return
    
    # Most common misclassifications
    print(f"\nMost Common Misclassifications:")
    misclass_pairs = {}
    for idx in error_indices:
        true_label = y_test_labels[idx]
        pred_label = y_pred[idx]
        pair = (true_label, pred_label)
        misclass_pairs[pair] = misclass_pairs.get(pair, 0) + 1
    
    sorted_pairs = sorted(misclass_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, ((true_label, pred_label), count) in enumerate(sorted_pairs[:10]):
        print(f"  {i+1}. {true_label} → {pred_label}: {count} times")
    
    # Visualize errors
    print(f"\nVisualizing misclassified examples...")
    n_show = min(20, n_errors)
    selected_errors = np.random.choice(error_indices, n_show, replace=False)
    
    fig, axes = plt.subplots(4, 5, figsize=(14, 11))
    fig.suptitle('Misclassified Examples', fontsize=16, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis('off')
            continue
        
        idx = selected_errors[i]
        image = X_test[:, idx].reshape(28, 28)
        true_label = y_test_labels[idx]
        pred_label = y_pred[idx]
        
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(f'True: {true_label}, Predicted: {pred_label}', 
                     color='red', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# PART 6: MAIN EXECUTION
# ============================================================================

def main():
    """Main project execution"""
    print("\n" + "="*70)
    print("STARTING PROJECT: MNIST DIGIT CLASSIFIER")
    print("="*70 + "\n")
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist()
    
    # Visualize samples
    visualize_samples(X_train, y_train, n_samples=20)
    
    # Preprocess
    X_train_prep, y_train_onehot, X_test_prep, y_test_onehot = preprocess_data(
        X_train, y_train, X_test, y_test
    )
    
    # Build network
    print("\nBuilding Neural Network Architecture")
    print("-" * 70)
    network = NeuralNetwork()
    network.add_layer(128, activation='relu', n_inputs=784)
    network.add_layer(64, activation='relu')
    network.add_layer(10, activation='softmax')
    print("Architecture: 784 → 128 (ReLU) → 64 (ReLU) → 10 (Softmax)")
    
    # Calculate parameters
    total_params = sum(layer.weights.size + layer.biases.size for layer in network.layers)
    print(f"Total parameters: {total_params:,}")
    
    # Train network
    train_simple(
        network, 
        X_train_prep, 
        y_train_onehot, 
        X_test_prep, 
        y_test, 
        epochs=5,  # Small number for demo
        batch_size=128,
        learning_rate=0.01
    )
    
    # Plot training history
    plot_training_history(network)
    
    # Evaluate
    test_accuracy, y_pred = evaluate_model(network, X_test_prep, y_test_onehot, y_test)
    
    # Visualize predictions
    visualize_predictions(X_test_prep, y_test, y_pred, n_samples=20)
    
    # Analyze errors
    analyze_errors(X_test_prep, y_test, y_pred)
    
    # Final summary
    print("\n" + "="*70)
    print("PROJECT COMPLETE!")
    print("="*70)
    print(f"\nFinal Test Accuracy: {test_accuracy:.2%}")
    print("\nKey Takeaways:")
    print("1. Built a neural network from scratch using NumPy")
    print("2. Implemented forward propagation through multiple layers")
    print("3. Trained on MNIST dataset (60,000 training images)")
    print("4. Achieved classification on 10 digit classes")
    print("\nNote: This used simplified training. Chapter 02 implements proper")
    print("      backpropagation and gradient descent for much better results!")
    print("\nNext Steps:")
    print("- Experiment with different architectures (more/fewer layers)")
    print("- Try different activation functions")
    print("- Move to Chapter 02 for proper training with backpropagation")
    print("- Aim for 98%+ accuracy with full implementation")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
