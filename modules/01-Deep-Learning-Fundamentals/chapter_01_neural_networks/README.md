# Chapter 01: Neural Networks from Scratch
## Building Your First Neural Network Using Only NumPy

<div align="center">

**The Foundation of All Deep Learning**

*Understanding what happens inside a neural network*

</div>

---

## 🎯 Chapter Objectives

By the end of this chapter, you will:

- ✅ Understand the **mathematical foundations** of neural networks
- ✅ Know what happens in **every single line** of a neural network forward pass
- ✅ Implement a **complete neural network** using only NumPy (no deep learning libraries)
- ✅ Train your network on **real data** (MNIST handwritten digits)
- ✅ Achieve **95%+ accuracy** on a real-world classification problem
- ✅ Develop **intuition** for how neural networks learn

---

## 📖 Table of Contents

1. [Biological Inspiration](#1-biological-inspiration)
2. [The Perceptron: Simplest Neural Network](#2-the-perceptron)
3. [Activation Functions](#3-activation-functions)
4. [Multi-Layer Perceptron (MLP)](#4-multi-layer-perceptron)
5. [Forward Propagation](#5-forward-propagation)
6. [Loss Functions](#6-loss-functions)
7. [Why We Need Multiple Layers](#7-why-we-need-multiple-layers)
8. [Implementation Strategy](#8-implementation-strategy)
9. [MNIST Dataset](#9-mnist-dataset)
10. [Training Loop Structure](#10-training-loop-structure)

---

## 1. Biological Inspiration

### 1.1 The Human Brain

The human brain contains approximately **86 billion neurons**, each connected to thousands of other neurons through structures called **synapses**. This is where artificial neural networks get their inspiration (though they work very differently).

#### How Biological Neurons Work:

```
Inputs (Dendrites) → Cell Body (Processing) → Output (Axon) → Synapses → Next Neurons
```

**Key Properties**:
- **Inputs**: Each neuron receives signals from multiple other neurons
- **Summation**: The cell body sums up all incoming signals
- **Activation**: If the sum exceeds a threshold, the neuron "fires" (sends a signal)
- **Output**: The signal travels down the axon to other neurons
- **Learning**: Synaptic weights change based on experience (Hebbian learning)

### 1.2 Artificial Neurons

Artificial neurons are **mathematical approximations** of biological neurons:

```
Inputs (x₁, x₂, ..., xₙ) → Weighted Sum + Bias → Activation Function → Output (y)
```

**Mathematical Formula**:

$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(w^T x + b)
$$

**Let's break down EVERY symbol**:

- **$y$** = Output of the neuron (the prediction or signal it sends)
- **$f$** = Activation function (decides IF and HOW MUCH the neuron fires)
- **$\sum_{i=1}^{n}$** = Summation symbol (add up all inputs from i=1 to i=n)
- **$n$** = Total number of input features
- **$w_i$** = Weight for input i (how important is this input?)
  - If $w_i$ is large and positive → this input strongly increases output
  - If $w_i$ is large and negative → this input strongly decreases output
  - If $w_i$ is near zero → this input is ignored
- **$x_i$** = Value of input feature i (the actual data)
- **$b$** = Bias term (shifts the activation threshold)
  - Positive bias → neuron fires more easily
  - Negative bias → neuron fires less easily
- **$w^T x$** = Compact notation for $\sum_{i=1}^{n} w_i x_i$ (dot product)
  - $w^T$ means transpose of weight vector: $[w_1, w_2, ..., w_n]$
  - $x$ is input vector: $[x_1, x_2, ..., x_n]^T$
  - Dot product: $w^T x = w_1 x_1 + w_2 x_2 + ... + w_n x_n$

**Example**: Predicting if a student will pass an exam

```python
# Inputs
hours_studied = 5  # x₁
previous_score = 85  # x₂
sleep_hours = 7  # x₃

# Learned weights (importance of each factor)
w₁ = 0.6  # hours_studied is important
w₂ = 0.3  # previous_score is somewhat important
w₃ = 0.1  # sleep_hours is slightly important

# Bias (base prediction)
b = -2.0

# Weighted sum
z = w₁ * hours_studied + w₂ * previous_score + w₃ * sleep_hours + b
z = 0.6 * 5 + 0.3 * 85 + 0.1 * 7 + (-2.0)
z = 3 + 25.5 + 0.7 - 2.0 = 27.2

# Activation (sigmoid function for probability)
y = 1 / (1 + e^(-z)) ≈ 0.9999... ≈ 1.0  # Very likely to pass!
```

---

## 2. The Perceptron

### 2.1 What is a Perceptron?

The **perceptron** is the simplest neural network, invented by Frank Rosenblatt in 1958. It's a **single artificial neuron** that can learn to classify data into two categories.

**Architecture**:
```
     x₁ ──w₁──╲
     x₂ ──w₂──╱ ⊕ ─→ f(z) ─→ y
     x₃ ──w₃──╲    ↑
              ╱    bias (b)
```

**Components**:
1. **Inputs** ($x_1, x_2, ..., x_n$): Features of your data
2. **Weights** ($w_1, w_2, ..., w_n$): Learned parameters
3. **Bias** ($b$): Shifts the decision boundary
4. **Activation** ($f$): Converts weighted sum to output

### 2.2 Perceptron Mathematics

**Step 1: Weighted Sum**
$$
z = \sum_{i=1}^{n} w_i x_i + b = w^T x + b
$$

**Step 2: Activation (Step Function)**
$$
y = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

**Example**: AND gate

```python
# Truth table for AND gate
# x₁  x₂  | y
#  0   0  | 0
#  0   1  | 0
#  1   0  | 0
#  1   1  | 1

# Perceptron weights for AND
w₁ = 1.0
w₂ = 1.0
b = -1.5

# Test (x₁=1, x₂=1)
z = 1.0 * 1 + 1.0 * 1 + (-1.5) = 0.5
y = 1 (since z ≥ 0)  ✓ Correct!

# Test (x₁=0, x₂=1)
z = 1.0 * 0 + 1.0 * 1 + (-1.5) = -0.5
y = 0 (since z < 0)  ✓ Correct!
```

### 2.3 Perceptron Limitations

**Problem**: Perceptrons can only learn **linearly separable** functions.

**What does "linearly separable" mean?**
- You can draw a **straight line** (or hyperplane in higher dimensions) to separate the two classes

**Examples**:
- ✅ AND gate: Linearly separable
- ✅ OR gate: Linearly separable
- ❌ XOR gate: **NOT** linearly separable

**XOR Problem** (why single-layer perceptrons fail):
```
Truth table:
x₁  x₂  | y
 0   0  | 0
 0   1  | 1
 1   0  | 1
 1   1  | 0

No single straight line can separate this!
```

**Solution**: Multi-layer neural networks (we'll cover this in section 7)

---

## 3. Activation Functions

### 3.1 Why We Need Activation Functions

**Without activation functions**, neural networks would just be:
$$
y = W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W_{combined} x + b_{combined}
$$

This is just a **linear transformation**! No matter how many layers, it's equivalent to a single-layer linear model.

**Activation functions introduce non-linearity**, allowing networks to learn complex patterns.

---

### 3.2 Common Activation Functions

#### **3.2.1 Sigmoid (Logistic)**

**Formula**:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Breaking down the formula**:
- **$\sigma$** = Greek letter "sigma", name for sigmoid function
- **$z$** = Input to the function (the weighted sum: $w^T x + b$)
- **$e$** = Euler's number ≈ 2.71828 (mathematical constant)
- **$e^{-z}$** = Exponential of negative z
  - When $z$ is large positive (e.g., 10): $e^{-10} \approx 0$ → $\sigma(10) \approx 1$
  - When $z$ is large negative (e.g., -10): $e^{10} \approx 22026$ → $\sigma(-10) \approx 0$
  - When $z = 0$: $e^0 = 1$ → $\sigma(0) = 1/(1+1) = 0.5$
- **$1 + e^{-z}$** = Denominator that normalizes the output
- **Entire fraction** = Always between 0 and 1 (perfect for probabilities!)

**Output Range**: $(0, 1)$ - never exactly 0 or 1, but gets very close

**Why this formula?**
- It's an S-shaped curve (sigmoid = S-shaped)
- Smoothly transitions from 0 to 1
- Has nice mathematical properties (easy to differentiate)

**Concrete Examples**:
- $\sigma(0) = \frac{1}{1 + e^{0}} = \frac{1}{1 + 1} = 0.5$
- $\sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135} = 0.88$
- $\sigma(-2) = \frac{1}{1 + e^{2}} = \frac{1}{1 + 7.39} = 0.12$
- $\sigma(10) = \frac{1}{1 + e^{-10}} \approx 0.9999$ (almost 1)

**Properties**:
- Smooth and differentiable everywhere
- Outputs can be interpreted as **probabilities** (0-1 range)
- Squashes large values to 0 or 1 (handles any input range)

**Graph**:
```
 1.0 |           ╱────  (approaches 1 as z→∞)
     |         ╱
 0.5 |       ╱  (exactly 0.5 at z=0)
     |     ╱
 0.0 |────╱  (approaches 0 as z→-∞)
     |_____|_____|_____|
    -6    0     6
```

**Derivative** (needed for backpropagation):
$$
\frac{d\sigma}{dz} = \sigma(z) \cdot (1 - \sigma(z))
$$

**Understanding the derivative**:
- **$\frac{d\sigma}{dz}$** = "How much does sigmoid change when z changes slightly?"
- **$\sigma(z)$** = The sigmoid value itself
- **$(1 - \sigma(z))$** = One minus the sigmoid value
- **Beautiful property**: Derivative only needs the sigmoid output!

**Why this derivative makes sense**:
- At $z = 0$: $\sigma(0) = 0.5$ → derivative = $0.5 \times 0.5 = 0.25$ (max slope)
- At $z = 5$: $\sigma(5) \approx 0.99$ → derivative = $0.99 \times 0.01 = 0.0099$ (flat)
- At $z = -5$: $\sigma(-5) \approx 0.01$ → derivative = $0.01 \times 0.99 = 0.0099$ (flat)
- The function changes fastest at z=0 and barely changes at extremes!

**Problem**: Vanishing gradients
- When $z$ is very large or very small, derivative ≈ 0
- Gradient "vanishes" during backpropagation
- Network learns very slowly or stops learning

**When to Use**:
- Output layer for **binary classification** (probability of class 1)
- When you need outputs between 0 and 1
- **Avoid in hidden layers** (use ReLU instead due to vanishing gradients)

**Downsides**:
- **Vanishing gradient problem**: For large |z|, gradient becomes very small
- **Not zero-centered**: All outputs are positive
- **Slow computation**: Exponential function

---

#### **3.2.2 Tanh (Hyperbolic Tangent)**

**Formula**:
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = \frac{2}{1 + e^{-2z}} - 1
$$

**Breaking down the formula**:
- **$\tanh$** = "hyperbolic tangent" (pronounced "tansh" or "tan-h")
- **$e^z$** = Exponential of z (e.g., if z=2, this is 7.39)
- **$e^{-z}$** = Exponential of negative z (e.g., if z=2, this is 0.135)
- **Numerator**: $e^z - e^{-z}$
  - When z is positive: $e^z$ is large, $e^{-z}$ is small → numerator is positive
  - When z is negative: $e^z$ is small, $e^{-z}$ is large → numerator is negative
  - When z = 0: $e^0 - e^0 = 1 - 1 = 0$ → output is 0
- **Denominator**: $e^z + e^{-z}$ (always positive, normalizes output)
- **Alternative form**: $\frac{2}{1 + e^{-2z}} - 1$ (shows it's a scaled, shifted sigmoid)

**Concrete Examples**:
- $\tanh(0) = \frac{1-1}{1+1} = 0$ (zero at origin)
- $\tanh(2) = \frac{e^2 - e^{-2}}{e^2 + e^{-2}} = \frac{7.39 - 0.135}{7.39 + 0.135} = 0.96$
- $\tanh(-2) = -0.96$ (symmetric around origin)
- $\tanh(5) \approx 0.9999$ (approaches 1)
- $\tanh(-5) \approx -0.9999$ (approaches -1)

**Output Range**: $(-1, 1)$ - notice it's **zero-centered** unlike sigmoid!

**Properties**:
- **Zero-centered**: Outputs can be positive or negative (better than sigmoid)
- Stronger gradients than sigmoid (range is 2x larger: 2 vs 1)
- Still suffers from vanishing gradients at extremes
- Essentially a **scaled sigmoid**: $\tanh(z) = 2\sigma(2z) - 1$

**Graph**:
```
 1.0 |           ╱────  (approaches 1)
     |         ╱
 0.0 |       ╱  (crosses zero at z=0)
     |     ╱
-1.0 |────╱  (approaches -1)
     |_____|_____|_____|
    -6    0     6
```

**Derivative**:
$$
\frac{d\tanh}{dz} = 1 - \tanh^2(z)
$$

**Understanding the derivative**:
- **Maximum at z=0**: $\tanh(0) = 0$ → derivative = $1 - 0^2 = 1$ (steepest slope)
- **At extremes**: $\tanh(\pm 5) \approx \pm 1$ → derivative = $1 - 1 = 0$ (flat)
- **Range**: (0, 1) - always positive, max value of 1

**When to Use**:
- Hidden layers (better than sigmoid in most cases)
- When you want zero-centered outputs
- RNNs and LSTMs often use tanh
- **Still not as popular as ReLU** (vanishing gradient issue remains)

---

#### **3.2.3 ReLU (Rectified Linear Unit)**

**Formula**:
$$
\text{ReLU}(z) = \max(0, z) = \begin{cases}
z & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Breaking down the formula**:
- **ReLU** = "Rectified Linear Unit" (rectified = corrected/straightened)
- **$\max(0, z)$** = "Take the maximum of 0 and z"
  - If z = 5: $\max(0, 5) = 5$ (keep positive values as-is)
  - If z = -3: $\max(0, -3) = 0$ (negative values become 0)
  - If z = 0: $\max(0, 0) = 0$ (boundary case)
- **Piecewise definition**:
  - **When z > 0**: Output = z (identity function, slope = 1)
  - **When z ≤ 0**: Output = 0 (flat, slope = 0)

**Concrete Examples**:
- $\text{ReLU}(3.5) = 3.5$ (positive values pass through unchanged)
- $\text{ReLU}(-2.7) = 0$ (negative values get "killed")
- $\text{ReLU}(0) = 0$ (exactly at threshold)
- $\text{ReLU}(100) = 100$ (no upper bound!)

**Output Range**: $[0, \infty)$ - from 0 to positive infinity

**Why it's so popular**:
1. **Dead simple**: Just one comparison and one operation
2. **No vanishing gradient**: When z > 0, gradient is exactly 1 (constant!)
3. **Computationally cheap**: No exponentials, just max()
4. **Sparse activation**: About 50% of neurons output exactly zero
5. **Empirically works great**: Best performance in most deep networks

**Properties**:
- **Very simple** to compute (fastest of all activations)
- **No vanishing gradient** for positive values (gradient = 1)
- **Sparse activation**: About 50% of neurons output zero (efficient!)
- **Not zero-centered**: All outputs are ≥ 0
- Most popular activation in modern deep learning (90%+ of networks)

**Graph**:
```
     |
 4   |         ╱  (slope = 1 for z > 0)
     |       ╱
 2   |     ╱
     |   ╱
 0   |──────────  (flat line = 0 for z ≤ 0)
     |_____|_____|
    -2    0    2
```

**Derivative**:
$$
\frac{d\text{ReLU}}{dz} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**Understanding the derivative**:
- **When z > 0**: Derivative = 1 (gradient flows through perfectly!)
- **When z ≤ 0**: Derivative = 0 (gradient is blocked, neuron is "dead")
- **At z = 0**: Technically undefined, but we set it to 0 in practice
- **No vanishing gradient problem** for positive values!

**Problem: "Dying ReLU"**:
- If a neuron's output is always negative, its gradient is always 0
- The neuron stops learning forever (it's "dead")
- Can happen with large negative biases or large learning rates
- Solution: Use Leaky ReLU or other variants

**When to Use**:
- **Default choice** for hidden layers in feedforward networks
- CNNs (almost always ReLU)
- Most modern architectures
- When you want fast training and good performance

**Derivative**:
$$
\frac{d\text{ReLU}}{dz} = \begin{cases}
1 & \text{if } z > 0 \\
0 & \text{if } z \leq 0
\end{cases}
$$

**When to Use**:
- **Default choice** for hidden layers
- Computer vision tasks (CNNs)
- Most deep networks

**Downsides**:
- **Dying ReLU problem**: If neuron outputs 0, gradient is 0, neuron never updates
- **Not zero-centered**
- **Unbounded output**: Can lead to large activations

---

#### **3.2.4 Leaky ReLU**

**Formula**:
$$
\text{Leaky ReLU}(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}
$$

Where $\alpha$ is a small constant (typically 0.01)

**Properties**:
- Fixes the **dying ReLU** problem
- Small gradient for negative values instead of zero

**Derivative**:
$$
\frac{d\text{Leaky ReLU}}{dz} = \begin{cases}
1 & \text{if } z > 0 \\
\alpha & \text{if } z \leq 0
\end{cases}
$$

**When to Use**:
- When you suspect dying ReLU is a problem
- Experiments show slightly better results sometimes

---

#### **3.2.5 Softmax (For Multi-Class Classification)**

**Formula** (for output vector $z$):
$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

**Breaking down EVERY symbol**:
- **$z$** = Input vector of "logits" (raw scores), e.g., $z = [2.1, 1.0, 0.3]$
- **$K$** = Number of classes (e.g., 10 for MNIST digits 0-9)
- **$z_i$** = Score for class i (the i-th element of vector z)
- **$e^{z_i}$** = Exponential of score for class i
  - Makes all values positive
  - Amplifies differences (larger scores get much larger)
- **$\sum_{j=1}^{K}$** = Sum over all K classes (j goes from 1 to K)
- **$\sum_{j=1}^{K} e^{z_j}$** = Sum of exponentials of ALL class scores
  - Example: If $z = [2, 1, 0]$, then sum = $e^2 + e^1 + e^0 = 7.39 + 2.72 + 1.00 = 11.11$
- **Entire fraction** = Normalized probability for class i
  - Numerator: How much class i "wants" to be chosen
  - Denominator: Total "wanting" across all classes

**Concrete Example**:
Let's say we have 3 classes (cat, dog, bird) with scores:
```
z = [2.0, 1.0, 0.1]  (raw scores/logits)
```

Step by step:
1. **Compute exponentials**:
   - $e^{2.0} = 7.39$
   - $e^{1.0} = 2.72$
   - $e^{0.1} = 1.11$

2. **Sum them**: $7.39 + 2.72 + 1.11 = 11.22$

3. **Divide each by sum**:
   - P(cat) = $\frac{7.39}{11.22} = 0.659$ (65.9%)
   - P(dog) = $\frac{2.72}{11.22} = 0.242$ (24.2%)
   - P(bird) = $\frac{1.11}{11.22} = 0.099$ (9.9%)

4. **Verify**: $0.659 + 0.242 + 0.099 = 1.000$ ✓ (sums to 1!)

**Why exponential?**
- Converts any real number to positive (probabilities must be positive)
- Amplifies differences: larger scores get much larger probabilities
- Has nice mathematical properties for derivatives

**Output**: Vector of probabilities that sum to **exactly 1.0**

**Properties**:
- Converts logits (raw scores) to **valid probability distribution**
- Used **ONLY in output layer** for multi-class classification
- **Each output represents**: P(input belongs to class i)
- Differentiable everywhere
- **Temperature**: Can control how "sharp" the distribution is

**Example**:
```python
# Raw scores (logits) for 3 classes
z = [2.0, 1.0, 0.1]

# Apply softmax
exp_z = [e^2.0, e^1.0, e^0.1] = [7.39, 2.72, 1.11]
sum_exp = 7.39 + 2.72 + 1.11 = 11.22

probabilities = [7.39/11.22, 2.72/11.22, 1.11/11.22]
              = [0.659, 0.242, 0.099]
              
# Sum = 1.0 ✓
# Largest logit (2.0) → Largest probability (0.659) ✓
```

**When to Use**:
- **Output layer** for multi-class classification (mutually exclusive classes)
- When you need class probabilities

---

### 3.3 Choosing Activation Functions

| Layer | Task | Recommended Activation |
|-------|------|------------------------|
| Hidden Layers | General | **ReLU** |
| Hidden Layers | Avoiding dying ReLU | **Leaky ReLU** or **ELU** |
| Hidden Layers | Recurrent networks (RNNs) | **Tanh** |
| Output Layer | Binary Classification | **Sigmoid** |
| Output Layer | Multi-Class Classification | **Softmax** |
| Output Layer | Regression | **None (Linear)** |

---

## 4. Multi-Layer Perceptron (MLP)

### 4.1 Architecture

A **Multi-Layer Perceptron** (also called a **feedforward neural network**) stacks multiple layers of neurons:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Output Layer
```

**Example: 3-Layer Network**
```
     Input (784)      Hidden (128)     Hidden (64)      Output (10)
     
     x₁ ──╲           h₁₁ ──╲          h₂₁ ──╲          y₁ (0)
     x₂ ───╲──→ h₁₂ ──╱──→ h₂₂ ──╱──→       y₂ (1)
     ...    ╱    h₁₃               h₂₃          ...
     x₇₈₄ ─╱     ...               ...          y₁₀ (9)
                 h₁₁₂₈             h₂₆₄
```

**For MNIST digit classification**:
- **Input**: 784 neurons (28×28 pixel image flattened)
- **Hidden 1**: 128 neurons with ReLU activation
- **Hidden 2**: 64 neurons with ReLU activation
- **Output**: 10 neurons with Softmax (one per digit 0-9)

### 4.2 Why "Deep" Learning?

**Deep** = Multiple hidden layers

**Benefits of depth**:
1. **Hierarchical Feature Learning**: Early layers learn simple features (edges), deeper layers learn complex features (faces, objects)
2. **Fewer Parameters**: Deep networks can represent complex functions with fewer parameters than shallow wide networks
3. **Better Generalization**: Empirically, deeper networks generalize better

**Historical Note**: Before 2012, training deep networks was very difficult (vanishing gradients, no good initialization, slow computers). Breakthroughs in optimization, regularization, and GPUs made deep learning practical.

---

## 5. Forward Propagation

### 5.1 What is Forward Propagation?

**Forward propagation** is the process of computing the output of a neural network given an input. Data "flows forward" through the network layer by layer.

### 5.2 Step-by-Step Forward Pass

Let's walk through a **2-layer network** (1 hidden, 1 output):

**Architecture**:
```
Input (3) → Hidden (4, ReLU) → Output (2, Softmax)
```

**Notation**:
- $x$ = input vector (size 3)
- $W^{[1]}$ = weights from input to hidden (size 4×3)
- $b^{[1]}$ = biases for hidden layer (size 4)
- $h$ = hidden layer activations (size 4)
- $W^{[2]}$ = weights from hidden to output (size 2×4)
- $b^{[2]}$ = biases for output layer (size 2)
- $y$ = output probabilities (size 2)

**Step 1: Input to Hidden**
$$
z^{[1]} = W^{[1]} x + b^{[1]} \quad \text{(size 4)}
$$
$$
h = \text{ReLU}(z^{[1]}) \quad \text{(size 4)}
$$

**Step 2: Hidden to Output**
$$
z^{[2]} = W^{[2]} h + b^{[2]} \quad \text{(size 2)}
$$
$$
y = \text{softmax}(z^{[2]}) \quad \text{(size 2)}
$$

### 5.3 Concrete Example

```python
import numpy as np

# Input: 3 features
x = np.array([1.0, 2.0, 3.0])

# Layer 1 weights (4×3) and biases (4)
W1 = np.random.randn(4, 3) * 0.1
b1 = np.zeros(4)

# Layer 2 weights (2×4) and biases (2)
W2 = np.random.randn(2, 4) * 0.1
b2 = np.zeros(2)

# Forward pass
z1 = W1 @ x + b1  # Shape: (4,)
h = np.maximum(0, z1)  # ReLU activation, shape: (4,)

z2 = W2 @ h + b2  # Shape: (2,)
exp_z2 = np.exp(z2)
y = exp_z2 / np.sum(exp_z2)  # Softmax, shape: (2,)

print(f"Output probabilities: {y}")  # e.g., [0.45, 0.55]
```

**Key Points**:
- Forward pass is just **matrix multiplications** and **element-wise activations**
- Very fast on GPUs (massively parallel)
- No loops needed (vectorized operations)

---

## 6. Loss Functions

### 6.1 What is a Loss Function?

A **loss function** (or **cost function**) measures how "wrong" the network's predictions are. Training minimizes this loss.

**Notation**:
- $y$ = true label (what the output should be)
- $\hat{y}$ = predicted output (what the network produces)
- $L(y, \hat{y})$ = loss (scalar value indicating error)

---

### 6.2 Mean Squared Error (MSE)

**Use Case**: Regression problems (predicting continuous values like house prices, temperature, stock prices)

**Formula**:
$$
L_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

**Breaking down every symbol**:
- **$L_{MSE}$** = Mean Squared Error loss value (single number)
- **$\frac{1}{n}$** = Averaging factor (divide by number of samples)
- **$n$** = Number of samples/predictions
- **$\sum_{i=1}^{n}$** = Sum over all n samples (i goes from 1 to n)
- **$y_i$** = True value for sample i (ground truth, actual value)
- **$\hat{y}_i$** = Predicted value for sample i (model's guess)
  - The hat (^) symbol means "predicted" or "estimated"
- **$(y_i - \hat{y}_i)$** = Error for sample i (how far off we are)
  - If y=100 and ŷ=110, error = -10
  - If y=100 and ŷ=90, error = +10
- **$(y_i - \hat{y}_i)^2$** = Squared error for sample i
  - Squaring makes all errors positive
  - Penalizes large errors more than small ones
  - Example: error of 10 → squared error = 100
  - Example: error of 2 → squared error = 4

**Why square the errors?**
1. **All positive**: (-10)² = 100 (same penalty as +10)
2. **Penalizes large errors**: Error of 100 → penalty of 10,000 (much worse than 2 errors of 50)
3. **Smooth derivative**: $\frac{d}{dy}(y-\hat{y})^2 = 2(y-\hat{y})$ (easy to compute gradient)
4. **Mathematical convenience**: Connects to Gaussian distributions

**Example**: Predicting house prices
```python
y_true = [300000, 450000, 200000]  # Actual prices ($)
y_pred = [310000, 430000, 195000]  # Predicted prices ($)

# Step by step:
# Sample 1: (300000 - 310000)² = (-10000)² = 100,000,000
# Sample 2: (450000 - 430000)² = (20000)² = 400,000,000
# Sample 3: (200000 - 195000)² = (5000)² = 25,000,000

# Sum: 100M + 400M + 25M = 525,000,000
# Average: 525M / 3 = 175,000,000

errors = [(300000-310000)**2, (450000-430000)**2, (200000-195000)**2]
       = [100000000, 400000000, 25000000]
       
MSE = (100000000 + 400000000 + 25000000) / 3
    = 175000000

RMSE = sqrt(MSE) = 13228.76  # On average, off by ~$13k
```

**Properties**:
- Penalizes large errors more (due to squaring)
- Differentiable everywhere
- Sensitive to outliers

---

### 6.3 Binary Cross-Entropy

**Use Case**: Binary classification (2 classes)

**Formula**:
$$
L_{BCE} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

**Breaking down EVERY symbol:**

- $L_{BCE}$ = **Loss** for **B**inary **C**ross-**E**ntropy (the final error value)
- $-$ = **Negative sign** (to make loss positive, since log of probabilities is negative)
- $\frac{1}{n}$ = **Average** over all samples
  - $n$ = total number of samples in the batch
  - Makes loss independent of batch size
- $\sum_{i=1}^{n}$ = **Sum** from sample 1 to sample n
  - $i$ = index of current sample (goes from 1, 2, 3, ..., n)
  - Adds up loss from every sample in the batch
- $y_i$ = **True label** for sample i
  - $y_i \in \{0, 1\}$ means can ONLY be 0 or 1
  - If $y_i = 1$ → sample belongs to positive class (e.g., spam, cat, disease)
  - If $y_i = 0$ → sample belongs to negative class (e.g., not spam, dog, healthy)
- $\hat{y}_i$ = **Predicted probability** for sample i
  - $\hat{y}_i \in (0, 1)$ means between 0 and 1 (exclusive)
  - Output from sigmoid activation: $\sigma(z)$
  - If $\hat{y}_i = 0.9$ → model is 90% confident it's positive class
  - If $\hat{y}_i = 0.1$ → model is 10% confident it's positive class
- $\log$ = **Natural logarithm** (base e ≈ 2.71828)
  - $\log(0.5) = -0.693$
  - $\log(0.9) = -0.105$ (confident correct prediction)
  - $\log(0.1) = -2.303$ (confident wrong prediction)
- $y_i \log(\hat{y}_i)$ = **Loss when true class is 1**
  - If $y_i = 1$, this term is active
  - If $y_i = 0$, this term becomes 0 (since $0 \times \log(\hat{y}_i) = 0$)
- $(1 - y_i)$ = **Switch** that activates the other term
  - If $y_i = 1$ → $(1-1) = 0$ → second term deactivated
  - If $y_i = 0$ → $(1-0) = 1$ → second term activated
- $\log(1 - \hat{y}_i)$ = **Loss when true class is 0**
  - $(1 - \hat{y}_i)$ = probability of negative class
  - If $\hat{y}_i = 0.1$ → $(1 - 0.1) = 0.9$ → confident negative
  - If $\hat{y}_i = 0.9$ → $(1 - 0.9) = 0.1$ → confident positive (wrong!)

**Why this formula works:**

The brilliant part: The formula automatically picks the right term:
- When true label is 1: Only $y_i \log(\hat{y}_i)$ matters (other term zeroed out)
- When true label is 0: Only $(1-y_i) \log(1-\hat{y}_i)$ matters (first term zeroed out)

**Concrete numerical example:**

Let's say model predicts $\hat{y} = 0.9$ (90% confident positive):
- If true label $y = 1$ (actually positive):
  - Loss = $-[1 \times \log(0.9) + 0 \times \log(0.1)]$
  - Loss = $-[\log(0.9)] = -(-0.105) = 0.105$ ✓ **Low loss (good prediction!)**
  
- If true label $y = 0$ (actually negative):
  - Loss = $-[0 \times \log(0.9) + 1 \times \log(0.1)]$
  - Loss = $-[\log(0.1)] = -(-2.303) = 2.303$ ✗ **High loss (bad prediction!)**

Now if model predicts $\hat{y} = 0.5$ (50% uncertain):
- If true label $y = 1$:
  - Loss = $-\log(0.5) = 0.693$ (medium loss)
- If true label $y = 0$:
  - Loss = $-\log(0.5) = 0.693$ (medium loss)

**Why negative sign?**
- $\log(0.9) = -0.105$ (negative!)
- $\log(0.1) = -2.303$ (very negative!)
- We want loss to be positive, so we add negative sign
- This makes confident wrong predictions have HIGH positive loss

**Example**: Email spam detection
```python
y_true = [1, 0, 1, 0]  # 1 = spam, 0 = not spam
y_pred = [0.9, 0.1, 0.8, 0.3]  # Predicted probabilities

# For y=1, want y_pred→1, so loss = -log(y_pred)
# For y=0, want y_pred→0, so loss = -log(1 - y_pred)

losses = [
    -log(0.9),        # y=1, pred=0.9: -(-0.105) = 0.105 (good)
    -log(1-0.1),      # y=0, pred=0.1: -(-0.105) = 0.105 (good)
    -log(0.8),        # y=1, pred=0.8: -(-0.223) = 0.223 (ok)
    -log(1-0.3),      # y=0, pred=0.3: -(-0.357) = 0.357 (worse)
]

BCE = (0.105 + 0.105 + 0.223 + 0.357) / 4 = 0.197
```

**Properties**:
- Outputs are probabilities (use sigmoid activation in output layer)
- Heavily penalizes confident wrong predictions
- Convex and differentiable

---

### 6.4 Categorical Cross-Entropy

**Use Case**: Multi-class classification (K classes, mutually exclusive)

**Formula**:
$$
L_{CCE} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})
$$

**Breaking down EVERY symbol:**

- $L_{CCE}$ = **Loss** for **C**ategorical **C**ross-**E**ntropy
- $-$ = **Negative sign** (makes loss positive)
- $\frac{1}{n}$ = **Average** over all samples
  - $n$ = total number of samples in batch
- $\sum_{i=1}^{n}$ = **Outer sum** over all samples
  - $i$ = sample index (1st sample, 2nd sample, ..., nth sample)
- $\sum_{k=1}^{K}$ = **Inner sum** over all classes
  - $k$ = class index (class 1, class 2, ..., class K)
  - $K$ = total number of classes (e.g., 10 for MNIST digits)
  - This means: "For each sample, look at ALL possible classes"
- $y_{i,k}$ = **True label** for sample i and class k (one-hot encoded)
  - $y_{i,k} = 1$ if sample i truly belongs to class k
  - $y_{i,k} = 0$ if sample i does NOT belong to class k
  - Example for digit "3": $y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$
  - Only ONE element is 1, rest are 0 (that's "one-hot")
- $\hat{y}_{i,k}$ = **Predicted probability** that sample i belongs to class k
  - Output from softmax activation
  - All probabilities sum to 1: $\sum_{k=1}^{K} \hat{y}_{i,k} = 1$
  - Example: $\hat{y} = [0.05, 0.05, 0.10, 0.60, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]$
- $\log(\hat{y}_{i,k})$ = **Natural log** of predicted probability

**Why this formula works:**

The magic: Since $y_{i,k}$ is one-hot encoded, only ONE term survives!
- If $y_{i,k} = 0$ → $0 \times \log(\hat{y}_{i,k}) = 0$ (term disappears)
- If $y_{i,k} = 1$ → $1 \times \log(\hat{y}_{i,k}) = \log(\hat{y}_{i,k})$ (only this term matters)

So even though we have a double sum over all samples and all classes, only the predicted probability of the TRUE class contributes to the loss!

**Concrete numerical example - MNIST digit classification:**

Say true label is digit "3" (4th class in 0-indexed array):
- True label (one-hot): $y = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]$

**Case 1: Good prediction (confident and correct)**
- Predictions: $\hat{y} = [0.01, 0.01, 0.02, 0.92, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002]$
- Let's compute the sum $\sum_{k=1}^{10} y_k \log(\hat{y}_k)$:
  - $k=1$: $0 \times \log(0.01) = 0$ (term vanishes)
  - $k=2$: $0 \times \log(0.01) = 0$ (term vanishes)
  - $k=3$: $0 \times \log(0.02) = 0$ (term vanishes)
  - $k=4$: $1 \times \log(0.92) = -0.083$ ← **Only this survives!**
  - $k=5$ through $k=10$: all $0 \times \log(...) = 0$ (terms vanish)
- Sum = $-0.083$
- Loss = $-(-0.083) = 0.083$ ✓ **Low loss (excellent!)**

**Case 2: Okay prediction (less confident but correct)**
- Predictions: $\hat{y} = [0.05, 0.05, 0.10, 0.60, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]$
- Only $k=4$ matters: $1 \times \log(0.60) = -0.511$
- Loss = $-(-0.511) = 0.511$ (medium loss)

**Case 3: Wrong prediction (confident but wrong)**
- Predictions: $\hat{y} = [0.05, 0.05, 0.10, 0.10, 0.50, 0.05, 0.05, 0.05, 0.03, 0.02]$
- Model thinks it's digit "4" (5th class) with 50% confidence
- But true class is still digit "3" (4th class)
- Only $k=4$ matters: $1 \times \log(0.10) = -2.303$
- Loss = $-(-2.303) = 2.303$ ✗ **High loss (very bad!)**

**Why use log?**
- When model is confident AND correct: $\log(0.99) = -0.01$ → Loss ≈ 0.01 (reward!)
- When model is confident BUT wrong: $\log(0.01) = -4.605$ → Loss = 4.605 (huge penalty!)
- This heavily penalizes confident mistakes
- Forces model to be honest about uncertainty

**Example**: MNIST digit classification
```python
# True label: digit "3"
y_true = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # One-hot encoded

# Predicted probabilities
y_pred = [0.05, 0.05, 0.10, 0.60, 0.05, 0.05, 0.05, 0.03, 0.01, 0.01]

# Only the true class contributes to loss
loss = -log(y_pred[3]) = -log(0.60) = 0.511

# If prediction was very confident and correct:
y_pred_good = [0.01, 0.01, 0.02, 0.92, 0.01, 0.01, 0.01, 0.005, 0.003, 0.002]
loss_good = -log(0.92) = 0.083  # Much lower loss!

# If prediction was wrong:
y_pred_wrong = [0.05, 0.05, 0.10, 0.10, 0.50, 0.05, 0.05, 0.05, 0.03, 0.02]
loss_wrong = -log(0.10) = 2.303  # Very high loss!
```

**Properties**:
- Use with softmax activation in output layer
- Encourages network to output high probability for correct class
- Gradient has nice form: $\hat{y}_k - y_k$

---

## 7. Why We Need Multiple Layers

### 7.1 The XOR Problem Revisited

Remember the XOR problem that single perceptrons can't solve? Let's solve it with a 2-layer network.

**XOR Truth Table**:
```
x₁  x₂  | y
 0   0  | 0
 0   1  | 1
 1   0  | 1
 1   1  | 0
```

**Network Architecture**:
```
Input (2) → Hidden (2, tanh) → Output (1, sigmoid)
```

**Hidden Layer**: Learns to separate the space into regions
- **Neuron 1**: Fires when (x₁ OR x₂)
- **Neuron 2**: Fires when (x₁ AND x₂)

**Output Layer**: Combines hidden neurons
- **Output**: (Neuron 1) AND NOT (Neuron 2)

This is equivalent to: (x₁ OR x₂) AND NOT (x₁ AND x₂) = XOR!

### 7.2 Universal Approximation Theorem

**Theorem**: A neural network with at least one hidden layer with enough neurons can approximate **any continuous function** to arbitrary accuracy.

**What This Means**:
- Even 2-layer networks are incredibly powerful
- With enough hidden neurons, you can fit any pattern

**Why We Use Deep Networks**:
- Depth is more **parameter-efficient** than width
- Deep networks learn **hierarchical features** naturally
- Empirically, deep networks **generalize better**

### 7.3 Feature Hierarchies Example

**Computer Vision** (recognizing faces):
```
Layer 1 (edges): Horizontal lines, vertical lines, diagonal edges
        ↓
Layer 2 (textures): Corners, curves, simple shapes
        ↓
Layer 3 (parts): Eyes, noses, mouths
        ↓
Layer 4 (objects): Faces, cars, animals
```

Each layer builds on the previous layer's features!

---

## 8. Implementation Strategy

### 8.1 What We'll Implement

**File**: `neural_network_from_scratch.py`

**Components**:
1. **Activation functions**: Sigmoid, ReLU, Softmax (with derivatives)
2. **Dense Layer class**: Forward pass, stores inputs for backprop
3. **Loss functions**: Binary cross-entropy, categorical cross-entropy
4. **Neural Network class**: Combines layers, implements forward pass
5. **Training loop**: Iterates through data, computes loss

**NOT in this chapter** (coming in Chapter 02):
- Backpropagation (gradient computation)
- Weight updates

We'll use **placeholder gradients** for now and implement proper backprop in the next chapter.

### 8.2 Design Philosophy

**Principles**:
- **Clarity over performance**: Readable code with extensive comments
- **NumPy only**: Understand the math, no black boxes
- **Modular design**: Each component is independent
- **Extensible**: Easy to add new layers/activations later

---

## 9. MNIST Dataset

### 9.1 What is MNIST?

**MNIST** (Modified National Institute of Standards and Technology) is the "Hello World" of deep learning.

**Dataset Details**:
- **Task**: Classify handwritten digits (0-9)
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28 pixels, grayscale
- **Classes**: 10 (digits 0-9)

**Example Images**:
```
Label: 5          Label: 0          Label: 4
  
 ░░░░░░░░░░░        ░░░░░░░░░░░        ░░░░░░░░░░░
 ░░█████░░░░        ░░████░░░░░        ░░░░░░█░░░░
 ░██░░░░░░░░        ░██░░██░░░░        ░░░░░██░░░░
 ░█████░░░░░        ██░░░░██░░░        ░░░░██░░░░░
 ░░░░░░██░░░        ██░░░░██░░░        ░░░██░░░░░░
 ░█████░░░░░        ░██░░██░░░░        ░░████████░
 ░░░░░░░░░░░        ░░████░░░░░        ░░░░░██░░░░
```

### 9.2 Data Preprocessing

**Steps**:
1. **Load data**: Download from `keras.datasets` or `torchvision`
2. **Flatten**: Reshape 28×28 images to 784-dimensional vectors
3. **Normalize**: Scale pixel values from [0, 255] to [0, 1]
4. **One-hot encode labels**: Convert class labels to vectors

**Example**:
```python
# Original
image_shape = (28, 28)  # 2D image
pixel_values = [0, 15, 230, 255, ...]  # Range 0-255
label = 5  # Class index

# After preprocessing
flattened_image = (784,)  # 1D vector
normalized_pixels = [0.0, 0.059, 0.902, 1.0, ...]  # Range 0-1
one_hot_label = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]  # One-hot encoding
```

### 9.3 Why MNIST is Easy

**Characteristics**:
- Images are centered and size-normalized
- Digits are well-separated from background
- Training data is large (60k samples)
- Task is clear (10 well-defined classes)

**State-of-the-art accuracy**: 99.79% (human-level is ~99.8%)

**Our Goal**: 95%+ accuracy with simple MLP (achievable!)

---

## 10. Training Loop Structure

### 10.1 High-Level Training Process

```python
# Pseudocode
for epoch in range(num_epochs):
    for batch in training_data:
        # 1. Forward pass: compute predictions
        predictions = network.forward(batch_inputs)
        
        # 2. Compute loss: how wrong are we?
        loss = loss_function(predictions, batch_labels)
        
        # 3. Backward pass: compute gradients (Chapter 02)
        gradients = network.backward(loss)
        
        # 4. Update weights: improve parameters (Chapter 02)
        optimizer.step(gradients)
    
    # 5. Validation: check performance on validation set
    val_accuracy = evaluate(network, validation_data)
    print(f"Epoch {epoch}: Loss={loss:.4f}, Val Acc={val_accuracy:.2%}")
```

### 10.2 Key Concepts

**Epoch**: One complete pass through the entire training dataset

**Batch**: A subset of training data processed together
- **Batch size**: Number of samples per batch (e.g., 32, 64, 128)
- **Why batches?**: Memory efficiency, faster computation, better gradients

**Iteration**: Processing one batch

**Example**:
```
Dataset: 60,000 samples
Batch size: 64
Iterations per epoch: 60,000 / 64 = 937
Epochs: 10
Total iterations: 937 × 10 = 9,370
```

### 10.3 Monitoring Training

**Metrics to Track**:
1. **Training Loss**: Should decrease over time
2. **Validation Accuracy**: Should increase over time
3. **Validation Loss**: Should decrease, but may start increasing (overfitting)
4. **Time per Epoch**: For planning

**Healthy Training**:
```
Epoch 1: Loss=0.542, Val Acc=89.2%
Epoch 2: Loss=0.234, Val Acc=93.1%  ✓ Both improving
Epoch 3: Loss=0.156, Val Acc=94.8%  ✓ Both improving
Epoch 4: Loss=0.112, Val Acc=95.6%  ✓ Both improving
Epoch 5: Loss=0.089, Val Acc=96.0%  ✓ Both improving
```

**Overfitting Warning**:
```
Epoch 6: Loss=0.072, Val Acc=96.1%  ⚠ Loss↓ but Acc plateaued
Epoch 7: Loss=0.061, Val Acc=95.9%  ❌ Acc decreasing! Stop training
```

---

## 📝 Summary

### What You Learned:

1. ✅ **Biological neurons** inspire artificial neurons
2. ✅ **Perceptrons** are single neurons that learn linear decision boundaries
3. ✅ **Activation functions** introduce non-linearity (Sigmoid, ReLU, Softmax)
4. ✅ **Multi-layer networks** can learn any function (Universal Approximation Theorem)
5. ✅ **Forward propagation** computes predictions layer by layer
6. ✅ **Loss functions** measure prediction errors (MSE, Cross-Entropy)
7. ✅ **Multiple layers** enable hierarchical feature learning
8. ✅ **MNIST dataset** is the standard benchmark for beginners
9. ✅ **Training loops** iterate through data to improve the network

### Next Steps:

1. **Read the implementation**: Open `neural_network_from_scratch.py`
2. **Understand every line**: Read all comments carefully
3. **Run the code**: Train a network on MNIST
4. **Experiment**: Change hyperparameters, observe effects
5. **Build the project**: Complete `project_mnist_classifier.py`

---

## 🎓 Self-Check Questions

Before moving to the implementation, make sure you can answer:

1. What is the mathematical formula for a single neuron's output?
2. Why do we need activation functions? What happens without them?
3. What's the difference between sigmoid, tanh, and ReLU?
4. What is forward propagation? Describe it step-by-step.
5. What's the difference between MSE and cross-entropy loss?
6. Why can't a single perceptron learn XOR?
7. What does the softmax function do? When do we use it?
8. How do we preprocess MNIST images for a neural network?
9. What's the difference between an epoch and a batch?
10. How do you know if your network is overfitting?

If you can answer these confidently, you're ready for the code! 🚀

---

**Next File**: Open `neural_network_from_scratch.py` to see these concepts in code!

