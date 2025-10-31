# Chapter 03: Optimization Algorithms

## Welcome to Optimization Techniques! 🚀

In Chapter 02, you learned basic gradient descent: `W -= learning_rate * dW`. But this simple approach has problems:
- **Slow convergence** on complex loss landscapes
- **Gets stuck** in local minima and saddle points
- **Same learning rate** for all parameters (inefficient)
- **Oscillates** in narrow valleys

In this chapter, you'll learn the **advanced optimizers** used in production:
- **SGD with Momentum** (90% of papers before 2014)
- **RMSprop** (Geoffrey Hinton's adaptive method)
- **Adam** (The king of optimizers - 95% of modern papers)
- **Learning rate schedules** (improve convergence)

These techniques will **reduce training time by 10-100x** and achieve **better final accuracy**!

---

## Table of Contents

1. [The Optimization Problem](#1-the-optimization-problem)
2. [Gradient Descent Variants](#2-gradient-descent-variants)
3. [SGD with Momentum](#3-sgd-with-momentum)
4. [RMSprop (Root Mean Square Propagation)](#4-rmsprop)
5. [Adam (Adaptive Moment Estimation)](#5-adam)
6. [Learning Rate Schedules](#6-learning-rate-schedules)
7. [Comparing All Optimizers](#7-comparing-all-optimizers)
8. [Practical Guidelines](#8-practical-guidelines)
9. [Implementation Strategy](#9-implementation-strategy)
10. [Self-Check Questions](#10-self-check-questions)

---

## 1. The Optimization Problem

### What Are We Optimizing?

We want to minimize the loss function:

$$L(\theta) = \frac{1}{m} \sum_{i=1}^{m} \mathcal{L}(f(x^{(i)}; \theta), y^{(i)})$$

Where:
- $\theta$ = all parameters (weights and biases)
- $f(x; \theta)$ = neural network output
- $\mathcal{L}$ = loss function (cross-entropy, MSE, etc.)
- $m$ = number of samples

### The Challenge: Loss Landscapes

Real neural network loss landscapes are **extremely complex**:

1. **High-dimensional** (millions of parameters)
2. **Non-convex** (many local minima)
3. **Saddle points** everywhere (gradient = 0 but not minimum)
4. **Ravines** (steep in some directions, flat in others)
5. **Plateaus** (flat regions with tiny gradients)

**Example: Why vanilla GD struggles**

Imagine a loss landscape like a narrow valley:
```
         |  steep
         |  walls
    Loss |  
         |     flat
         |     bottom
         +------------→ parameters
```

- **Vertical direction**: Steep gradient → large updates → oscillation
- **Horizontal direction**: Small gradient → tiny updates → slow progress

Vanilla gradient descent oscillates back and forth while slowly moving toward the minimum!

### Optimization Goals

A good optimizer should:
1. **Converge quickly** (fewer epochs to reach minimum)
2. **Escape saddle points** (don't get stuck where gradient = 0)
3. **Handle different scales** (some parameters need big updates, some small)
4. **Generalize well** (don't overfit to training data)
5. **Be robust** (work with different architectures and datasets)

---

## 2. Gradient Descent Variants

### Batch Gradient Descent

Use **all training samples** to compute gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where $\nabla L$ is computed over the entire dataset.

**Advantages:**
- Stable gradient estimate
- Converges smoothly

**Disadvantages:**
- Very slow (one update per epoch)
- Doesn't work for large datasets
- Can't leverage GPU parallelism well
- Gets stuck in local minima

### Stochastic Gradient Descent (SGD)

Use **one random sample** to compute gradient:

$$\theta_{t+1} = \theta_t - \eta \nabla \mathcal{L}(f(x^{(i)}; \theta_t), y^{(i)})$$

**Advantages:**
- Fast updates (one per sample)
- Noise helps escape local minima
- Can train on streaming data

**Disadvantages:**
- Very noisy gradients
- Erratic convergence (jumps around)
- Hard to parallelize

### Mini-Batch Gradient Descent

Use **small batch** of samples (typically 32-256):

$$\theta_{t+1} = \theta_t - \eta \frac{1}{B} \sum_{i \in \text{batch}} \nabla \mathcal{L}(f(x^{(i)}; \theta_t), y^{(i)})$$

**This is the standard in practice!**

**Advantages:**
- Reduces gradient noise (more stable than SGD)
- Fast updates (more frequent than batch GD)
- Efficient GPU parallelism (matrix operations)
- Good balance of speed and stability

**Typical batch sizes:**
- Small models: 32-64
- Medium models: 128-256
- Large models (GPT, etc.): 512-2048
- Rule of thumb: Largest that fits in GPU memory

---

## 3. SGD with Momentum

### The Problem with Vanilla SGD

Imagine rolling a ball down a hill with many bumps:
- Without momentum: Ball stops at every bump
- With momentum: Ball rolls through small bumps

**Mathematical problem:**

Vanilla SGD: $\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$

Issues:
1. **Oscillations** in steep directions (gradient flip-flops)
2. **Slow progress** in flat directions (tiny gradients)
3. **Gets stuck** in local minima easily

### The Momentum Solution

Add a **velocity term** that accumulates gradients:

$$v_t = \beta v_{t-1} + \eta \nabla L(\theta_t)$$

**Breaking down EVERY symbol:**

- $v_t$ = **Velocity** at time step t
  - Accumulates momentum from past gradients
  - Shape: Same as parameters $\theta$
  - Initialized to 0: $v_0 = 0$
  - Think of it as "speed" of parameter movement
  
- $=$ = **Assignment**

- $\beta$ = **Momentum coefficient** (Greek letter "beta")
  - Controls how much past velocity to keep
  - Typical value: $\beta = 0.9$ (keep 90% of old velocity)
  - Range: $[0, 1]$
  - $\beta = 0$ → no momentum (vanilla SGD)
  - $\beta = 0.99$ → very high momentum (slow to change direction)
  
- $v_{t-1}$ = **Previous velocity** (from last time step)
  - The velocity we computed in the previous iteration
  - Subscript $t-1$ means "one step ago"
  
- $+$ = **Addition**

- $\eta$ = **Learning rate** (Greek letter "eta")
  - Controls step size
  - Typical values: 0.001, 0.01, 0.1
  
- $\nabla L(\theta_t)$ = **Gradient** of loss at current parameters
  - $\nabla$ = "nabla", gradient operator
  - $L$ = loss function
  - $\theta_t$ = current parameters
  - This is what we computed via backpropagation

**What this means:**
- Take 90% ($\beta$) of previous velocity
- Add 100% ($\eta \times$) of current gradient
- New velocity = weighted sum of old velocity + new gradient

**Concrete numerical example:**

Let's track velocity for one parameter with $\beta = 0.9$, $\eta = 0.1$:

Initial: $v_0 = 0$, $\theta_0 = 1.0$

Step 1:
- Gradient: $\nabla L = -2.0$ (wants to decrease parameter)
- $v_1 = 0.9 \times 0 + 0.1 \times (-2.0) = -0.2$

Step 2:
- Gradient: $\nabla L = -2.0$ (still decreasing)
- $v_2 = 0.9 \times (-0.2) + 0.1 \times (-2.0)$
- $v_2 = -0.18 - 0.2 = -0.38$
- Notice: Velocity increased from -0.2 to -0.38 (acceleration!)

Step 3:
- Gradient: $\nabla L = -2.0$ (consistent direction)
- $v_3 = 0.9 \times (-0.38) + 0.1 \times (-2.0)$
- $v_3 = -0.342 - 0.2 = -0.542$
- Velocity growing! (Building momentum)

Step 4:
- Gradient: $\nabla L = -2.0$
- $v_4 = 0.9 \times (-0.542) + 0.1 \times (-2.0)$
- $v_4 = -0.4878 - 0.2 = -0.6878$

After many steps with consistent gradient -2.0:
- $v_{\infty} \approx -2.0$ (converges to terminal velocity)

**Now what if gradient oscillates?**

Step 1: $\nabla L = -2.0$ → $v_1 = -0.2$
Step 2: $\nabla L = +1.5$ → $v_2 = 0.9(-0.2) + 0.1(1.5) = -0.18 + 0.15 = -0.03$
Step 3: $\nabla L = -2.0$ → $v_3 = 0.9(-0.03) + 0.1(-2.0) = -0.027 - 0.2 = -0.227$

Notice: Velocity dampened! Oscillating gradients partially cancel out.

$$\theta_{t+1} = \theta_t - v_t$$

**Breaking down:**

- $\theta_{t+1}$ = **New parameters** (updated)
- $=$ = **Assignment**
- $\theta_t$ = **Current parameters**
- $-$ = **Subtraction**
- $v_t$ = **Velocity** we just computed

**Why subtract velocity?**
- Velocity points in direction of accumulated gradients
- Gradients point uphill (direction of loss increase)
- We want to go downhill, so we subtract
- Same logic as vanilla gradient descent

Or equivalently (different formulation):

$$v_t = \beta v_{t-1} + (1 - \beta) \nabla L(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

Where:
- $v_t$ = velocity (running average of gradients)
- $\beta$ = momentum coefficient (typically 0.9)
- $\eta$ = learning rate

### How Momentum Works

Think of $v_t$ as the **speed** of a ball rolling down the loss landscape:

1. **Consistent directions**: Velocity builds up (accelerates)
2. **Oscillating directions**: Velocities cancel out (dampens oscillation)
3. **Past information**: Momentum "remembers" previous gradients

**Example:**

```
Step    Gradient    Velocity (β=0.9)    Update
1       -2.0        -2.0                -2.0
2       -2.0        -3.8 = 0.9*(-2.0) + (-2.0)    -3.8
3       -2.0        -5.42 = 0.9*(-3.8) + (-2.0)   -5.42
4       -2.0        -6.878                        -6.878
```

Notice: Even though gradient is constant (-2.0), the update grows!
This is **acceleration** in consistent directions.

### Momentum Hyperparameter β

The momentum coefficient $\beta$ controls how much history to keep:

- **β = 0**: No momentum (vanilla SGD)
- **β = 0.9**: Standard (90% of previous velocity + 10% new gradient)
- **β = 0.99**: High momentum (slow to change direction)

**Rule of thumb:** Start with β = 0.9

**Exponential moving average:**

The velocity is an exponentially-weighted average:

$$v_t = (1-\beta)\sum_{i=0}^{\infty} \beta^i \nabla L(\theta_{t-i})$$

This means:
- Recent gradients have weight $(1-\beta)$
- Gradient from 1 step ago has weight $(1-\beta)\beta$
- Gradient from 2 steps ago has weight $(1-\beta)\beta^2$
- etc.

With $\beta=0.9$, we're averaging over roughly $\frac{1}{1-\beta} = 10$ steps.

### Nesterov Accelerated Gradient (NAG)

A clever improvement to momentum:

$$v_t = \beta v_{t-1} + \eta \nabla L(\theta_t - \beta v_{t-1})$$

$$\theta_{t+1} = \theta_t - v_t$$

**Key difference:** Compute gradient at **lookahead position** $\theta_t - \beta v_{t-1}$

**Intuition:** Look ahead to where momentum will take you, then compute gradient there.

This provides **better convergence** in convex problems, but the difference is small in deep learning.

---

## 4. RMSprop

### The Problem: Different Learning Rates for Different Parameters

Consider a neural network with both:
- **Frequent features** (word "the" in text) → large gradients
- **Rare features** (word "quantum" in text) → small gradients

Using the same learning rate for both is inefficient!

**We need adaptive learning rates.**

### RMSprop (Root Mean Square Propagation)

Invented by Geoffrey Hinton in his Coursera course (2012).

**Algorithm:**

$$s_t = \beta s_{t-1} + (1-\beta) (\nabla L(\theta_t))^2$$

**Breaking down EVERY symbol:**

- $s_t$ = **Squared gradient accumulator** at time t
  - Accumulates squared gradients (like variance)
  - Shape: Same as parameters $\theta$
  - Initialized to 0: $s_0 = 0$
  - "s" stands for "squared"
  
- $=$ = **Assignment**

- $\beta$ = **Decay rate**
  - Controls how much history to keep
  - Typical values: 0.9 or 0.99
  - Similar role to momentum's $\beta$, but for squared gradients
  
- $s_{t-1}$ = **Previous squared gradient accumulator**
  - Value from last iteration
  
- $+$ = **Addition**

- $(1-\beta)$ = **Weight for current gradient**
  - If $\beta = 0.9$ → $(1-\beta) = 0.1$ (10% of new info)
  - Ensures weighted average has proper scale
  
- $(\nabla L(\theta_t))^2$ = **Squared gradient**
  - $\nabla L(\theta_t)$ = gradient (can be negative)
  - $()^2$ = element-wise square (makes all values positive)
  - Example: If $\nabla L = [-2.0, 0.5]$ → $(\nabla L)^2 = [4.0, 0.25]$
  - Measures magnitude of gradient regardless of direction

**What this does:** Maintains exponentially-weighted average of squared gradients. Large gradients → large $s_t$. Small gradients → small $s_t$.

**Concrete numerical example:**

Let's track $s_t$ for one parameter with $\beta = 0.9$:

Step 1:
- Gradient: $\nabla L = -2.0$
- $s_1 = 0.9 \times 0 + 0.1 \times (-2.0)^2 = 0 + 0.1 \times 4.0 = 0.4$

Step 2:
- Gradient: $\nabla L = -2.0$ (consistent)
- $s_2 = 0.9 \times 0.4 + 0.1 \times 4.0 = 0.36 + 0.4 = 0.76$

Step 3:
- Gradient: $\nabla L = -2.0$
- $s_3 = 0.9 \times 0.76 + 0.1 \times 4.0 = 0.684 + 0.4 = 1.084$

Converges to: $s_{\infty} = \frac{(1-\beta) \times 4.0}{1-\beta} = 4.0$

**Now if gradient changes:**

Step 1: $\nabla L = 10.0$ → $s_1 = 0.1 \times 100 = 10.0$ (large!)
Step 2: $\nabla L = 0.1$ → $s_2 = 0.9 \times 10 + 0.1 \times 0.01 = 9.001$ (stays large)

$s_t$ remembers that this parameter has had large gradients recently.

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{s_t + \epsilon}} \nabla L(\theta_t)$$

**Breaking down EVERY symbol:**

- $\theta_{t+1}$ = **Updated parameters**
- $=$ = **Assignment**
- $\theta_t$ = **Current parameters**
- $-$ = **Subtraction**

- $\frac{\eta}{\sqrt{s_t + \epsilon}}$ = **Adaptive learning rate** (different for each parameter!)
  - $\eta$ = base learning rate (like 0.001)
  - $\sqrt{s_t + \epsilon}$ = square root of accumulated squared gradients
  - $\epsilon$ = tiny constant (like 1e-8) to prevent division by zero
  - Result: larger $s_t$ → smaller effective learning rate
  
- $\nabla L(\theta_t)$ = **Current gradient**

**Why this works:**

Parameters with large typical gradients:
- $s_t$ is large (accumulated many large squared gradients)
- $\sqrt{s_t}$ is large
- $\frac{\eta}{\sqrt{s_t}}$ is SMALL effective learning rate
- Update is dampened (prevents overshooting)

Parameters with small typical gradients:
- $s_t$ is small
- $\sqrt{s_t}$ is small  
- $\frac{\eta}{\sqrt{s_t}}$ is LARGE effective learning rate
- Update is amplified (makes faster progress)

**Concrete example:**

Say $\eta = 0.1$, $\epsilon = 10^{-8}$ (negligible)

Parameter A (has large gradients):
- Current: $\nabla L = 10.0$
- Accumulated: $s_t = 100.0$
- Update: $0.1 \times \frac{10.0}{\sqrt{100}} = 0.1 \times \frac{10}{10} = 0.1$

Parameter B (has small gradients):
- Current: $\nabla L = 0.1$
- Accumulated: $s_t = 0.01$
- Update: $0.1 \times \frac{0.1}{\sqrt{0.01}} = 0.1 \times \frac{0.1}{0.1} = 0.1$

Both get similar updates (≈ 0.1) despite 100× difference in gradient! This is the magic of adaptive learning rates.

### How RMSprop Works

**Key insight:** Divide each parameter's gradient by the **RMS (root mean square)** of recent gradients.

**For parameters with:**

1. **Large, frequent gradients:**
   - $s_t$ is large
   - Effective learning rate = $\frac{\eta}{\sqrt{s_t}}$ is small
   - Updates are dampened (prevents overshooting)

2. **Small, infrequent gradients:**
   - $s_t$ is small
   - Effective learning rate = $\frac{\eta}{\sqrt{s_t}}$ is large
   - Updates are amplified (makes progress faster)

**This automatically balances learning rates across parameters!**

### Mathematical Intuition

The term $\frac{1}{\sqrt{s_t + \epsilon}}$ is approximately the inverse of the gradient's standard deviation:

$$\frac{1}{\sqrt{E[g^2]}} \approx \frac{1}{\text{std}(g)}$$

This normalizes gradients to have similar magnitudes.

**Example:**

```
Parameter A:  gradient = 10     →  s_t ≈ 100  →  update ≈ η * 10/10 = η
Parameter B:  gradient = 0.1    →  s_t ≈ 0.01 →  update ≈ η * 0.1/0.1 = η
```

Both parameters get similar effective updates despite 100x difference in gradient magnitude!

### Why Call it "RMSprop"?

RMS = Root Mean Square:

$$\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^n x_i^2}$$

We're dividing by $\sqrt{s_t}$, which is an exponentially-weighted RMS of gradients.

---

## 5. Adam (Adaptive Moment Estimation)

### The Best of Both Worlds

**Adam** = **Momentum** + **RMSprop**

Published by Kingma & Ba (2014). Now the **most popular optimizer** in deep learning!

### Adam Algorithm

Combines two ideas:
1. **First moment** (mean) estimation → momentum
2. **Second moment** (variance) estimation → RMSprop

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \quad \text{(momentum)}$$

**Breaking down EVERY symbol:**

- $m_t$ = **First moment estimate** (momentum)
  - Exponentially-weighted average of gradients
  - "m" stands for "mean" or "momentum"
  - Shape: Same as parameters $\theta$
  - Initialized: $m_0 = 0$
  
- $=$ = **Assignment**

- $\beta_1$ = **Momentum decay rate**
  - Typical value: $\beta_1 = 0.9$
  - Controls how much history to keep
  - Subscript "1" = first moment
  
- $m_{t-1}$ = **Previous first moment**
  - Value from last iteration
  
- $+$ = **Addition**

- $(1-\beta_1)$ = **Weight for current gradient**
  - If $\beta_1 = 0.9$ → $(1-\beta_1) = 0.1$
  - Ensures proper normalization
  
- $\nabla L(\theta_t)$ = **Current gradient**

**What this does:** Same as momentum! Accumulates gradients with exponential decay.

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_t))^2 \quad \text{(RMSprop)}$$

**Breaking down EVERY symbol:**

- $v_t$ = **Second moment estimate** (variance)
  - Exponentially-weighted average of SQUARED gradients
  - "v" stands for "variance"
  - Shape: Same as parameters $\theta$
  - Initialized: $v_0 = 0$
  
- $=$ = **Assignment**

- $\beta_2$ = **RMSprop decay rate**
  - Typical value: $\beta_2 = 0.999$ (note: different from $\beta_1$!)
  - Higher than $\beta_1$ because we want more stable variance estimates
  - Subscript "2" = second moment
  
- $v_{t-1}$ = **Previous second moment**

- $+$ = **Addition**

- $(1-\beta_2)$ = **Weight for current squared gradient**
  - If $\beta_2 = 0.999$ → $(1-\beta_2) = 0.001$ (very small!)
  - Variance changes slowly
  
- $(\nabla L(\theta_t))^2$ = **Squared gradient** (element-wise)

**What this does:** Same as RMSprop! Tracks squared gradients for adaptive learning rates.

**Bias correction** (important in early iterations):

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$

**Breaking down:**

- $\hat{m}_t$ = **Bias-corrected first moment**
  - Hat symbol (^) indicates bias-corrected version
  - This is what we actually use for updates
  
- $=$ = **Equals**

- $m_t$ = **Uncorrected first moment** (biased toward 0 initially)

- $1 - \beta_1^t$ = **Bias correction factor**
  - $t$ = current iteration number (1, 2, 3, ...)
  - $\beta_1^t$ = $\beta_1$ raised to power t
  - At $t=1$: $\beta_1^1 = 0.9$ → correction = $1-0.9 = 0.1$ (large!)
  - At $t=10$: $\beta_1^{10} \approx 0.349$ → correction ≈ $0.651$
  - At $t=100$: $\beta_1^{100} \approx 0.000027$ → correction ≈ $1.0$ (minimal)
  
**Why bias correction?**

Initially, $m_0 = 0$, so:
- $m_1 = 0.9 \times 0 + 0.1 \times g_1 = 0.1 \cdot g_1$ (biased toward 0!)
- $\hat{m}_1 = \frac{0.1 \cdot g_1}{1-0.9} = \frac{0.1 \cdot g_1}{0.1} = g_1$ ✓ Corrected!

Without correction, early updates would be artificially small.

$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Breaking down:**

- $\hat{v}_t$ = **Bias-corrected second moment**
- $v_t$ = **Uncorrected second moment**
- $1 - \beta_2^t$ = **Bias correction for variance**
  - At $t=1$: $1-0.999^1 = 0.001$ (huge correction!)
  - At $t=1000$: $1-0.999^{1000} \approx 0.632$
  - At $t=10000$: $\approx 1.0$ (no correction needed)

Same reasoning as first moment, but $\beta_2$ is larger (0.999 vs 0.9), so correction lasts longer.

**Parameter update:**

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**Breaking down EVERY symbol:**

- $\theta_{t+1}$ = **Updated parameters**
- $=$ = **Assignment**
- $\theta_t$ = **Current parameters**
- $-$ = **Subtraction**

- $\frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}$ = **Adaptive learning rate**
  - $\eta$ = base learning rate (like 0.001)
  - $\sqrt{\hat{v}_t}$ = square root of bias-corrected variance
  - $\epsilon$ = small constant (1e-8) for numerical stability
  - This part is from RMSprop (adaptive per parameter)
  
- $\hat{m}_t$ = **Bias-corrected momentum**
  - This part is from momentum (direction and magnitude)
  
**Concrete numerical example:**

Let's trace one parameter through 3 steps with:
- $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 0.1$, $\epsilon = 10^{-8}$

Step 1: $\nabla L = 2.0$
- $m_1 = 0.9(0) + 0.1(2.0) = 0.2$
- $v_1 = 0.999(0) + 0.001(4.0) = 0.004$
- $\hat{m}_1 = \frac{0.2}{1-0.9^1} = \frac{0.2}{0.1} = 2.0$
- $\hat{v}_1 = \frac{0.004}{1-0.999^1} = \frac{0.004}{0.001} = 4.0$
- Update: $\frac{0.1}{\sqrt{4.0}} \times 2.0 = \frac{0.1}{2.0} \times 2.0 = 0.1$

Step 2: $\nabla L = 2.0$
- $m_2 = 0.9(0.2) + 0.1(2.0) = 0.38$
- $v_2 = 0.999(0.004) + 0.001(4.0) = 0.007996$
- Bias corrections applied...
- Update calculated with both momentum and adaptive LR

Adam automatically:
- Uses momentum for consistent direction
- Adapts learning rate per parameter based on gradient history
- Corrects for initialization bias

### Hyperparameters

**Standard values (from the paper):**
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (RMSprop decay)
- $\eta = 0.001$ (learning rate)
- $\epsilon = 10^{-8}$ (numerical stability)

**These defaults work well 90% of the time!**

### Why Bias Correction?

Without bias correction, early estimates are biased toward zero.

**Problem:** 
- Initialize: $m_0 = 0$, $v_0 = 0$
- First update: $m_1 = (1-\beta_1) g_1 = 0.1 \cdot g_1$ (much smaller than $g_1$!)

**Solution:** Divide by $(1 - \beta_1^t)$
- At $t=1$: $(1 - 0.9^1) = 0.1$ → $\hat{m}_1 = \frac{0.1 \cdot g_1}{0.1} = g_1$ ✓
- At $t=10$: $(1 - 0.9^{10}) \approx 0.65$ → correction is small
- At $t=\infty$: $(1 - 0.9^{\infty}) = 1$ → no correction needed

### How Adam Works

**Step-by-step for one parameter:**

1. **Compute gradient:** $g_t = \nabla L(\theta_t)$
2. **Update momentum:** $m_t = 0.9 \cdot m_{t-1} + 0.1 \cdot g_t$
3. **Update variance:** $v_t = 0.999 \cdot v_{t-1} + 0.001 \cdot g_t^2$
4. **Bias correction:** $\hat{m}_t = \frac{m_t}{1-0.9^t}$, $\hat{v}_t = \frac{v_t}{1-0.999^t}$
5. **Update parameter:** $\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

**Intuition:**
- **Numerator** $\hat{m}_t$: Smoothed gradient (momentum) → direction
- **Denominator** $\sqrt{\hat{v}_t}$: Gradient scale (RMSprop) → step size adaptation

### Adam vs Others

| Optimizer | Momentum | Adaptive LR | Bias Correction |
|-----------|----------|-------------|-----------------|
| SGD       | ✗        | ✗           | N/A             |
| Momentum  | ✓        | ✗           | N/A             |
| RMSprop   | ✗        | ✓           | ✗               |
| Adam      | ✓        | ✓           | ✓               |

**Adam combines the best features of all previous optimizers!**

---

## 6. Learning Rate Schedules

### Why Decay Learning Rate?

**At the start of training:**
- Far from minimum → need large steps → high learning rate

**Near the end of training:**
- Close to minimum → need small steps → low learning rate

**Analogy:** When driving to a destination:
- Highway: Drive fast (high LR)
- Parking lot: Drive slowly (low LR)

### Common Schedules

#### 1. Step Decay

Reduce learning rate by factor every N epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / k \rfloor}$$

Where:
- $\eta_0$ = initial learning rate
- $\gamma$ = decay factor (e.g., 0.5)
- $k$ = step size (e.g., 10 epochs)

**Example:** $\eta_0 = 0.1$, $\gamma = 0.5$, $k = 10$
- Epochs 0-9: lr = 0.1
- Epochs 10-19: lr = 0.05
- Epochs 20-29: lr = 0.025

#### 2. Exponential Decay

Smooth exponential decrease:

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

Or discrete version:

$$\eta_t = \eta_0 \cdot \gamma^t$$

Where $\gamma \approx 0.99$ to 0.9999 (decay rate).

#### 3. Polynomial Decay

$$\eta_t = \eta_0 \cdot \left(1 - \frac{t}{T}\right)^p$$

Where:
- $T$ = total training steps
- $p$ = polynomial degree (typically 0.5 or 1.0)

#### 4. Cosine Annealing

Smooth decrease following cosine curve:

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Popular in modern vision models (ResNet, ViT).**

Advantages:
- Smooth decrease (no abrupt changes)
- Fast decrease at start, slow at end
- Can restart (cosine annealing with warm restarts)

#### 5. Warm-up + Decay

**Linear warm-up** for first few epochs, then decay:

$$\eta_t = \begin{cases}
\frac{t}{T_{\text{warmup}}} \cdot \eta_0 & \text{if } t < T_{\text{warmup}} \\
\eta_0 \cdot \text{schedule}(t - T_{\text{warmup}}) & \text{otherwise}
\end{cases}$$

**Why warm-up?**
- Large initial learning rates can cause instability
- Especially important for large batch training
- Used in BERT, GPT, and most transformer models

### Choosing a Schedule

**Simple rule:**

1. **No schedule**: Try first (Adam often doesn't need scheduling)
2. **Step decay**: Easy to implement, works well
3. **Cosine annealing**: Best for vision tasks
4. **Warm-up + decay**: Best for transformers/LLMs

**Typical values:**
- Initial LR: 0.1 (SGD), 0.001 (Adam)
- Decay factor: 0.1 (step) or 0.95-0.999 (exponential)
- Warm-up: 5-10% of total training

---

## 7. Comparing All Optimizers

### Convergence Speed

On typical deep learning tasks:

**Speed (epochs to 95% accuracy):**
1. **Adam**: 10-30 epochs ⚡ (fastest)
2. **RMSprop**: 15-40 epochs
3. **SGD + Momentum**: 20-50 epochs
4. **Vanilla SGD**: 50-200 epochs 🐌 (slowest)

### Final Performance

**Best test accuracy:**
1. **SGD + Momentum + LR schedule**: 98.5% (best generalization)
2. **Adam**: 98.2%
3. **RMSprop**: 98.0%
4. **Vanilla SGD**: 97.5%

**Interesting fact:** Adam converges faster but SGD+Momentum often generalizes better!

### Memory Requirements

Per-parameter memory overhead:

- **SGD**: 0 extra
- **Momentum**: 1x (velocity)
- **RMSprop**: 1x (squared gradients)
- **Adam**: 2x (momentum + squared gradients)

For a model with 100M parameters:
- SGD: 400 MB
- Adam: 1.2 GB (3x more!)

### Hyperparameter Sensitivity

**Robustness to LR choice:**
1. **Adam**: Very robust (0.001 works 90% of time)
2. **RMSprop**: Fairly robust
3. **Momentum**: Moderate
4. **SGD**: Very sensitive (need careful tuning)

---

## 8. Practical Guidelines

### Which Optimizer to Use?

**Default choice: Adam**
- Learning rate: 0.001
- Betas: (0.9, 0.999)
- Works well out-of-the-box

**For best performance: SGD + Momentum + LR schedule**
- Learning rate: 0.1 (tune this!)
- Momentum: 0.9
- Use step decay or cosine annealing
- Requires more hyperparameter tuning

**For RNNs/LSTMs: RMSprop or Adam**
- RNNs have exploding/vanishing gradients
- Adaptive learning rates help

### Learning Rate Tuning

**Method 1: Grid search**
```python
learning_rates = [0.001, 0.003, 0.01, 0.03, 0.1]
# Try each, pick the best
```

**Method 2: Learning rate finder**
1. Start with very small LR (1e-7)
2. Increase exponentially each batch
3. Plot loss vs LR
4. Choose LR where loss decreases fastest
5. Common in fastai library

**Rule of thumb:**
- Too high: Loss explodes or oscillates
- Too low: Learning is very slow
- Just right: Steady, fast decrease in loss

### Batch Size Effects

**Larger batch size:**
- More stable gradients (less noise)
- Faster computation (GPU parallelism)
- Requires lower learning rate (gradients are averaged)

**Smaller batch size:**
- More noise in gradients (helps escape local minima)
- Better generalization (regularization effect)
- Requires higher learning rate

**Linear scaling rule:** If you multiply batch size by N, multiply learning rate by N.

Example:
- Batch=32, LR=0.001 → Batch=256, LR=0.008

---

## 9. Implementation Strategy

### What We'll Build

**File 1: `optimization_algorithms.py`**
- Implement all optimizers from scratch
- Base Optimizer class
- SGD, Momentum, RMSprop, Adam
- Learning rate schedules
- Train MNIST with each optimizer
- Compare convergence

**File 2: `project_optimizer_visualizer.py`**
- Interactive visualization of optimizers
- 2D loss landscapes (Rosenbrock, Beale functions)
- Animate optimization paths
- Compare all optimizers side-by-side
- Training curves comparison
- Hyperparameter exploration

### Key Concepts to Implement

1. **Optimizer base class** with update() method
2. **State tracking** (momentum, squared gradients)
3. **Per-parameter updates** (vectorized operations)
4. **Learning rate scheduling** (callbacks during training)
5. **Gradient clipping** (prevent exploding gradients)

---

## 10. Self-Check Questions

Test your understanding:

1. **Why does momentum help?** 
   - What problem does it solve?
   - How does the β parameter affect behavior?

2. **What's the difference between RMSprop and Adam?**
   - Which one has momentum?
   - Why does Adam need bias correction but RMSprop doesn't?

3. **When would you use SGD instead of Adam?**
   - What are the tradeoffs?
   - Which generalizes better? Which converges faster?

4. **How do learning rate schedules help?**
   - Why decay the learning rate?
   - What happens if you don't decay?

5. **Calculate by hand:**
   ```
   Given: θ₀ = 5, lr = 0.1, β = 0.9
   Gradients: g₁ = 2, g₂ = 2, g₃ = -1
   
   Using momentum, what is θ₃?
   ```

6. **Why is Adam the default choice?**
   - What problems does it solve?
   - When might it not be the best choice?

7. **What's the purpose of ε in Adam?**
   - What would happen without it?
   - Why is it so small (1e-8)?

8. **Explain the bias correction term** $(1 - \beta^t)$
   - Why is it needed?
   - What happens as t → ∞?

9. **How does batch size affect learning rate?**
   - What's the linear scaling rule?
   - Why does this relationship exist?

10. **Design an optimizer:**
    - If you had to combine features from SGD, Momentum, RMSprop, and Adam differently, what would you create?
    - What's the minimal set of features for good performance?

---

## Summary

You've learned the **optimization techniques that power modern deep learning**:

✅ **Problem**: Vanilla GD is slow and gets stuck  
✅ **Momentum**: Accelerates in consistent directions  
✅ **RMSprop**: Adapts learning rate per parameter  
✅ **Adam**: Combines momentum + RMSprop (best overall)  
✅ **LR Schedules**: Improve convergence and final accuracy  

**Key Takeaways:**
1. Use **Adam with default parameters** as your starting point
2. For best performance, try **SGD + Momentum + LR decay**
3. Always use **mini-batches** (32-256)
4. **Tune learning rate** first before other hyperparameters
5. **Larger batches** need lower learning rates

**Next Steps:**
- Implement all optimizers from scratch
- Visualize optimization paths on 2D landscapes
- Compare convergence speed on MNIST
- Understand when each optimizer excels

Now let's build these optimizers! 💪
