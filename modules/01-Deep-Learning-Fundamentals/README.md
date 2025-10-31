# Module 01: Deep Learning Fundamentals
## Master Neural Networks, CNNs & RNNs from First Principles

<div align="center">

**Duration**: 6-8 weeks | **Difficulty**: Beginner to Intermediate

*Build unshakeable foundations in deep learning*

</div>

---

## 🎯 Module Objectives

By the end of this module, you will:

- ✅ **Understand neural networks at the mathematical level** - no black boxes!
- ✅ **Implement neural networks from scratch** using only NumPy
- ✅ **Master backpropagation** and automatic differentiation
- ✅ **Train models effectively** with modern optimization techniques
- ✅ **Build Convolutional Neural Networks** for computer vision
- ✅ **Implement Recurrent Neural Networks** for sequence modeling
- ✅ **Achieve competitive results** on benchmark datasets (MNIST, CIFAR-10)
- ✅ **Develop intuition** for hyperparameter tuning and debugging

---

## 📚 Chapter Structure

### **Chapter 01: Neural Networks from Scratch** (Week 1)
**Goal**: Build your first neural network using only NumPy

**Theory** (`README.md`):
- What is a neuron? Biological inspiration
- Perceptron: The simplest neural network
- Activation functions: Sigmoid, ReLU, Tanh
- Forward propagation: Computing predictions
- Loss functions: MSE, Cross-Entropy
- Why we need multiple layers

**Implementation** (`neural_network_from_scratch.py`):
- Complete NumPy implementation of multilayer perceptron
- Dense layer class with forward pass
- Activation functions from scratch
- Loss calculation
- Training loop structure
- Every line explained in detail

**Project** (`project_mnist_classifier.py`):
- Build handwritten digit classifier for MNIST dataset
- Achieve 95%+ accuracy with 2-hidden-layer network
- Visualize predictions and errors
- Analyze where the model fails

**Key Concepts**:
- Neurons, weights, biases
- Matrix operations for efficiency
- Vectorization techniques
- Debugging neural networks

---

### **Chapter 02: Backpropagation & Gradient Descent** (Week 1-2)
**Goal**: Understand how neural networks learn

**Theory** (`README.md`):
- The learning problem: Minimizing loss
- Calculus refresher: Derivatives and chain rule
- Computational graphs
- Backpropagation algorithm step-by-step
- Gradient descent variants: Batch, Mini-batch, Stochastic
- Learning rate: The most important hyperparameter
- Automatic differentiation (how PyTorch works internally)

**Implementation** (`backpropagation_from_scratch.py`):
- Implement backward pass for each layer
- Compute gradients using chain rule
- Update weights with gradient descent
- Build mini autograd engine (micrograd style)
- Visualize gradient flow

**Project** (`project_custom_autograd.py`):
- Build your own automatic differentiation engine
- Support basic operations: +, *, -, /, exp, log
- Implement computational graph
- Compare with PyTorch gradients

**Key Concepts**:
- Chain rule mathematics
- Jacobians and gradients
- Vanishing/exploding gradients (introduction)
- Computational efficiency

---

### **Chapter 03: Training Techniques & Optimization** (Week 2-3)
**Goal**: Train neural networks faster and more reliably

**Theory** (`README.md`):
- Problems with vanilla gradient descent
- Momentum: Accelerating convergence
- RMSprop: Adaptive learning rates
- Adam optimizer: Best of both worlds
- Learning rate schedules: Step decay, cosine annealing
- Batch normalization: Faster training
- Weight initialization: Xavier, He initialization
- Gradient clipping

**Implementation** (`optimization_algorithms.py`):
- SGD with momentum from scratch
- RMSprop implementation
- Adam optimizer from scratch
- Learning rate schedulers
- Compare convergence speeds
- Visualize optimization trajectories

**Project** (`project_training_dynamics_visualizer.py`):
- Interactive visualization of different optimizers
- 2D loss landscape visualization
- Compare SGD vs Momentum vs Adam
- Hyperparameter sensitivity analysis

**Key Concepts**:
- First and second-order optimization
- Adaptive learning rates
- Momentum and velocity
- Training stability

---

### **Chapter 04: Regularization & Generalization** (Week 3-4)
**Goal**: Build models that generalize well to unseen data

**Theory** (`README.md`):
- Overfitting vs Underfitting
- Bias-variance tradeoff
- L1 regularization (Lasso): Sparsity
- L2 regularization (Ridge): Weight decay
- Dropout: Random neuron dropping
- Data augmentation techniques
- Early stopping
- Cross-validation for model selection
- Validation curves and learning curves

**Implementation** (`regularization_techniques.py`):
- L1/L2 regularization from scratch
- Dropout layer implementation
- Data augmentation pipeline
- Early stopping callback
- Model checkpointing

**Project** (`project_overfit_detector.py`):
- Intentionally overfit a model
- Apply different regularization techniques
- Measure generalization gap
- Visualize overfitting in action

**Key Concepts**:
- Train vs validation vs test sets
- Regularization strength tuning
- Detecting overfitting early
- Model selection strategies

---

### **Chapter 05: Convolutional Neural Networks (CNNs)** (Week 4-5)
**Goal**: Master the architecture that powers computer vision

**Theory** (`README.md`):
- Limitations of fully connected networks for images
- Convolution operation: How it works
- Filters and feature maps
- Pooling layers: Max pooling, average pooling
- Receptive fields
- Parameter sharing and translation equivariance
- CNN architectures: LeNet, AlexNet
- Understanding what CNNs learn (feature visualization)

**Implementation** (`cnn_from_scratch.py`):
- Convolution operation in NumPy
- Backpropagation through convolution
- Max pooling forward and backward
- Complete CNN implementation
- PyTorch CNN for comparison

**Project** (`project_cifar10_classifier.py`):
- Build CNN for CIFAR-10 image classification
- Achieve 75%+ accuracy
- Data augmentation for images
- Visualize learned filters
- Error analysis on misclassified images

**Key Concepts**:
- 2D convolutions
- Feature hierarchies (edges → textures → objects)
- Parameter efficiency
- Inductive biases

---

### **Chapter 06: Advanced CNN Architectures** (Week 5-6)
**Goal**: Learn state-of-the-art CNN designs

**Theory** (`README.md`):
- VGGNet: Depth and small filters
- ResNet: Skip connections and residual learning
- Inception: Multi-scale processing
- DenseNet: Dense connections
- EfficientNet: Compound scaling
- MobileNet: Efficient mobile architectures
- Transfer learning and fine-tuning
- Pre-trained models: When and how to use

**Implementation** (`advanced_cnn_architectures.py`):
- ResNet from scratch (ResNet-18)
- Residual blocks implementation
- Bottleneck layers
- Transfer learning with PyTorch
- Feature extraction vs fine-tuning

**Project** (`project_transfer_learning.py`):
- Use pre-trained ResNet50 for custom classification
- Fine-tune on small dataset (e.g., cats vs dogs)
- Compare training from scratch vs transfer learning
- Visualize intermediate activations

**Key Concepts**:
- Skip connections
- Batch normalization
- Residual learning theory
- Transfer learning strategies

---

### **Chapter 07: Recurrent Neural Networks (RNNs)** (Week 6-7)
**Goal**: Process sequential data with neural networks

**Theory** (`README.md`):
- Sequential data and temporal dependencies
- Vanilla RNN architecture
- Hidden states and unrolling
- Backpropagation through time (BPTT)
- Vanishing gradient problem in RNNs
- Long Short-Term Memory (LSTM): Gates and cell state
- Gated Recurrent Unit (GRU): Simplified LSTM
- When to use RNNs vs other architectures

**Implementation** (`rnn_from_scratch.py`):
- Vanilla RNN cell implementation
- LSTM cell from scratch (all gates)
- GRU cell implementation
- BPTT implementation
- PyTorch RNN/LSTM comparison

**Project** (`project_text_generation.py`):
- Character-level language model
- Train on Shakespeare text
- Generate new text samples
- Visualize hidden state activations
- Temperature sampling for diversity

**Key Concepts**:
- Sequential processing
- Memory in neural networks
- Gating mechanisms
- Teacher forcing

---

### **Chapter 08: Advanced RNNs & Seq2Seq** (Week 7-8)
**Goal**: Build encoder-decoder models for complex sequence tasks

**Theory** (`README.md`):
- Sequence-to-sequence (Seq2Seq) models
- Encoder-decoder architecture
- Attention mechanism (preview for Module 02)
- Bidirectional RNNs
- Many-to-one, one-to-many, many-to-many tasks
- Beam search for decoding
- Evaluation metrics: BLEU, perplexity

**Implementation** (`seq2seq_attention.py`):
- Encoder RNN implementation
- Decoder RNN with attention
- Training with teacher forcing
- Inference with beam search
- Attention weight visualization

**Project** (`project_neural_machine_translation.py`):
- English to French translation
- Train on Multi30k dataset
- Implement attention mechanism
- Visualize attention weights
- Evaluate with BLEU score

**Key Concepts**:
- Sequence transduction
- Attention weights
- Teacher forcing vs autoregressive
- Decoding strategies

---

## 🛠️ Tools & Libraries

### **Module Requirements**:

```bash
# Core libraries
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Deep learning frameworks
torch>=2.0.0
torchvision>=0.15.0

# Data processing
pandas>=2.0.0
scikit-learn>=1.2.0

# Visualization
plotly>=5.14.0
tensorboard>=2.13.0

# Utilities
tqdm>=4.65.0
jupyter>=1.0.0
```

### **Install All Dependencies**:
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets Used

1. **MNIST**: Handwritten digits (70K images, 28x28 grayscale)
2. **Fashion-MNIST**: Clothing items (70K images, 28x28 grayscale)
3. **CIFAR-10**: Natural images (60K images, 32x32 RGB)
4. **CIFAR-100**: 100-class image dataset
5. **Shakespeare Text**: Character-level text generation
6. **Multi30k**: English-German/French translation pairs

All datasets will be automatically downloaded by the code.

---

## 🎯 Learning Outcomes

### **Technical Skills**:
- Implement neural networks from scratch (without libraries)
- Understand backpropagation mathematically
- Train models effectively with modern techniques
- Build CNNs for computer vision tasks
- Implement RNNs for sequence modeling
- Debug training issues (overfitting, vanishing gradients)

### **Practical Skills**:
- Data preprocessing and augmentation
- Hyperparameter tuning strategies
- Model evaluation and metrics
- Visualization of training dynamics
- Code organization for deep learning projects

### **Conceptual Understanding**:
- Why neural networks work
- Tradeoffs in architecture design
- When to use which technique
- How to approach new problems

---

## 📈 Success Metrics

By the end of this module, you should achieve:

- ✅ **MNIST Classification**: 98%+ accuracy
- ✅ **CIFAR-10 Classification**: 80%+ accuracy
- ✅ **Text Generation**: Coherent character-level samples
- ✅ **Neural Machine Translation**: BLEU score > 20

---

## 🚀 How to Progress Through This Module

### **Recommended Approach**:

1. **Read Theory First** (2-3 hours per chapter)
   - Read the README.md completely
   - Work through mathematical derivations on paper
   - Understand intuition before code

2. **Study Implementation** (3-4 hours per chapter)
   - Read every line of the Python file
   - Run code cell by cell (Jupyter notebook style)
   - Modify hyperparameters and observe effects

3. **Build Project** (4-5 hours per chapter)
   - Start from scratch, referring to code as needed
   - Experiment with different architectures
   - Analyze results and iterate

4. **Review & Practice** (2-3 hours per chapter)
   - Reattempt implementation without looking at code
   - Explain concepts to yourself/others
   - Connect to research papers

### **Weekly Time Commitment**:
- **15-20 hours/week**: Complete module in 8 weeks
- **20-25 hours/week**: Complete module in 6 weeks
- **25-30 hours/week**: Complete module in 5 weeks (intense)

---

## 🤔 Common Questions

**Q: Do I need a GPU for this module?**
A: Not required initially. Google Colab's free tier is sufficient. GPU recommended for Chapter 05 onwards.

**Q: Can I skip the from-scratch implementations?**
A: **No**. Understanding the internals is crucial for debugging and advanced work. From-scratch = deep understanding.

**Q: How much math do I need?**
A: High school calculus (derivatives) and basic linear algebra (matrix multiplication). We explain everything.

**Q: What if I get stuck?**
A: Read the theory again, check the code comments, experiment with print statements, ask in discussion forum.

**Q: Can I use TensorFlow instead of PyTorch?**
A: The course uses PyTorch, but concepts transfer. You can implement in TensorFlow if comfortable.

---

## 🎓 Next Steps

After completing this module, you'll be ready for:
- **Module 02**: Transformers & Attention Mechanisms
- **Module 03**: Large Language Models
- Or dive into specialized computer vision/NLP courses

---

## 📂 Module Structure

```
01-Deep-Learning-Fundamentals/
│
├── README.md (this file)
├── requirements.txt
│
├── chapter_01_neural_networks/
│   ├── README.md (Theory: Perceptrons, activation functions)
│   ├── neural_network_from_scratch.py (NumPy implementation)
│   └── project_mnist_classifier.py (MNIST digit classification)
│
├── chapter_02_backpropagation/
│   ├── README.md (Theory: Chain rule, computational graphs)
│   ├── backpropagation_from_scratch.py (Backward pass implementation)
│   └── project_custom_autograd.py (Build your own autograd)
│
├── chapter_03_optimization/
│   ├── README.md (Theory: SGD, Adam, learning rate schedules)
│   ├── optimization_algorithms.py (Optimizers from scratch)
│   └── project_training_dynamics_visualizer.py (Optimizer comparison)
│
├── chapter_04_regularization/
│   ├── README.md (Theory: Overfitting, dropout, L1/L2)
│   ├── regularization_techniques.py (Regularization from scratch)
│   └── project_overfit_detector.py (Detect and fix overfitting)
│
├── chapter_05_cnns/
│   ├── README.md (Theory: Convolution, pooling, CNN architectures)
│   ├── cnn_from_scratch.py (CNN in NumPy and PyTorch)
│   └── project_cifar10_classifier.py (CIFAR-10 image classification)
│
├── chapter_06_advanced_cnns/
│   ├── README.md (Theory: ResNet, DenseNet, transfer learning)
│   ├── advanced_cnn_architectures.py (ResNet implementation)
│   └── project_transfer_learning.py (Fine-tune pre-trained models)
│
├── chapter_07_rnns/
│   ├── README.md (Theory: RNNs, LSTM, GRU, BPTT)
│   ├── rnn_from_scratch.py (RNN/LSTM from scratch)
│   └── project_text_generation.py (Character-level language model)
│
└── chapter_08_advanced_rnns/
    ├── README.md (Theory: Seq2Seq, attention, bidirectional RNNs)
    ├── seq2seq_attention.py (Encoder-decoder with attention)
    └── project_neural_machine_translation.py (English→French translation)
```

---

## 💪 Let's Build Your Foundation!

**Start with Chapter 01**: `chapter_01_neural_networks/`

Remember: **Deep learning is a superpower you can learn**. Every expert started exactly where you are now.

The difference? They kept going. 🚀

---

*"There is nothing more powerful than a properly trained neural network."* - Andrew Ng (paraphrased)

