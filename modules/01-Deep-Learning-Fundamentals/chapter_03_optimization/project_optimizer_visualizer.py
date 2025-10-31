"""
PROJECT: INTERACTIVE OPTIMIZER VISUALIZER
==========================================

This project visualizes how different optimizers navigate loss landscapes!

You'll see:
- 2D loss landscape visualization
- Animated optimization paths
- Side-by-side comparison of all optimizers
- How each optimizer handles different landscape types
- Real-time parameter updates

This helps build intuition for why Adam is so popular!

File structure:
---------------
Part 1: Loss landscape functions
Part 2: Optimizer implementations (simplified for visualization)
Part 3: Path tracking and animation
Part 4: Interactive visualization
Part 5: Main execution

Author: Deep Learning Master Course
Purpose: Visual understanding of optimization algorithms
"""

# ============================================================================
# PART 1: IMPORTS AND LOSS LANDSCAPES
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List, Callable
import time

# Set style
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')

print("=" * 70)
print("INTERACTIVE OPTIMIZER VISUALIZER - Chapter 03 Project")
print("=" * 70)
print("\nVisualizing optimization algorithms on 2D loss landscapes!")
print("=" * 70 + "\n")


# ============================================================================
# LOSS LANDSCAPE FUNCTIONS
# ============================================================================

def rosenbrock(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Rosenbrock function: f(x,y) = (a-x)² + b(y-x²)²
    
    Classic optimization test function with:
    - Global minimum at (1, 1)
    - Narrow valley (hard to optimize)
    - Tests optimizer's ability to handle ravines
    
    This is a HARD function for optimizers!
    """
    a = 1.0
    b = 100.0
    return (a - x)**2 + b * (y - x**2)**2


def beale(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Beale function: f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    
    Features:
    - Global minimum at (3, 0.5)
    - Flat regions and steep regions
    - Multiple local minima
    - Tests escaping plateaus
    """
    return ((1.5 - x + x*y)**2 + 
            (2.25 - x + x*y**2)**2 + 
            (2.625 - x + x*y**3)**2)


def booth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Booth function: f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
    
    Features:
    - Global minimum at (1, 3)
    - Relatively simple (good for testing)
    - Convex function
    """
    return (x + 2*y - 7)**2 + (2*x + y - 5)**2


def sphere(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Sphere function: f(x,y) = x² + y²
    
    The simplest test function:
    - Global minimum at (0, 0)
    - Perfectly convex
    - All directions equally important
    """
    return x**2 + y**2


def himmelblau(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Himmelblau function: f(x,y) = (x²+y-11)² + (x+y²-7)²
    
    Features:
    - Four identical local minima (multi-modal)
    - Tests optimizer's starting point sensitivity
    - Symmetric structure
    """
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2


# Dictionary of available loss functions
LOSS_FUNCTIONS = {
    'rosenbrock': {
        'func': rosenbrock,
        'bounds': (-2, 2, -1, 3),
        'start': (-1.5, -0.5),
        'title': 'Rosenbrock Function'
    },
    'beale': {
        'func': beale,
        'bounds': (-4.5, 4.5, -4.5, 4.5),
        'start': (3.0, 3.0),
        'title': 'Beale Function'
    },
    'booth': {
        'func': booth,
        'bounds': (-10, 10, -10, 10),
        'start': (-5, -5),
        'title': 'Booth Function'
    },
    'sphere': {
        'func': sphere,
        'bounds': (-5, 5, -5, 5),
        'start': (4, 4),
        'title': 'Sphere Function'
    },
    'himmelblau': {
        'func': himmelblau,
        'bounds': (-5, 5, -5, 5),
        'start': (-3, -3),
        'title': 'Himmelblau Function'
    }
}


# ============================================================================
# PART 2: SIMPLIFIED OPTIMIZERS FOR VISUALIZATION
# ============================================================================

class VisualizationOptimizer:
    """Base class for visualization optimizers"""
    
    def __init__(self, learning_rate: float = 0.01):
        self.lr = learning_rate
        self.path = []  # Track (x, y, loss) at each step
    
    def compute_gradient(self, func: Callable, x: float, y: float, epsilon: float = 1e-5) -> Tuple[float, float]:
        """
        Compute numerical gradient using finite differences.
        
        Formula:
            ∂f/∂x ≈ [f(x+ε, y) - f(x-ε, y)] / (2ε)
            ∂f/∂y ≈ [f(x, y+ε) - f(x, y-ε)] / (2ε)
        """
        # Gradient w.r.t. x
        dx = (func(x + epsilon, y) - func(x - epsilon, y)) / (2 * epsilon)
        
        # Gradient w.r.t. y
        dy = (func(x, y + epsilon) - func(x, y - epsilon)) / (2 * epsilon)
        
        return dx, dy
    
    def step(self, func: Callable, x: float, y: float) -> Tuple[float, float]:
        """Take one optimization step"""
        raise NotImplementedError


class SGDVisualizer(VisualizationOptimizer):
    """Vanilla SGD for visualization"""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.name = "SGD"
    
    def step(self, func: Callable, x: float, y: float) -> Tuple[float, float]:
        """Update: x_new = x - lr * gradient"""
        dx, dy = self.compute_gradient(func, x, y)
        x_new = x - self.lr * dx
        y_new = y - self.lr * dy
        return x_new, y_new


class MomentumVisualizer(VisualizationOptimizer):
    """SGD with Momentum for visualization"""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.vx = 0.0  # Velocity in x direction
        self.vy = 0.0  # Velocity in y direction
        self.name = "Momentum"
    
    def step(self, func: Callable, x: float, y: float) -> Tuple[float, float]:
        """Update with momentum"""
        dx, dy = self.compute_gradient(func, x, y)
        
        # Update velocity
        self.vx = self.momentum * self.vx + (1 - self.momentum) * dx
        self.vy = self.momentum * self.vy + (1 - self.momentum) * dy
        
        # Update position using velocity
        x_new = x - self.lr * self.vx
        y_new = y - self.lr * self.vy
        
        return x_new, y_new


class RMSpropVisualizer(VisualizationOptimizer):
    """RMSprop for visualization"""
    
    def __init__(self, learning_rate: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.sx = 0.0  # Squared gradient accumulator for x
        self.sy = 0.0  # Squared gradient accumulator for y
        self.name = "RMSprop"
    
    def step(self, func: Callable, x: float, y: float) -> Tuple[float, float]:
        """Update with adaptive learning rate"""
        dx, dy = self.compute_gradient(func, x, y)
        
        # Update squared gradient moving average
        self.sx = self.beta * self.sx + (1 - self.beta) * dx**2
        self.sy = self.beta * self.sy + (1 - self.beta) * dy**2
        
        # Adaptive learning rate
        x_new = x - (self.lr / (np.sqrt(self.sx) + self.epsilon)) * dx
        y_new = y - (self.lr / (np.sqrt(self.sy) + self.epsilon)) * dy
        
        return x_new, y_new


class AdamVisualizer(VisualizationOptimizer):
    """Adam for visualization"""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.mx = 0.0  # First moment for x
        self.my = 0.0  # First moment for y
        self.vx = 0.0  # Second moment for x
        self.vy = 0.0  # Second moment for y
        self.t = 0     # Time step
        self.name = "Adam"
    
    def step(self, func: Callable, x: float, y: float) -> Tuple[float, float]:
        """Update with Adam"""
        dx, dy = self.compute_gradient(func, x, y)
        self.t += 1
        
        # Update biased first moment
        self.mx = self.beta1 * self.mx + (1 - self.beta1) * dx
        self.my = self.beta1 * self.my + (1 - self.beta1) * dy
        
        # Update biased second moment
        self.vx = self.beta2 * self.vx + (1 - self.beta2) * dx**2
        self.vy = self.beta2 * self.vy + (1 - self.beta2) * dy**2
        
        # Bias correction
        mx_hat = self.mx / (1 - self.beta1**self.t)
        my_hat = self.my / (1 - self.beta1**self.t)
        vx_hat = self.vx / (1 - self.beta2**self.t)
        vy_hat = self.vy / (1 - self.beta2**self.t)
        
        # Update position
        x_new = x - (self.lr / (np.sqrt(vx_hat) + self.epsilon)) * mx_hat
        y_new = y - (self.lr / (np.sqrt(vy_hat) + self.epsilon)) * my_hat
        
        return x_new, y_new


# ============================================================================
# PART 3: OPTIMIZATION PATH TRACKING
# ============================================================================

def optimize_with_path(
    optimizer: VisualizationOptimizer,
    func: Callable,
    start: Tuple[float, float],
    max_steps: int = 200,
    tolerance: float = 1e-6
) -> List[Tuple[float, float, float]]:
    """
    Run optimizer and track full path.
    
    Returns:
        List of (x, y, loss) tuples showing optimization trajectory
    """
    x, y = start
    path = [(x, y, func(x, y))]
    
    for step in range(max_steps):
        # Take optimization step
        x_new, y_new = optimizer.step(func, x, y)
        loss_new = func(x_new, y_new)
        
        # Record path
        path.append((x_new, y_new, loss_new))
        
        # Check convergence
        if abs(x_new - x) < tolerance and abs(y_new - y) < tolerance:
            break
        
        x, y = x_new, y_new
    
    return path


# ============================================================================
# PART 4: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_3d_landscape(landscape_name: str = 'rosenbrock'):
    """Plot 3D visualization of loss landscape"""
    config = LOSS_FUNCTIONS[landscape_name]
    func = config['func']
    x_min, x_max, y_min, y_max = config['bounds']
    
    # Create mesh
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    # Plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('f(x, y)', fontsize=12)
    ax.set_title(f'{config["title"]} - 3D View', fontsize=14, fontweight='bold')
    
    plt.colorbar(surf, ax=ax, shrink=0.5)
    plt.show()


def plot_contour_with_paths(
    landscape_name: str = 'rosenbrock',
    learning_rates: Dict[str, float] = None
):
    """
    Plot contour map with optimization paths for all optimizers.
    
    This is the KEY visualization showing how different optimizers navigate!
    """
    print(f"\nOptimizing on {landscape_name.capitalize()} function...")
    print("-" * 70)
    
    config = LOSS_FUNCTIONS[landscape_name]
    func = config['func']
    x_min, x_max, y_min, y_max = config['bounds']
    start = config['start']
    
    # Default learning rates
    if learning_rates is None:
        learning_rates = {
            'sgd': 0.001,
            'momentum': 0.001,
            'rmsprop': 0.01,
            'adam': 0.01
        }
    
    # Create mesh for contour plot
    x = np.linspace(x_min, x_max, 200)
    y = np.linspace(y_min, y_max, 200)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    
    # Create optimizers
    optimizers = [
        SGDVisualizer(learning_rates['sgd']),
        MomentumVisualizer(learning_rates['momentum']),
        RMSpropVisualizer(learning_rates['rmsprop']),
        AdamVisualizer(learning_rates['adam'])
    ]
    
    # Optimize with each
    paths = {}
    for opt in optimizers:
        path = optimize_with_path(opt, func, start, max_steps=300)
        paths[opt.name] = path
        print(f"{opt.name:10s}: {len(path):3d} steps, "
              f"final loss = {path[-1][2]:.6f}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Contour plot (use log scale for better visualization)
    contour = ax.contour(X, Y, np.log10(Z + 1), levels=30, cmap='gray', alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8)
    
    # Plot starting point
    ax.plot(start[0], start[1], 'k*', markersize=20, label='Start', zorder=5)
    
    # Plot optimization paths
    colors = ['red', 'blue', 'green', 'orange']
    for (name, path), color in zip(paths.items(), colors):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        
        # Plot path
        ax.plot(xs, ys, color=color, linewidth=2, alpha=0.7, label=name)
        
        # Plot end point
        ax.plot(xs[-1], ys[-1], 'o', color=color, markersize=10, 
               markeredgecolor='black', markeredgewidth=2, zorder=4)
        
        # Add arrows to show direction
        for i in range(0, len(xs)-1, max(1, len(xs)//10)):
            dx = xs[i+1] - xs[i]
            dy = ys[i+1] - ys[i]
            ax.arrow(xs[i], ys[i], dx, dy, head_width=0.1, head_length=0.1,
                    fc=color, ec=color, alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title(f'{config["title"]} - Optimizer Comparison', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(f'optimizer_paths_{landscape_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_convergence_comparison(landscape_name: str = 'rosenbrock'):
    """Plot loss vs steps for all optimizers"""
    config = LOSS_FUNCTIONS[landscape_name]
    func = config['func']
    start = config['start']
    
    # Learning rates
    learning_rates = {
        'sgd': 0.001,
        'momentum': 0.001,
        'rmsprop': 0.01,
        'adam': 0.01
    }
    
    # Create optimizers
    optimizers = [
        SGDVisualizer(learning_rates['sgd']),
        MomentumVisualizer(learning_rates['momentum']),
        RMSpropVisualizer(learning_rates['rmsprop']),
        AdamVisualizer(learning_rates['adam'])
    ]
    
    # Optimize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['red', 'blue', 'green', 'orange']
    
    for opt, color in zip(optimizers, colors):
        path = optimize_with_path(opt, func, start, max_steps=200)
        
        steps = list(range(len(path)))
        losses = [p[2] for p in path]
        
        # Loss plot (log scale)
        ax1.semilogy(steps, losses, color=color, linewidth=2, label=opt.name)
        
        # Loss plot (linear scale - first 50 steps)
        if len(steps) > 50:
            ax2.plot(steps[:50], losses[:50], color=color, linewidth=2, label=opt.name)
        else:
            ax2.plot(steps, losses, color=color, linewidth=2, label=opt.name)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss (log scale)', fontsize=12)
    ax1.set_title('Convergence Speed - Full Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Convergence Speed - First 50 Steps', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'convergence_comparison_{landscape_name}.png', dpi=150, bbox_inches='tight')
    plt.show()


def compare_all_landscapes():
    """Compare optimizers on all landscapes side-by-side"""
    landscapes = ['sphere', 'booth', 'rosenbrock', 'beale', 'himmelblau']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()
    
    for idx, landscape_name in enumerate(landscapes):
        ax = axes[idx]
        
        config = LOSS_FUNCTIONS[landscape_name]
        func = config['func']
        x_min, x_max, y_min, y_max = config['bounds']
        start = config['start']
        
        # Create mesh
        x = np.linspace(x_min, x_max, 150)
        y = np.linspace(y_min, y_max, 150)
        X, Y = np.meshgrid(x, y)
        Z = func(X, Y)
        
        # Contour
        ax.contour(X, Y, np.log10(Z + 1), levels=20, cmap='gray', alpha=0.3)
        
        # Optimize with Adam (best overall)
        opt = AdamVisualizer(0.01)
        path = optimize_with_path(opt, func, start, max_steps=200)
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        
        # Plot path
        ax.plot(start[0], start[1], 'r*', markersize=15, label='Start')
        ax.plot(xs, ys, 'b-', linewidth=2, alpha=0.7, label=f'Adam ({len(path)} steps)')
        ax.plot(xs[-1], ys[-1], 'go', markersize=10, label='End')
        
        ax.set_title(config['title'], fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('all_landscapes_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# PART 5: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution with interactive exploration"""
    print("\n" + "=" * 70)
    print("OPTIMIZER VISUALIZATION PROJECT")
    print("=" * 70)
    print("\nExploring how different optimizers navigate loss landscapes!")
    print("=" * 70 + "\n")
    
    # 1. Show 3D landscape
    print("1. Visualizing 3D loss landscape...")
    plot_3d_landscape('rosenbrock')
    
    # 2. Compare optimizers on Rosenbrock (hard problem)
    print("\n2. Comparing optimizers on challenging Rosenbrock function...")
    plot_contour_with_paths('rosenbrock')
    plot_convergence_comparison('rosenbrock')
    
    # 3. Compare on simpler function
    print("\n3. Comparing optimizers on simpler Booth function...")
    plot_contour_with_paths('booth')
    
    # 4. Multi-modal function
    print("\n4. Comparing optimizers on multi-modal Himmelblau function...")
    plot_contour_with_paths('himmelblau')
    
    # 5. All landscapes comparison
    print("\n5. Adam's performance across all landscapes...")
    compare_all_landscapes()
    
    # Summary
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print("\nKey Observations:")
    print("\n1. ROSENBROCK (narrow valley):")
    print("   - SGD: Oscillates and makes slow progress")
    print("   - Momentum: Overshoots but eventually converges")
    print("   - RMSprop: Adapts step size, smooth convergence")
    print("   - Adam: Best - combines benefits of both")
    
    print("\n2. CONVERGENCE SPEED:")
    print("   - Adam converges fastest (fewest steps)")
    print("   - RMSprop is second best")
    print("   - Momentum helps SGD significantly")
    print("   - Vanilla SGD is slowest")
    
    print("\n3. ADAPTIVE LEARNING RATES:")
    print("   - RMSprop and Adam handle different scales well")
    print("   - They take large steps in flat regions")
    print("   - They take small steps in steep regions")
    print("   - This automatic adaptation is very powerful!")
    
    print("\n4. PRACTICAL TAKEAWAY:")
    print("   - Start with Adam (lr=0.001) for most problems")
    print("   - If you have time to tune: try SGD+Momentum with LR schedule")
    print("   - Adam is more robust to hyperparameter choices")
    
    print("\n" + "=" * 70)
    print("You now understand WHY Adam is the default choice!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
