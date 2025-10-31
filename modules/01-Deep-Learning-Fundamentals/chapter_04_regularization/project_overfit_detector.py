"""
PROJECT: Overfit Detector and Regularization Recommender
=========================================================

This project builds an intelligent system that:
1. Trains multiple neural networks with different regularization strategies
2. Automatically detects overfitting by analyzing learning curves
3. Compares regularization techniques visually
4. Recommends which techniques to use for a given dataset
5. Generates comprehensive diagnostic reports

Learning Goals:
- Understand how to detect overfitting programmatically
- Compare regularization techniques systematically
- Learn hyperparameter tuning strategies
- Practice model selection and evaluation

Author: Teaching by Doing AI Course
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.datasets import make_classification, make_moons, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our regularization techniques
from regularization_techniques import RegularizedNeuralNetwork

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

# Random seed for reproducibility
np.random.seed(42)


# ============================================================================
# OVERFITTING DETECTION ALGORITHMS
# ============================================================================

class OverfitDetector:
    """
    Automated overfitting detection system.
    
    This class analyzes training history and detects signs of overfitting:
    1. Train-validation gap analysis
    2. Validation loss trend analysis
    3. Early stopping point identification
    4. Severity classification (mild, moderate, severe)
    
    Methods:
        detect_overfitting: Main detection algorithm
        compute_gap_score: Quantify train-val performance gap
        analyze_val_curve: Check if validation loss is increasing
        classify_severity: Categorize overfitting severity
        generate_report: Create human-readable diagnostic report
    """
    
    def __init__(self, threshold_gap: float = 0.1, threshold_trend: int = 5):
        """
        Initialize overfit detector with detection thresholds.
        
        Args:
            threshold_gap: Maximum acceptable train-val gap (default: 0.1 = 10%)
            threshold_trend: Number of epochs for trend analysis (default: 5)
            
        Example thresholds:
        - threshold_gap = 0.05: Strict (detect small gaps)
        - threshold_gap = 0.15: Lenient (only detect large gaps)
        - threshold_trend = 3: Quick detection (may be noisy)
        - threshold_trend = 10: Robust detection (slower)
        """
        self.threshold_gap = threshold_gap
        self.threshold_trend = threshold_trend
    
    def detect_overfitting(self, history: Dict) -> Dict:
        """
        Main overfitting detection algorithm.
        
        Analyzes training history using multiple signals:
        1. Final gap: train_loss - val_loss at end of training
        2. Maximum gap: largest train_loss - val_loss ever observed
        3. Val trend: is validation loss increasing?
        4. Gap trend: is the gap widening over time?
        
        Args:
            history: Training history dict with keys:
                - 'train_loss': List of training losses
                - 'val_loss': List of validation losses
                - 'train_acc': List of training accuracies
                - 'val_acc': List of validation accuracies
                
        Returns:
            Detection report dict with:
                - 'is_overfitting': Boolean (True = overfitting detected)
                - 'severity': String ('none', 'mild', 'moderate', 'severe')
                - 'gap_score': Float (quantified train-val gap)
                - 'best_epoch': Int (when to stop training)
                - 'signals': Dict (detailed diagnostic signals)
                - 'recommendations': List[str] (what to do about it)
        """
        train_loss = np.array(history['train_loss'])
        val_loss = np.array(history['val_loss'])
        train_acc = np.array(history['train_acc'])
        val_acc = np.array(history['val_acc'])
        
        # Initialize report
        report = {
            'is_overfitting': False,
            'severity': 'none',
            'gap_score': 0.0,
            'best_epoch': 0,
            'signals': {},
            'recommendations': []
        }
        
        # ===== Signal 1: Final gap analysis =====
        # Compare final train vs validation performance
        # Large gap indicates overfitting
        
        final_loss_gap = train_loss[-1] - val_loss[-1]
        final_acc_gap = train_acc[-1] - val_acc[-1]
        
        report['signals']['final_loss_gap'] = final_loss_gap
        report['signals']['final_acc_gap'] = final_acc_gap
        
        # Check if gap exceeds threshold
        # Negative gap is OK (val sometimes better than train)
        if final_loss_gap < -self.threshold_gap:
            report['signals']['final_gap_check'] = 'PASS'
        elif final_loss_gap < self.threshold_gap:
            report['signals']['final_gap_check'] = 'WARN'
        else:
            report['signals']['final_gap_check'] = 'FAIL'
            report['is_overfitting'] = True
        
        # ===== Signal 2: Maximum gap analysis =====
        # Check largest gap ever observed during training
        # Even if final gap is small, large historical gap indicates problems
        
        loss_gaps = train_loss - val_loss
        max_loss_gap = np.max(loss_gaps)
        max_gap_epoch = np.argmax(loss_gaps)
        
        report['signals']['max_loss_gap'] = max_loss_gap
        report['signals']['max_gap_epoch'] = max_gap_epoch
        
        if max_loss_gap > 2 * self.threshold_gap:
            report['signals']['max_gap_check'] = 'FAIL'
            report['is_overfitting'] = True
        elif max_loss_gap > self.threshold_gap:
            report['signals']['max_gap_check'] = 'WARN'
        else:
            report['signals']['max_gap_check'] = 'PASS'
        
        # ===== Signal 3: Validation loss trend =====
        # Is validation loss increasing over last N epochs?
        # This is classic overfitting: train improves, val degrades
        
        if len(val_loss) >= self.threshold_trend:
            recent_val = val_loss[-self.threshold_trend:]
            val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
            
            report['signals']['val_loss_trend'] = val_trend
            
            # Positive trend = validation loss increasing = BAD
            if val_trend > 0.001:
                report['signals']['val_trend_check'] = 'FAIL'
                report['is_overfitting'] = True
            elif val_trend > 0.0001:
                report['signals']['val_trend_check'] = 'WARN'
            else:
                report['signals']['val_trend_check'] = 'PASS'
        
        # ===== Signal 4: Gap widening trend =====
        # Is train-val gap increasing over time?
        # Even if both losses decrease, widening gap = overfitting
        
        if len(loss_gaps) >= self.threshold_trend:
            recent_gaps = loss_gaps[-self.threshold_trend:]
            gap_trend = np.polyfit(range(len(recent_gaps)), recent_gaps, 1)[0]
            
            report['signals']['gap_trend'] = gap_trend
            
            # Positive trend = gap widening = BAD
            if gap_trend > 0.005:
                report['signals']['gap_trend_check'] = 'FAIL'
                report['is_overfitting'] = True
            elif gap_trend > 0.001:
                report['signals']['gap_trend_check'] = 'WARN'
            else:
                report['signals']['gap_trend_check'] = 'PASS'
        
        # ===== Compute gap score =====
        # Unified metric combining all signals
        # Higher score = more severe overfitting
        
        gap_score = 0.0
        gap_score += max(0, final_loss_gap) * 10  # Final gap (weight: 10x)
        gap_score += max(0, max_loss_gap) * 5     # Max gap (weight: 5x)
        gap_score += max(0, report['signals'].get('val_loss_trend', 0)) * 100  # Val trend (weight: 100x)
        gap_score += max(0, report['signals'].get('gap_trend', 0)) * 50  # Gap trend (weight: 50x)
        
        report['gap_score'] = gap_score
        
        # ===== Classify severity =====
        # Categorize overfitting into levels
        
        if gap_score < 0.5:
            report['severity'] = 'none'
        elif gap_score < 1.0:
            report['severity'] = 'mild'
        elif gap_score < 2.0:
            report['severity'] = 'moderate'
        else:
            report['severity'] = 'severe'
        
        # ===== Find best epoch =====
        # When should training have stopped?
        # Use validation loss minimum
        
        best_epoch = np.argmin(val_loss)
        report['best_epoch'] = best_epoch
        
        # ===== Generate recommendations =====
        
        if report['severity'] == 'none':
            report['recommendations'].append("✓ No overfitting detected. Model is well-regularized.")
            report['recommendations'].append("  Consider: Increasing model capacity or training longer.")
        
        elif report['severity'] == 'mild':
            report['recommendations'].append("⚠ Mild overfitting detected.")
            report['recommendations'].append("  1. Use early stopping (stop at epoch {})".format(best_epoch))
            report['recommendations'].append("  2. Slight increase in L2 regularization (e.g., λ × 1.5)")
            report['recommendations'].append("  3. Add light dropout (p=0.2-0.3) if not already present")
        
        elif report['severity'] == 'moderate':
            report['recommendations'].append("⚠⚠ Moderate overfitting detected.")
            report['recommendations'].append("  1. Definitely use early stopping (stop at epoch {})".format(best_epoch))
            report['recommendations'].append("  2. Increase L2 regularization significantly (e.g., λ × 3-5)")
            report['recommendations'].append("  3. Add/increase dropout (p=0.4-0.5)")
            report['recommendations'].append("  4. Consider batch normalization")
            report['recommendations'].append("  5. Get more training data if possible")
        
        else:  # severe
            report['recommendations'].append("❌ SEVERE overfitting detected!")
            report['recommendations'].append("  1. CRITICAL: Use early stopping (stop at epoch {})".format(best_epoch))
            report['recommendations'].append("  2. Reduce model size (fewer/smaller layers)")
            report['recommendations'].append("  3. Strong L2 regularization (λ = 0.01-0.1)")
            report['recommendations'].append("  4. Heavy dropout (p=0.5-0.7)")
            report['recommendations'].append("  5. Batch normalization (essential)")
            report['recommendations'].append("  6. Data augmentation")
            report['recommendations'].append("  7. Get significantly more training data")
        
        return report
    
    def visualize_detection(self, history: Dict, report: Dict, save_path: Optional[str] = None):
        """
        Visualize overfitting detection results.
        
        Creates a comprehensive 2×2 plot:
        1. Learning curves (loss over epochs)
        2. Accuracy curves (accuracy over epochs)
        3. Train-val gap evolution
        4. Detection summary (text)
        
        Args:
            history: Training history
            report: Detection report from detect_overfitting()
            save_path: Optional path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = np.arange(1, len(history['train_loss']) + 1)
        
        # ===== Plot 1: Loss curves =====
        ax = axes[0, 0]
        ax.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2, color='blue')
        ax.plot(epochs, history['val_loss'], label='Val Loss', linewidth=2, color='orange')
        
        # Mark best epoch
        best_epoch = report['best_epoch']
        ax.axvline(best_epoch + 1, color='red', linestyle='--', 
                   label=f'Best Epoch ({best_epoch + 1})', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Learning Curves: Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 2: Accuracy curves =====
        ax = axes[0, 1]
        ax.plot(epochs, history['train_acc'], label='Train Acc', linewidth=2, color='blue')
        ax.plot(epochs, history['val_acc'], label='Val Acc', linewidth=2, color='orange')
        
        # Mark best epoch
        ax.axvline(best_epoch + 1, color='red', linestyle='--', 
                   label=f'Best Epoch ({best_epoch + 1})', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Learning Curves: Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 3: Gap evolution =====
        ax = axes[1, 0]
        
        loss_gap = np.array(history['train_loss']) - np.array(history['val_loss'])
        acc_gap = np.array(history['train_acc']) - np.array(history['val_acc'])
        
        ax.plot(epochs, loss_gap, label='Loss Gap', linewidth=2, color='red')
        ax.axhline(0, color='black', linestyle='-', linewidth=1)
        ax.axhline(self.threshold_gap, color='orange', linestyle='--', 
                   label=f'Warning Threshold ({self.threshold_gap})', linewidth=1)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Train - Val Gap', fontsize=12)
        ax.set_title('Overfitting Gap Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # ===== Plot 4: Detection summary =====
        ax = axes[1, 1]
        ax.axis('off')
        
        # Format summary text
        severity_colors = {
            'none': 'green',
            'mild': 'yellow',
            'moderate': 'orange',
            'severe': 'red'
        }
        
        summary_text = "OVERFITTING DETECTION REPORT\n"
        summary_text += "=" * 40 + "\n\n"
        summary_text += f"Severity: {report['severity'].upper()}\n"
        summary_text += f"Gap Score: {report['gap_score']:.3f}\n"
        summary_text += f"Best Epoch: {report['best_epoch'] + 1}\n"
        summary_text += f"Early Stop Savings: {len(epochs) - report['best_epoch'] - 1} epochs\n\n"
        
        summary_text += "DIAGNOSTIC SIGNALS:\n"
        summary_text += "-" * 40 + "\n"
        for key, value in report['signals'].items():
            if isinstance(value, (int, float)):
                summary_text += f"  {key}: {value:.4f}\n"
            else:
                summary_text += f"  {key}: {value}\n"
        
        summary_text += "\nRECOMMENDATIONS:\n"
        summary_text += "-" * 40 + "\n"
        for rec in report['recommendations']:
            summary_text += f"{rec}\n"
        
        # Display text
        ax.text(0.05, 0.95, summary_text, 
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor=severity_colors[report['severity']], alpha=0.3))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()


# ============================================================================
# REGULARIZATION COMPARISON FRAMEWORK
# ============================================================================

class RegularizationComparison:
    """
    Systematic comparison of regularization techniques.
    
    This class trains multiple models with different regularization
    strategies and compares their performance.
    
    Configurations tested:
    1. Baseline (no regularization)
    2. L1 only
    3. L2 only
    4. Dropout only
    5. Batch Normalization only
    6. L2 + Dropout
    7. L2 + Batch Norm
    8. L2 + Dropout + Batch Norm (full)
    
    Methods:
        run_comparison: Train all configurations
        visualize_comparison: Plot comparative results
        recommend_best: Suggest best configuration
    """
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        layer_sizes: List[int]
    ):
        """
        Initialize comparison framework with data.
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            X_test, y_test: Test data
            layer_sizes: Network architecture [input, hidden1, ..., output]
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.layer_sizes = layer_sizes
        
        # Storage for results
        self.models = {}
        self.histories = {}
        self.test_metrics = {}
    
    def run_comparison(
        self,
        epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 0.01,
        verbose: bool = True
    ):
        """
        Run comprehensive comparison of all regularization strategies.
        
        Trains 8 different model configurations and stores results.
        
        Args:
            epochs: Maximum training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate
            verbose: Print progress
        """
        # Define configurations to test
        configs = {
            '1. Baseline (No Reg)': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.0,
                'dropout_rate': 0.0,
                'use_batch_norm': False
            },
            '2. L1 Only': {
                'l1_lambda': 0.001,
                'l2_lambda': 0.0,
                'dropout_rate': 0.0,
                'use_batch_norm': False
            },
            '3. L2 Only': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.01,
                'dropout_rate': 0.0,
                'use_batch_norm': False
            },
            '4. Dropout Only': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.0,
                'dropout_rate': 0.5,
                'use_batch_norm': False
            },
            '5. BatchNorm Only': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.0,
                'dropout_rate': 0.0,
                'use_batch_norm': True
            },
            '6. L2 + Dropout': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.01,
                'dropout_rate': 0.5,
                'use_batch_norm': False
            },
            '7. L2 + BatchNorm': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.01,
                'dropout_rate': 0.0,
                'use_batch_norm': True
            },
            '8. Full (L2+Drop+BN)': {
                'l1_lambda': 0.0,
                'l2_lambda': 0.01,
                'dropout_rate': 0.3,
                'use_batch_norm': True
            }
        }
        
        print("=" * 70)
        print("REGULARIZATION TECHNIQUES COMPARISON")
        print("=" * 70)
        print(f"\nTraining {len(configs)} different configurations...")
        print(f"Architecture: {self.layer_sizes}")
        print(f"Training samples: {self.X_train.shape[0]}")
        print(f"Validation samples: {self.X_val.shape[0]}")
        print(f"Test samples: {self.X_test.shape[0]}")
        print()
        
        # Train each configuration
        for name, config in configs.items():
            if verbose:
                print(f"\n{'-'*70}")
                print(f"Training: {name}")
                print(f"Config: {config}")
                print(f"{'-'*70}")
            
            # Create model
            model = RegularizedNeuralNetwork(
                layer_sizes=self.layer_sizes,
                **config
            )
            
            # Train model
            history = model.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                early_stopping_patience=15,
                verbose=False
            )
            
            # Evaluate on test set
            model.eval_mode()
            test_acc = model.accuracy(self.X_test, self.y_test)
            test_loss = model.compute_loss(self.X_test, self.y_test, include_regularization=False)
            
            # Store results
            self.models[name] = model
            self.histories[name] = history
            self.test_metrics[name] = {
                'accuracy': test_acc,
                'loss': test_loss,
                'final_train_acc': history['train_acc'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_acc': max(history['val_acc']),
                'epochs_trained': len(history['train_loss'])
            }
            
            if verbose:
                print(f"\nResults:")
                print(f"  Final Train Acc: {history['train_acc'][-1]:.4f}")
                print(f"  Final Val Acc: {history['val_acc'][-1]:.4f}")
                print(f"  Test Acc: {test_acc:.4f}")
                print(f"  Test Loss: {test_loss:.4f}")
                print(f"  Epochs: {len(history['train_loss'])}")
        
        if verbose:
            print("\n" + "=" * 70)
            print("COMPARISON COMPLETE!")
            print("=" * 70)
    
    def visualize_comparison(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization comparing all models.
        
        Generates 4 plots:
        1. Test accuracy comparison (bar chart)
        2. Learning curves (all models overlaid)
        3. Train-val gap comparison
        4. Training efficiency (epochs needed)
        
        Args:
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Extract data
        names = list(self.models.keys())
        test_accs = [self.test_metrics[name]['accuracy'] for name in names]
        test_losses = [self.test_metrics[name]['loss'] for name in names]
        epochs_trained = [self.test_metrics[name]['epochs_trained'] for name in names]
        
        # ===== Plot 1: Test Accuracy Comparison =====
        ax1 = fig.add_subplot(gs[0, 0])
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        bars = ax1.barh(names, test_accs, color=colors)
        
        # Highlight best model
        best_idx = np.argmax(test_accs)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('black')
        bars[best_idx].set_linewidth(2)
        
        ax1.set_xlabel('Test Accuracy', fontsize=12)
        ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add values on bars
        for i, (name, acc) in enumerate(zip(names, test_accs)):
            ax1.text(acc + 0.01, i, f'{acc:.4f}', va='center', fontsize=9)
        
        # ===== Plot 2: Learning Curves =====
        ax2 = fig.add_subplot(gs[0, 1])
        
        for name, color in zip(names, colors):
            history = self.histories[name]
            epochs = np.arange(1, len(history['val_loss']) + 1)
            ax2.plot(epochs, history['val_loss'], label=name.split('.')[1].strip(), 
                    color=color, linewidth=2, alpha=0.7)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Validation Loss Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=8, loc='best')
        ax2.grid(True, alpha=0.3)
        
        # ===== Plot 3: Train-Val Gap =====
        ax3 = fig.add_subplot(gs[1, :])
        
        x_pos = np.arange(len(names))
        train_accs = [self.test_metrics[name]['final_train_acc'] for name in names]
        val_accs = [self.test_metrics[name]['final_val_acc'] for name in names]
        gaps = np.array(train_accs) - np.array(val_accs)
        
        width = 0.35
        ax3.bar(x_pos - width/2, train_accs, width, label='Train Acc', color='skyblue')
        ax3.bar(x_pos + width/2, val_accs, width, label='Val Acc', color='lightcoral')
        
        # Add gap annotations
        for i, gap in enumerate(gaps):
            color = 'red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green'
            ax3.text(i, max(train_accs[i], val_accs[i]) + 0.02, 
                    f'Gap: {gap:.3f}', ha='center', fontsize=8, color=color, fontweight='bold')
        
        ax3.set_xlabel('Model', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Train-Val Accuracy Gap (Lower Gap = Better Generalization)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([name.split('.')[1].strip() for name in names], 
                            rotation=45, ha='right')
        ax3.legend(fontsize=10)
        ax3.grid(axis='y', alpha=0.3)
        
        # ===== Plot 4: Training Efficiency =====
        ax4 = fig.add_subplot(gs[2, 0])
        
        bars = ax4.barh(names, epochs_trained, color=colors)
        
        ax4.set_xlabel('Epochs Trained', fontsize=12)
        ax4.set_title('Training Efficiency (Fewer Epochs = Faster Convergence)', 
                     fontsize=14, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
        
        # Add values
        for i, (name, ep) in enumerate(zip(names, epochs_trained)):
            ax4.text(ep + 1, i, f'{ep}', va='center', fontsize=9)
        
        # ===== Plot 5: Summary Table =====
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis('off')
        
        # Create summary table
        table_data = []
        table_data.append(['Model', 'Test Acc', 'Val Gap', 'Epochs', 'Rank'])
        
        # Rank models by test accuracy
        rankings = np.argsort(test_accs)[::-1]  # Descending order
        
        for rank, idx in enumerate(rankings):
            name = names[idx]
            metrics = self.test_metrics[name]
            gap = metrics['final_train_acc'] - metrics['final_val_acc']
            
            table_data.append([
                name.split('.')[1].strip(),
                f"{metrics['accuracy']:.4f}",
                f"{gap:.4f}",
                f"{metrics['epochs_trained']}",
                f"#{rank + 1}"
            ])
        
        # Display table
        table = ax5.table(cellText=table_data, cellLoc='left', loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            cell = table[(0, i)]
            cell.set_facecolor('#4CAF50')
            cell.set_text_props(weight='bold', color='white')
        
        # Highlight best model (rank #1)
        for i in range(5):
            cell = table[(1, i)]  # Second row = rank 1
            cell.set_facecolor('#FFD700')
            cell.set_text_props(weight='bold')
        
        plt.suptitle('COMPREHENSIVE REGULARIZATION COMPARISON', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        plt.show()
    
    def recommend_best(self) -> Dict:
        """
        Recommend best regularization strategy based on results.
        
        Returns:
            Recommendation dict with:
                - 'best_model': Name of best performing model
                - 'test_accuracy': Test accuracy of best model
                - 'reason': Why this model is best
                - 'alternatives': Other good options
        """
        # Find model with best test accuracy
        names = list(self.models.keys())
        test_accs = [self.test_metrics[name]['accuracy'] for name in names]
        best_idx = np.argmax(test_accs)
        best_name = names[best_idx]
        
        # Find models with good generalization (small train-val gap)
        gaps = []
        for name in names:
            metrics = self.test_metrics[name]
            gap = metrics['final_train_acc'] - metrics['final_val_acc']
            gaps.append(gap)
        
        # Model with smallest gap
        best_gap_idx = np.argmin(gaps)
        best_gap_name = names[best_gap_idx]
        
        # Create recommendation
        recommendation = {
            'best_overall': best_name,
            'test_accuracy': test_accs[best_idx],
            'best_generalization': best_gap_name,
            'generalization_gap': gaps[best_gap_idx],
            'summary': []
        }
        
        recommendation['summary'].append("=" * 70)
        recommendation['summary'].append("REGULARIZATION RECOMMENDATION")
        recommendation['summary'].append("=" * 70)
        recommendation['summary'].append(f"\n🏆 BEST OVERALL MODEL: {best_name}")
        recommendation['summary'].append(f"   Test Accuracy: {test_accs[best_idx]:.4f}")
        recommendation['summary'].append(f"\n🎯 BEST GENERALIZATION: {best_gap_name}")
        recommendation['summary'].append(f"   Train-Val Gap: {gaps[best_gap_idx]:.4f}")
        
        # Add insights
        recommendation['summary'].append("\n📊 KEY INSIGHTS:")
        
        baseline_acc = test_accs[0]  # First model is always baseline
        improvement = test_accs[best_idx] - baseline_acc
        
        if improvement > 0.05:
            recommendation['summary'].append(f"   ✓ Regularization helped significantly (+{improvement:.2%})")
        elif improvement > 0:
            recommendation['summary'].append(f"   ✓ Regularization helped moderately (+{improvement:.2%})")
        else:
            recommendation['summary'].append("   ⚠ Regularization didn't improve performance")
            recommendation['summary'].append("     → Model may be underfitting (needs more capacity)")
        
        # Check if combination is best
        if '8. Full' in best_name or '7. L2 + BatchNorm' in best_name or '6. L2 + Dropout' in best_name:
            recommendation['summary'].append("   ✓ Combining regularization techniques works best")
        
        # Practical advice
        recommendation['summary'].append("\n💡 PRACTICAL ADVICE:")
        recommendation['summary'].append(f"   1. Use configuration: {best_name}")
        recommendation['summary'].append(f"   2. Implement early stopping (saved {100 - self.test_metrics[best_name]['epochs_trained']} epochs)")
        recommendation['summary'].append("   3. Monitor train-val gap during training")
        recommendation['summary'].append("   4. Consider data augmentation if still overfitting")
        
        return recommendation


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """
    Main demonstration of overfit detector and regularization comparison.
    
    Workflow:
    1. Load and prepare dataset
    2. Train baseline model
    3. Detect overfitting
    4. Run comprehensive regularization comparison
    5. Recommend best strategy
    """
    print("=" * 70)
    print("PROJECT: OVERFIT DETECTOR & REGULARIZATION RECOMMENDER")
    print("=" * 70)
    
    # ===== STEP 1: Load and prepare data =====
    print("\n[1/5] Loading and preparing dataset...")
    
    # Using digits dataset (0-9 handwritten digits)
    # Alternative: use make_classification() for synthetic data
    from sklearn.datasets import load_digits
    
    digits = load_digits()
    X = digits.data / 16.0  # Normalize pixel values to [0, 1]
    y = digits.target
    
    # One-hot encode labels
    num_classes = 10
    y_one_hot = np.eye(num_classes)[y]
    
    # Split into train, validation, test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_one_hot, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, random_state=42, 
        stratify=np.argmax(y_train_val, axis=1)
    )
    
    print(f"   Dataset: MNIST Digits (8×8 images, 10 classes)")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Validation samples: {X_val.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    print(f"   Input features: {X_train.shape[1]}")
    
    # ===== STEP 2: Train baseline model (prone to overfitting) =====
    print("\n[2/5] Training baseline model (no regularization)...")
    
    baseline_model = RegularizedNeuralNetwork(
        layer_sizes=[64, 128, 128, 10],  # Large network (easy to overfit)
        l1_lambda=0.0,
        l2_lambda=0.0,
        dropout_rate=0.0,
        use_batch_norm=False
    )
    
    baseline_history = baseline_model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        early_stopping_patience=50,  # Very patient (allow overfitting)
        verbose=False
    )
    
    print(f"   Training complete!")
    print(f"   Final train accuracy: {baseline_history['train_acc'][-1]:.4f}")
    print(f"   Final validation accuracy: {baseline_history['val_acc'][-1]:.4f}")
    
    # ===== STEP 3: Detect overfitting =====
    print("\n[3/5] Running overfitting detection...")
    
    detector = OverfitDetector(threshold_gap=0.1, threshold_trend=5)
    report = detector.detect_overfitting(baseline_history)
    
    print(f"\n   Detection Results:")
    print(f"   {'='*60}")
    print(f"   Overfitting detected: {report['is_overfitting']}")
    print(f"   Severity: {report['severity'].upper()}")
    print(f"   Gap score: {report['gap_score']:.3f}")
    print(f"   Best epoch: {report['best_epoch'] + 1}")
    print(f"\n   Recommendations:")
    for rec in report['recommendations']:
        print(f"   {rec}")
    
    # Visualize detection
    print("\n   Generating detection visualization...")
    detector.visualize_detection(baseline_history, report, 
                                 save_path='overfit_detection.png')
    
    # ===== STEP 4: Run comprehensive comparison =====
    print("\n[4/5] Running comprehensive regularization comparison...")
    print("   (This may take a few minutes...)")
    
    comparison = RegularizationComparison(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        layer_sizes=[64, 128, 128, 10]
    )
    
    comparison.run_comparison(
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )
    
    # Visualize comparison
    print("\n   Generating comparison visualization...")
    comparison.visualize_comparison(save_path='regularization_comparison.png')
    
    # ===== STEP 5: Get recommendation =====
    print("\n[5/5] Generating final recommendation...")
    
    recommendation = comparison.recommend_best()
    
    print()
    for line in recommendation['summary']:
        print(line)
    
    print("\n" + "=" * 70)
    print("PROJECT COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - overfit_detection.png: Baseline model analysis")
    print("  - regularization_comparison.png: Technique comparison")
    print("\nNext steps:")
    print("  1. Try with your own dataset")
    print("  2. Experiment with different hyperparameters")
    print("  3. Implement custom regularization techniques")
    print("  4. Combine with data augmentation")
    print("=" * 70)


if __name__ == "__main__":
    main()
