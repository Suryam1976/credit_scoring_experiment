#!/usr/bin/env python3
"""
Temperature Scaling Demo Script
==============================

This script demonstrates how temperature scaling works with a simple example
using the credit scoring experiment framework.

Usage:
    python temperature_scaling_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add models directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'models'))

def demonstrate_temperature_effect():
    """
    Show how different temperatures affect probability distributions
    """
    print("🌡️ Temperature Scaling Demonstration")
    print("=" * 50)
    
    # Simulate overconfident model logits (like XGBoost)
    np.random.seed(42)
    logits = np.array([-4.2, -2.1, -0.5, 0.8, 2.3, 3.9, 5.1])
    
    # Different temperature values
    temperatures = [0.5, 1.0, 2.0, 4.0]
    
    print("Original Logits:", logits)
    print("\nEffect of Different Temperatures:")
    print("-" * 40)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Probability curves
    logit_range = np.linspace(-6, 6, 100)
    
    for i, T in enumerate(temperatures):
        # Calculate probabilities for this temperature
        probs = 1 / (1 + np.exp(-logits / T))
        
        # Calculate probability curve
        prob_curve = 1 / (1 + np.exp(-logit_range / T))
        
        print(f"T = {T:3.1f}: {probs}")
        
        # Plot the curve
        ax1.plot(logit_range, prob_curve, label=f'T = {T}', linewidth=2)
        
        # Plot the specific points
        ax2.bar(np.arange(len(logits)) + i*0.15, probs, width=0.15, 
                alpha=0.7, label=f'T = {T}')
    
    ax1.set_xlabel('Logits')
    ax1.set_ylabel('Probability')
    ax1.set_title('Temperature Effect on Sigmoid Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Probability')
    ax2.set_title('Temperature Effect on Sample Predictions')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/visualizations/temperature_scaling_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Key Insights:")
    print("• T < 1.0: Makes model MORE confident (sharper)")
    print("• T = 1.0: Original model (no change)")  
    print("• T > 1.0: Makes model LESS confident (smoother)")
    print("• Higher T values spread probabilities toward 0.5")

def simulate_calibration_improvement():
    """
    Simulate how temperature scaling improves calibration
    """
    print("\n🎯 Calibration Improvement Simulation")
    print("=" * 50)
    
    # Simulate overconfident model (like Random Forest)
    np.random.seed(123)
    n_samples = 1000
    
    # Generate true probabilities (well-distributed)
    true_probs = np.random.beta(2, 2, n_samples)  # Nice distribution
    
    # Generate actual outcomes
    y_true = np.random.binomial(1, true_probs)
    
    # Simulate overconfident model logits
    # Overconfident model: amplifies the true logits
    true_logits = np.log(true_probs / (1 - true_probs + 1e-15))
    overconfident_logits = true_logits * 2.5  # Make it overconfident
    
    # Original (overconfident) probabilities
    original_probs = 1 / (1 + np.exp(-overconfident_logits))
    
    # Find optimal temperature (simplified)
    def ece_loss(temperature):
        calibrated_probs = 1 / (1 + np.exp(-overconfident_logits / temperature))
        
        # Calculate ECE
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (calibrated_probs > bin_boundaries[i]) & (calibrated_probs <= bin_boundaries[i+1])
            if mask.sum() > 0:
                bin_accuracy = y_true[mask].mean()
                bin_confidence = calibrated_probs[mask].mean()
                bin_prop = mask.mean()
                ece += abs(bin_confidence - bin_accuracy) * bin_prop
        
        return ece
    
    # Find best temperature
    temperatures = np.linspace(0.5, 5.0, 50)
    eces = [ece_loss(t) for t in temperatures]
    optimal_temp = temperatures[np.argmin(eces)]
    
    # Apply optimal temperature
    calibrated_probs = 1 / (1 + np.exp(-overconfident_logits / optimal_temp))
    
    # Calculate improvements
    original_ece = ece_loss(1.0)
    calibrated_ece = ece_loss(optimal_temp)
    improvement = (original_ece - calibrated_ece) / original_ece * 100
    
    print(f"Original ECE (overconfident): {original_ece:.4f}")
    print(f"Optimal Temperature: {optimal_temp:.2f}")
    print(f"Calibrated ECE: {calibrated_ece:.4f}")
    print(f"ECE Improvement: {improvement:.1f}%")
    
    # Visualize probability distributions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original distribution
    axes[0].hist(original_probs, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[0].set_title(f'Original (Overconfident)\nECE = {original_ece:.4f}')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Frequency')
    
    # Calibrated distribution
    axes[1].hist(calibrated_probs, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[1].set_title(f'Temperature Scaled (T={optimal_temp:.2f})\nECE = {calibrated_ece:.4f}')
    axes[1].set_xlabel('Predicted Probability')
    axes[1].set_ylabel('Frequency')
    
    # ECE vs Temperature curve
    axes[2].plot(temperatures, eces, linewidth=2, color='blue')
    axes[2].axvline(optimal_temp, color='red', linestyle='--', linewidth=2, label=f'Optimal T = {optimal_temp:.2f}')
    axes[2].set_title('ECE vs Temperature')
    axes[2].set_xlabel('Temperature')
    axes[2].set_ylabel('Expected Calibration Error')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/visualizations/calibration_improvement_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Run the complete temperature scaling demonstration
    """
    print("🚀 Temperature Scaling Interactive Demo")
    print("=" * 60)
    
    # Create visualizations directory if it doesn't exist
    Path('../results/visualizations').mkdir(parents=True, exist_ok=True)
    
    # Run demonstrations
    demonstrate_temperature_effect()
    simulate_calibration_improvement()
    
    print("\n🎉 Demo completed!")
    print("📁 Visualizations saved to: ../results/visualizations/")
    print("\n💡 Key Takeaways:")
    print("• Temperature scaling is simple but very effective")
    print("• It's particularly good for overconfident models (GBDTs)")
    print("• Only requires one parameter to learn")
    print("• Can achieve dramatic calibration improvements")
    print("• Essential tool for production ML systems")

if __name__ == "__main__":
    main()
