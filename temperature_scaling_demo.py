#!/usr/bin/env python3
"""
Temperature Scaling Demonstration Script
======================================

This script demonstrates temperature scaling on the credit scoring experiment
and shows the dramatic calibration improvements it can achieve.

Usage:
    python temperature_scaling_demo.py
"""

import sys
import os
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def demonstrate_temperature_scaling():
    """
    Demonstrate temperature scaling with synthetic overconfident predictions
    """
    print("🌡️ Temperature Scaling Demonstration")
    print("=" * 50)
    
    # Simulate overconfident model predictions (like XGBoost)
    np.random.seed(42)
    n_samples = 1000
    
    # Create realistic but overconfident logits
    true_logits = np.random.normal(0, 1.5, n_samples)  # True underlying logits
    overconfident_logits = true_logits * 2.5  # Model is 2.5x overconfident
    
    # Generate true labels based on true logits
    true_probabilities = 1 / (1 + np.exp(-true_logits))
    y_true = np.random.binomial(1, true_probabilities)
    
    # Original overconfident predictions
    original_probs = 1 / (1 + np.exp(-overconfident_logits))
    
    # Temperature scaling
    def temperature_scaling(logits, temperature):
        return 1 / (1 + np.exp(-logits / temperature))
    
    def calculate_ece(y_true, y_prob, n_bins=10):
        """Calculate Expected Calibration Error"""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    # Test different temperatures
    temperatures = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    eces = []
    
    for T in temperatures:
        calibrated_probs = temperature_scaling(overconfident_logits, T)
        ece = calculate_ece(y_true, calibrated_probs)
        eces.append(ece)
    
    # Find optimal temperature
    optimal_idx = np.argmin(eces)
    optimal_temp = temperatures[optimal_idx]
    optimal_ece = eces[optimal_idx]
    
    # Calculate original ECE
    original_ece = calculate_ece(y_true, original_probs)
    
    print(f"📊 Results:")
    print(f"   Original ECE: {original_ece:.4f}")
    print(f"   Optimal Temperature: {optimal_temp:.1f}")
    print(f"   Calibrated ECE: {optimal_ece:.4f}")
    print(f"   Improvement: {((original_ece - optimal_ece) / original_ece * 100):.1f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: ECE vs Temperature
    axes[0, 0].plot(temperatures, eces, 'b-o', linewidth=2, markersize=8)
    axes[0, 0].axvline(optimal_temp, color='red', linestyle='--', alpha=0.7, label=f'Optimal T = {optimal_temp}')
    axes[0, 0].axhline(original_ece, color='orange', linestyle='--', alpha=0.7, label=f'Original ECE = {original_ece:.3f}')
    axes[0, 0].set_xlabel('Temperature')
    axes[0, 0].set_ylabel('Expected Calibration Error (ECE)')
    axes[0, 0].set_title('ECE vs Temperature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Probability distributions
    optimal_probs = temperature_scaling(overconfident_logits, optimal_temp)
    
    axes[0, 1].hist(original_probs, bins=30, alpha=0.6, label='Original (Overconfident)', color='red', density=True)\n    axes[0, 1].hist(optimal_probs, bins=30, alpha=0.6, label=f'Temperature Scaled (T={optimal_temp})', color='blue', density=True)\n    axes[0, 1].set_xlabel('Predicted Probability')\n    axes[0, 1].set_ylabel('Density')\n    axes[0, 1].set_title('Probability Distributions')\n    axes[0, 1].legend()\n    axes[0, 1].grid(True, alpha=0.3)\n    \n    # Plot 3: Reliability diagram - Original\n    def plot_reliability_diagram(ax, y_true, y_prob, title, color):\n        from sklearn.calibration import calibration_curve\n        \n        fraction_of_positives, mean_predicted_value = calibration_curve(\n            y_true, y_prob, n_bins=10\n        )\n        \n        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', \n                linewidth=2, markersize=8, color=color, label=title)\n        ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=2, label='Perfect Calibration')\n        ax.set_xlabel('Mean Predicted Probability')\n        ax.set_ylabel('Fraction of Positives')\n        ax.set_title(f'Reliability Diagram - {title}')\n        ax.legend()\n        ax.grid(True, alpha=0.3)\n        ax.set_xlim([0, 1])\n        ax.set_ylim([0, 1])\n    \n    plot_reliability_diagram(axes[1, 0], y_true, original_probs, 'Original', 'red')\n    plot_reliability_diagram(axes[1, 1], y_true, optimal_probs, f'Temperature Scaled (T={optimal_temp})', 'blue')\n    \n    plt.tight_layout()\n    \n    # Save plot if results directory exists\n    results_dir = Path('results/visualizations')\n    if results_dir.exists():\n        plt.savefig(results_dir / 'temperature_scaling_demo.png', dpi=300, bbox_inches='tight')\n        print(f\"📊 Visualization saved to {results_dir / 'temperature_scaling_demo.png'}\")\n    \n    plt.show()\n    \n    # Business impact simulation\n    print(\"\\n💼 Business Impact Simulation:\")\n    print(\"-\" * 30)\n    \n    # Simulate loan portfolio\n    portfolio_size = 10000\n    avg_loan_amount = 10000\n    approval_threshold = 0.3\n    \n    # Original model decisions\n    original_approved = np.sum(original_probs < approval_threshold)\n    original_approved_probs = original_probs[original_probs < approval_threshold]\n    original_approved_labels = y_true[original_probs < approval_threshold]\n    \n    if len(original_approved_probs) > 0:\n        original_predicted_default_rate = original_approved_probs.mean()\n        original_actual_default_rate = original_approved_labels.mean()\n        \n        # Temperature scaled model decisions\n        temp_approved = np.sum(optimal_probs < approval_threshold)\n        temp_approved_probs = optimal_probs[optimal_probs < approval_threshold]\n        temp_approved_labels = y_true[optimal_probs < approval_threshold]\n        \n        if len(temp_approved_probs) > 0:\n            temp_predicted_default_rate = temp_approved_probs.mean()\n            temp_actual_default_rate = temp_approved_labels.mean()\n            \n            # Calculate financial impact\n            original_volume = original_approved * avg_loan_amount\n            temp_volume = temp_approved * avg_loan_amount\n            \n            original_predicted_loss = original_predicted_default_rate * original_volume\n            original_actual_loss = original_actual_default_rate * original_volume\n            original_surprise = original_actual_loss - original_predicted_loss\n            \n            temp_predicted_loss = temp_predicted_default_rate * temp_volume\n            temp_actual_loss = temp_actual_default_rate * temp_volume\n            temp_surprise = temp_actual_loss - temp_predicted_loss\n            \n            print(f\"Original Model:\")\n            print(f\"  Loans Approved: {original_approved:,}\")\n            print(f\"  Predicted Loss: ${original_predicted_loss/1e6:.1f}M\")\n            print(f\"  Actual Loss: ${original_actual_loss/1e6:.1f}M\")\n            print(f\"  Surprise: ${original_surprise/1e6:.1f}M\")\n            \n            print(f\"\\nTemperature Scaled Model:\")\n            print(f\"  Loans Approved: {temp_approved:,}\")\n            print(f\"  Predicted Loss: ${temp_predicted_loss/1e6:.1f}M\")\n            print(f\"  Actual Loss: ${temp_actual_loss/1e6:.1f}M\")\n            print(f\"  Surprise: ${temp_surprise/1e6:.1f}M\")\n            \n            risk_reduction = original_surprise - temp_surprise\n            print(f\"\\n💰 Risk Reduction: ${risk_reduction/1e6:.1f}M\")\n            \n            if abs(risk_reduction) > 0:\n                roi = (abs(risk_reduction) / 50000) * 100  # Assuming $50K calibration cost\n                print(f\"📈 ROI of Temperature Scaling: {roi:.0f}%\")\n    \n    print(\"\\n🎯 Key Takeaways:\")\n    print(\"  • Temperature scaling can dramatically improve calibration\")\n    print(\"  • Single parameter optimization is simple but effective\")\n    print(\"  • Business impact can be substantial ($millions)\")\n    print(\"  • Works especially well for overconfident models\")\n    print(\"  • Should be standard practice for tree-based and ensemble models\")\n\ndef main():\n    \"\"\"\n    Main function\n    \"\"\"\n    try:\n        demonstrate_temperature_scaling()\n        print(\"\\n✅ Temperature scaling demonstration completed!\")\n        \n    except Exception as e:\n        print(f\"❌ Error in demonstration: {e}\")\n        raise\n\nif __name__ == \"__main__\":\n    main()