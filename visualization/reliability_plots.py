"""
Visualization Module for Credit Scoring Calibration Experiment
============================================================

This module creates comprehensive visualizations to demonstrate the difference
between accuracy and calibration in machine learning models.

Author: Credit Scoring Experiment
Date: June 2025
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# ML libraries for calibration curves
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class CalibrationVisualizer:
    """
    Creates comprehensive visualizations for calibration analysis
    """
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization subdirectory
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(exist_ok=True)
        
        # Set up plotting parameters
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        self.figsize = (12, 8)
        
    def load_results(self) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Load calibration results and comparison data
        
        Returns:
            Tuple of (calibration_results, comparison_dataframe)
        """
        print("📂 Loading calibration results...")
        
        # Load calibrated models
        calibrated_models_path = self.results_dir / "calibrated_models.pkl"
        if not calibrated_models_path.exists():
            raise FileNotFoundError("No calibrated models found. Run calibration analysis first.")
            
        with open(calibrated_models_path, 'rb') as f:
            calibration_results = pickle.load(f)
        
        # Load comparison dataframe
        comparison_path = self.results_dir / "calibration_comparison.csv"
        if not comparison_path.exists():
            raise FileNotFoundError("No calibration comparison found. Run calibration analysis first.")
            
        comparison_df = pd.read_csv(comparison_path)
        
        print("✅ Results loaded successfully")
        return calibration_results, comparison_df
    
    def load_data_splits(self) -> Dict[str, Any]:
        """
        Load data splits for generating predictions
        
        Returns:
            Dictionary containing data splits
        """
        splits_path = Path("../data/processed/train_test_split.pkl")
        if not splits_path.exists():
            raise FileNotFoundError("No data splits found.")
            
        with open(splits_path, 'rb') as f:
            data_splits = pickle.load(f)
            
        return data_splits
    
    def create_reliability_diagram(self, calibration_results: Dict[str, Any], 
                                 data_splits: Dict[str, Any], save: bool = True) -> None:
        """
        Create reliability diagrams comparing original and calibrated models
        
        Args:
            calibration_results: Results from calibration analysis
            data_splits: Data splits dictionary
            save: Whether to save the plot
        """
        print("📈 Creating reliability diagrams...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Set up subplots
        n_models = len(calibration_results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, model_variants) in enumerate(calibration_results.items()):
            ax = axes[i]
            
            # Plot calibration curves for each variant
            for j, (variant_name, variant_data) in enumerate(model_variants.items()):
                if 'model' in variant_data:
                    model = variant_data['model']
                    y_prob = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate calibration curve
                    fraction_of_positives, mean_predicted_value = calibration_curve(
                        y_test, y_prob, n_bins=10
                    )
                    
                    # Plot with different styles
                    if variant_name == 'original':
                        ax.plot(mean_predicted_value, fraction_of_positives, 
                                marker='s', linewidth=3, markersize=8,
                                color=self.colors[0], label='Original', alpha=0.8)
                    elif variant_name == 'platt':
                        ax.plot(mean_predicted_value, fraction_of_positives, 
                                marker='o', linewidth=2, markersize=6,
                                color=self.colors[1], label='Platt Scaling', alpha=0.8)
                    elif variant_name == 'isotonic':
                        ax.plot(mean_predicted_value, fraction_of_positives, 
                                marker='^', linewidth=2, markersize=6,
                                color=self.colors[2], label='Isotonic Regression', alpha=0.8)
            
            # Perfect calibration line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.8, linewidth=2, label='Perfect Calibration')
            
            # Formatting
            ax.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax.set_ylabel('Fraction of Positives', fontsize=12)
            ax.set_title(f'{model_name.replace("_", " ")}\nReliability Diagram', fontsize=14, fontweight='bold')
            ax.legend(frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.viz_dir / "reliability_diagrams.png", dpi=300, bbox_inches='tight')
            plt.savefig(self.viz_dir / "reliability_diagrams.pdf", bbox_inches='tight')
        
        plt.show()
        print("✅ Reliability diagrams created")
    
    def generate_all_visualizations(self) -> None:
        """
        Generate all visualizations for the calibration experiment
        """
        print("🎨 Generating all visualizations...")
        print("=" * 50)
        
        try:
            # Load results
            calibration_results, comparison_df = self.load_results()
            data_splits = self.load_data_splits()
            
            # Create reliability diagrams
            self.create_reliability_diagram(calibration_results, data_splits)
            
            print("\n🎉 All visualizations generated successfully!")
            print(f"📁 Check {self.viz_dir} for all visualization files")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure to run the calibration analysis first.")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

def main():
    """
    Main function for visualization generation
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Calibration Visualizations')
    parser.add_argument('--results-dir', default='../results',
                       help='Directory containing calibration results')
    parser.add_argument('--reliability-only', action='store_true',
                       help='Generate only reliability diagrams')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = CalibrationVisualizer(args.results_dir)
    
    try:
        if args.reliability_only:
            calibration_results, _ = visualizer.load_results()
            data_splits = visualizer.load_data_splits()
            visualizer.create_reliability_diagram(calibration_results, data_splits)
        else:
            # Generate all visualizations
            visualizer.generate_all_visualizations()
            
    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        raise

if __name__ == "__main__":
    main()