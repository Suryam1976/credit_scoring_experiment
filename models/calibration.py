"""
Calibration Methods for Machine Learning Models
==============================================

This module implements various calibration techniques for improving the reliability 
of predicted probabilities from machine learning models.

Author: Credit Scoring Experiment
Date: June 2025
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# ML Libraries
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss
import scipy.stats as stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

class CalibrationAnalyzer:
    """
    Comprehensive calibration analysis and improvement toolkit
    """
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibrated_models = {}
        self.calibration_metrics = {}
        
    def load_trained_models(self, models_path: str = None) -> Dict[str, Any]:
        """
        Load pre-trained models from pickle file
        
        Args:
            models_path: Path to trained models pickle file
            
        Returns:
            Dictionary of loaded models
        """
        if models_path is None:
            # Try different possible paths
            possible_paths = [
                "../results/trained_models.pkl",
                "./results/trained_models.pkl", 
                "results/trained_models.pkl"
            ]
            
            models_file = None
            for path in possible_paths:
                if Path(path).exists():
                    models_file = Path(path)
                    break
            
            if models_file is None:
                raise FileNotFoundError(f"No trained models found. Tried: {possible_paths}")
        else:
            models_file = Path(models_path)
            if not models_file.exists():
                raise FileNotFoundError(f"No trained models found at {models_path}")
        
        with open(models_file, 'rb') as f:
            models = pickle.load(f)
        
        print(f"✅ Loaded {len(models)} trained models from {models_file}")
        return models
    
    def load_data_splits(self, splits_path: str = None) -> Dict[str, Any]:
        """
        Load data splits from pickle file
        
        Args:
            splits_path: Path to data splits pickle file
            
        Returns:
            Dictionary containing data splits
        """
        if splits_path is None:
            # Try different possible paths
            possible_paths = [
                "../data/processed/train_test_split.pkl",
                "./data/processed/train_test_split.pkl",
                "data/processed/train_test_split.pkl"
            ]
            
            splits_file = None
            for path in possible_paths:
                if Path(path).exists():
                    splits_file = Path(path)
                    break
            
            if splits_file is None:
                raise FileNotFoundError(f"No data splits found. Tried: {possible_paths}")
        else:
            splits_file = Path(splits_path)
            if not splits_file.exists():
                raise FileNotFoundError(f"No data splits found at {splits_path}")
        
        with open(splits_file, 'rb') as f:
            splits = pickle.load(f)
        
        print(f"✅ Loaded data splits from {splits_file}")
        return splits
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """
        Calculate comprehensive calibration metrics
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Dictionary of calibration metrics
        """
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0  # Maximum Calibration Error
        bin_accuracies = []
        bin_confidences = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_sizes.append(prop_in_bin)
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_sizes.append(0)
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Brier Score Decomposition: BS = Reliability - Resolution + Uncertainty
        y_mean = y_true.mean()
        uncertainty = y_mean * (1 - y_mean)
        
        bins = np.digitize(y_prob, bin_boundaries) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        
        reliability = 0
        resolution = 0
        
        for i in range(n_bins):
            mask = bins == i
            if mask.sum() > 0:
                n_i = mask.sum()
                o_i = y_true[mask].mean()
                p_i = y_prob[mask].mean()
                
                reliability += n_i * (p_i - o_i) ** 2
                resolution += n_i * (o_i - y_mean) ** 2
        
        reliability /= len(y_true)
        resolution /= len(y_true)
        
        # Hosmer-Lemeshow Test
        hl_stat, hl_pvalue = self._hosmer_lemeshow_test(y_true, y_prob, n_bins)
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'uncertainty': uncertainty,
            'log_loss': log_loss(y_true, y_prob),
            'hosmer_lemeshow_stat': hl_stat,
            'hosmer_lemeshow_pvalue': hl_pvalue,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_sizes': bin_sizes
        }
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """
        Perform Hosmer-Lemeshow goodness-of-fit test for calibration
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins
            
        Returns:
            Tuple of (test statistic, p-value)
        """
        # Create bins based on predicted probabilities
        bin_boundaries = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        bin_boundaries[0] = 0  # Ensure first boundary is 0
        bin_boundaries[-1] = 1  # Ensure last boundary is 1
        
        bins = np.digitize(y_prob, bin_boundaries) - 1
        bins = np.clip(bins, 0, n_bins - 1)
        
        # Calculate Hosmer-Lemeshow statistic
        hl_stat = 0
        
        for i in range(n_bins):
            mask = bins == i
            if mask.sum() > 0:
                n_i = mask.sum()
                o_i = y_true[mask].sum()  # Observed events
                e_i = y_prob[mask].sum()  # Expected events
                
                if e_i > 0 and (n_i - e_i) > 0:
                    hl_stat += ((o_i - e_i) ** 2) / e_i + (((n_i - o_i) - (n_i - e_i)) ** 2) / (n_i - e_i)
        
        # Calculate p-value using chi-square distribution
        df = n_bins - 2  # degrees of freedom
        p_value = 1 - stats.chi2.cdf(hl_stat, df)
        
        return hl_stat, p_value
    
    def apply_platt_scaling(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
        """
        Apply Platt scaling (sigmoid calibration) to a model
        
        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration labels
            
        Returns:
            Calibrated model using Platt scaling
        """
        print("🔧 Applying Platt scaling...")
        
        # Use CalibratedClassifierCV with sigmoid method
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='sigmoid', 
            cv='prefit'
        )
        
        calibrated_model.fit(X_cal, y_cal)
        
        print("✅ Platt scaling applied")
        return calibrated_model
    
    def apply_isotonic_regression(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
        """
        Apply isotonic regression calibration to a model
        
        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration labels
            
        Returns:
            Calibrated model using isotonic regression
        """
        print("🔧 Applying isotonic regression...")
        
        # Use CalibratedClassifierCV with isotonic method
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='isotonic', 
            cv='prefit'
        )
        
        calibrated_model.fit(X_cal, y_cal)
        
        print("✅ Isotonic regression applied")
        return calibrated_model
    
    def generate_calibration_report(self, models_path: str = None, splits_path: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive calibration analysis report
        
        Args:
            models_path: Path to trained models (optional)
            splits_path: Path to data splits (optional)
            
        Returns:
            Dictionary containing all analysis results
        """
        print("📋 Generating comprehensive calibration report...")
        print("=" * 60)
        
        # Load data
        if models_path is None:
            models_path = "../results/trained_models.pkl"
        if splits_path is None:
            splits_path = "../data/processed/train_test_split.pkl"
            
        models = self.load_trained_models(models_path)
        data_splits = self.load_data_splits(splits_path)
        
        # Perform calibration analysis
        calibration_results = self.calibrate_all_models(models, data_splits)
        
        print("\n🎉 Comprehensive calibration report generated!")
        return calibration_results
    
    def calibrate_all_models(self, models: Dict[str, Any], data_splits: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Apply all calibration methods to all models
        
        Args:
            models: Dictionary of trained models
            data_splits: Dictionary containing data splits
            
        Returns:
            Dictionary of calibrated models and their metrics
        """
        print("🎯 Starting comprehensive calibration analysis...")
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        calibration_results = {}
        
        for model_name, model in models.items():
            print(f"\n📊 Calibrating {model_name}...")
            
            model_results = {
                'original': {
                    'model': model,
                    'name': f"{model_name}_Original"
                }
            }
            
            # Get original predictions
            original_probs = model.predict_proba(X_test)[:, 1]
            original_metrics = self.calculate_calibration_metrics(y_test, original_probs)
            model_results['original']['metrics'] = original_metrics
            
            # Apply Platt scaling
            try:
                platt_model = self.apply_platt_scaling(model, X_val, y_val)
                platt_probs = platt_model.predict_proba(X_test)[:, 1]
                platt_metrics = self.calculate_calibration_metrics(y_test, platt_probs)
                
                model_results['platt'] = {
                    'model': platt_model,
                    'name': f"{model_name}_Platt",
                    'metrics': platt_metrics
                }
            except Exception as e:
                print(f"⚠️ Platt scaling failed for {model_name}: {e}")
            
            # Apply Isotonic regression
            try:
                isotonic_model = self.apply_isotonic_regression(model, X_val, y_val)
                isotonic_probs = isotonic_model.predict_proba(X_test)[:, 1]
                isotonic_metrics = self.calculate_calibration_metrics(y_test, isotonic_probs)
                
                model_results['isotonic'] = {
                    'model': isotonic_model,
                    'name': f"{model_name}_Isotonic",
                    'metrics': isotonic_metrics
                }
            except Exception as e:
                print(f"⚠️ Isotonic regression failed for {model_name}: {e}")
            
            calibration_results[model_name] = model_results
        
        # Save calibrated models
        self.calibrated_models = calibration_results
        
        # Save to file
        with open(self.results_dir / "calibrated_models.pkl", 'wb') as f:
            pickle.dump(calibration_results, f)
        
        # Create summary table and save
        self._create_summary_table(calibration_results)
        
        print("\n🎉 Calibration analysis completed!")
        return calibration_results
    
    def _create_summary_table(self, calibration_results: Dict[str, Dict[str, Any]]) -> None:
        """Create and save summary table of calibration results"""
        
        comparison_data = []
        
        for model_name, model_variants in calibration_results.items():
            for variant_name, variant_data in model_variants.items():
                if 'metrics' in variant_data:
                    metrics = variant_data['metrics']
                    
                    comparison_data.append({
                        'Model': model_name,
                        'Calibration_Method': variant_name.title(),
                        'Full_Name': variant_data['name'],
                        'ECE': metrics['ece'],
                        'MCE': metrics['mce'],
                        'Brier_Score': metrics['brier_score'],
                        'Log_Loss': metrics['log_loss'],
                        'Reliability': metrics['reliability'],
                        'Resolution': metrics['resolution'],
                        'HL_Statistic': metrics['hosmer_lemeshow_stat'],
                        'HL_P_Value': metrics['hosmer_lemeshow_pvalue']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ECE')
        
        # Save to CSV
        comparison_df.to_csv(self.results_dir / "calibration_comparison.csv", index=False)
        
        print("\n🏆 Best Calibrated Models (by ECE):")
        print(comparison_df[['Full_Name', 'ECE', 'Brier_Score']].head().to_string(index=False))

def main():
    """
    Main function for calibration analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Calibration Analysis')
    parser.add_argument('--models-path', default='../results/trained_models.pkl',
                       help='Path to trained models pickle file')
    parser.add_argument('--splits-path', default='../data/processed/train_test_split.pkl',
                       help='Path to data splits pickle file')
    parser.add_argument('--results-dir', default='../results',
                       help='Directory to save calibration results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CalibrationAnalyzer(args.results_dir)
    
    # Generate comprehensive report
    try:
        results = analyzer.generate_calibration_report(
            models_path=args.models_path,
            splits_path=args.splits_path
        )
        print("\n✅ Calibration analysis completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure to run train_models.py first to generate the required files.")
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    main()