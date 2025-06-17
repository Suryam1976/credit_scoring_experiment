"""
Calibration Methods for Machine Learning Models
==============================================

This module implements various calibration techniques for improving the reliability 
of predicted probabilities from machine learning models.

Author: Credit Scoring Experiment
Date: June 2025
"""

import os
import sys
# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from scipy.optimize import minimize_scalar
import scipy.stats as stats

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import the shared model class
from models.model_utils import TemperatureScaledModel

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
    
    def _get_model_logits(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Extract logits from different model types
        
        Args:
            model: Trained model
            X: Input features
            
        Returns:
            Raw logits (before sigmoid/softmax)
        """
        if hasattr(model, 'decision_function'):
            # For SVM, Logistic Regression
            return model.decision_function(X)
        elif hasattr(model, 'predict_proba'):
            # For tree-based models, convert probabilities to logits
            probs = model.predict_proba(X)[:, 1]
            # Clip probabilities to avoid log(0) or log(1)
            probs = np.clip(probs, 1e-15, 1 - 1e-15)
            logits = np.log(probs / (1 - probs))
            return logits
        else:
            raise ValueError(f"Model {type(model)} doesn't support logit extraction")
    
    def _temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to logits
        
        Args:
            logits: Raw logits from model
            temperature: Temperature parameter T
            
        Returns:
            Calibrated probabilities
        """
        return 1 / (1 + np.exp(-logits / temperature))
    
    def _find_optimal_temperature(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """
        Find optimal temperature using validation set
        
        Args:
            val_logits: Validation set logits
            val_labels: Validation set true labels
            
        Returns:
            Optimal temperature value
        """
        def negative_log_likelihood(temperature):
            # Apply temperature scaling
            probs = self._temperature_scaling(val_logits, temperature)
            
            # Calculate negative log-likelihood
            eps = 1e-15  # For numerical stability
            probs = np.clip(probs, eps, 1 - eps)
            
            nll = -np.mean(
                val_labels * np.log(probs) + 
                (1 - val_labels) * np.log(1 - probs)
            )
            return nll
        
        # Optimize temperature between 0.1 and 10.0
        result = minimize_scalar(
            negative_log_likelihood, 
            bounds=(0.1, 10.0), 
            method='bounded'
        )
        
        return result.x
    
    def apply_temperature_scaling(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Tuple[Any, float]:
        """
        Apply temperature scaling calibration to a model
        
        Args:
            model: Trained model
            X_cal: Calibration features
            y_cal: Calibration labels
        
        Returns:
            Tuple of (calibrated_model, optimal_temperature)
        """
        print("🌡️ Applying temperature scaling...")
        
        # Get logits from validation set
        val_logits = self._get_model_logits(model, X_cal)
        
        # Find optimal temperature
        optimal_temperature = self._find_optimal_temperature(val_logits, y_cal)
        
        # Create calibrated model using the class defined at module level
        calibrated_model = TemperatureScaledModel(
            model, 
            optimal_temperature,
            self._get_model_logits,  # Pass the method reference
            self._temperature_scaling  # Pass the method reference
        )
        
        print(f"✅ Temperature scaling applied (T = {optimal_temperature:.3f})")
        return calibrated_model, optimal_temperature  # Return both model and temperature
    
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
            
            # Apply Temperature scaling
            try:
                temp_model, optimal_temp = self.apply_temperature_scaling(model, X_val, y_val)
                temp_probs = temp_model.predict_proba(X_test)[:, 1]
                temp_metrics = self.calculate_calibration_metrics(y_test, temp_probs)
                
                # Add temperature value to metrics for analysis
                temp_metrics['temperature'] = optimal_temp
                
                model_results['temperature'] = {
                    'model': temp_model,
                    'name': f"{model_name}_Temperature",
                    'metrics': temp_metrics,
                    'optimal_temperature': optimal_temp
                }
            except Exception as e:
                print(f"⚠️ Temperature scaling failed for {model_name}: {e}")
            
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
        
        # Save model predictions for visualization
        data_splits = self.load_data_splits()
        X_test = data_splits['X_test']
        
        # Create a dictionary to store predictions
        model_predictions = {}
        
        for model_name, model_variants in calibration_results.items():
            for variant_name, variant_data in model_variants.items():
                if 'model' in variant_data:
                    # Generate and store predictions
                    model = variant_data['model']
                    try:
                        y_prob = model.predict_proba(X_test)[:, 1]
                        model_predictions[variant_data['name']] = y_prob
                    except Exception as e:
                        print(f"⚠️ Could not generate predictions for {variant_data['name']}: {e}")
        
        # Save predictions separately
        with open(self.results_dir / "model_predictions.pkl", 'wb') as f:
            pickle.dump(model_predictions, f)
        
        for model_name, model_variants in calibration_results.items():
            for variant_name, variant_data in model_variants.items():
                if 'metrics' in variant_data:
                    metrics = variant_data['metrics']
                    
                    row_data = {
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
                    }
                    
                    # Add temperature value if available
                    if 'temperature' in metrics:
                        row_data['Temperature_Value'] = metrics['temperature']
                    else:
                        row_data['Temperature_Value'] = None
                    
                    comparison_data.append(row_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ECE')
        
        # Save to CSV
        comparison_df.to_csv(self.results_dir / "calibration_comparison.csv", index=False)
        
        print("\n🏆 Best Calibrated Models (by ECE):")
        display_cols = ['Full_Name', 'ECE', 'Brier_Score']
        if 'Temperature_Value' in comparison_df.columns:
            display_cols.append('Temperature_Value')
        print(comparison_df[display_cols].head().to_string(index=False))
        
        # Print temperature scaling analysis
        temp_results = comparison_df[comparison_df['Calibration_Method'] == 'Temperature']
        if len(temp_results) > 0:
            print("\n🌡️ Temperature Scaling Analysis:")
            for _, row in temp_results.iterrows():
                temp_val = row['Temperature_Value']
                if temp_val is not None:
                    if temp_val > 3.0:
                        confidence_level = "severely overconfident"
                    elif temp_val > 1.5:
                        confidence_level = "moderately overconfident"
                    elif temp_val < 0.7:
                        confidence_level = "underconfident"
                    else:
                        confidence_level = "well-calibrated"
                    
                    print(f"   {row['Model']}: T = {temp_val:.3f} (📊 {confidence_level})")

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








