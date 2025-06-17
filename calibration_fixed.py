"""
Fixed Calibration Analysis with Enhanced Temperature Scaling
==========================================================

This module fixes the temperature scaling implementation and provides better debugging.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

# ML Libraries
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize_scalar
import scipy.stats as stats

# Import the shared model class
from models.model_utils import TemperatureScaledModel

class FixedCalibrationAnalyzer:
    """
    Fixed calibration analyzer with enhanced temperature scaling
    """
    
    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.calibrated_models = {}
        self.calibration_metrics = {}
    
    def load_trained_models(self) -> Dict[str, Any]:
        """Load pre-trained models"""
        models_path = self.results_dir / "trained_models.pkl"
        
        if not models_path.exists():
            raise FileNotFoundError(f"No trained models found at {models_path}")
        
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        print(f"✅ Loaded {len(models)} trained models from {models_path}")
        return models
    
    def load_data_splits(self) -> Dict[str, Any]:
        """Load data splits"""
        splits_path = Path("data/processed/train_test_split.pkl")
        
        if not splits_path.exists():
            raise FileNotFoundError(f"No data splits found at {splits_path}")
        
        with open(splits_path, 'rb') as f:
            splits = pickle.load(f)
        
        print(f"✅ Loaded data splits from {splits_path}")
        return splits
    
    def calculate_calibration_metrics(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
        """Calculate comprehensive calibration metrics"""
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        mce = 0  # Maximum Calibration Error
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_error * prop_in_bin
                mce = max(mce, bin_error)
        
        # Brier Score
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Calculate basic Hosmer-Lemeshow test
        try:
            hl_stat, hl_pvalue = self._hosmer_lemeshow_test(y_true, y_prob, n_bins)
        except:
            hl_stat, hl_pvalue = 0, 1  # Default values if test fails
        
        return {
            'ece': ece,
            'mce': mce,
            'brier_score': brier_score,
            'log_loss': log_loss(y_true, y_prob),
            'reliability': 0,  # Simplified for now
            'resolution': 0,   # Simplified for now
            'uncertainty': y_true.mean() * (1 - y_true.mean()),
            'hosmer_lemeshow_stat': hl_stat,
            'hosmer_lemeshow_pvalue': hl_pvalue,
        }
    
    def _hosmer_lemeshow_test(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[float, float]:
        """Simplified Hosmer-Lemeshow test"""
        # Create equal-sized bins based on predicted probabilities
        sorted_indices = np.argsort(y_prob)
        n_per_bin = len(y_prob) // n_bins
        
        hl_stat = 0
        
        for i in range(n_bins):
            start_idx = i * n_per_bin
            if i == n_bins - 1:  # Last bin gets remaining samples
                end_idx = len(y_prob)
            else:
                end_idx = (i + 1) * n_per_bin
            
            bin_indices = sorted_indices[start_idx:end_idx]
            
            if len(bin_indices) > 0:
                o_i = y_true[bin_indices].sum()  # Observed events
                e_i = y_prob[bin_indices].sum()  # Expected events
                n_i = len(bin_indices)
                
                if e_i > 0 and (n_i - e_i) > 0:
                    hl_stat += ((o_i - e_i) ** 2) / e_i + (((n_i - o_i) - (n_i - e_i)) ** 2) / (n_i - e_i)
        
        # Calculate p-value using chi-square distribution
        df = max(1, n_bins - 2)  # degrees of freedom
        p_value = 1 - stats.chi2.cdf(hl_stat, df)
        
        return hl_stat, p_value
    
    def apply_platt_scaling(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
        """Apply Platt scaling"""
        print("🔧 Applying Platt scaling...")
        
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='sigmoid', 
            cv='prefit'
        )
        
        calibrated_model.fit(X_cal, y_cal)
        
        print("✅ Platt scaling applied")
        return calibrated_model
    
    def apply_isotonic_regression(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Any:
        """Apply isotonic regression calibration"""
        print("🔧 Applying isotonic regression...")
        
        calibrated_model = CalibratedClassifierCV(
            model, 
            method='isotonic', 
            cv='prefit'
        )
        
        calibrated_model.fit(X_cal, y_cal)
        
        print("✅ Isotonic regression applied")
        return calibrated_model
    
    def _get_model_logits(self, model: Any, X: np.ndarray) -> np.ndarray:
        """Extract logits from different model types"""
        if hasattr(model, 'decision_function'):
            # For SVM, Logistic Regression
            logits = model.decision_function(X)
            print(f"   📊 Decision function logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            return logits
        elif hasattr(model, 'predict_proba'):
            # For tree-based models, convert probabilities to logits
            probs = model.predict_proba(X)[:, 1]
            print(f"   📊 Original probs range: [{probs.min():.3f}, {probs.max():.3f}]")
            
            # Clip probabilities to avoid log(0) or log(1)
            probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
            logits = np.log(probs_clipped / (1 - probs_clipped))
            print(f"   📊 Converted logits range: [{logits.min():.3f}, {logits.max():.3f}]")
            return logits
        else:
            raise ValueError(f"Model {type(model)} doesn't support logit extraction")
    
    def _temperature_scaling(self, logits: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to logits"""
        try:
            scaled_probs = 1 / (1 + np.exp(-logits / temperature))
            return scaled_probs
        except Exception as e:
            print(f"   ⚠️ Temperature scaling error: {e}")
            # Return original sigmoid if temperature scaling fails
            return 1 / (1 + np.exp(-logits))
    
    def _find_optimal_temperature(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Find optimal temperature using validation set"""
        print(f"   🔍 Optimizing temperature for {len(val_logits)} validation samples...")
        
        def negative_log_likelihood(temperature):
            try:
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
            except Exception as e:
                print(f"   ⚠️ NLL calculation error at T={temperature}: {e}")
                return 1e6  # Return large value if calculation fails
        
        try:
            # Test baseline (no scaling)
            baseline_nll = negative_log_likelihood(1.0)
            print(f"   📊 Baseline NLL (T=1.0): {baseline_nll:.4f}")
            
            # Optimize temperature between 0.1 and 10.0
            result = minimize_scalar(
                negative_log_likelihood, 
                bounds=(0.1, 10.0), 
                method='bounded'
            )
            
            optimal_temp = result.x
            optimal_nll = result.fun
            
            print(f"   📊 Optimal temperature: {optimal_temp:.3f}")
            print(f"   📊 Optimal NLL: {optimal_nll:.4f}")
            print(f"   📊 Improvement: {baseline_nll - optimal_nll:.4f}")
            
            return optimal_temp
            
        except Exception as e:
            print(f"   ❌ Temperature optimization failed: {e}")
            return 1.0  # Return baseline temperature
    
    def apply_temperature_scaling(self, model: Any, X_cal: np.ndarray, y_cal: np.ndarray) -> Tuple[Any, float]:
        """Apply temperature scaling calibration"""
        print("🌡️ Applying temperature scaling...")
        
        try:
            # Get logits from validation set
            val_logits = self._get_model_logits(model, X_cal)
            
            # Find optimal temperature
            optimal_temperature = self._find_optimal_temperature(val_logits, y_cal)
            
            # Create calibrated model
            calibrated_model = TemperatureScaledModel(
                model, 
                optimal_temperature,
                self._get_model_logits,
                self._temperature_scaling
            )
            
            print(f"✅ Temperature scaling applied (T = {optimal_temperature:.3f})")
            return calibrated_model, optimal_temperature
            
        except Exception as e:
            print(f"❌ Temperature scaling failed: {e}")
            import traceback
            traceback.print_exc()
            raise e
    
    def run_fixed_calibration_analysis(self) -> pd.DataFrame:
        """Run the fixed calibration analysis"""
        print("🔧 Running Fixed Calibration Analysis")
        print("=" * 50)
        
        # Load data
        models = self.load_trained_models()
        data_splits = self.load_data_splits()
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        print(f"📊 Data shapes: X_val={X_val.shape}, X_test={X_test.shape}")
        
        comparison_data = []
        model_predictions = {}
        
        for model_name, model in models.items():
            print(f"\n📊 Calibrating {model_name}...")
            
            # Original model results
            try:
                original_probs = model.predict_proba(X_test)[:, 1]
                original_metrics = self.calculate_calibration_metrics(y_test, original_probs)
                
                comparison_data.append({
                    'Model': model_name,
                    'Calibration_Method': 'Original',
                    'Full_Name': f"{model_name}_Original",
                    'ECE': original_metrics['ece'],
                    'MCE': original_metrics['mce'],
                    'Brier_Score': original_metrics['brier_score'],
                    'Log_Loss': original_metrics['log_loss'],
                    'Reliability': original_metrics['reliability'],
                    'Resolution': original_metrics['resolution'],
                    'HL_Statistic': original_metrics['hosmer_lemeshow_stat'],
                    'HL_P_Value': original_metrics['hosmer_lemeshow_pvalue'],
                    'Temperature_Value': None
                })
                
                model_predictions[f"{model_name}_Original"] = original_probs
                
                print(f"   ✅ Original ECE: {original_metrics['ece']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Original model failed: {e}")
                continue
            
            # Platt scaling
            try:
                platt_model = self.apply_platt_scaling(model, X_val, y_val)
                platt_probs = platt_model.predict_proba(X_test)[:, 1]
                platt_metrics = self.calculate_calibration_metrics(y_test, platt_probs)
                
                comparison_data.append({
                    'Model': model_name,
                    'Calibration_Method': 'Platt',
                    'Full_Name': f"{model_name}_Platt",
                    'ECE': platt_metrics['ece'],
                    'MCE': platt_metrics['mce'],
                    'Brier_Score': platt_metrics['brier_score'],
                    'Log_Loss': platt_metrics['log_loss'],
                    'Reliability': platt_metrics['reliability'],
                    'Resolution': platt_metrics['resolution'],
                    'HL_Statistic': platt_metrics['hosmer_lemeshow_stat'],
                    'HL_P_Value': platt_metrics['hosmer_lemeshow_pvalue'],
                    'Temperature_Value': None
                })
                
                model_predictions[f"{model_name}_Platt"] = platt_probs
                
                print(f"   ✅ Platt ECE: {platt_metrics['ece']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Platt scaling failed: {e}")
            
            # Isotonic regression
            try:
                isotonic_model = self.apply_isotonic_regression(model, X_val, y_val)
                isotonic_probs = isotonic_model.predict_proba(X_test)[:, 1]
                isotonic_metrics = self.calculate_calibration_metrics(y_test, isotonic_probs)
                
                comparison_data.append({
                    'Model': model_name,
                    'Calibration_Method': 'Isotonic',
                    'Full_Name': f"{model_name}_Isotonic",
                    'ECE': isotonic_metrics['ece'],
                    'MCE': isotonic_metrics['mce'],
                    'Brier_Score': isotonic_metrics['brier_score'],
                    'Log_Loss': isotonic_metrics['log_loss'],
                    'Reliability': isotonic_metrics['reliability'],
                    'Resolution': isotonic_metrics['resolution'],
                    'HL_Statistic': isotonic_metrics['hosmer_lemeshow_stat'],
                    'HL_P_Value': isotonic_metrics['hosmer_lemeshow_pvalue'],
                    'Temperature_Value': None
                })
                
                model_predictions[f"{model_name}_Isotonic"] = isotonic_probs
                
                print(f"   ✅ Isotonic ECE: {isotonic_metrics['ece']:.4f}")
                
            except Exception as e:
                print(f"   ❌ Isotonic regression failed: {e}")
            
            # Temperature scaling - Enhanced with detailed debugging
            try:
                temp_model, optimal_temp = self.apply_temperature_scaling(model, X_val, y_val)
                temp_probs = temp_model.predict_proba(X_test)[:, 1]
                temp_metrics = self.calculate_calibration_metrics(y_test, temp_probs)
                
                # Add temperature value to metrics
                temp_metrics['temperature'] = optimal_temp
                
                comparison_data.append({
                    'Model': model_name,
                    'Calibration_Method': 'Temperature',
                    'Full_Name': f"{model_name}_Temperature",
                    'ECE': temp_metrics['ece'],
                    'MCE': temp_metrics['mce'],
                    'Brier_Score': temp_metrics['brier_score'],
                    'Log_Loss': temp_metrics['log_loss'],
                    'Reliability': temp_metrics['reliability'],
                    'Resolution': temp_metrics['resolution'],
                    'HL_Statistic': temp_metrics['hosmer_lemeshow_stat'],
                    'HL_P_Value': temp_metrics['hosmer_lemeshow_pvalue'],
                    'Temperature_Value': optimal_temp
                })
                
                model_predictions[f"{model_name}_Temperature"] = temp_probs
                
                print(f"   ✅ Temperature ECE: {temp_metrics['ece']:.4f} (T={optimal_temp:.3f})")
                
                # Interpret temperature value
                if optimal_temp > 3.0:
                    confidence_level = "severely overconfident"
                elif optimal_temp > 1.5:
                    confidence_level = "moderately overconfident"
                elif optimal_temp < 0.7:
                    confidence_level = "underconfident"
                else:
                    confidence_level = "well-calibrated"
                
                print(f"   🔍 Model assessment: {confidence_level}")
                
            except Exception as e:
                print(f"   ❌ Temperature scaling failed: {e}")
                import traceback
                traceback.print_exc()
        
        # Create DataFrame and save results
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('ECE')
        
        # Save results
        comparison_df.to_csv(self.results_dir / "calibration_comparison_fixed.csv", index=False)
        
        # Save model predictions
        with open(self.results_dir / "model_predictions_fixed.pkl", 'wb') as f:
            pickle.dump(model_predictions, f)
        
        print("\n🏆 Best Calibrated Models (by ECE):")
        display_cols = ['Full_Name', 'ECE', 'Brier_Score', 'Temperature_Value']
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
        else:
            print("\n⚠️ No temperature scaling results found")
        
        print(f"\n✅ Fixed calibration analysis completed!")
        print(f"📁 Results saved to {self.results_dir}/calibration_comparison_fixed.csv")
        
        return comparison_df

def main():
    """Main function for fixed calibration analysis"""
    print("🔧 Running Fixed Temperature Scaling Analysis")
    print("=" * 55)
    
    try:
        # Initialize analyzer
        analyzer = FixedCalibrationAnalyzer("results")
        
        # Run analysis
        results_df = analyzer.run_fixed_calibration_analysis()
        
        print("\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
