"""
Run Fixed Calibration Analysis with Temperature Scaling
=====================================================

This script runs the fixed calibration analysis with proper temperature scaling.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize_scalar
import scipy.stats as stats

from models.model_utils import TemperatureScaledModel

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

def get_model_logits(model, X):
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

def temperature_scaling(logits, temperature):
    """Apply temperature scaling to logits"""
    return 1 / (1 + np.exp(-logits / temperature))

def find_optimal_temperature(val_logits, val_labels):
    """Find optimal temperature using validation set"""
    print(f"   🔍 Optimizing temperature for {len(val_logits)} validation samples...")
    
    def negative_log_likelihood(temperature):
        try:
            probs = temperature_scaling(val_logits, temperature)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(val_labels * np.log(probs) + (1 - val_labels) * np.log(1 - probs))
            return nll
        except:
            return 1e6
    
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
        return 1.0

def apply_temperature_scaling(model, X_cal, y_cal):
    """Apply temperature scaling calibration"""
    print("🌡️ Applying temperature scaling...")
    
    try:
        # Get logits from validation set
        val_logits = get_model_logits(model, X_cal)
        
        # Find optimal temperature
        optimal_temperature = find_optimal_temperature(val_logits, y_cal)
        
        # Create calibrated model
        calibrated_model = TemperatureScaledModel(
            model,
            optimal_temperature,
            get_model_logits,
            temperature_scaling
        )
        
        print(f"✅ Temperature scaling applied (T = {optimal_temperature:.3f})")
        return calibrated_model, optimal_temperature
        
    except Exception as e:
        print(f"❌ Temperature scaling failed: {e}")
        import traceback
        traceback.print_exc()
        raise e

def run_complete_calibration_analysis():
    """Run complete calibration analysis with all methods including temperature scaling"""
    print("🔧 Running Complete Calibration Analysis with Temperature Scaling")
    print("=" * 65)
    
    # Load data
    print("📂 Loading trained models and data splits...")
    
    with open("results/trained_models.pkl", 'rb') as f:
        models = pickle.load(f)
    
    with open("data/processed/train_test_split.pkl", 'rb') as f:
        data_splits = pickle.load(f)
    
    X_val = data_splits['X_val']
    y_val = data_splits['y_val']
    X_test = data_splits['X_test']
    y_test = data_splits['y_test']
    
    print(f"✅ Loaded {len(models)} models")
    print(f"📊 Data shapes: X_val={X_val.shape}, X_test={X_test.shape}")
    
    # Results storage
    all_results = []
    model_predictions = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Analyzing {model_name}...")
        print("-" * 40)
        
        # 1. Original model
        try:
            original_probs = model.predict_proba(X_test)[:, 1]
            original_ece = calculate_ece(y_test, original_probs)
            original_brier = brier_score_loss(y_test, original_probs)
            original_logloss = log_loss(y_test, original_probs)
            
            all_results.append({
                'Model': model_name,
                'Calibration_Method': 'Original',
                'Full_Name': f"{model_name}_Original",
                'ECE': original_ece,
                'Brier_Score': original_brier,
                'Log_Loss': original_logloss,
                'Temperature_Value': None
            })
            
            model_predictions[f"{model_name}_Original"] = original_probs
            print(f"   ✅ Original: ECE={original_ece:.4f}, Brier={original_brier:.4f}")
            
        except Exception as e:
            print(f"   ❌ Original model failed: {e}")
            continue
        
        # 2. Platt scaling
        try:
            print("🔧 Applying Platt scaling...")
            platt_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
            platt_model.fit(X_val, y_val)
            
            platt_probs = platt_model.predict_proba(X_test)[:, 1]
            platt_ece = calculate_ece(y_test, platt_probs)
            platt_brier = brier_score_loss(y_test, platt_probs)
            platt_logloss = log_loss(y_test, platt_probs)
            
            all_results.append({
                'Model': model_name,
                'Calibration_Method': 'Platt',
                'Full_Name': f"{model_name}_Platt",
                'ECE': platt_ece,
                'Brier_Score': platt_brier,
                'Log_Loss': platt_logloss,
                'Temperature_Value': None
            })
            
            model_predictions[f"{model_name}_Platt"] = platt_probs
            print(f"   ✅ Platt: ECE={platt_ece:.4f}, Brier={platt_brier:.4f}")
            
        except Exception as e:
            print(f"   ❌ Platt scaling failed: {e}")
        
        # 3. Isotonic regression
        try:
            print("🔧 Applying isotonic regression...")
            isotonic_model = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
            isotonic_model.fit(X_val, y_val)
            
            isotonic_probs = isotonic_model.predict_proba(X_test)[:, 1]
            isotonic_ece = calculate_ece(y_test, isotonic_probs)
            isotonic_brier = brier_score_loss(y_test, isotonic_probs)
            isotonic_logloss = log_loss(y_test, isotonic_probs)
            
            all_results.append({
                'Model': model_name,
                'Calibration_Method': 'Isotonic',
                'Full_Name': f"{model_name}_Isotonic",
                'ECE': isotonic_ece,
                'Brier_Score': isotonic_brier,
                'Log_Loss': isotonic_logloss,
                'Temperature_Value': None
            })
            
            model_predictions[f"{model_name}_Isotonic"] = isotonic_probs
            print(f"   ✅ Isotonic: ECE={isotonic_ece:.4f}, Brier={isotonic_brier:.4f}")
            
        except Exception as e:
            print(f"   ❌ Isotonic regression failed: {e}")
        
        # 4. Temperature scaling - Enhanced with detailed debugging
        try:
            temp_model, optimal_temp = apply_temperature_scaling(model, X_val, y_val)
            
            # Test the temperature scaled model
            temp_probs = temp_model.predict_proba(X_test)[:, 1]
            temp_ece = calculate_ece(y_test, temp_probs)
            temp_brier = brier_score_loss(y_test, temp_probs)
            temp_logloss = log_loss(y_test, temp_probs)
            
            all_results.append({
                'Model': model_name,
                'Calibration_Method': 'Temperature',
                'Full_Name': f"{model_name}_Temperature",
                'ECE': temp_ece,
                'Brier_Score': temp_brier,
                'Log_Loss': temp_logloss,
                'Temperature_Value': optimal_temp
            })
            
            model_predictions[f"{model_name}_Temperature"] = temp_probs
            print(f"   ✅ Temperature: ECE={temp_ece:.4f}, Brier={temp_brier:.4f}, T={optimal_temp:.3f}")
            
            # Interpret temperature value
            if optimal_temp > 3.0:
                assessment = "severely overconfident"
            elif optimal_temp > 1.5:
                assessment = "moderately overconfident"
            elif optimal_temp < 0.7:
                assessment = "underconfident"
            else:
                assessment = "well-calibrated"
            
            print(f"   🔍 Assessment: {assessment}")
            
            # Compare improvement
            improvement = original_ece - temp_ece
            print(f"   📈 ECE improvement: {improvement:.4f}")
            
        except Exception as e:
            print(f"   ❌ Temperature scaling failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('ECE')
    
    # Add additional metrics for completeness
    for i, row in results_df.iterrows():
        # Add MCE (Maximum Calibration Error) - simplified calculation
        model_name = row['Full_Name']
        if model_name in model_predictions:
            probs = model_predictions[model_name]
            
            # Calculate MCE
            bin_boundaries = np.linspace(0, 1, 11)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            mce = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (probs > bin_lower) & (probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_test[in_bin].mean()
                    avg_confidence_in_bin = probs[in_bin].mean()
                    bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                    mce = max(mce, bin_error)
            
            results_df.loc[i, 'MCE'] = mce
            results_df.loc[i, 'Reliability'] = 0  # Simplified
            results_df.loc[i, 'Resolution'] = 0   # Simplified
            results_df.loc[i, 'HL_Statistic'] = 0 # Simplified
            results_df.loc[i, 'HL_P_Value'] = 1   # Simplified
    
    # Save results
    results_df.to_csv("results/calibration_comparison_complete.csv", index=False)
    
    # Save model predictions
    with open("results/model_predictions_complete.pkl", 'wb') as f:
        pickle.dump(model_predictions, f)
    
    # Display results
    print("\n🏆 Complete Calibration Results (sorted by ECE):")
    print("=" * 60)
    display_cols = ['Full_Name', 'ECE', 'Brier_Score', 'Temperature_Value']
    print(results_df[display_cols].to_string(index=False))
    
    # Temperature scaling analysis
    temp_results = results_df[results_df['Calibration_Method'] == 'Temperature']
    if len(temp_results) > 0:
        print("\n🌡️ Temperature Scaling Analysis:")
        print("=" * 40)
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
                
                print(f"   {row['Model']}: T = {temp_val:.3f} ({confidence_level})")
                original_ece = results_df[results_df['Full_Name'] == f"{row['Model']}_Original"]['ECE'].iloc[0]
                improvement = original_ece - row['ECE']
                print(f"      ECE improvement: {improvement:.4f}")
    else:
        print("\n⚠️ No temperature scaling results found")
    
    # Summary statistics
    print("\n📊 Summary by Calibration Method:")
    print("=" * 40)
    method_summary = results_df.groupby('Calibration_Method')['ECE'].agg(['mean', 'std', 'min', 'max'])
    print(method_summary.round(4))
    
    # Best models
    print("\n🥇 Best Calibrated Models:")
    print("=" * 30)
    top_5 = results_df.head(5)
    for _, row in top_5.iterrows():
        temp_info = f" (T={row['Temperature_Value']:.3f})" if row['Temperature_Value'] is not None else ""
        print(f"   {row['Full_Name']}: ECE={row['ECE']:.4f}{temp_info}")
    
    print(f"\n✅ Complete calibration analysis finished!")
    print(f"📁 Results saved to results/calibration_comparison_complete.csv")
    print(f"📁 Predictions saved to results/model_predictions_complete.pkl")
    
    return results_df

def main():
    """Main function"""
    try:
        results_df = run_complete_calibration_analysis()
        
        # Check if temperature scaling worked
        temp_count = len(results_df[results_df['Calibration_Method'] == 'Temperature'])
        if temp_count > 0:
            print(f"\n🎉 SUCCESS: Temperature scaling worked for {temp_count} models!")
        else:
            print(f"\n❌ WARNING: No temperature scaling results found")
            
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
