"""
Debug Temperature Scaling Implementation
=======================================

This script tests the temperature scaling implementation to identify and fix issues.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize_scalar

from models.model_utils import TemperatureScaledModel

def load_test_data():
    """Load test data and models"""
    print("📂 Loading test data...")
    
    # Load data splits
    splits_path = Path("data/processed/train_test_split.pkl")
    with open(splits_path, 'rb') as f:
        data_splits = pickle.load(f)
    
    # Load trained models
    models_path = Path("results/trained_models.pkl")
    with open(models_path, 'rb') as f:
        models = pickle.load(f)
    
    return data_splits, models

def test_get_model_logits(model, X):
    """Test logit extraction for different model types"""
    print(f"🔍 Testing logit extraction for {type(model).__name__}...")
    
    if hasattr(model, 'decision_function'):
        print(f"   ✅ Using decision_function")
        logits = model.decision_function(X)
        print(f"   📊 Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        return logits
    elif hasattr(model, 'predict_proba'):
        print(f"   ✅ Converting from predict_proba")
        probs = model.predict_proba(X)[:, 1]
        print(f"   📊 Probs range: [{probs.min():.3f}, {probs.max():.3f}]")
        
        # Clip probabilities to avoid log(0) or log(1)
        probs_clipped = np.clip(probs, 1e-15, 1 - 1e-15)
        logits = np.log(probs_clipped / (1 - probs_clipped))
        print(f"   📊 Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
        return logits
    else:
        raise ValueError(f"Model {type(model)} doesn't support logit extraction")

def test_temperature_scaling_function(logits, temperature):
    """Test temperature scaling function"""
    print(f"🌡️ Testing temperature scaling with T={temperature}")
    
    try:
        scaled_probs = 1 / (1 + np.exp(-logits / temperature))
        print(f"   📊 Scaled probs range: [{scaled_probs.min():.3f}, {scaled_probs.max():.3f}]")
        return scaled_probs
    except Exception as e:
        print(f"   ❌ Error in temperature scaling: {e}")
        return None

def test_temperature_optimization(val_logits, val_labels):
    """Test temperature optimization"""
    print("🔧 Testing temperature optimization...")
    
    def negative_log_likelihood(temperature):
        # Apply temperature scaling
        probs = 1 / (1 + np.exp(-val_logits / temperature))
        
        # Calculate negative log-likelihood
        eps = 1e-15  # For numerical stability
        probs = np.clip(probs, eps, 1 - eps)
        
        nll = -np.mean(
            val_labels * np.log(probs) + 
            (1 - val_labels) * np.log(1 - probs)
        )
        return nll
    
    try:
        # Test a few temperature values manually
        test_temps = [0.5, 1.0, 2.0, 5.0]
        print("   📊 Testing different temperatures:")
        for temp in test_temps:
            nll = negative_log_likelihood(temp)
            print(f"      T={temp}: NLL={nll:.4f}")
        
        # Optimize temperature between 0.1 and 10.0
        result = minimize_scalar(
            negative_log_likelihood, 
            bounds=(0.1, 10.0), 
            method='bounded'
        )
        
        print(f"   ✅ Optimal temperature: {result.x:.3f}")
        print(f"   📊 Optimal NLL: {result.fun:.4f}")
        return result.x
    except Exception as e:
        print(f"   ❌ Temperature optimization failed: {e}")
        return None

def test_temperature_scaled_model(model, optimal_temp, X_test, y_test):
    """Test the TemperatureScaledModel class"""
    print("🧪 Testing TemperatureScaledModel class...")
    
    try:
        # Create helper functions
        def get_model_logits_func(model, X):
            return test_get_model_logits(model, X)
        
        def temperature_scaling_func(logits, temperature):
            return test_temperature_scaling_function(logits, temperature)
        
        # Create temperature scaled model
        temp_model = TemperatureScaledModel(
            model, 
            optimal_temp,
            get_model_logits_func,
            temperature_scaling_func
        )
        
        # Test prediction
        print("   🔮 Testing predict_proba...")
        temp_probs_full = temp_model.predict_proba(X_test)
        temp_probs = temp_probs_full[:, 1]
        
        print(f"   📊 Temperature-scaled probs range: [{temp_probs.min():.3f}, {temp_probs.max():.3f}]")
        
        # Test discrete prediction
        print("   🔮 Testing predict...")
        temp_pred = temp_model.predict(X_test)
        print(f"   📊 Predictions: {np.bincount(temp_pred)}")
        
        # Calculate metrics
        brier = brier_score_loss(y_test, temp_probs)
        logloss = log_loss(y_test, temp_probs)
        
        print(f"   📊 Brier Score: {brier:.4f}")
        print(f"   📊 Log Loss: {logloss:.4f}")
        
        return temp_model, temp_probs
        
    except Exception as e:
        print(f"   ❌ TemperatureScaledModel test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

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

def main():
    """Main debugging function"""
    print("🐛 Debugging Temperature Scaling Implementation")
    print("=" * 55)
    
    try:
        # Load data
        data_splits, models = load_test_data()
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        print(f"📊 Data shapes: X_val={X_val.shape}, X_test={X_test.shape}")
        
        # Test each model
        for model_name, model in models.items():
            print(f"\n🧪 Testing Temperature Scaling for {model_name}")
            print("-" * 50)
            
            # Step 1: Test logit extraction
            try:
                val_logits = test_get_model_logits(model, X_val)
                test_logits = test_get_model_logits(model, X_test)
            except Exception as e:
                print(f"❌ Logit extraction failed: {e}")
                continue
            
            # Step 2: Test temperature optimization
            optimal_temp = test_temperature_optimization(val_logits, y_val)
            if optimal_temp is None:
                continue
            
            # Step 3: Test temperature scaling function
            scaled_probs = test_temperature_scaling_function(test_logits, optimal_temp)
            if scaled_probs is None:
                continue
            
            # Step 4: Test TemperatureScaledModel class
            temp_model, temp_probs = test_temperature_scaled_model(model, optimal_temp, X_test, y_test)
            if temp_model is None:
                continue
            
            # Step 5: Compare with original model
            original_probs = model.predict_proba(X_test)[:, 1]
            
            original_ece = calculate_ece(y_test, original_probs)
            temp_ece = calculate_ece(y_test, temp_probs)
            
            print(f"\n📊 Results Comparison:")
            print(f"   Original ECE: {original_ece:.4f}")
            print(f"   Temperature ECE: {temp_ece:.4f}")
            print(f"   Improvement: {original_ece - temp_ece:.4f}")
            print(f"   Optimal Temperature: {optimal_temp:.3f}")
            
            # Interpret temperature value
            if optimal_temp > 3.0:
                confidence_level = "severely overconfident"
            elif optimal_temp > 1.5:
                confidence_level = "moderately overconfident"
            elif optimal_temp < 0.7:
                confidence_level = "underconfident"
            else:
                confidence_level = "well-calibrated"
            
            print(f"   Model Assessment: {confidence_level}")
            
        print("\n✅ Temperature scaling debugging completed!")
        
    except Exception as e:
        print(f"❌ Debugging failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
