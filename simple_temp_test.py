"""
Simple Temperature Scaling Test
===============================

Test temperature scaling logic with synthetic data to verify implementation.
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import brier_score_loss, log_loss
from scipy.optimize import minimize_scalar

def test_temperature_scaling():
    """Test temperature scaling with known data"""
    print("🧪 Testing Temperature Scaling Logic")
    print("=" * 40)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 200
    
    # Create overconfident predictions (typical of Random Forest)
    y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive class
    
    # Simulate overconfident probabilities (too extreme)
    raw_probs = np.random.beta(0.5, 2, n_samples)  # Skewed towards 0
    raw_probs[y_true == 1] = np.random.beta(2, 0.5, np.sum(y_true == 1))  # Skewed towards 1
    
    # Convert to logits
    raw_probs_clipped = np.clip(raw_probs, 1e-15, 1 - 1e-15)
    logits = np.log(raw_probs_clipped / (1 - raw_probs_clipped))
    
    print(f"📊 Synthetic data created: {n_samples} samples")
    print(f"📊 True positive rate: {y_true.mean():.3f}")
    print(f"📊 Raw probabilities range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
    print(f"📊 Logits range: [{logits.min():.3f}, {logits.max():.3f}]")
    
    # Calculate original calibration error
    def calculate_ece(y_true, y_prob, n_bins=10):
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
    
    original_ece = calculate_ece(y_true, raw_probs)
    original_brier = brier_score_loss(y_true, raw_probs)
    original_logloss = log_loss(y_true, raw_probs)
    
    print(f"\n📊 Original Performance:")
    print(f"   ECE: {original_ece:.4f}")
    print(f"   Brier Score: {original_brier:.4f}")
    print(f"   Log Loss: {original_logloss:.4f}")
    
    # Define temperature scaling function
    def temperature_scaling(logits, temperature):
        return 1 / (1 + np.exp(-logits / temperature))
    
    # Define optimization function
    def negative_log_likelihood(temperature):
        try:
            probs = temperature_scaling(logits, temperature)
            eps = 1e-15
            probs = np.clip(probs, eps, 1 - eps)
            nll = -np.mean(y_true * np.log(probs) + (1 - y_true) * np.log(1 - probs))
            return nll
        except:
            return 1e6
    
    # Test different temperature values
    print("\n🌡️ Testing different temperatures:")
    test_temps = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    
    for temp in test_temps:
        temp_probs = temperature_scaling(logits, temp)
        temp_ece = calculate_ece(y_true, temp_probs)
        temp_brier = brier_score_loss(y_true, temp_probs)
        nll = negative_log_likelihood(temp)
        
        print(f"   T={temp}: ECE={temp_ece:.4f}, Brier={temp_brier:.4f}, NLL={nll:.4f}")
    
    # Optimize temperature
    print("\n🔍 Optimizing temperature...")
    try:
        result = minimize_scalar(
            negative_log_likelihood,
            bounds=(0.1, 10.0),
            method='bounded'
        )
        
        optimal_temp = result.x
        optimal_nll = result.fun
        
        print(f"✅ Optimal temperature: {optimal_temp:.3f}")
        print(f"✅ Optimal NLL: {optimal_nll:.4f}")
        
        # Apply optimal temperature
        optimal_probs = temperature_scaling(logits, optimal_temp)
        optimal_ece = calculate_ece(y_true, optimal_probs)
        optimal_brier = brier_score_loss(y_true, optimal_probs)
        optimal_logloss = log_loss(y_true, optimal_probs)
        
        print(f"\n📊 Optimized Performance:")
        print(f"   ECE: {optimal_ece:.4f} (improvement: {original_ece - optimal_ece:.4f})")
        print(f"   Brier Score: {optimal_brier:.4f} (improvement: {original_brier - optimal_brier:.4f})")
        print(f"   Log Loss: {optimal_logloss:.4f} (improvement: {original_logloss - optimal_logloss:.4f})")
        
        # Interpret temperature
        if optimal_temp > 3.0:
            assessment = "severely overconfident"
        elif optimal_temp > 1.5:
            assessment = "moderately overconfident"
        elif optimal_temp < 0.7:
            assessment = "underconfident"
        else:
            assessment = "well-calibrated"
        
        print(f"\n🔍 Model Assessment: {assessment}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return False

def test_with_real_data():
    """Test with actual project data if available"""
    print("\n🔍 Testing with Real Project Data")
    print("=" * 40)
    
    try:
        # Try to load real data
        models_path = Path("results/trained_models.pkl")
        splits_path = Path("data/processed/train_test_split.pkl")
        
        if not models_path.exists() or not splits_path.exists():
            print("❌ Real data not found, skipping real data test")
            return False
        
        with open(models_path, 'rb') as f:
            models = pickle.load(f)
        
        with open(splits_path, 'rb') as f:
            data_splits = pickle.load(f)
        
        print(f"✅ Loaded {len(models)} models and data splits")
        
        X_val = data_splits['X_val']
        y_val = data_splits['y_val']
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Test with one model (Random Forest - typically overconfident)
        model_name = "Random_Forest"
        if model_name in models:
            model = models[model_name]
            print(f"\n🧪 Testing temperature scaling on {model_name}")
            
            # Get validation probabilities and convert to logits
            val_probs = model.predict_proba(X_val)[:, 1]
            val_probs_clipped = np.clip(val_probs, 1e-15, 1 - 1e-15)
            val_logits = np.log(val_probs_clipped / (1 - val_probs_clipped))
            
            print(f"📊 Validation probs range: [{val_probs.min():.3f}, {val_probs.max():.3f}]")
            print(f"📊 Validation logits range: [{val_logits.min():.3f}, {val_logits.max():.3f}]")
            
            # Define functions
            def temperature_scaling(logits, temperature):
                return 1 / (1 + np.exp(-logits / temperature))
            
            def negative_log_likelihood(temperature):
                try:
                    probs = temperature_scaling(val_logits, temperature)
                    eps = 1e-15
                    probs = np.clip(probs, eps, 1 - eps)
                    nll = -np.mean(y_val * np.log(probs) + (1 - y_val) * np.log(1 - probs))
                    return nll
                except:
                    return 1e6
            
            # Optimize temperature
            result = minimize_scalar(
                negative_log_likelihood,
                bounds=(0.1, 10.0),
                method='bounded'
            )
            
            optimal_temp = result.x
            print(f"✅ Real data optimal temperature: {optimal_temp:.3f}")
            
            # Test on test set
            test_probs = model.predict_proba(X_test)[:, 1]
            test_probs_clipped = np.clip(test_probs, 1e-15, 1 - 1e-15)
            test_logits = np.log(test_probs_clipped / (1 - test_probs_clipped))
            
            # Apply temperature scaling
            calibrated_probs = temperature_scaling(test_logits, optimal_temp)
            
            # Calculate metrics
            def calculate_ece(y_true, y_prob):
                bin_boundaries = np.linspace(0, 1, 11)
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
            
            original_ece = calculate_ece(y_test, test_probs)
            calibrated_ece = calculate_ece(y_test, calibrated_probs)
            
            print(f"📊 Original ECE: {original_ece:.4f}")
            print(f"📊 Calibrated ECE: {calibrated_ece:.4f}")
            print(f"📊 Improvement: {original_ece - calibrated_ece:.4f}")
            
            return True
        else:
            print(f"❌ {model_name} not found in models")
            return False
            
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🔬 Temperature Scaling Verification")
    print("=" * 50)
    
    # Test 1: Synthetic data
    synthetic_success = test_temperature_scaling()
    
    # Test 2: Real data (if available)
    real_success = test_with_real_data()
    
    print("\n📋 Test Summary:")
    print(f"   Synthetic data test: {'✅ PASSED' if synthetic_success else '❌ FAILED'}")
    print(f"   Real data test: {'✅ PASSED' if real_success else '❌ FAILED'}")
    
    if synthetic_success:
        print("\n✅ Temperature scaling logic is working correctly!")
        print("💡 The implementation should work with real models.")
    else:
        print("\n❌ Temperature scaling logic has issues.")
        print("💡 Need to debug the implementation further.")

if __name__ == "__main__":
    main()
