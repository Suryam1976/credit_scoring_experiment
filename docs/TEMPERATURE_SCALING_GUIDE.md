# Temperature Scaling: Complete Guide

## 🌡️ What is Temperature Scaling?

Temperature Scaling is a **post-hoc calibration method** that improves probability calibration by adding a single learnable parameter called "temperature" (T) to the model's output layer.

## 🧠 The Core Idea

The concept comes from **statistical mechanics** and **neural network training**, where "temperature" controls how "sharp" or "smooth" probability distributions are.

### Mathematical Foundation

```python
# Standard prediction
P(class) = softmax(logits) = exp(logit_i) / sum(exp(logit_j))

# Temperature Scaling
P(class) = softmax(logits / T) = exp(logit_i / T) / sum(exp(logit_j / T))

# For binary classification
P(positive) = sigmoid(logit / T) = 1 / (1 + exp(-logit / T))
```

Where:
- **T > 1**: Makes probabilities **less confident** (smoother distribution)
- **T < 1**: Makes probabilities **more confident** (sharper distribution)  
- **T = 1**: No change (original model)

## 🎯 Visual Intuition

### Effect of Different Temperatures

```python
import numpy as np
import matplotlib.pyplot as plt

# Original logits from a model
logits = np.array([-2, -1, 0, 1, 2, 3, 4])

# Different temperatures
temperatures = [0.5, 1.0, 2.0, 5.0]

for T in temperatures:
    probabilities = 1 / (1 + np.exp(-logits / T))
    plt.plot(logits, probabilities, label=f'T = {T}')

plt.legend()
plt.xlabel('Logits')
plt.ylabel('Probabilities')
plt.title('Effect of Temperature on Sigmoid Function')
```

**Result:**
```
Probability
     ↑
1.0  |     T=0.5 (sharp)
     |    ╱╲
0.8  |   ╱  ╲    T=1.0 (original)
     |  ╱    ╲   ╱╲
0.6  | ╱      ╲ ╱  ╲
     |╱        ╱    ╲   T=2.0 (smooth)
0.4  |        ╱      ╲ ╱╲
     |       ╱        ╱  ╲    T=5.0 (very smooth)
0.2  |      ╱        ╱    ╲ ╱╲
0.0  └─────────────────────────→
     -3    -1     0     1    3
              Logits

Lower T = Sharper, more confident
Higher T = Smoother, less confident
```

## 🔬 Mathematical Deep Dive

### How Temperature Affects Probabilities

```python
def temperature_scaling_demo():
    logit = 3.0  # A confident prediction
    
    temperatures = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for T in temperatures:
        prob = 1 / (1 + np.exp(-logit / T))
        print(f"T = {T:4.1f}: logit/T = {logit/T:5.2f}, P = {prob:.4f}")

# Output:
# T =  0.1: logit/T = 30.00, P = 1.0000  (extremely confident)
# T =  0.5: logit/T =  6.00, P = 0.9975  (very confident)
# T =  1.0: logit/T =  3.00, P = 0.9526  (original)
# T =  2.0: logit/T =  1.50, P = 0.8176  (less confident)
# T =  5.0: logit/T =  0.60, P = 0.6457  (much less confident)
# T = 10.0: logit/T =  0.30, P = 0.5744  (close to uncertain)
```

### Key Properties

1. **Monotonicity Preserved**: If logit_A > logit_B, then P_A > P_B (ranking unchanged)
2. **Single Parameter**: Only one parameter T to learn (simple, robust)
3. **Model Agnostic**: Works with any model that outputs logits
4. **Differentiable**: Can be optimized using gradient descent

## 🛠️ Implementation

### Basic Implementation

```python
import numpy as np
from scipy.optimize import minimize_scalar

def temperature_scaling(logits, temperature):
    """
    Apply temperature scaling to logits
    
    Args:
        logits: Raw model outputs (before sigmoid/softmax)
        temperature: Temperature parameter T
    
    Returns:
        Calibrated probabilities
    """
    return 1 / (1 + np.exp(-logits / temperature))

def find_optimal_temperature(val_logits, val_labels):
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
        probs = temperature_scaling(val_logits, temperature)
        
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

# Usage example
def calibrate_model_with_temperature(model, X_val, y_val, X_test):
    """
    Complete temperature scaling calibration pipeline
    """
    # Get logits from validation set
    if hasattr(model, 'decision_function'):
        # For SVM, logistic regression
        val_logits = model.decision_function(X_val)
        test_logits = model.decision_function(X_test)
    elif hasattr(model, 'predict_proba'):
        # Convert probabilities back to logits (approximation)
        val_probs = model.predict_proba(X_val)[:, 1]
        test_probs = model.predict_proba(X_test)[:, 1]
        val_logits = np.log(val_probs / (1 - val_probs + 1e-15))
        test_logits = np.log(test_probs / (1 - test_probs + 1e-15))
    
    # Find optimal temperature
    optimal_temp = find_optimal_temperature(val_logits, y_val)
    
    # Apply temperature scaling to test set
    calibrated_probs = temperature_scaling(test_logits, optimal_temp)
    
    return calibrated_probs, optimal_temp
```

### Advanced Implementation with Cross-Validation

```python
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin

class TemperatureScaledClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper that adds temperature scaling to any classifier
    """
    
    def __init__(self, base_classifier, cv=5):
        self.base_classifier = base_classifier
        self.cv = cv
        self.temperature_ = None
        
    def fit(self, X, y):
        """
        Fit base classifier and learn temperature using cross-validation
        """
        # Fit base classifier
        self.base_classifier.fit(X, y)
        
        # Get cross-validated logits for temperature learning
        cv_logits = self._get_cv_logits(X, y)
        
        # Learn optimal temperature
        self.temperature_ = find_optimal_temperature(cv_logits, y)
        
        return self
    
    def _get_cv_logits(self, X, y):
        """
        Get cross-validated logits to avoid overfitting
        """
        cv_logits = np.zeros(len(y))
        
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        for train_idx, val_idx in kf.split(X):
            # Train fold model
            fold_model = clone(self.base_classifier)
            fold_model.fit(X[train_idx], y[train_idx])
            
            # Get logits for validation fold
            if hasattr(fold_model, 'decision_function'):
                cv_logits[val_idx] = fold_model.decision_function(X[val_idx])
            else:
                probs = fold_model.predict_proba(X[val_idx])[:, 1]
                cv_logits[val_idx] = np.log(probs / (1 - probs + 1e-15))
        
        return cv_logits
    
    def predict_proba(self, X):
        """
        Predict calibrated probabilities
        """
        # Get base model logits
        if hasattr(self.base_classifier, 'decision_function'):
            logits = self.base_classifier.decision_function(X)
        else:
            probs = self.base_classifier.predict_proba(X)[:, 1]
            logits = np.log(probs / (1 - probs + 1e-15))
        
        # Apply temperature scaling
        calibrated_probs = temperature_scaling(logits, self.temperature_)
        
        # Return in sklearn format
        return np.column_stack([1 - calibrated_probs, calibrated_probs])
    
    def predict(self, X):
        """
        Predict classes using calibrated probabilities
        """
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# Usage
temp_scaled_model = TemperatureScaledClassifier(XGBClassifier())
temp_scaled_model.fit(X_train, y_train)
calibrated_probs = temp_scaled_model.predict_proba(X_test)[:, 1]
```

## 🎯 When Temperature Scaling Works Best

### ✅ Ideal Scenarios

1. **Neural Networks**: Originally designed for deep neural networks
2. **Gradient Boosting**: XGBoost, LightGBM with extreme confidence
3. **Large Models**: Complex models prone to overconfidence
4. **Sufficient Data**: Need enough validation data to learn temperature
5. **Systematic Miscalibration**: When model is consistently over/under-confident

### ❌ Less Effective When

1. **Already Well-Calibrated**: If ECE < 0.05, may not help much
2. **Very Small Datasets**: Not enough data to reliably estimate temperature
3. **Non-systematic Miscalibration**: Random calibration errors
4. **Models Without Logits**: Some models don't naturally output logits

## 📊 Comparison with Other Calibration Methods

| Aspect | Temperature Scaling | Platt Scaling | Isotonic Regression |
|--------|-------------------|---------------|-------------------|
| **Parameters** | 1 (temperature T) | 2 (A and B) | Many (non-parametric) |
| **Flexibility** | Low | Medium | High |
| **Overfitting Risk** | Very Low | Low | Medium |
| **Data Requirements** | Low | Low | High |
| **Computational Cost** | Very Fast | Fast | Medium |
| **Preserves Ranking** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Works with Extremes** | ✅ Excellent | ⚠️ Limited | ✅ Good |

## 🧪 Experimental Results

### Credit Scoring Example

```python
# Original XGBoost results
original_logits = [4.2, -3.8, 2.1, -2.9, 3.7]
original_probs = [0.985, 0.022, 0.891, 0.052, 0.976]  # Very extreme!
original_ece = 0.134

# Temperature scaling results
optimal_temperature = 2.3  # Learned from validation set
scaled_logits = [1.83, -1.65, 0.91, -1.26, 1.61]  # logits / T
calibrated_probs = [0.862, 0.161, 0.713, 0.221, 0.833]  # Much better!
calibrated_ece = 0.043  # 68% improvement!
```

### Business Impact

```python
# Portfolio analysis: 10,000 loans, $10K each
threshold = 0.3  # Approve if P(default) < 30%

# Original XGBoost
original_approved = 8200  # Very confident → approve many
original_expected_loss = 0.05 * 82M = $4.1M
original_actual_loss = 0.18 * 82M = $14.8M  # Surprise!
original_unexpected_loss = $10.7M  # 😱

# After Temperature Scaling  
calibrated_approved = 6800  # More realistic → approve fewer
calibrated_expected_loss = 0.15 * 68M = $10.2M
calibrated_actual_loss = 0.16 * 68M = $10.9M
calibrated_unexpected_loss = $0.7M  # Much better! 🎯

# Risk Reduction: $10.0M saved!
```

## 💡 Why Temperature Scaling is So Effective

### 1. **Addresses Root Cause**
- Many models output overconfident logits
- Temperature scaling directly addresses this at the logit level
- Simple but theoretically sound

### 2. **Minimal Overfitting**
- Only 1 parameter to learn
- Hard to overfit with single parameter
- Generalizes well to test data

### 3. **Preserves Model Performance**
- Doesn't change ranking of predictions
- Maintains all accuracy metrics (precision, recall, F1)
- Only improves calibration

### 4. **Fast and Simple**
- No complex optimization
- Easy to implement and understand
- Computationally efficient

## 🎯 Best Practices

### 1. **Validation Set Strategy**
```python
# Use separate calibration set
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)

# Train on X_train
# Learn temperature on X_val
# Evaluate on X_test
```

### 2. **Cross-Validation for Small Datasets**
```python
# Use CV when validation set would be too small
temp_model = TemperatureScaledClassifier(base_model, cv=5)
temp_model.fit(X_train, y_train)  # Uses CV internally
```

### 3. **Monitor Temperature Value**
```python
if optimal_temperature > 5.0:
    print("⚠️ Very high temperature - model is severely overconfident")
elif optimal_temperature < 0.5:
    print("⚠️ Very low temperature - model may be underconfident")
else:
    print(f"✅ Reasonable temperature: {optimal_temperature:.2f}")
```

### 4. **Combine with Other Methods**
```python
# Multi-step calibration for very poorly calibrated models
def advanced_calibration(model, X_val, y_val, X_test):
    # Step 1: Temperature scaling
    temp_probs, temp = calibrate_with_temperature(model, X_val, y_val, X_test)
    
    # Step 2: Isotonic regression on temperature-scaled output
    iso_calibrator = IsotonicRegression()
    iso_calibrator.fit(temp_probs, y_test)  # Use held-out test for demo
    
    final_probs = iso_calibrator.predict(temp_probs)
    return final_probs
```

## 🎯 Key Takeaways

1. **Temperature Scaling is Simple**: Just one parameter, but very effective

2. **Perfect for Overconfident Models**: Especially neural networks and GBDTs

3. **Preserves Accuracy**: Maintains all ranking-based metrics

4. **Low Overfitting Risk**: Single parameter is hard to overfit

5. **Fast and Reliable**: Quick to compute, stable results

6. **Not a Silver Bullet**: Works best for systematic over/under-confidence

7. **Often First Choice**: Try temperature scaling before more complex methods

**Temperature Scaling turns overconfident models into well-calibrated ones with minimal complexity - it's the "Occam's Razor" of calibration methods!** 🌡️🎯