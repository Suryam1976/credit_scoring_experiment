# Platt Scaling vs Isotonic Regression: Detailed Comparison

## 📚 Overview

Both Platt Scaling and Isotonic Regression are **post-hoc calibration methods** used to improve the reliability of predicted probabilities from machine learning models. While they serve the same purpose, they use fundamentally different mathematical approaches.

## 🔬 Mathematical Foundations

### Platt Scaling (Sigmoid Calibration)

**Mathematical Form:**
```
P_calibrated = 1 / (1 + exp(A * P_uncalibrated + B))
```

Where:
- `A` and `B` are parameters learned from validation data
- Uses a **parametric sigmoid function**
- Assumes the calibration mapping follows a specific S-shaped curve

**Learning Process:**
1. Take uncalibrated probabilities from validation set
2. Fit sigmoid parameters A and B using maximum likelihood estimation
3. Apply the learned sigmoid transformation to new predictions

### Isotonic Regression (Non-parametric Calibration)

**Mathematical Form:**
```
P_calibrated = f(P_uncalibrated)
```

Where:
- `f` is a **non-parametric monotonic function**
- No assumption about the specific shape of the calibration curve
- Only constraint: the function must be monotonically increasing

**Learning Process:**
1. Sort uncalibrated probabilities from validation set
2. Find the best monotonic mapping that minimizes prediction error
3. Use this learned mapping (stored as a lookup table) for new predictions

## 🎯 Key Differences

| Aspect | Platt Scaling | Isotonic Regression |
|--------|---------------|-------------------|
| **Approach** | Parametric (2 parameters) | Non-parametric |
| **Shape Assumption** | Sigmoid/S-curve | Any monotonic shape |
| **Flexibility** | Less flexible | More flexible |
| **Data Requirements** | Works with small datasets | Needs more data points |
| **Overfitting Risk** | Lower (only 2 parameters) | Higher (more complex) |
| **Computational Cost** | Fast | Moderate |
| **Interpretation** | Simple sigmoid transform | Complex lookup function |

## 📊 When to Use Each Method

### Use **Platt Scaling** When:
✅ **Small validation sets** (< 1000 samples)  
✅ **Simple calibration needed** (sigmoid-like miscalibration)  
✅ **Computational efficiency important**  
✅ **SVM models** (originally designed for SVMs)  
✅ **Interpretability matters** (simple 2-parameter model)  
✅ **Smooth prediction surfaces desired**  

### Use **Isotonic Regression** When:
✅ **Large validation sets** (> 1000 samples)  
✅ **Complex miscalibration patterns**  
✅ **Unknown calibration shape**  
✅ **Tree-based models** (Random Forest, XGBoost)  
✅ **Maximum flexibility needed**  
✅ **Non-sigmoid miscalibration patterns**  

## 🔬 Practical Examples

### Example 1: SVM with Extreme Probabilities

**Problem:** SVM outputs probabilities like [0.01, 0.05, 0.95, 0.99]
```python
# Platt Scaling approach
from sklearn.calibration import CalibratedClassifierCV

# Fits sigmoid: P_cal = 1/(1 + exp(A*P + B))
platt_calibrated = CalibratedClassifierCV(svm_model, method='sigmoid', cv='prefit')
platt_calibrated.fit(X_val, y_val)

# Result: Smooth sigmoid mapping
# [0.01, 0.05, 0.95, 0.99] → [0.12, 0.25, 0.75, 0.88]
```

### Example 2: Random Forest with Complex Patterns

**Problem:** Random Forest has non-linear miscalibration
```python
# Isotonic Regression approach
isotonic_calibrated = CalibratedClassifierCV(rf_model, method='isotonic', cv='prefit')
isotonic_calibrated.fit(X_val, y_val)

# Result: Flexible monotonic mapping
# Can handle any monotonic miscalibration pattern
```

## 📈 Calibration Curves Comparison

### Platt Scaling Calibration Curve
```
Actual Probability
     ↑
1.0  |     ╭─────╮
     |   ╭─╯     ╰─╮
0.5  | ╭─╯         ╰─╮
     |╱             ╲
0.0  └─────────────────→
     0.0           1.0
         Predicted Probability

Shape: Smooth S-curve (sigmoid)
Constraint: Must follow sigmoid shape
```

### Isotonic Regression Calibration Curve
```
Actual Probability
     ↑
1.0  |   ╭─╮  ╭────╮
     |  ╱   ╲╱     ╲
0.5  | ╱           ╲
     |╱             ╲
0.0  └─────────────────→
     0.0           1.0
         Predicted Probability

Shape: Any monotonic curve
Constraint: Only monotonically increasing
```

## 🧪 Code Implementation Examples

### Platt Scaling Implementation
```python
import numpy as np
from scipy.optimize import minimize
from sklearn.calibration import calibration_curve

def platt_scaling(y_true, y_prob):
    """
    Implement Platt Scaling manually
    """
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def objective(params):
        A, B = params
        y_pred = sigmoid(A * y_prob + B)
        # Negative log-likelihood
        return -np.sum(y_true * np.log(y_pred + 1e-15) + 
                      (1 - y_true) * np.log(1 - y_pred + 1e-15))
    
    # Optimize A and B parameters
    result = minimize(objective, [1.0, 0.0], method='BFGS')
    A, B = result.x
    
    return A, B

# Usage
A, B = platt_scaling(y_val, prob_val)
prob_calibrated = 1 / (1 + np.exp(-(A * prob_test + B)))
```

### Isotonic Regression Implementation
```python
from sklearn.isotonic import IsotonicRegression

def isotonic_calibration(y_true, y_prob):
    """
    Implement Isotonic Regression calibration
    """
    # Fit isotonic regression
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(y_prob, y_true)
    
    return iso_reg

# Usage
iso_calibrator = isotonic_calibration(y_val, prob_val)
prob_calibrated = iso_calibrator.predict(prob_test)
```

## 📊 Performance Comparison in Credit Scoring

### Typical Results on Credit Data

| Model | Original ECE | Platt Scaling ECE | Isotonic ECE | Best Method |
|-------|--------------|-------------------|--------------|-------------|
| **Logistic Regression** | 0.087 | 0.045 | 0.052 | Platt |
| **Random Forest** | 0.156 | 0.078 | 0.048 | Isotonic |
| **SVM** | 0.143 | 0.041 | 0.055 | Platt |
| **XGBoost** | 0.134 | 0.089 | 0.043 | Isotonic |

### Observations:
- **Tree-based models** (RF, XGBoost) → Isotonic works better
- **Linear models** (LR, SVM) → Platt scaling works better
- **Complex patterns** → Isotonic more flexible
- **Simple patterns** → Platt less prone to overfitting

## ⚖️ Advantages and Disadvantages

### Platt Scaling

**Advantages:**
✅ Simple and interpretable (2 parameters)  
✅ Fast computation  
✅ Works well with small datasets  
✅ Less prone to overfitting  
✅ Smooth, continuous output  
✅ Originally designed for SVMs  

**Disadvantages:**
❌ Assumes sigmoid-shaped miscalibration  
❌ Less flexible for complex patterns  
❌ May not capture non-sigmoid relationships  
❌ Can fail when assumption is violated  

### Isotonic Regression

**Advantages:**
✅ Very flexible - any monotonic shape  
✅ No parametric assumptions  
✅ Can capture complex calibration patterns  
✅ Often works better for tree-based models  
✅ Optimal for non-sigmoid miscalibration  

**Disadvantages:**
❌ Requires more data  
❌ More prone to overfitting  
❌ Less interpretable  
❌ Can produce non-smooth outputs  
❌ Computationally more expensive  

## 🎯 Decision Framework

### Choose Platt Scaling If:
```python
if (dataset_size < 1000 or 
    model_type in ['SVM', 'LogisticRegression'] or
    calibration_pattern == 'sigmoid-like' or
    interpretability_required):
    use_platt_scaling()
```

### Choose Isotonic Regression If:
```python
if (dataset_size > 1000 and 
    model_type in ['RandomForest', 'XGBoost', 'DecisionTree'] and
    flexibility_needed and
    calibration_pattern == 'complex'):
    use_isotonic_regression()
```

## 🔬 Advanced Considerations

### Combining Both Methods
```python
# Ensemble calibration approach
platt_probs = platt_calibrator.predict_proba(X_test)[:, 1]
isotonic_probs = isotonic_calibrator.predict_proba(X_test)[:, 1]

# Weighted combination
alpha = 0.6  # Learned from validation
ensemble_probs = alpha * platt_probs + (1 - alpha) * isotonic_probs
```

### Cross-Validation Strategy
```python
from sklearn.model_selection import cross_val_predict

# Use cross-validation to avoid overfitting
cv_probs = cross_val_predict(model, X_train, y_train, 
                           cv=5, method='predict_proba')[:, 1]

# Fit calibration on CV predictions
platt_calibrator.fit(cv_probs.reshape(-1, 1), y_train)
```

## 💡 Key Takeaways

1. **Model Type Matters**: Tree models → Isotonic, Linear models → Platt
2. **Data Size Matters**: Small datasets → Platt, Large datasets → Isotonic  
3. **Pattern Complexity**: Simple sigmoid → Platt, Complex patterns → Isotonic
4. **Always Validate**: Test both methods on your specific problem
5. **Consider Ensemble**: Combining both can sometimes work better

## 🎯 Business Impact Example

### Credit Scoring Scenario:
```
Original Random Forest ECE: 0.156
- Predicted Loss: $1.2M
- Actual Loss: $3.8M  
- Surprise: $2.6M

After Platt Scaling ECE: 0.078
- Predicted Loss: $2.1M
- Actual Loss: $2.4M
- Surprise: $300K (88% improvement)

After Isotonic Regression ECE: 0.048  
- Predicted Loss: $2.3M
- Actual Loss: $2.4M
- Surprise: $100K (96% improvement)

Winner: Isotonic Regression saves additional $200K
```

The choice between Platt Scaling and Isotonic Regression can have significant business impact - choose wisely based on your data characteristics and model type! 🎯