# 🚀 Quick Start Guide

This guide will help you run the Credit Scoring Calibration Experiment in just a few steps.

## ⚡ Quick Execution

### Option 1: Full Automated Run
```bash
# Navigate to the project directory
cd C:\Users\avssm\credit_scoring_experiment

# Install dependencies
pip install -r requirements.txt

# Set up directories (ensures all folders exist)
python setup_dirs.py

# Run the complete experiment pipeline
python run_calibration_pipeline.py
```

### Option 2: Step-by-Step Execution
```bash
# Navigate to the project directory
cd C:\Users\avssm\credit_scoring_experiment

# Install dependencies
pip install -r requirements.txt

# Set up directories
python setup_dirs.py

# Step 1: Download and prepare data
python models/train_models.py --prepare-data

# Step 2: Train models
python models/train_models.py --train-all

# Step 3: Run calibration analysis
python models/calibration.py

# Step 4: Generate visualizations
python visualization/reliability_plots.py
python visualization/business_impact.py
```

### Option 3: Interactive Jupyter Notebook
```bash
# Navigate to the project directory
cd C:\Users\avssm\credit_scoring_experiment

# Install dependencies
pip install -r requirements.txt

# Set up directories
python setup_dirs.py

# Launch Jupyter
jupyter notebook notebooks/01_calibration_experiment.ipynb
```

## 📋 Expected Output

After running the pipeline, you should see the following outputs:

### 1. Console Output
```
🚀 Running calibration pipeline...

📊 Step 1: Running calibration analysis...
📂 Loading trained models...
✅ Models loaded successfully
🔍 Calibrating LogisticRegression...
🔍 Calibrating RandomForest...
🔍 Calibrating SVM_RBF...
📊 Creating summary table...
📊 Saving calibration results...
✅ Calibration analysis completed successfully!

🎨 Step 2: Generating visualizations...
📂 Loading calibration results...
✅ Results loaded successfully
📈 Creating reliability diagrams...
✅ Reliability diagrams created
🎉 All visualizations generated successfully!
📁 Check results/visualizations for all visualization files

✅ Pipeline completed successfully!
```

### 2. Generated Files
```
results/
├── model_metrics.csv              # Basic model performance comparison
├── model_predictions.pkl          # Saved model predictions
├── calibration_comparison.csv     # Detailed calibration analysis
├── business_impact_analysis.csv   # Financial impact assessment  
└── visualizations/
    ├── reliability_diagrams.png   # Calibration curves
    └── [other generated plots]
```

## 🔍 Viewing Results

### Reliability Diagrams
Open `results/visualizations/reliability_diagrams.png` to see how well each model's probabilities match actual outcomes.

### Calibration Metrics
Open `results/calibration_comparison.csv` to see detailed metrics for each model and calibration method.

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```
ModuleNotFoundError: No module named 'models'
```

**Solution**: Make sure to run scripts from the project root directory, not from inside subdirectories.

#### 2. Missing Directories
```
FileNotFoundError: [Errno 2] No such file or directory: 'results/trained_models.pkl'
```

**Solution**: Run `python setup_dirs.py` to create all required directories.

#### 3. Serialization Errors
```
AttributeError: Can't get attribute 'TemperatureScaledModel' on <module '__main__'...
```

**Solution**: Run the full pipeline with `python run_calibration_pipeline.py` to ensure consistent serialization.

## 🎓 Next Steps

After running the experiment:

1. Examine the reliability diagrams to understand calibration differences
2. Compare the metrics in `calibration_comparison.csv`
3. Explore the business impact analysis to see financial implications
4. Try modifying model parameters to see how they affect calibration

For more detailed information, refer to the documentation in the `docs/` directory.

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Solution: Install missing packages
   pip install -r requirements.txt
   ```

2. **Directory Not Found Errors**
   ```bash
   # Solution: Run setup script first
   python setup_dirs.py
   ```

3. **File Not Found Errors**
   ```bash
   # Solution: Run steps in order
   python models/train_models.py --prepare-data
   python models/train_models.py --train-all
   python models/calibration.py
   ```

4. **Calibration Script Can't Find Models**
   ```bash
   # Solution: Check if models were saved properly
   dir results\trained_models.pkl  # Windows
   ls results/trained_models.pkl   # Mac/Linux
   # If missing, re-run training:
   python models/train_models.py --train-all
   ```

5. **Conda Environment Issues**
   ```bash
   # Solution: Use pip directly instead of conda
   pip install -r requirements.txt
   # Or initialize conda first:
   conda init cmd.exe
   ```

6. **Visualization Issues**
   ```bash
   # Solution: Install visualization dependencies
   pip install matplotlib seaborn plotly
   ```

7. **Memory Issues**
   ```bash
   # Solution: The dataset is small (1000 samples), but if you have issues:
   # - Close other applications
   # - Use a smaller subset of models
   ```

### Path-Related Issues

1. **Working Directory Problems**
   ```bash
   # Always run from the project root:
   cd C:\Users\avssm\credit_scoring_experiment
   # Then run commands from there
   ```

2. **Relative Path Issues**
   ```bash
   # The scripts now auto-detect paths, but if issues persist:
   # Check that you're in the right directory:
   cd  # Shows current directory
   ```

### Recent Fixes Applied

✅ **Fixed Model Save/Load Issues**: Models now save to `results/trained_models.pkl`  
✅ **Fixed Directory Creation**: Added `setup_dirs.py` to create all necessary folders  
✅ **Fixed Path Resolution**: Scripts now work from different working directories  
✅ **Fixed Conda Issues**: Added pip-based installation as primary option

### Jupyter Notebook Issues

1. **Kernel Not Found**
   ```bash
   python -m ipykernel install --user
   ```

2. **Module Import Errors**
   ```python
   # Add these lines at the top of notebooks
   import sys
   sys.path.append('../models')
   sys.path.append('../visualization')
   ```

## 🎓 Educational Value

This experiment demonstrates:

1. **Accuracy ≠ Calibration**: High accuracy doesn't guarantee reliable probabilities
2. **Business Impact**: Poor calibration has real financial consequences
3. **Calibration Methods**: Practical techniques to improve probability reliability
4. **Measurement**: How to assess and validate model calibration
5. **Decision Making**: Why calibration matters for probability-based decisions

## 🔧 Customization

### Using Your Own Dataset
1. Replace the data loading code in `models/train_models.py`
2. Ensure your target is binary (0/1)
3. Adjust feature preprocessing as needed
4. Update business scenario parameters in `visualization/business_impact.py`

### Adding Models
1. Add new model configurations in `models/train_models.py`
2. Update the `model_configs` dictionary
3. Ensure the model has `predict_proba()` method

### Modifying Business Scenario
Edit these parameters in `visualization/business_impact.py`:
```python
portfolio_size = 10000      # Number of loans
avg_loan_amount = 10000     # Average loan size
approval_threshold = 0.3    # Risk threshold for approval
```

## 📚 Next Steps

After completing this experiment:

1. **Apply to your domain**: Use this framework on your own data
2. **Explore advanced methods**: Temperature scaling, ensemble calibration
3. **Production considerations**: Monitor calibration drift over time
4. **Fairness**: Ensure calibration across different demographic groups
5. **MLOps integration**: Include calibration in your model pipeline

## 💡 Key Takeaway

> **In probability-sensitive applications, a well-calibrated model with 84% accuracy is often more valuable than a poorly calibrated model with 90% accuracy.**

This experiment proves why calibration should be a standard part of every machine learning evaluation! 🎯
