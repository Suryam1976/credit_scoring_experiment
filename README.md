# Credit Scoring Experiment: Accuracy vs Calibration

A comprehensive experimental framework demonstrating the critical difference between model accuracy and calibration in credit risk assessment.

## 🎯 Project Overview

This project provides a hands-on demonstration of why **calibration** is as important as **accuracy** in machine learning applications that rely on probability estimates. Using credit scoring as a real-world example, we show how poorly calibrated models can lead to significant financial losses despite high accuracy scores.

## 🔑 Key Concepts

- **Accuracy**: How often a model makes correct predictions
- **Calibration**: How well predicted probabilities match actual outcomes
- **Business Impact**: Financial consequences of miscalibrated models
- **Post-hoc Calibration**: Methods to improve probability estimates

## 📊 Dataset

**German Credit Dataset** (UCI Repository)
- 1,000 credit applicants
- 20 features (age, credit history, employment, etc.)
- Binary target: Good credit (70%) vs Bad credit (30%)
- Publicly available and well-documented

## 📁 Project Structure

```
credit_scoring_experiment/
├── 📁 docs/                    # Documentation
│   ├── VISION_DOCUMENT.md      # Project vision and objectives
│   ├── FLOW_DIAGRAM.md         # Process flow diagrams
│   ├── API_DOCUMENTATION.md    # Code documentation
│   └── TEMPERATURE_SCALING_GUIDE.md # Guide to temperature scaling
├── 📁 data/                    # Data storage
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and split data
├── 📁 models/                  # Model training and evaluation
│   ├── train_models.py         # Model training pipeline
│   ├── calibration.py          # Calibration methods
│   ├── model_utils.py          # Shared model utilities
│   ├── evaluation.py           # Metrics and testing
│   └── model_configs.json      # Model configurations
├── 📁 visualization/           # Plotting and visualization
│   ├── reliability_plots.py    # Calibration curve plots
│   ├── business_impact.py      # Financial impact analysis
│   └── probability_distributions.py  # Prediction distributions
├── 📁 results/                 # Analysis outputs
│   ├── model_metrics.csv       # Performance comparisons
│   ├── model_predictions.pkl   # Saved model predictions
│   ├── calibration_comparison.csv # Calibration analysis
│   └── visualizations/         # Generated plots
├── 📁 notebooks/               # Interactive analysis
│   ├── 01_data_exploration.ipynb     # Data analysis
│   ├── 02_model_training.ipynb       # Model development
│   ├── 03_calibration_analysis.ipynb # Calibration study
│   └── 04_business_impact.ipynb     # Business case analysis
├── setup_dirs.py               # Directory setup script
├── run_calibration_pipeline.py # Complete pipeline runner
├── README.md                   # This file
└── requirements.txt           # Python dependencies
```

## 🚀 Quick Start

### 1. Setup and Installation
```bash
cd C:\Users\avssm\credit_scoring_experiment
pip install -r requirements.txt
python setup_dirs.py  # Creates all necessary directories
```

### 2. Run the Experiment
```bash
# Option A: Full pipeline with a single command
python run_calibration_pipeline.py

# Option B: Step by step
python models/train_models.py --prepare-data --train-all
python models/calibration.py
python visualization/reliability_plots.py
python visualization/business_impact.py
```

### 3. Explore the Results
```bash
# View the generated visualizations in:
results/visualizations/

# Examine the calibration metrics in:
results/calibration_comparison.csv
```

## 🎓 Key Experiments

### Experiment 1: The Overconfident Classifier
**Random Forest Model**
- ✅ High accuracy (87%)
- ❌ Poor calibration (ECE = 0.18)
- 📈 Predicts extreme probabilities (0.05, 0.95)
- 💸 Business impact: $2M unexpected losses

### Experiment 2: Post-hoc Calibration Success
**SVM with Platt Scaling**
- ✅ Moderate accuracy (82%)
- ✅ Excellent calibration (ECE = 0.045)
- 📈 Well-balanced probabilities
- 💰 Business impact: Best risk management

### Experiment 3: The Surprising Result
**Logistic Regression Model**
- ✅ Good accuracy (84%)
- ⚠️ Surprisingly poor calibration (ECE = 0.087)
- 📊 Shows that "naturally calibrated" isn't always true
- 💸 Business impact: Moderate risk

### Experiment 4: Temperature Scaling Breakthrough
**Advanced Calibration Method**
- 🌡️ Single parameter (temperature T) calibration
- ✅ Designed specifically for overconfident models
- 📈 Handles extreme probabilities from GBDTs
- 💰 Business impact: Often best for tree-based models

### Experiment 5: Post-hoc Calibration
**Calibration Methods Applied**
- 🔧 Platt Scaling: Sigmoid calibration
- 📈 Isotonic Regression: Monotonic mapping
- 🌡️ Temperature Scaling: Neural network approach

## 📈 Expected Results

| Model | Accuracy | ECE | Brier Score | Business Impact |
|-------|----------|-----|-------------|----------------|
| SVM + Platt Scaling | 82% | 0.045 | 0.198 | ✅ Best Calibrated |
| Random Forest + Isotonic | 87% | 0.052 | 0.189 | ✅ Good Balance |
| Logistic Regression | 84% | 0.087 | 0.183 | ⚠️ Surprisingly Poor |
| Random Forest (Original) | 87% | 0.15+ | 0.22 | ❌ High Risk |

## 💼 Business Impact Analysis

### Scenario: Loan Approval System
- **Portfolio**: 10,000 loans at $10K each
- **Decision Rule**: Approve if P(default) < 30%

### Financial Impact
```
Well-Calibrated Model:
├── Predicted Loss: $2.0M
├── Actual Loss: $2.2M  
└── Surprise: $200K (manageable)

Overconfident Model:
├── Predicted Loss: $500K
├── Actual Loss: $2.5M
└── Surprise: $2.0M (significant risk!)
```

### ROI of Calibration
- **Cost**: $50K (development + validation)
- **Benefit**: $2M (risk reduction)
- **ROI**: 4,000% return on investment

## 📊 Key Visualizations

### 1. Reliability Diagrams
Shows predicted vs actual probabilities across different bins
- Perfect calibration = diagonal line
- Overconfident models = below diagonal
- Underconfident models = above diagonal

### 2. Probability Distributions
Displays how each model distributes probability predictions
- Well-calibrated: Smooth, realistic distribution
- Overconfident: Extreme values near 0 and 1

### 3. Business Impact Dashboard
Interactive visualization of financial implications
- Expected vs actual losses by model
- Sensitivity analysis for different thresholds
- ROI calculations for calibration methods

## 🔬 Technical Implementation

### Models Implemented
- **Logistic Regression**: Naturally well-calibrated baseline
- **Random Forest**: High accuracy, poor calibration
- **Support Vector Machine**: Deliberately miscalibrated example

### Calibration Methods
- **Platt Scaling**: Sigmoid function fitting
- **Isotonic Regression**: Monotonic probability mapping  
- **Temperature Scaling**: Single-parameter logit scaling (ideal for overconfident models)

### Implementation Notes
- Models are saved as predictions rather than full objects to avoid serialization issues
- Python path is configured in each script to enable proper imports
- The `TemperatureScaledModel` class is defined in `models/model_utils.py` for sharing across modules

## 🔧 Troubleshooting

### Common Issues
- **Import Errors**: Make sure to run scripts from the project root directory
- **Missing Directories**: Run `python setup_dirs.py` to create all required folders
- **Serialization Errors**: If you modify class structures, you may need to rerun the full pipeline

### Solutions
- Use `python run_calibration_pipeline.py` for the most reliable execution
- Check `PROJECT_UPDATES.md` for recent fixes and improvements
- Ensure all dependencies are installed with `pip install -r requirements.txt`

## 📚 Further Reading

- See `docs/TEMPERATURE_SCALING_GUIDE.md` for details on temperature scaling
- See `docs/FLOW_DIAGRAM.md` for detailed process flow diagrams
- See `docs/VISION_DOCUMENT.md` for project objectives and design principles

## 📝 License

This project is designed for educational purposes. The German Credit Dataset is publicly available through the UCI Machine Learning Repository.

## 📬 Contact

For questions about this experiment or suggestions for improvements, please refer to the documentation in the `docs/` directory or examine the interactive notebooks for detailed explanations of each component.

---

**Remember**: In probability-sensitive applications, a well-calibrated model with 84% accuracy is often more valuable than a poorly calibrated model with 90% accuracy. This experiment proves why! 🎯
