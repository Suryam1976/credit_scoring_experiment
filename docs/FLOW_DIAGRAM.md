# Experimental Flow Diagram

This document contains the complete flow diagrams for the Credit Scoring Experiment, showing the data flow, model training pipeline, and evaluation process.

## Overall System Architecture

```mermaid
graph TB
    A[German Credit Dataset] --> B[Data Preprocessing]
    B --> C[Feature Engineering]
    C --> D[Train/Validation/Test Split]
    
    D --> E[Model Training Pipeline]
    E --> F[Logistic Regression<br/>Well-Calibrated]
    E --> G[Random Forest<br/>Overconfident]
    E --> H[SVM<br/>Poorly Calibrated]
    
    F --> I[Probability Predictions]
    G --> I
    H --> I
    
    I --> J[Calibration Assessment]
    J --> K[Post-hoc Calibration]
    K --> L[Platt Scaling]
    K --> M[Isotonic Regression]
    K --> N[Temperature Scaling]
    
    I --> O[Evaluation Framework]
    O --> P[Accuracy Metrics]
    O --> Q[Calibration Metrics]
    
    P --> R[Business Impact Analysis]
    Q --> R
    L --> R
    M --> R
    N --> R
    
    R --> S[Visualization Suite]
    R --> T[Statistical Testing]
    
    S --> U[Final Report]
    T --> U
```

## Detailed Data Pipeline

```mermaid
graph LR
    A[Raw German Credit Data<br/>1000 samples, 20 features] --> B[Data Loading<br/>pandas.read_csv]
    B --> C{Data Quality Check}
    C -->|Missing Values?| D[Handle Missing Data<br/>Imputation/Removal]
    C -->|Clean Data| E[Feature Engineering]
    D --> E
    
    E --> F[Categorical Encoding<br/>One-hot/Label Encoding]
    F --> G[Numerical Scaling<br/>StandardScaler]
    G --> H[Feature Selection<br/>Correlation Analysis]
    
    H --> I[Stratified Split<br/>60% Train / 20% Val / 20% Test]
    I --> J[Training Set<br/>600 samples]
    I --> K[Validation Set<br/>200 samples]
    I --> L[Test Set<br/>200 samples]
```

## Model Training and Evaluation Pipeline

```mermaid
graph TD
    A[Training Data] --> B[Model Training Phase]
    
    B --> C[Logistic Regression<br/>C=1.0, max_iter=1000]
    B --> D[Random Forest<br/>n_estimators=100, max_depth=10]
    B --> E[SVM with RBF<br/>C=1.0, gamma=scale]
    
    C --> F[LR Predictions<br/>Well-calibrated probabilities]
    D --> G[RF Predictions<br/>Overconfident probabilities]
    E --> H[SVM Predictions<br/>Poorly calibrated probabilities]
    
    F --> I[Validation Set Evaluation]
    G --> I
    H --> I
    
    I --> J{Calibration Quality Check}
    J -->|Well Calibrated<br/>ECE < 0.05| K[Keep Original Model]
    J -->|Poorly Calibrated<br/>ECE > 0.05| L[Apply Calibration Methods]
    
    L --> M[Platt Scaling<br/>Sigmoid fitting]
    L --> N[Isotonic Regression<br/>Monotonic mapping]
    L --> O[Temperature Scaling<br/>Neural network approach]
    
    K --> P[Final Test Set Evaluation]
    M --> P
    N --> P
    O --> P
```

## Evaluation and Analysis Framework

```mermaid
graph TB
    A[Model Predictions on Test Set] --> B[Accuracy Metrics Calculation]
    A --> C[Calibration Metrics Calculation]
    
    B --> D[Classification Accuracy<br/>Precision, Recall, F1]
    B --> E[ROC-AUC Analysis<br/>Discrimination Ability]
    
    C --> F[Reliability Diagram<br/>Bin-based Analysis]
    C --> G[Expected Calibration Error<br/>ECE Calculation]
    C --> H[Brier Score<br/>Probability Accuracy]
    C --> I[Hosmer-Lemeshow Test<br/>Statistical Significance]
    
    D --> J[Comparative Analysis]
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    
    J --> K[Business Impact Modeling]
    K --> L[Loan Approval Simulation<br/>Risk Threshold Analysis]
    K --> M[Financial Impact Calculation<br/>Expected vs Actual Losses]
    
    L --> N[Visualization Generation]
    M --> N
    
    N --> O[Reliability Plots<br/>Calibration Curves]
    N --> P[Probability Histograms<br/>Distribution Analysis]
    N --> Q[Business Impact Dashboard<br/>ROI Analysis]
    
    O --> R[Statistical Testing<br/>Significance Analysis]
    P --> R
    Q --> R
    
    R --> S[Final Report Generation<br/>Results Summary]
```

## Calibration Methods Detail Flow

```mermaid
graph LR
    A[Uncalibrated Model Predictions] --> B{Calibration Method Selection}
    
    B --> C[Platt Scaling]
    C --> C1[Fit Sigmoid Function<br/>P_cal = 1/(1+exp(A*P+B))]
    C1 --> C2[Optimize A and B parameters<br/>on validation set]
    C2 --> C3[Apply to test predictions]
    
    B --> D[Isotonic Regression]
    D --> D1[Fit Monotonic Function<br/>Non-parametric approach]
    D1 --> D2[Preserve probability ordering<br/>while improving calibration]
    D2 --> D3[Apply to test predictions]
    
    B --> E[Temperature Scaling]
    E --> E1[Add temperature parameter T<br/>P_cal = softmax(logits/T)]
    E1 --> E2[Optimize T on validation set<br/>minimize NLL]
    E2 --> E3[Apply to test predictions]
    
    C3 --> F[Calibrated Predictions]
    D3 --> F
    E3 --> F
    
    F --> G[Post-calibration Evaluation]
    G --> H[Compare ECE before/after]
    G --> I[Assess impact on accuracy]
    G --> J[Business impact analysis]
```

## Business Impact Analysis Flow

```mermaid
graph TD
    A[Calibrated vs Uncalibrated Predictions] --> B[Define Business Scenario]
    B --> C[Credit Approval Decision<br/>Threshold: P(default) < 0.3]
    
    C --> D[Simulate Loan Portfolio<br/>10,000 loans, $10K average]
    
    D --> E[Well-Calibrated Model Scenario]
    D --> F[Overconfident Model Scenario]
    
    E --> E1[Predicted Risk: 20%<br/>Actual Risk: 22%]
    E1 --> E2[Expected Loss: $2M<br/>Actual Loss: $2.2M]
    E2 --> E3[Difference: $200K<br/>Manageable Risk]
    
    F --> F1[Predicted Risk: 5%<br/>Actual Risk: 25%]
    F1 --> F2[Expected Loss: $500K<br/>Actual Loss: $2.5M]
    F2 --> F3[Difference: $2M<br/>Significant Risk]
    
    E3 --> G[ROI Calculation for Calibration]
    F3 --> G
    
    G --> H[Cost of Calibration<br/>Development + Validation]
    G --> I[Benefit of Calibration<br/>Risk Reduction]
    
    H --> J[ROI = (Benefit - Cost) / Cost]
    I --> J
    
    J --> K[Business Recommendation<br/>Implement Calibration]
```

## Visualization Pipeline

```mermaid
graph TB
    A[Analysis Results] --> B[Visualization Generation]
    
    B --> C[Reliability Diagrams<br/>matplotlib/seaborn]
    C --> C1[Perfect Calibration Line<br/>y = x reference]
    C1 --> C2[Observed vs Predicted<br/>by probability bins]
    C2 --> C3[Confidence Intervals<br/>Bootstrap sampling]
    
    B --> D[Probability Distributions<br/>histogram plots]
    D --> D1[Show prediction confidence<br/>distribution shape]
    D1 --> D2[Compare across models<br/>side-by-side plots]
    
    B --> E[Business Impact Dashboard<br/>plotly interactive]
    E --> E1[Financial metrics by model<br/>loss calculations]
    E1 --> E2[ROI scenarios<br/>sensitivity analysis]
    
    C3 --> F[Static Report Figures<br/>PNG/PDF export]
    D2 --> F
    E2 --> G[Interactive Dashboard<br/>HTML export]
    
    F --> H[Integrated Report<br/>Jupyter notebook]
    G --> H
    
    H --> I[Final Deliverable<br/>Complete analysis]
```

## File Organization Structure

```mermaid
graph TD
    A[credit_scoring_experiment/] --> B[docs/]
    A --> C[data/]
    A --> D[models/]
    A --> E[visualization/]
    A --> F[results/]
    A --> G[notebooks/]
    A --> H[README.md]
    A --> I[requirements.txt]
    
    B --> B1[VISION_DOCUMENT.md]
    B --> B2[FLOW_DIAGRAM.md]
    B --> B3[API_DOCUMENTATION.md]
    
    C --> C1[raw/german_credit.csv]
    C --> C2[processed/train_test_split.pkl]
    
    D --> D1[train_models.py]
    D --> D2[calibration.py]
    D --> D3[evaluation.py]
    D --> D4[model_configs.json]
    
    E --> E1[reliability_plots.py]
    E --> E2[business_impact.py]
    E --> E3[probability_distributions.py]
    
    F --> F1[model_metrics.csv]
    F --> F2[calibration_results.json]
    F --> F3[business_impact_analysis.pdf]
    
    G --> G1[01_data_exploration.ipynb]
    G --> G2[02_model_training.ipynb]
    G --> G3[03_calibration_analysis.ipynb]
    G --> G4[04_business_impact.ipynb]
```

---

*These diagrams provide a comprehensive view of the experimental design and implementation flow. Each component is designed to be modular and testable, ensuring reproducibility and educational value.*