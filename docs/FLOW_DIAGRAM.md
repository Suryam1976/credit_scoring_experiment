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

## Detailed Implementation Architecture

```mermaid
graph TB
    A[train_models.py] --> B[Data Loading & Preprocessing]
    B --> C[Model Training]
    C --> D[Save Trained Models<br/>results/trained_models.pkl]
    
    D --> E[calibration.py]
    E --> F[Load Trained Models]
    F --> G[Apply Calibration Methods]
    G --> G1[Platt Scaling]
    G --> G2[Isotonic Regression]
    G --> G3[Temperature Scaling<br/>TemperatureScaledModel]
    
    G1 --> H[Calculate Calibration Metrics]
    G2 --> H
    G3 --> H
    
    H --> I[Save Calibration Results]
    I --> I1[Save Model Predictions<br/>model_predictions.pkl]
    I --> I2[Save Comparison Table<br/>calibration_comparison.csv]
    
    I1 --> J[reliability_plots.py]
    I2 --> J
    J --> K[Load Predictions & Results]
    K --> L[Generate Reliability Diagrams]
    K --> M[Generate Business Impact Analysis]
    
    L --> N[Save Visualizations<br/>results/visualizations/]
    M --> N
```

## Module Structure and Dependencies

```mermaid
graph TB
    A[models/model_utils.py] --> B[TemperatureScaledModel Class]
    
    C[models/calibration.py] --> A
    C --> D[CalibrationAnalyzer Class]
    D --> D1[apply_platt_scaling]
    D --> D2[apply_isotonic_regression]
    D --> D3[apply_temperature_scaling]
    D --> D4[calculate_calibration_metrics]
    
    E[visualization/reliability_plots.py] --> A
    E --> F[CalibrationVisualizer Class]
    F --> F1[load_results]
    F --> F2[create_reliability_diagram]
    F --> F3[generate_all_visualizations]
    
    G[run_calibration_pipeline.py] --> C
    G --> E
```

## Data Flow and Serialization

```mermaid
graph LR
    A[Raw Data] --> B[train_models.py]
    B --> C[Trained Models<br/>trained_models.pkl]
    
    C --> D[calibration.py]
    D --> E[Model Predictions<br/>model_predictions.pkl]
    D --> F[Calibration Metrics<br/>calibration_comparison.csv]
    
    E --> G[reliability_plots.py]
    F --> G
    G --> H[Visualizations<br/>reliability_diagrams.png]
```

## Python Path and Import Resolution

```mermaid
graph TB
    A[Project Root Directory] --> B[sys.path.append]
    B --> C[Absolute Imports<br/>from models.model_utils import ...]
    
    D[models/calibration.py] --> B
    E[visualization/reliability_plots.py] --> B
    
    F[run_calibration_pipeline.py] --> G[Subprocess Execution<br/>os.system]
    G --> D
    G --> E
```

## Execution Flow

```mermaid
graph TD
    A[Setup Environment] --> B[Install Dependencies<br/>pip install -r requirements.txt]
    B --> C[Create Directories<br/>python setup_dirs.py]
    
    C --> D[Run Complete Pipeline<br/>python run_calibration_pipeline.py]
    
    D --> E[Train Models<br/>python models/calibration.py]
    E --> F[Apply Calibration<br/>python models/calibration.py]
    F --> G[Generate Visualizations<br/>python visualization/reliability_plots.py]
    
    H[Alternative: Step-by-Step] --> I[Train Models<br/>python models/train_models.py --prepare-data --train-all]
    I --> J[Apply Calibration<br/>python models/calibration.py]
    J --> K[Generate Visualizations<br/>python visualization/reliability_plots.py]
```

## Temperature Scaling Implementation

```mermaid
graph TB
    A[apply_temperature_scaling] --> B[Get Model Logits<br/>_get_model_logits]
    B --> C[Find Optimal Temperature<br/>_find_optimal_temperature]
    C --> D[Create TemperatureScaledModel]
    
    E[TemperatureScaledModel Class] --> F[__init__<br/>Store model & temperature]
    E --> G[predict_proba<br/>Apply temperature scaling]
    G --> H[Get Logits<br/>_get_model_logits]
    H --> I[Scale Logits<br/>_temperature_scaling]
    I --> J[Return Calibrated Probabilities]
```

## Visualization Pipeline

```mermaid
graph TB
    A[CalibrationVisualizer] --> B[load_results<br/>Load model_predictions.pkl]
    A --> C[load_data_splits<br/>Load test data]
    
    B --> D[create_reliability_diagram]
    C --> D
    
    D --> E[Group Models by Base Name]
    E --> F[Calculate Calibration Curves]
    F --> G[Plot Reliability Diagrams]
    G --> H[Save Visualizations<br/>reliability_diagrams.png]
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
