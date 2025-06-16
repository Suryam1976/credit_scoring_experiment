"""
Credit Scoring Model Training Pipeline
=====================================

This module implements the core model training pipeline for the accuracy vs calibration experiment.
It includes data loading, preprocessing, model training, and initial evaluation.

Author: Credit Scoring Experiment
Date: June 2025
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, List, Any
from dataclasses import dataclass
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Statistical and utility libraries
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

@dataclass
class ModelConfig:
    """Configuration class for model parameters"""
    name: str
    model: Any
    params: Dict[str, Any]
    description: str

@dataclass
class ExperimentResults:
    """Container for experiment results"""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    brier_score: float
    log_loss: float
    ece: float  # Expected Calibration Error
    predictions: np.ndarray
    probabilities: np.ndarray

class CreditScoringExperiment:
    """
    Main experiment class for comparing accuracy vs calibration in credit scoring models.
    """
    
    def __init__(self, data_dir: str = "../data", results_dir: str = "../results"):
        """
        Initialize the experiment with data and results directories.
        
        Args:
            data_dir: Directory containing the dataset
            results_dir: Directory to save results
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        
        # Initialize results storage
        self.models = {}
        self.results = {}
        
        # Define model configurations
        self.model_configs = self._define_model_configs()
        
    def _define_model_configs(self) -> List[ModelConfig]:
        """Define the models to be trained and compared."""
        return [
            ModelConfig(
                name="Logistic_Regression",
                model=LogisticRegression,
                params={
                    'random_state': RANDOM_STATE,
                    'max_iter': 1000,
                    'C': 1.0
                },
                description="Well-calibrated baseline model"
            ),
            ModelConfig(
                name="Random_Forest",
                model=RandomForestClassifier,
                params={
                    'random_state': RANDOM_STATE,
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 20,
                    'min_samples_leaf': 10
                },
                description="High accuracy but typically overconfident"
            ),
            ModelConfig(
                name="SVM_RBF",
                model=SVC,
                params={
                    'random_state': RANDOM_STATE,
                    'C': 1.0,
                    'gamma': 'scale',
                    'probability': True
                },
                description="Often poorly calibrated without post-processing"
            )
        ]
    
    def download_data(self) -> None:
        """Download the German Credit Dataset from UCI repository."""
        print("📥 Downloading German Credit Dataset...")
        
        # UCI German Credit Dataset URL
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Define column names for the German Credit Dataset
            columns = [
                'checking_account', 'duration', 'credit_history', 'purpose', 'credit_amount',
                'savings_account', 'employment', 'installment_rate', 'personal_status',
                'other_debtors', 'residence_since', 'property', 'age', 'other_installments',
                'housing', 'existing_credits', 'job', 'num_dependents', 'telephone',
                'foreign_worker', 'target'
            ]
            
            # Read the data
            data = pd.read_csv(StringIO(response.text), sep=' ', header=None, names=columns)
            
            # Convert target to binary (1 = good credit, 2 = bad credit -> 0 = good, 1 = bad)
            data['target'] = (data['target'] == 2).astype(int)
            
            # Save the raw data
            raw_data_path = self.data_dir / "raw"
            raw_data_path.mkdir(exist_ok=True)
            data.to_csv(raw_data_path / "german_credit.csv", index=False)
            
            print(f"✅ Data downloaded successfully! Shape: {data.shape}")
            print(f"📁 Saved to: {raw_data_path / 'german_credit.csv'}")
            
        except Exception as e:
            print(f"❌ Error downloading data: {e}")
            print("💡 You can manually download from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
            raise
    
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Load and preprocess the German Credit Dataset."""
        print("🔄 Loading and preprocessing data...")
        
        # Load the data
        data_path = self.data_dir / "raw" / "german_credit.csv"
        
        if not data_path.exists():
            print("❌ Data file not found. Downloading...")
            self.download_data()
        
        df = pd.read_csv(data_path)
        print(f"📊 Original data shape: {df.shape}")
        
        # Basic data info
        print(f"🎯 Target distribution:")
        print(df['target'].value_counts(normalize=True))
        
        # Handle categorical variables
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        print(f"📝 Categorical columns: {len(categorical_columns)}")
        
        # Simple encoding for categorical variables
        label_encoders = {}
        for col in categorical_columns:
            if col != 'target':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Save label encoders for future use
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        with open(processed_dir / "label_encoders.pkl", "wb") as f:
            pickle.dump(label_encoders, f)
        
        # Handle any missing values
        if df.isnull().sum().sum() > 0:
            print("⚠️ Found missing values, filling with median/mode...")
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
        
        print("✅ Data preprocessing completed!")
        return df
    
    def create_train_test_split(self, df: pd.DataFrame) -> None:
        """Create stratified train/validation/test splits."""
        print("🔪 Creating train/validation/test splits...")
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Create train/temp split (60% train, 40% temp)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
        )
        
        # Split temp into validation and test (20% each)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
        )
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Store the data
        self.X_train = X_train_scaled
        self.X_val = X_val_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        
        # Save the scaler
        processed_dir = self.data_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        with open(processed_dir / "scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        
        # Save the splits
        splits = {
            'X_train': X_train_scaled,
            'X_val': X_val_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test
        }
        
        with open(processed_dir / "train_test_split.pkl", "wb") as f:
            pickle.dump(splits, f)
        
        print(f"✅ Data splits created:")
        print(f"   📊 Train: {X_train_scaled.shape[0]} samples")
        print(f"   📊 Validation: {X_val_scaled.shape[0]} samples") 
        print(f"   📊 Test: {X_test_scaled.shape[0]} samples")
        
    def calculate_expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            y_true: True binary labels
            y_prob: Predicted probabilities
            n_bins: Number of bins for calibration
            
        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find predictions in this bin
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                
                # Add to ECE
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def train_model(self, config: ModelConfig) -> Tuple[Any, ExperimentResults]:
        """Train a single model and evaluate its performance."""
        print(f"🚂 Training {config.name}...")
        
        # Initialize and train the model
        model = config.model(**config.params)
        model.fit(self.X_train, self.y_train)
        
        # Make predictions on validation set
        y_pred = model.predict(self.X_val)
        y_prob = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_val, y_pred)
        precision = precision_score(self.y_val, y_pred)
        recall = recall_score(self.y_val, y_pred)
        f1 = f1_score(self.y_val, y_pred)
        roc_auc = roc_auc_score(self.y_val, y_prob)
        brier = brier_score_loss(self.y_val, y_prob)
        logloss = log_loss(self.y_val, y_prob)
        ece = self.calculate_expected_calibration_error(self.y_val, y_prob)
        
        # Create results object
        results = ExperimentResults(
            model_name=config.name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            brier_score=brier,
            log_loss=logloss,
            ece=ece,
            predictions=y_pred,
            probabilities=y_prob
        )
        
        print(f"✅ {config.name} trained - Accuracy: {accuracy:.3f}, ECE: {ece:.3f}")
        
        return model, results
    
    def train_all_models(self) -> None:
        """Train all configured models."""
        print("🎯 Starting model training pipeline...")
        
        for config in self.model_configs:
            model, results = self.train_model(config)
            self.models[config.name] = model
            self.results[config.name] = results
        
        print("🎉 All models trained successfully!")
        
        # Save models to results directory in the expected format
        with open(self.results_dir / "trained_models.pkl", "wb") as f:
            pickle.dump(self.models, f)
        
        print(f"✅ Models saved to {self.results_dir / 'trained_models.pkl'}")
    
    def generate_results_summary(self) -> pd.DataFrame:
        """Generate a summary DataFrame of all model results."""
        print("📊 Generating results summary...")
        
        summary_data = []
        for name, results in self.results.items():
            summary_data.append({
                'Model': name,
                'Accuracy': results.accuracy,
                'Precision': results.precision,
                'Recall': results.recall,
                'F1-Score': results.f1_score,
                'ROC-AUC': results.roc_auc,
                'Brier Score': results.brier_score,
                'Log Loss': results.log_loss,
                'ECE': results.ece
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save results
        summary_df.to_csv(self.results_dir / "model_metrics.csv", index=False)
        
        print("✅ Results summary generated!")
        print(summary_df.round(4))
        
        return summary_df
    
    def run_experiment(self) -> None:
        """Run the complete experiment pipeline."""
        print("🚀 Starting Credit Scoring Experiment: Accuracy vs Calibration")
        print("=" * 70)
        
        # Step 1: Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Step 2: Create train/test splits
        self.create_train_test_split(df)
        
        # Step 3: Train all models
        self.train_all_models()
        
        # Step 4: Generate summary
        summary = self.generate_results_summary()
        
        print("\n🎯 Experiment completed successfully!")
        print("📁 Check the results directory for detailed outputs")
        print("🔍 Key finding: Notice the difference between accuracy and ECE!")
        
        return summary


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='Credit Scoring Experiment: Accuracy vs Calibration')
    parser.add_argument('--prepare-data', action='store_true', 
                       help='Download and prepare the dataset')
    parser.add_argument('--train-all', action='store_true',
                       help='Train all models and generate results')
    parser.add_argument('--data-dir', default='../data',
                       help='Directory for data storage')
    parser.add_argument('--results-dir', default='../results',
                       help='Directory for results storage')
    
    args = parser.parse_args()
    
    # Initialize experiment
    experiment = CreditScoringExperiment(
        data_dir=args.data_dir,
        results_dir=args.results_dir
    )
    
    if args.prepare_data:
        experiment.download_data()
        df = experiment.load_and_preprocess_data()
        experiment.create_train_test_split(df)
        print("✅ Data preparation completed!")
        
    elif args.train_all:
        experiment.run_experiment()
        
    else:
        print("Please specify --prepare-data or --train-all")
        print("Run 'python train_models.py --help' for more options")


if __name__ == "__main__":
    main()
