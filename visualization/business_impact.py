"""
Business Impact Analysis for Credit Scoring Calibration
=====================================================

This module analyzes the financial impact of model calibration on business decisions.

Author: Credit Scoring Experiment
Date: June 2025
"""

import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class BusinessImpactAnalyzer:
    """
    Analyzes the business impact of model calibration
    """
    
    def __init__(self, results_dir: str = "../results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def load_calibration_results(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load calibration results and data splits
        
        Returns:
            Tuple of (calibration_results, data_splits)
        """
        # Load calibrated models
        calibrated_models_path = self.results_dir / "calibrated_models.pkl"
        if not calibrated_models_path.exists():
            raise FileNotFoundError("No calibrated models found. Run calibration analysis first.")
            
        with open(calibrated_models_path, 'rb') as f:
            calibration_results = pickle.load(f)
        
        # Load data splits
        splits_path = Path("../data/processed/train_test_split.pkl")
        if not splits_path.exists():
            raise FileNotFoundError("No data splits found.")
            
        with open(splits_path, 'rb') as f:
            data_splits = pickle.load(f)
            
        return calibration_results, data_splits
    
    def analyze_business_impact(self, calibration_results: Dict[str, Any], 
                              data_splits: Dict[str, Any]) -> pd.DataFrame:
        """
        Analyze business impact of calibration improvements
        
        Args:
            calibration_results: Results from calibration analysis
            data_splits: Dictionary containing data splits
            
        Returns:
            DataFrame with business impact analysis
        """
        print("💼 Analyzing business impact...")
        
        X_test = data_splits['X_test']
        y_test = data_splits['y_test']
        
        # Business scenario parameters
        portfolio_size = 10000
        avg_loan_amount = 10000
        approval_threshold = 0.3  # Approve if P(default) < 30%
        
        business_results = []
        
        for model_name, model_variants in calibration_results.items():
            for variant_name, variant_data in model_variants.items():
                if 'model' in variant_data:
                    model = variant_data['model']
                    probs = model.predict_proba(X_test)[:, 1]
                    
                    # Simulate loan approval decisions
                    approved_mask = probs < approval_threshold
                    n_approved = np.sum(approved_mask)
                    
                    if n_approved > 0:
                        # Calculate actual vs predicted default rates for approved loans
                        actual_default_rate = y_test[approved_mask].mean()
                        predicted_default_rate = probs[approved_mask].mean()
                        
                        # Scale to portfolio size
                        portfolio_approved = int(n_approved / len(y_test) * portfolio_size)
                        total_loan_volume = portfolio_approved * avg_loan_amount
                        
                        # Calculate losses
                        predicted_losses = predicted_default_rate * total_loan_volume
                        actual_losses = actual_default_rate * total_loan_volume
                        unexpected_loss = actual_losses - predicted_losses
                        
                        # Calculate metrics
                        approval_rate = n_approved / len(y_test)
                        calibration_error = abs(predicted_default_rate - actual_default_rate)
                        
                        business_results.append({
                            'Model': model_name,
                            'Method': variant_name.title(),
                            'Full_Name': variant_data['name'],
                            'Approval_Rate': approval_rate,
                            'Portfolio_Approved': portfolio_approved,
                            'Total_Loan_Volume_M': total_loan_volume / 1e6,  # in millions
                            'Predicted_Default_Rate': predicted_default_rate,
                            'Actual_Default_Rate': actual_default_rate,
                            'Calibration_Error': calibration_error,
                            'Predicted_Losses_M': predicted_losses / 1e6,
                            'Actual_Losses_M': actual_losses / 1e6,
                            'Unexpected_Loss_M': unexpected_loss / 1e6,
                            'Loss_Surprise_Pct': (unexpected_loss / predicted_losses * 100) if predicted_losses > 0 else 0
                        })
        
        business_df = pd.DataFrame(business_results)
        
        # Save results
        business_df.to_csv(self.results_dir / "business_impact_analysis.csv", index=False)
        
        print("\n💰 BUSINESS IMPACT SUMMARY")
        print("=" * 40)
        
        if len(business_df) > 0:
            # Find best and worst calibration errors
            best_calibration = business_df.loc[business_df['Calibration_Error'].idxmin()]
            worst_calibration = business_df.loc[business_df['Calibration_Error'].idxmax()]
            
            print(f"\n🏆 Best Calibrated Model:")
            print(f"   {best_calibration['Full_Name']}")
            print(f"   Calibration Error: {best_calibration['Calibration_Error']:.1%}")
            print(f"   Unexpected Loss: ${best_calibration['Unexpected_Loss_M']:.1f}M")
            
            print(f"\n⚠️  Worst Calibrated Model:")
            print(f"   {worst_calibration['Full_Name']}")
            print(f"   Calibration Error: {worst_calibration['Calibration_Error']:.1%}")
            print(f"   Unexpected Loss: ${worst_calibration['Unexpected_Loss_M']:.1f}M")
            
            risk_difference = worst_calibration['Unexpected_Loss_M'] - best_calibration['Unexpected_Loss_M']
            print(f"\n💸 Risk Difference: ${risk_difference:.1f}M")
            if worst_calibration['Predicted_Losses_M'] > 0:
                print(f"   ({risk_difference/worst_calibration['Predicted_Losses_M']*100:.0f}% of predicted losses)")
        
        print("\n✅ Business impact analysis completed")
        return business_df
    
    def generate_business_report(self) -> None:
        """
        Generate comprehensive business impact report
        """
        print("📋 Generating business impact report...")
        print("=" * 50)
        
        try:
            # Load results
            calibration_results, data_splits = self.load_calibration_results()
            
            # Analyze business impact
            business_df = self.analyze_business_impact(calibration_results, data_splits)
            
            print(f"\n📁 Business impact analysis saved to {self.results_dir}")
            
        except FileNotFoundError as e:
            print(f"❌ Error: {e}")
            print("💡 Make sure to run calibration analysis first.")
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            raise

def main():
    """
    Main function for business impact analysis
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Business Impact Analysis')
    parser.add_argument('--results-dir', default='../results',
                       help='Directory containing calibration results')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = BusinessImpactAnalyzer(args.results_dir)
    
    # Generate business impact report
    analyzer.generate_business_report()

if __name__ == "__main__":
    main()