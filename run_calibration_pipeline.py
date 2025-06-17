"""
Run the complete calibration pipeline
"""

import os
import sys
from pathlib import Path

def run_pipeline():
    """Run the complete calibration pipeline"""
    print("🚀 Running calibration pipeline...")
    
    # Run calibration
    print("\n📊 Step 1: Running calibration analysis...")
    os.system("python models/calibration.py")
    
    # Run visualization
    print("\n🎨 Step 2: Generating visualizations...")
    os.system("python visualization/reliability_plots.py")
    
    print("\n✅ Pipeline completed successfully!")

if __name__ == "__main__":
    run_pipeline()