"""
Setup Script for Credit Scoring Experiment
==========================================

This script ensures all necessary directories exist before running the experiment.
"""

from pathlib import Path

def setup_directories():
    """Create all necessary directories for the experiment"""
    
    base_dir = Path(__file__).parent
    
    # Create directory structure
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed", 
        base_dir / "results",
        base_dir / "results" / "visualizations",
        base_dir / "models" / "trained"
    ]
    
    print("🔧 Setting up project directories...")
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    print("\n🎉 Directory setup completed!")
    print("📁 Project structure ready for experiment")

if __name__ == "__main__":
    setup_directories()
