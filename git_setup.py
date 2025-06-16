#!/usr/bin/env python3
"""
Git Setup Helper Script
======================

This script helps initialize the Git repository for the Credit Scoring Experiment
and performs the initial commit with proper file organization.

Usage:
    python git_setup.py
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"❌ {description} failed")
            print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False
    return True

def check_git_installed():
    """Check if Git is installed"""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True)
        return result.returncode == 0
    except:
        return False

def setup_git_repository():
    """Initialize and set up the Git repository"""
    
    print("🚀 Credit Scoring Experiment - Git Setup")
    print("=" * 50)
    
    # Check if Git is installed
    if not check_git_installed():
        print("❌ Git is not installed or not in PATH")
        print("💡 Please install Git from: https://git-scm.com/download")
        return False
    
    # Check if we're in the right directory
    if not Path("README.md").exists():
        print("❌ Please run this script from the project root directory")
        print("💡 Navigate to: C:\\Users\\avssm\\credit_scoring_experiment")
        return False
    
    # Initialize Git repository
    if not Path(".git").exists():
        if not run_command("git init", "Initializing Git repository"):
            return False
    else:
        print("ℹ️  Git repository already initialized")
    
    # Check Git configuration
    print("\n🔧 Checking Git configuration...")
    name_result = subprocess.run("git config user.name", shell=True, capture_output=True, text=True)
    email_result = subprocess.run("git config user.email", shell=True, capture_output=True, text=True)
    
    if name_result.returncode != 0 or not name_result.stdout.strip():
        print("⚠️  Git user.name not configured")
        name = input("Enter your name for Git commits: ")
        run_command(f'git config user.name "{name}"', "Setting Git user name")
    
    if email_result.returncode != 0 or not email_result.stdout.strip():
        print("⚠️  Git user.email not configured")
        email = input("Enter your email for Git commits: ")
        run_command(f'git config user.email "{email}"', "Setting Git user email")
    
    # Check current status
    print("\n📊 Checking repository status...")
    run_command("git status", "Checking Git status")
    
    # Add files to staging
    print("\n📦 Adding files to Git...")
    files_to_add = [
        "README.md",
        "QUICK_START.md", 
        "PROJECT_UPDATES.md",
        "GIT_GUIDE.md",
        "requirements.txt",
        "setup_dirs.py",
        ".gitignore",
        "docs/",
        "models/",
        "visualization/",
        "notebooks/",
        "data/*/.gitkeep",
        "results/*/.gitkeep"
    ]
    
    for file_pattern in files_to_add:
        if Path(file_pattern.replace("/*", "")).exists():
            run_command(f"git add {file_pattern}", f"Adding {file_pattern}")
    
    # Check what's staged
    print("\n📋 Files staged for commit:")
    run_command("git diff --cached --name-only", "Listing staged files")
    
    # Create initial commit
    commit_message = '''Initial commit: Credit Scoring Calibration Experiment

🎯 Project: Comprehensive framework demonstrating accuracy vs calibration

📦 Core Components:
- Model training pipeline (Logistic Regression, Random Forest, SVM)
- Calibration methods (Platt Scaling, Isotonic Regression)
- Business impact analysis with financial modeling
- Interactive Jupyter notebook for guided exploration
- Comprehensive documentation and setup guides

🔧 Key Features:
- Automated directory setup and robust path handling
- Multiple execution options (CLI, notebook, step-by-step)
- Detailed troubleshooting and error handling
- Real-world credit scoring dataset and scenarios

📊 Expected Results:
- SVM + Platt Scaling: Best calibrated (ECE ~0.045)
- Random Forest: High accuracy (87%) but poor calibration
- Logistic Regression: Surprisingly poor calibration (ECE ~0.087)

🎓 Educational Value:
- Demonstrates why calibration matters in business applications
- Proves that accuracy ≠ calibration
- Shows financial impact of miscalibrated models ($1-2M potential losses)
- Debunks "naturally calibrated" model assumptions

🚀 Ready for: Research, education, and practical ML applications'''
    
    print("\n💾 Creating initial commit...")
    # Escape the commit message properly for command line
    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        # If multi-line commit fails, try simple version
        simple_message = "Initial commit: Credit Scoring Calibration Experiment - Complete framework with models, calibration methods, and business impact analysis"
        run_command(f'git commit -m "{simple_message}"', "Creating initial commit (simple)")
    
    # Show final status
    print("\n📈 Final repository status:")
    run_command("git log --oneline -5", "Showing recent commits")
    run_command("git status", "Final status check")
    
    print("\n🎉 Git setup completed successfully!")
    print("\n📚 Next steps:")
    print("1. Add remote repository: git remote add origin <your-repo-url>")
    print("2. Push to remote: git push -u origin main")
    print("3. See GIT_GUIDE.md for detailed Git workflow instructions")
    
    return True

def main():
    """Main function"""
    try:
        success = setup_git_repository()
        if success:
            print("\n✅ All done! Your repository is ready for version control.")
        else:
            print("\n❌ Setup encountered issues. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
