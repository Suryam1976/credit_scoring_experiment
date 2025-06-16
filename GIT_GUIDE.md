# 📝 Git Setup and Usage Guide

This guide helps you set up version control for the Credit Scoring Experiment and provides common Git commands.

## 🚀 Initial Git Setup

### 1. Initialize Git Repository
```bash
cd C:\Users\avssm\credit_scoring_experiment

# Initialize git repository
git init

# Add remote repository (replace with your actual repository URL)
# git remote add origin https://github.com/yourusername/credit_scoring_experiment.git
```

### 2. Configure Git (First Time Setup)
```bash
# Set your name and email (required for commits)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Optional: Set default branch name
git config --global init.defaultBranch main

# Optional: Set preferred editor
git config --global core.editor "code --wait"  # For VS Code
# git config --global core.editor "notepad"    # For Notepad
```

### 3. Create .gitkeep Files for Empty Directories
```bash
# Create placeholder files to track empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/visualizations/.gitkeep

# On Windows, use:
# echo. > data\raw\.gitkeep
# echo. > data\processed\.gitkeep
# echo. > results\visualizations\.gitkeep
```

## 📦 Initial Commit

### 1. Add Files to Staging
```bash
# Check current status
git status

# Add all project files (respects .gitignore)
git add .

# Or add specific files/directories
git add README.md
git add requirements.txt
git add models/
git add docs/
git add notebooks/
git add visualization/
git add setup_dirs.py
git add .gitignore
```

### 2. Create Initial Commit
```bash
# Create initial commit
git commit -m "Initial commit: Credit Scoring Calibration Experiment

- Add comprehensive experimental framework
- Include model training pipeline (Logistic Regression, Random Forest, SVM)
- Implement calibration methods (Platt Scaling, Isotonic Regression)
- Add business impact analysis and visualizations
- Include documentation and setup scripts
- Add Jupyter notebook for interactive analysis"

# Check commit history
git log --oneline
```

## 🔄 Common Git Workflow

### Daily Development Commands
```bash
# Check repository status
git status

# See what changed
git diff

# Add changes to staging
git add <filename>        # Add specific file
git add .                # Add all changes
git add -u               # Add only modified files (no new files)

# Commit changes
git commit -m "Descriptive commit message"

# Push to remote repository
git push origin main

# Pull latest changes from remote
git pull origin main
```

### Working with Branches
```bash
# Create and switch to new branch
git checkout -b feature/new-calibration-method

# Switch between branches
git checkout main
git checkout feature/new-calibration-method

# List all branches
git branch -a

# Merge branch back to main
git checkout main
git merge feature/new-calibration-method

# Delete branch after merging
git branch -d feature/new-calibration-method
```

## 📊 Experiment-Specific Git Commands

### Tracking Experiment Results
```bash
# Add experiment results (only summary files, not large data)
git add results/model_metrics.csv
git add results/calibration_comparison.csv
git commit -m "Add experiment results: calibration analysis

- SVM + Platt Scaling achieved best calibration (ECE: 0.045)
- Random Forest shows poor calibration despite high accuracy
- Logistic Regression surprisingly poorly calibrated (ECE: 0.087)"
```

### Tagging Experiment Versions
```bash
# Tag important experiment versions
git tag -a v1.0 -m "Initial working experiment"
git tag -a v1.1 -m "Added business impact analysis"
git tag -a v2.0 -m "Corrected results documentation"

# Push tags to remote
git push origin --tags

# List tags
git tag -l
```

### Handling Large Files (if needed)
```bash
# If you need to track large model files, use Git LFS
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS tracking for model files"
```

## 🔍 Useful Git Commands

### Viewing History and Changes
```bash
# View commit history
git log --oneline --graph --all
git log --since="2 weeks ago"
git log --author="Your Name"

# View changes in specific commit
git show <commit-hash>

# View file history
git log --follow -- models/train_models.py

# Compare branches
git diff main..feature-branch
```

### Undoing Changes
```bash
# Undo changes in working directory
git checkout -- <filename>

# Unstage files
git reset HEAD <filename>

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1

# Revert a specific commit
git revert <commit-hash>
```

### Stashing Changes
```bash
# Temporarily save uncommitted changes
git stash

# Apply stashed changes
git stash pop

# List stashes
git stash list

# Apply specific stash
git stash apply stash@{0}
```

## 🌐 Remote Repository Setup

### GitHub Setup
```bash
# Create repository on GitHub, then:
git remote add origin https://github.com/yourusername/credit_scoring_experiment.git

# Push initial code
git push -u origin main

# Clone existing repository
git clone https://github.com/yourusername/credit_scoring_experiment.git
```

### Collaborating
```bash
# Fork workflow
git remote add upstream https://github.com/original-owner/credit_scoring_experiment.git
git fetch upstream
git merge upstream/main

# Pull request workflow
git checkout -b fix/calibration-bug
# Make changes
git add .
git commit -m "Fix calibration calculation bug"
git push origin fix/calibration-bug
# Create pull request on GitHub
```

## 📋 Commit Message Best Practices

### Good Commit Message Format
```
<type>(<scope>): <subject>

<body>

<footer>
```

### Examples
```bash
# Feature addition
git commit -m "feat(calibration): add temperature scaling method

- Implement temperature scaling for neural network calibration
- Add validation on held-out calibration set
- Update comparison analysis to include new method"

# Bug fix
git commit -m "fix(models): correct ECE calculation for edge cases

- Handle empty probability bins properly
- Add validation for probability ranges
- Fixes issue #123"

# Documentation
git commit -m "docs(readme): update expected results section

- Correct ECE values based on actual experiment results
- Update business impact calculations
- Add clarification about Logistic Regression calibration"

# Refactoring
git commit -m "refactor(training): extract model configuration to separate file

- Move model configs to JSON file for easier modification
- Add validation for configuration parameters
- Improve code organization and maintainability"
```

### Commit Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

## 🏷️ Branching Strategy

### Recommended Branch Structure
```
main                    # Production-ready code
├── develop            # Integration branch
├── feature/new-model  # New feature development
├── fix/calibration    # Bug fixes
└── experiment/v2      # Experimental changes
```

### Branch Commands
```bash
# Create feature branch
git checkout -b feature/add-ensemble-methods

# Work on feature
git add .
git commit -m "feat: add ensemble calibration methods"

# Push feature branch
git push origin feature/add-ensemble-methods

# Merge to main (after review)
git checkout main
git merge feature/add-ensemble-methods
git push origin main

# Clean up
git branch -d feature/add-ensemble-methods
git push origin --delete feature/add-ensemble-methods
```

## 📊 Project-Specific Workflow

### Typical Experiment Workflow
```bash
# 1. Start new experiment
git checkout -b experiment/new-dataset

# 2. Modify parameters
# Edit files...
git add .
git commit -m "experiment: test on healthcare dataset"

# 3. Run experiment
python models/train_models.py --prepare-data --train-all
python models/calibration.py

# 4. Commit results
git add results/model_metrics.csv
git commit -m "results: healthcare dataset calibration analysis"

# 5. Merge successful experiment
git checkout main
git merge experiment/new-dataset

# 6. Tag release
git tag -a v1.2 -m "Healthcare dataset experiment"
git push origin main --tags
```

## 🔒 Security Best Practices

### Protect Sensitive Information
```bash
# Never commit:
# - API keys
# - Passwords
# - Personal data
# - Large datasets
# - Temporary files

# Use .gitignore for:
echo "config/secrets.json" >> .gitignore
echo "*.env" >> .gitignore
echo "data/private/" >> .gitignore
```

### Environment Variables
```bash
# Use environment variables for sensitive config
echo "DATABASE_URL=your_database_url" > .env
echo ".env" >> .gitignore

# In Python:
# import os
# db_url = os.getenv('DATABASE_URL')
```

## 📚 Additional Resources

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)

---

**Remember**: Commit early, commit often, and always write descriptive commit messages! 🚀