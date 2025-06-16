# 🎯 Git Quick Reference - Credit Scoring Experiment

## 🚀 Initial Setup (One-time)
```bash
cd C:\Users\avssm\credit_scoring_experiment
python git_setup.py                    # Automated setup
# OR manual setup:
git init
git config user.name "Your Name"
git config user.email "your@email.com"
git add .
git commit -m "Initial commit"
```

## 📦 Daily Workflow
```bash
git status                             # Check what changed
git add .                              # Stage all changes
git commit -m "descriptive message"    # Commit changes
git push origin main                   # Push to remote
git pull origin main                   # Pull latest changes
```

## 🔄 Common Commands
```bash
# Viewing
git log --oneline                      # View commit history
git diff                               # See unstaged changes
git diff --cached                      # See staged changes

# Undoing
git checkout -- filename              # Discard file changes
git reset HEAD filename               # Unstage file
git reset --soft HEAD~1               # Undo last commit (keep changes)

# Branching
git checkout -b feature-name          # Create and switch to branch
git checkout main                     # Switch to main branch
git merge feature-name                # Merge branch to current
```

## 🧪 Experiment-Specific Workflow
```bash
# Starting new experiment
git checkout -b experiment/new-method
# Make changes to models/calibration.py
git add models/calibration.py
git commit -m "experiment: test new calibration method"

# Save experiment results
python models/train_models.py --train-all
python models/calibration.py
git add results/calibration_comparison.csv
git commit -m "results: new method shows ECE improvement of 15%"

# Merge successful experiment
git checkout main
git merge experiment/new-method
git tag -a v1.3 -m "New calibration method experiment"
```

## 📊 File Management
```bash
# Files to ALWAYS commit:
git add README.md requirements.txt
git add models/ docs/ notebooks/
git add .gitignore

# Files to NEVER commit (handled by .gitignore):
# - results/trained_models.pkl
# - data/raw/*.csv
# - results/visualizations/*.png
# - __pycache__/

# Check what's ignored:
git status --ignored
```

## 🏷️ Tagging Releases
```bash
git tag -a v1.0 -m "Initial working version"
git tag -a v1.1 -m "Added business impact analysis" 
git tag -a v2.0 -m "Corrected documentation results"
git push origin --tags
```

## 🌐 Remote Repository
```bash
# Connect to GitHub/GitLab
git remote add origin https://github.com/username/repo.git
git push -u origin main

# Clone existing repo
git clone https://github.com/username/credit_scoring_experiment.git
```

## 🔧 Configuration
```bash
# Set editor
git config --global core.editor "code --wait"  # VS Code
git config --global core.editor "notepad"      # Notepad

# Set default branch
git config --global init.defaultBranch main

# View all config
git config --list
```

## 🚨 Emergency Commands
```bash
# Accidentally committed large file
git reset --soft HEAD~1               # Undo commit, keep changes
git reset HEAD large_file.pkl         # Unstage large file
echo "large_file.pkl" >> .gitignore   # Add to gitignore
git add .gitignore
git commit -m "Add gitignore for large files"

# Committed to wrong branch
git log --oneline -5                  # Find commit hash
git reset --hard HEAD~1               # Remove commit from current branch
git checkout correct-branch
git cherry-pick <commit-hash>         # Apply commit to correct branch
```

## 📋 Commit Message Templates

### Feature Addition
```
feat(calibration): add temperature scaling method

- Implement temperature scaling for neural networks
- Add validation on calibration set
- Update comparison analysis
```

### Bug Fix
```
fix(models): correct ECE calculation edge case

- Handle empty probability bins
- Add input validation
- Fixes issue with small datasets
```

### Results Update
```
results(experiment): SVM Platt scaling achieves best calibration

- ECE improved from 0.089 to 0.045
- Beats Logistic Regression baseline
- $1.5M potential savings in business scenario
```

### Documentation
```
docs(readme): update expected results with actual findings

- Correct Logistic Regression ECE values
- Add note about calibration myths
- Update business impact calculations
```

## 💡 Best Practices

✅ **DO:**
- Commit frequently with descriptive messages
- Use branches for experiments
- Tag important versions
- Keep commits focused on single changes
- Write clear commit messages

❌ **DON'T:**
- Commit large data files or model artifacts
- Commit passwords or API keys
- Use generic commit messages like "fix" or "update"
- Commit broken code to main branch
- Force push to shared branches

---

**Need help?** See `GIT_GUIDE.md` for detailed instructions! 📚