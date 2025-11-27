# How to Use the Baseline Analysis and Fix

## Quick Start Guide

Follow these steps to train the Random Forest baseline and get a complete evaluation of your models.

---

## Step 1: Install Dependencies

If you haven't already, install the required Python packages:

```bash
# Option A: Use the setup script (recommended)
bash setup_test_environment.sh

# Option B: Manual installation
pip install -r requirements.txt
```

**Wait for installation to complete** (may take 2-5 minutes depending on your system).

---

## Step 2: Train the Random Forest Baseline

Run the training script I created:

```bash
python3 train_random_forest_baseline.py
```

**What this does:**
- Loads your training dataset
- Generates embeddings using Sentence-BERT
- Trains a Random Forest classifier (100 trees, max depth 20)
- Evaluates it on the test set
- Saves the model to `models/random_forest_baseline.pkl`
- Saves training data for future use

**Expected output:**
```
================================================================================
TRAINING RANDOM FOREST BASELINE MODEL
================================================================================

Loading dataset...
✓ Loaded 2849 activities

Generating embeddings...
Generated embeddings with shape: (2849, 384)

Creating labels...
Splitting data...
Train: 2279 samples
Val: 284 samples
Test: 189 samples

================================================================================
Training Random Forest (this may take a minute)...
================================================================================
[Parallel output from sklearn...]

================================================================================
Evaluating Random Forest...
================================================================================

Test Accuracy: 0.6772 (67.72%)

Classification Report:
              precision    recall  f1-score   support

   Toddler       1.00      0.18      0.31        11
  Preschool      0.50      0.04      0.08        24
 Elementary      0.55      0.30      0.39        40
      Teen+       0.69      0.99      0.82       114

    accuracy                           0.68       189

✓ Random Forest model saved to: models/random_forest_baseline.pkl
✓ Training embeddings saved to: models/train_embeddings.npy
✓ Training labels saved to: models/train_labels.npy

================================================================================
RANDOM FOREST BASELINE TRAINING COMPLETE!
================================================================================
```

**Time required:** 1-3 minutes

---

## Step 3: Re-run Evaluation with All Baselines

Now evaluate all three models on your new data:

```bash
python3 evaluate_new_data.py --new-data dataset/evaluation_dataset.csv
```

**What this does:**
- Evaluates the Neural Network on new data
- Evaluates the Random Forest baseline on new data
- Compares both against the majority class baseline
- Generates comprehensive reports and visualizations

**Expected output:**
```
================================================================================
STARTING MODEL EVALUATION ON NEW DATA
================================================================================

Loading new data from: dataset/evaluation_dataset.csv
Loaded 70 new samples

Class distribution in new data:
  Toddler (0-3): 7 samples (10.0%)
  Preschool (4-6): 21 samples (30.0%)
  Elementary (7-10): 7 samples (10.0%)
  Teen+ (11+): 35 samples (50.0%)

Generating embeddings for 70 samples...
Generated embeddings with shape: (70, 384)

Extracting numerical features (age_min, age_max, duration_mins)...
Combined features shape: (70, 387)

Evaluating baseline (majority class) on new data...
Baseline evaluation complete - Accuracy: 0.5000 (predicting Teen+ (11+))

Evaluating model on new data...
Evaluation complete - Accuracy: 0.5714

Comparing Neural Network vs Baseline (majority class) on new data...
Neural Network Accuracy: 0.5714 | Baseline Accuracy: 0.5000
Improvement: 0.0714 (14.29%)

Evaluating Random Forest baseline on new data...
Random Forest evaluation complete - Accuracy: 0.6571  ← THIS IS NEW!

Comparing Neural Network vs Random Forest baseline...
Neural Network Accuracy: 0.5714 | Random Forest Accuracy: 0.6571  ← IMPORTANT!
Winner: Random Forest

Generating visualizations...
  ✓ Confusion matrices saved
  ✓ Learning curves saved
  ✓ Per-class performance plots saved
  ✓ Model comparison plots saved

✓ Report saved to: new_data_evaluation/NEW_DATA_EVALUATION_REPORT.md
✓ Results saved to: new_data_evaluation/new_data_evaluation_results.json

================================================================================
EVALUATION COMPLETE
================================================================================
```

---

## Step 4: Review the Results

### A. Check the Evaluation Report

```bash
# View the report in your terminal
cat new_data_evaluation/NEW_DATA_EVALUATION_REPORT.md

# Or open it in a text editor
nano new_data_evaluation/NEW_DATA_EVALUATION_REPORT.md
```

**What to look for:**

1. **Section 2: Overall Performance**
   - How well does your Neural Network perform on new data?
   - Target: Should be >70% to be useful

2. **Section 4: Random Forest Baseline Performance**
   - How does the simpler model perform?
   - If Random Forest > Neural Network: Your NN needs improvement

3. **Section 5: Model Comparison**
   - Which model wins?
   - How big is the performance gap?

4. **Section 6: Per-Class Performance**
   - Which age groups does your model struggle with?
   - Look for classes with F1-score < 0.5

### B. Check the Visualizations

```bash
ls -lh new_data_evaluation/figures/
```

**Key visualizations:**
- `confusion_matrix_neural_network.png` - See where NN makes mistakes
- `confusion_matrix_random_forest.png` - See where RF makes mistakes
- `neural_network_vs_random_forest.png` - Direct comparison
- `nn_vs_baseline_accuracy.png` - NN vs majority class
- `per_class_performance_*.png` - Performance breakdown by age group

### C. Check the JSON Results

For detailed numerical results:

```bash
python3 -m json.tool new_data_evaluation/new_data_evaluation_results.json | less
```

---

## Understanding Your Results

### Scenario 1: Neural Network Wins ✓

```
Neural Network:  75%
Random Forest:   68%
Majority Class:  50%
```

**Interpretation:**
- ✓ Your NN is working well
- ✓ Complexity is justified
- Action: Monitor performance, deploy if >70%

### Scenario 2: Random Forest Wins ⚠

```
Neural Network:  57%
Random Forest:   68%
Majority Class:  50%
```

**Interpretation:**
- ✗ Your NN is underperforming
- ✗ Simpler model is better
- Action: Fix NN using recommendations in BASELINE_ACCURACY_ANALYSIS.md

### Scenario 3: Both Poor ⚠⚠

```
Neural Network:  57%
Random Forest:   55%
Majority Class:  50%
```

**Interpretation:**
- ✗ Both models barely beat guessing
- ✗ Fundamental data or feature problem
- Action: Review data quality, feature engineering

---

## Step 5: Fix Your Model (If Needed)

If your Neural Network is underperforming, read the recommendations in:

```bash
cat BASELINE_ACCURACY_ANALYSIS.md
```

**Quick fixes to try:**

1. **Reduce Dropout** (in `train_model.py`):
   ```python
   # Change from:
   dropout=0.5
   # To:
   dropout=0.2
   ```

2. **Add Class Weights** (in `train_model.py`):
   ```python
   # Calculate class weights
   from sklearn.utils.class_weight import compute_class_weight
   class_weights = compute_class_weight('balanced',
                                        classes=np.unique(y_train),
                                        y=y_train)
   # Use in loss function
   criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights))
   ```

3. **Collect More Data**:
   - Focus on under-represented classes (Elementary, Toddler)
   - Ensure diverse activity types

4. **Try Ensemble**:
   ```python
   # Combine predictions from both models
   final_prediction = (nn_prediction + rf_prediction) / 2
   ```

---

## Common Issues

### Issue 1: "ModuleNotFoundError: No module named 'numpy'"

**Solution:**
```bash
pip install -r requirements.txt
# or
bash setup_test_environment.sh
```

### Issue 2: "FileNotFoundError: dataset/evaluation_dataset.csv"

**Solution:**
```bash
# Check if the file exists
ls dataset/evaluation_dataset.csv

# If not, use the correct path
python3 evaluate_new_data.py --new-data path/to/your/data.csv
```

### Issue 3: Random Forest training is slow

**Solution:**
This is normal. Random Forest trains 100 trees on high-dimensional data (384 features).
Expected time: 1-3 minutes depending on your CPU.

### Issue 4: "Model not found at models/neural_classifier.pth"

**Solution:**
```bash
# Train your neural network first
python3 train_model.py
```

---

## Quick Reference

### File Purposes

| File | Purpose |
|------|---------|
| `train_random_forest_baseline.py` | Train and save RF baseline |
| `evaluate_new_data.py` | Evaluate all models on new data |
| `BASELINE_ACCURACY_ANALYSIS.md` | Understand the results and get recommendations |
| `HOW_TO_USE_BASELINE_FIX.md` | This file - step-by-step guide |

### Directory Structure After Running

```
activity-planner/
├── models/
│   ├── random_forest_baseline.pkl      ← RF model (created by step 2)
│   ├── train_embeddings.npy            ← Training data (created by step 2)
│   ├── train_labels.npy                ← Training labels (created by step 2)
│   └── neural_classifier.pth           ← NN model (should already exist)
│
└── new_data_evaluation/
    ├── NEW_DATA_EVALUATION_REPORT.md   ← Main report (updated by step 3)
    ├── new_data_evaluation_results.json ← Detailed results (updated by step 3)
    └── figures/                         ← Visualizations (updated by step 3)
        ├── confusion_matrix_neural_network.png
        ├── confusion_matrix_random_forest.png
        ├── neural_network_vs_random_forest.png
        └── ...
```

---

## Summary

1. Install dependencies: `bash setup_test_environment.sh`
2. Train RF baseline: `python3 train_random_forest_baseline.py`
3. Evaluate all models: `python3 evaluate_new_data.py --new-data dataset/evaluation_dataset.csv`
4. Review results: `cat new_data_evaluation/NEW_DATA_EVALUATION_REPORT.md`
5. Fix model if needed: Follow recommendations in `BASELINE_ACCURACY_ANALYSIS.md`

**Expected total time:** 5-10 minutes

---

## Questions?

If you encounter any issues or need clarification:
1. Check the "Common Issues" section above
2. Review `BASELINE_ACCURACY_ANALYSIS.md` for detailed explanations
3. Check the error messages carefully - they usually indicate what's wrong

Good luck!