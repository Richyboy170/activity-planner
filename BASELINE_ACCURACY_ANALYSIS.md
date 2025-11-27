# Baseline Accuracy Analysis - Complete Explanation

## Executive Summary

**The 50% baseline accuracy is NOT a bug** - it's the expected performance of a trivial "majority class" predictor. The **real concern** is that your Neural Network is only achieving 57% accuracy on new data, barely outperforming this trivial baseline.

---

## Understanding the Three "Baselines"

Your evaluation system uses three different comparison points, which can be confusing:

### 1. Majority Class Baseline: 50% (Trivial Baseline)

**What it is:**
- A naive strategy that always predicts the most common class
- In your evaluation dataset, "Teen+ (11+)" represents exactly 50% of the data (35/70 samples)
- This baseline literally just guesses "Teen+" for every single sample

**Why it shows 50% accuracy:**
```
Total samples: 70
Teen+ samples: 35
Accuracy if always predicting Teen+: 35/70 = 50%
```

**Purpose:**
- Establishes the absolute minimum performance
- Any useful model must significantly outperform this

**Verdict:** ✓ Working as designed

### 2. Random Forest Baseline: ~68% (ML Baseline)

**What it is:**
- A proper machine learning baseline using 100 decision trees
- On the original test set: 67.7% accuracy

**Current Status:**
- ✗ NOT evaluated on your new data yet
- ✗ Model file `random_forest_baseline.pkl` doesn't exist
- The file `train_random_forest_baseline.py` has been created to fix this

**Purpose:**
- Represents a strong, interpretable baseline
- Shows what a simpler ML model can achieve
- Helps determine if the complexity of a neural network is justified

### 3. Original Test Set Performance: 93% (Generalization Check)

**What it is:**
- Your Neural Network's performance on the original test set
- This was the "held-out" data during training

**Purpose:**
- Used to compare new data performance against original performance
- Helps detect if the model generalizes beyond the original test set

---

## Your Current Results Explained

### Data Distribution in Evaluation Set
```
Class                Count    Percentage
────────────────────────────────────────
Toddler (0-3)          7       10%
Preschool (4-6)       21       30%
Elementary (7-10)      7       10%
Teen+ (11+)           35       50%  ← Majority class
────────────────────────────────────────
Total                 70      100%
```

### Performance Comparison

| Model/Baseline              | Original Test | New Data | Gap    |
|-----------------------------|---------------|----------|--------|
| Majority Class (trivial)    | -             | 50.0%    | -      |
| Random Forest (ML baseline) | 67.7%         | *TBD*    | -      |
| Neural Network (your model) | 93.1%         | 57.1%    | -36.0% |

---

## The Real Problem: Poor Generalization

### What the Numbers Tell Us

1. **Barely Beats Trivial Baseline**
   - Neural Network: 57%
   - Majority Class: 50%
   - Improvement: Only 7 percentage points
   - **Conclusion:** Your sophisticated neural network is barely better than always guessing "Teen+"

2. **Massive Performance Drop**
   - Original test: 93%
   - New data: 57%
   - Drop: -36 percentage points (-38.6%)
   - **Conclusion:** The model is not generalizing well to new data

3. **Per-Class Performance Issues**
   ```
   Class              Precision  Recall   F1-Score  Support
   ─────────────────────────────────────────────────────────
   Toddler (0-3)        66.7%    28.6%     40.0%        7
   Preschool (4-6)      71.4%    23.8%     35.7%       21
   Elementary (7-10)     9.1%    14.3%     11.1%        7  ← Very poor!
   Teen+ (11+)          65.3%    91.4%     76.2%       35
   ```

   **Key Issues:**
   - Elementary class: Nearly failing (11% F1-score)
   - Low recall on Toddler and Preschool (only catching ~25% of cases)
   - Model is biased toward predicting "Teen+" (91% recall)

---

## Root Causes

### Likely Causes of Poor Generalization

1. **Overfitting to Training Data**
   - Model memorized training patterns rather than learning generalizable features
   - High performance on original test set may have been partially lucky

2. **Distribution Shift**
   - New evaluation data may differ from training data in important ways:
     - Different activity types
     - Different writing styles
     - Different age range distributions

3. **Class Imbalance**
   - Training data heavily biased toward certain age groups
   - Model learned to default to "Teen+" for uncertain cases
   - Under-represented classes (like Elementary) suffer most

4. **Feature Quality**
   - Text embeddings may not capture age-relevant information well
   - Numerical features (age_min, age_max, duration) may be more reliable

5. **Model Architecture**
   - Current architecture (384→128→64→4) with heavy dropout may be:
     - Too aggressive with regularization
     - Not capturing complex patterns effectively

---

## Recommendations

### Immediate Actions

1. **Complete the Random Forest Baseline Training**
   ```bash
   # Install dependencies (if not already done)
   pip install -r requirements.txt

   # Train Random Forest baseline
   python train_random_forest_baseline.py

   # Re-run evaluation with all baselines
   python evaluate_new_data.py --new-data dataset/evaluation_dataset.csv
   ```

   This will show if the Neural Network is even beating a simpler model.

2. **Analyze the Data Distribution**
   - Compare class distributions between training and evaluation data
   - Look for systematic differences in activity descriptions
   - Check if certain activity types are over/under-represented

3. **Review Training Process**
   - Check for signs of overfitting in training curves
   - Consider if validation set was truly independent
   - Verify that evaluation data is genuinely "new"

### Medium-term Solutions

1. **Improve Training Data Quality**
   - Balance class distribution (use techniques like SMOTE or class weighting)
   - Add more diverse examples for under-represented classes
   - Ensure training data covers full range of activity types

2. **Adjust Model Architecture**
   - Try reducing dropout (currently 0.5 → try 0.3 or 0.2)
   - Consider a simpler architecture if overfitting
   - Add batch normalization for training stability

3. **Feature Engineering**
   - Give more weight to numerical features (age_min, age_max, duration)
   - Try different text representations or embeddings
   - Consider domain-specific features (activity type, complexity indicators)

4. **Regularization Strategies**
   - Use class weights to handle imbalance
   - Try different regularization approaches (L1, L2)
   - Implement early stopping based on validation set

5. **Ensemble Methods**
   - Combine Neural Network with Random Forest
   - Use voting or stacking approaches
   - Leverage strengths of different model types

### Long-term Improvements

1. **Cross-validation**
   - Use k-fold cross-validation to get more reliable performance estimates
   - Identify which types of activities are hardest to classify

2. **Error Analysis**
   - Manually review misclassified examples
   - Identify patterns in failures
   - Adjust features or model based on findings

3. **Active Learning**
   - Identify uncertain predictions
   - Get labels for challenging cases
   - Iteratively improve the model

---

## Expected Results After Fixes

### Realistic Performance Targets

| Scenario | Majority Class | Random Forest | Neural Network |
|----------|----------------|---------------|----------------|
| Minimum acceptable | 50% | 65% | 70% |
| Good performance | 50% | 70% | 80% |
| Excellent performance | 50% | 75% | 90% |

**Current status:** Neural Network at 57% is **below minimum acceptable threshold**

---

## Next Steps

1. ✓ Understanding achieved - you now know why baseline is 50%
2. ⏳ Train Random Forest baseline (run `train_random_forest_baseline.py`)
3. ⏳ Re-evaluate all models on new data
4. ⏳ Implement recommended fixes based on comparison
5. ⏳ Re-train and validate improved model

---

## Files Created

- `train_random_forest_baseline.py` - Script to train and save RF baseline
- `BASELINE_ACCURACY_ANALYSIS.md` - This comprehensive analysis document

## Questions?

If you need clarification on:
- How to interpret any of these metrics
- Which recommendation to prioritize
- How to implement specific improvements

Please ask!