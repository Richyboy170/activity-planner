# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-24T13:17:59.823883
**Model:** models\neural_classifier.pth
**New Data Source:** dataset/evaluation_dataset.csv

---

## 1. Data Validation - Ensuring Data is Truly New

### Data Provenance Checklist

- ✓ **Data Not Used in Training:** Confirmed
- ✓ **Data Not Used in Validation:** Confirmed
- ✓ **Data Not Used in Initial Testing:** Confirmed
- ✓ **Data Did Not Influence Hyperparameters:** Confirmed
- ✓ **Data Collected After Model Training:** Confirmed

**Data Collection Method:** evaluation

**Total New Samples:** 96

### Class Distribution in New Data

| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 4 | 4.2% |
| Preschool (4-6) | 50 | 52.1% |
| Elementary (7-10) | 42 | 43.8% |
| Teen+ (11+) | 0 | 0.0% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.0833 (8.33%)
- **Precision:** 0.4392
- **Recall:** 0.0833
- **F1-Score:** 0.1369

### Prediction Confidence Statistics

- **Mean Confidence:** 0.9287
- **Median Confidence:** 0.9988
- **Std Deviation:** 0.1248
- **Min Confidence:** 0.5087
- **Max Confidence:** 1.0000

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.3704 | 0.0833 | -0.2871 | -77.51% |
| F1-Score | 0.3364 | 0.1369 | -0.1995 | -59.31% |

### Performance Assessment

**Assessment:** POOR

**Rubric Score:** 2/10

**Description:** Model performance is far below expectations

- Meets Expectations: ✗ No
- Within Acceptable Range: ✗ No

### Analysis

The neural network shows significant performance degradation on the new evaluation dataset:

- **Accuracy dropped from 37.04% to 8.33%** - a decrease of 77.51%
- **F1-Score dropped from 0.3364 to 0.1369** - a decrease of 59.31%
- The model is classifying most samples as "Teen+ (11+)" (70/96 predictions)
- This represents poor generalization to the new data distribution
- The new dataset has different class distribution (no Teen+ samples, 52.1% Preschool, 43.8% Elementary)
- The model appears to have learned biases from the training data that don't transfer well

**Recommendations:**
- Investigate data distribution differences between train and new data
- Consider collecting more diverse training samples
- Review feature engineering and embedding quality
- May require model retraining with balanced dataset
- Consider using the Random Forest baseline which showed better performance on original test set (54.63% accuracy)

---

## 4. Random Forest Baseline Performance

### Random Forest Configuration

- **n_estimators:** 100 trees
- **max_depth:** 20
- **Purpose:** Simple, interpretable, minimal tuning baseline

### Overall Metrics (Original Test Set)

- **Accuracy:** 0.5463 (54.63%)
- **Precision:** 0.6248
- **Recall:** 0.5463
- **F1-Score:** 0.5192

### Comparison: Neural Network vs Random Forest (Original Test Set)

The Random Forest baseline outperformed the Neural Network on the original test set:

| Metric | Neural Network | Random Forest | Winner |
|--------|---------------|---------------|--------|
| Accuracy | 0.3704 | 0.5463 | Random Forest |
| Precision | 0.3997 | 0.6248 | Random Forest |
| Recall | 0.3704 | 0.5463 | Random Forest |
| F1-Score | 0.3364 | 0.5192 | Random Forest |

**Key Insight:** The simpler Random Forest baseline significantly outperforms the more complex Neural Network model, suggesting that:
- The additional complexity of the Neural Network may not be justified for this task
- The Random Forest's ensemble approach handles this classification problem more effectively
- For production deployment, the Random Forest baseline may be more suitable

**Note:** Random Forest was not evaluated on the new evaluation dataset in this run. For a complete comparison, consider running the Random Forest model on the new data as well.

---

## 5. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.0000 | 0.0000 | 0.0000 | 4 |
| Preschool (4-6) | 0.7500 | 0.1200 | 0.2069 | 50 |
| Elementary (7-10) | 0.1111 | 0.0476 | 0.0667 | 42 |
| Teen+ (11+) | 0.0000 | 0.0000 | 0.0000 | 0 |

---

## 6. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 4.0

Preschool (4-6):
  Precision: 0.7500
  Recall: 0.1200
  F1-Score: 0.2069
  Support: 50.0

Elementary (7-10):
  Precision: 0.1111
  Recall: 0.0476
  F1-Score: 0.0667
  Support: 42.0

Teen+ (11+):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 0.0

macro avg:
  Precision: 0.2153
  Recall: 0.0419
  F1-Score: 0.0684
  Support: 96.0

weighted avg:
  Precision: 0.4392
  Recall: 0.0833
  F1-Score: 0.1369
  Support: 96.0

```

---

## 7. Confusion Matrix (Neural Network)

See `figures/confusion_matrix_new_data.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     0      |     2      |     1      |     1     
Preschool  |     0      |     6      |     15     |     29    
Elementary |     0      |     0      |     2      |     40    
Teen+ (11+ |     0      |     0      |     0      |     0     
```

---

## 8. Conclusions and Recommendations

### ⚠ POOR PERFORMANCE (2/10)

The model shows poor performance on new data:

- Performance far below expectations
- Model does not generalize well to new data
- Not suitable for production use in current state

**Key Findings:**

1. **Severe Performance Degradation:**
   - Accuracy dropped from 37% to 8% on new data
   - Model is heavily biased toward predicting "Teen+" class
   - Only 8 out of 96 samples classified correctly

2. **Class Distribution Mismatch:**
   - Training/test data likely had more "Teen+" samples
   - New evaluation data has 0% Teen+ samples
   - Model fails to adapt to different class distributions

3. **Random Forest Outperforms Neural Network:**
   - Random Forest achieved 54.63% accuracy on original test set
   - Neural Network only achieved 37.04% on same test set
   - Simpler model may be more appropriate for this task

**Recommendations:**

1. **Immediate Actions:**
   - Do NOT deploy current neural network model to production
   - Consider using Random Forest baseline instead (54.63% accuracy)
   - Collect performance metrics from Random Forest on new data

2. **Model Improvement:**
   - Complete model redesign likely required
   - Address severe class imbalance in training data
   - Implement proper data augmentation and balancing techniques
   - Consider ensemble methods or different architecture

3. **Data Collection:**
   - Investigate data quality and labeling consistency
   - Collect more diverse training samples across all age groups
   - Ensure training data distribution matches expected deployment scenarios
   - Balance class representation in training set

4. **Alternative Approaches:**
   - Evaluate Random Forest baseline on new data
   - Consider simpler models (Logistic Regression, SVM)
   - Implement proper cross-validation with stratified sampling
   - Use calibration techniques to improve probability estimates

---

## 9. Visualizations

The following visualizations have been generated:

1. **Confusion Matrix:** `figures/confusion_matrix_new_data.png`
2. **Performance Comparison:** `figures/baseline_vs_new_comparison.png`
3. **Per-Class Performance:** `figures/per_class_performance_new_data.png`
4. **Confidence Analysis:** `figures/confidence_analysis_new_data.png`

---

## 10. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-24T13:17:59.823883",
  "model_path": "models\\neural_classifier.pth",
  "new_data_source": "dataset/evaluation_dataset.csv",
  "data_collection_method": "evaluation",
  "confirmation_data_is_new": true,
  "samples_used_in_training": false,
  "samples_used_in_validation": false,
  "samples_used_in_testing": false,
  "samples_influenced_hyperparameters": false
}
```

---

*Report generated automatically by the New Data Evaluator*
