# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-26T16:50:23.947622
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

**Total New Samples:** 70

### Class Distribution in New Data

| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 7 | 10.0% |
| Preschool (4-6) | 21 | 30.0% |
| Elementary (7-10) | 7 | 10.0% |
| Teen+ (11+) | 35 | 50.0% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.5714 (57.14%)
- **Precision:** 0.6166
- **Recall:** 0.5714
- **F1-Score:** 0.5392

### Prediction Confidence Statistics

- **Mean Confidence:** 0.8121
- **Median Confidence:** 0.9230
- **Std Deviation:** 0.1982
- **Min Confidence:** 0.4174
- **Max Confidence:** 0.9978

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.9312 | 0.5714 | -0.3598 | -38.64% |
| F1-Score | 0.9317 | 0.5392 | -0.3925 | -42.13% |

### Performance Assessment

**Assessment:** INCONSISTENT

**Rubric Score:** 4/10

**Description:** Model performs inconsistently on new samples

- Meets Expectations: ✗ No
- Within Acceptable Range: ✗ No

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.6667 | 0.2857 | 0.4000 | 7 |
| Preschool (4-6) | 0.7143 | 0.2381 | 0.3571 | 21 |
| Elementary (7-10) | 0.0909 | 0.1429 | 0.1111 | 7 |
| Teen+ (11+) | 0.6531 | 0.9143 | 0.7619 | 35 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.6667
  Recall: 0.2857
  F1-Score: 0.4000
  Support: 7.0

Preschool (4-6):
  Precision: 0.7143
  Recall: 0.2381
  F1-Score: 0.3571
  Support: 21.0

Elementary (7-10):
  Precision: 0.0909
  Recall: 0.1429
  F1-Score: 0.1111
  Support: 7.0

Teen+ (11+):
  Precision: 0.6531
  Recall: 0.9143
  F1-Score: 0.7619
  Support: 35.0

macro avg:
  Precision: 0.5312
  Recall: 0.3952
  F1-Score: 0.4075
  Support: 70.0

weighted avg:
  Precision: 0.6166
  Recall: 0.5714
  F1-Score: 0.5392
  Support: 70.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     2      |     2      |     0      |     3     
Preschool  |     1      |     5      |     7      |     8     
Elementary |     0      |     0      |     1      |     6     
Teen+ (11+ |     0      |     0      |     3      |     32    
```

---

## 9. Conclusions and Recommendations

### ⚠ INCONSISTENT PERFORMANCE (4/10)

The model shows inconsistent performance on new data:

- Significant performance drop compared to baseline
- Model may be overfitting to training data
- Substantial improvement needed before production use

**Recommendations:**
- Investigate data distribution shift between train and new data
- Consider regularization techniques to improve generalization
- Expand training dataset with more diverse samples
- Review feature engineering approach
- Retrain model with adjusted hyperparameters

---

## 10. Visualizations

The following visualizations have been generated:

### Neural Network
1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`
2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`
3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`

### Comparisons
7. **Neural Network Baseline Comparison:** `figures/baseline_vs_new_comparison_neural_network.png`

---

## 11. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-26T16:50:23.947622",
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
