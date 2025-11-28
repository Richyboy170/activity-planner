# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-28T14:14:12.419677
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

- **Accuracy:** 0.7286 (72.86%)
- **Precision:** 0.7469
- **Recall:** 0.7286
- **F1-Score:** 0.7208

### Prediction Confidence Statistics

- **Mean Confidence:** 0.9200
- **Median Confidence:** 0.9865
- **Std Deviation:** 0.1250
- **Min Confidence:** 0.5037
- **Max Confidence:** 1.0000

### Baseline Comparison (Majority Class Predictor)

- **Baseline Accuracy:** 0.5000 (50.00%)
- **Baseline Method:** Always predicting **Teen+ (11+)**
- **Neural Network Improvement:** +0.2286 (45.71%)

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.9312 | 0.7286 | -0.2026 | -21.76% |
| F1-Score | 0.9317 | 0.7208 | -0.2109 | -22.64% |

### Performance Assessment

**Assessment:** ACCEPTABLE

**Rubric Score:** 7/10

**Description:** Model performance does not meet expectations but is reasonable

- Meets Expectations: ✗ No
- Within Acceptable Range: ✗ No

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.5714 | 0.5714 | 0.5714 | 7 |
| Preschool (4-6) | 0.7500 | 0.4286 | 0.5455 | 21 |
| Elementary (7-10) | 0.2727 | 0.4286 | 0.3333 | 7 |
| Teen+ (11+) | 0.8750 | 1.0000 | 0.9333 | 35 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.5714
  Recall: 0.5714
  F1-Score: 0.5714
  Support: 7.0

Preschool (4-6):
  Precision: 0.7500
  Recall: 0.4286
  F1-Score: 0.5455
  Support: 21.0

Elementary (7-10):
  Precision: 0.2727
  Recall: 0.4286
  F1-Score: 0.3333
  Support: 7.0

Teen+ (11+):
  Precision: 0.8750
  Recall: 1.0000
  F1-Score: 0.9333
  Support: 35.0

macro avg:
  Precision: 0.6173
  Recall: 0.6071
  F1-Score: 0.5959
  Support: 70.0

weighted avg:
  Precision: 0.7469
  Recall: 0.7286
  F1-Score: 0.7208
  Support: 70.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     4      |     3      |     0      |     0     
Preschool  |     3      |     9      |     8      |     1     
Elementary |     0      |     0      |     3      |     4     
Teen+ (11+ |     0      |     0      |     0      |     35    
```

---

## 9. Conclusions and Recommendations

### ⚠ ACCEPTABLE PERFORMANCE (7/10)

The model shows reasonable performance but does not fully meet expectations:

- Performance on new data is lower than baseline
- Some degradation in generalization capability
- Further investigation recommended

**Recommendations:**
- Analyze failure cases to identify patterns
- Consider collecting more training data from underperforming classes
- Review data distribution differences between train and new data
- May require model retraining with augmented dataset

---

## 10. Visualizations

The following visualizations have been generated:

### Neural Network
1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`
2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`
3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`

### Comparisons
7. **Baseline vs Neural Network:** `figures/nn_vs_baseline_accuracy.png`
8. **Neural Network Baseline Comparison:** `figures/baseline_vs_new_comparison_neural_network.png`

---

## 11. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-28T14:14:12.419677",
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
