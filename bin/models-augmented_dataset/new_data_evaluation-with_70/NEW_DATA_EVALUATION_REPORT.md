# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-25T11:23:29.524022
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
| Toddler (0-3) | 25 | 35.7% |
| Preschool (4-6) | 10 | 14.3% |
| Elementary (7-10) | 6 | 8.6% |
| Teen+ (11+) | 29 | 41.4% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.4429 (44.29%)
- **Precision:** 0.2784
- **Recall:** 0.4429
- **F1-Score:** 0.3413

### Prediction Confidence Statistics

- **Mean Confidence:** 0.8781
- **Median Confidence:** 0.9368
- **Std Deviation:** 0.1505
- **Min Confidence:** 0.4444
- **Max Confidence:** 0.9998

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.3704 | 0.4429 | +0.0725 | +19.57% |
| F1-Score | 0.3364 | 0.3413 | +0.0049 | +1.45% |

### Performance Assessment

**Assessment:** POOR

**Rubric Score:** 2/10

**Description:** Model performance is far below expectations

- Meets Expectations: ✗ No
- Within Acceptable Range: ✓ Yes

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.0000 | 0.0000 | 0.0000 | 25 |
| Preschool (4-6) | 0.1667 | 0.2000 | 0.1818 | 10 |
| Elementary (7-10) | 0.0909 | 0.1667 | 0.1176 | 6 |
| Teen+ (11+) | 0.5957 | 0.9655 | 0.7368 | 29 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 25.0

Preschool (4-6):
  Precision: 0.1667
  Recall: 0.2000
  F1-Score: 0.1818
  Support: 10.0

Elementary (7-10):
  Precision: 0.0909
  Recall: 0.1667
  F1-Score: 0.1176
  Support: 6.0

Teen+ (11+):
  Precision: 0.5957
  Recall: 0.9655
  F1-Score: 0.7368
  Support: 29.0

macro avg:
  Precision: 0.2133
  Recall: 0.3330
  F1-Score: 0.2591
  Support: 70.0

weighted avg:
  Precision: 0.2784
  Recall: 0.4429
  F1-Score: 0.3413
  Support: 70.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     0      |     9      |     9      |     7     
Preschool  |     0      |     2      |     0      |     8     
Elementary |     0      |     1      |     1      |     4     
Teen+ (11+ |     0      |     0      |     1      |     28    
```

---

## 9. Conclusions and Recommendations

### ✗ POOR PERFORMANCE (≤2/10)

The model shows poor performance on new data:

- Performance far below expectations
- Model does not generalize to new data
- Not suitable for production use

**Recommendations:**
- Complete model redesign likely required
- Review problem formulation and feature selection
- Investigate data quality and labeling consistency
- Consider alternative modeling approaches
- Increase training dataset size substantially

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
  "evaluation_date": "2025-11-25T11:23:29.524022",
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
