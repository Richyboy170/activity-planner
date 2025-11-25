# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-25T11:08:02.351165
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

**Total New Samples:** 147

### Class Distribution in New Data

| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 39 | 26.5% |
| Preschool (4-6) | 25 | 17.0% |
| Elementary (7-10) | 34 | 23.1% |
| Teen+ (11+) | 49 | 33.3% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.3673 (36.73%)
- **Precision:** 0.5011
- **Recall:** 0.3673
- **F1-Score:** 0.3164

### Prediction Confidence Statistics

- **Mean Confidence:** 0.7491
- **Median Confidence:** 0.7730
- **Std Deviation:** 0.1738
- **Min Confidence:** 0.3524
- **Max Confidence:** 0.9950

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.3704 | 0.3673 | -0.0030 | -0.82% |
| F1-Score | 0.3364 | 0.3164 | -0.0201 | -5.97% |

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
| Toddler (0-3) | 1.0000 | 0.0769 | 0.1429 | 39 |
| Preschool (4-6) | 0.0833 | 0.0800 | 0.0816 | 25 |
| Elementary (7-10) | 0.2821 | 0.3235 | 0.3014 | 34 |
| Teen+ (11+) | 0.4691 | 0.7755 | 0.5846 | 49 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 1.0000
  Recall: 0.0769
  F1-Score: 0.1429
  Support: 39.0

Preschool (4-6):
  Precision: 0.0833
  Recall: 0.0800
  F1-Score: 0.0816
  Support: 25.0

Elementary (7-10):
  Precision: 0.2821
  Recall: 0.3235
  F1-Score: 0.3014
  Support: 34.0

Teen+ (11+):
  Precision: 0.4691
  Recall: 0.7755
  F1-Score: 0.5846
  Support: 49.0

macro avg:
  Precision: 0.4586
  Recall: 0.3140
  F1-Score: 0.2776
  Support: 147.0

weighted avg:
  Precision: 0.5011
  Recall: 0.3673
  F1-Score: 0.3164
  Support: 147.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     3      |     16     |     9      |     11    
Preschool  |     0      |     2      |     11     |     12    
Elementary |     0      |     3      |     11     |     20    
Teen+ (11+ |     0      |     3      |     8      |     38    
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
  "evaluation_date": "2025-11-25T11:08:02.351165",
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
