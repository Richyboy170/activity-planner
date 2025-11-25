# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-25T16:30:27.324830
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
| Teen (11-17) | 2 | 2.9% |
| Young Adult (18-39) | 15 | 21.4% |
| Adult (40-64) | 6 | 8.6% |
| Senior (65+) | 6 | 8.6% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.2286 (22.86%)
- **Precision:** 0.5129
- **Recall:** 0.2286
- **F1-Score:** 0.2738

### Prediction Confidence Statistics

- **Mean Confidence:** 0.6042
- **Median Confidence:** 0.5737
- **Std Deviation:** 0.1672
- **Min Confidence:** 0.2860
- **Max Confidence:** 0.9766

---

## 3. Comparison with Baseline Performance

### Baseline (Original Test Set) vs New Data

| Metric | Baseline | New Data | Difference | Change % |
|--------|----------|----------|------------|----------|
| Accuracy | 0.7725 | 0.2286 | -0.5439 | -70.41% |
| F1-Score | 0.7731 | 0.2738 | -0.4993 | -64.58% |

### Performance Assessment

**Assessment:** POOR

**Rubric Score:** 2/10

**Description:** Model performance is far below expectations

- Meets Expectations: ✗ No
- Within Acceptable Range: ✗ No

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.8333 | 0.2000 | 0.3226 | 25 |
| Preschool (4-6) | 0.1429 | 0.1000 | 0.1176 | 10 |
| Elementary (7-10) | 0.0000 | 0.0000 | 0.0000 | 6 |
| Teen (11-17) | 0.0690 | 1.0000 | 0.1290 | 2 |
| Young Adult (18-39) | 0.5000 | 0.4000 | 0.4444 | 15 |
| Adult (40-64) | 1.0000 | 0.3333 | 0.5000 | 6 |
| Senior (65+) | 0.0000 | 0.0000 | 0.0000 | 6 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.8333
  Recall: 0.2000
  F1-Score: 0.3226
  Support: 25.0

Preschool (4-6):
  Precision: 0.1429
  Recall: 0.1000
  F1-Score: 0.1176
  Support: 10.0

Elementary (7-10):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 6.0

Teen (11-17):
  Precision: 0.0690
  Recall: 1.0000
  F1-Score: 0.1290
  Support: 2.0

Young Adult (18-39):
  Precision: 0.5000
  Recall: 0.4000
  F1-Score: 0.4444
  Support: 15.0

Adult (40-64):
  Precision: 1.0000
  Recall: 0.3333
  F1-Score: 0.5000
  Support: 6.0

Senior (65+):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 6.0

macro avg:
  Precision: 0.3636
  Recall: 0.2905
  F1-Score: 0.2162
  Support: 70.0

weighted avg:
  Precision: 0.5129
  Recall: 0.2286
  F1-Score: 0.2738
  Support: 70.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen (11-1 | Young Adul | Adult (40- | Senior (65
---------------------------------------------------------------------------------------------------
Toddler (0 |     5      |     5      |     9      |     4      |     2      |     0      |     0     
Preschool  |     1      |     1      |     2      |     5      |     1      |     0      |     0     
Elementary |     0      |     1      |     0      |     5      |     0      |     0      |     0     
Teen (11-1 |     0      |     0      |     0      |     2      |     0      |     0      |     0     
Young Adul |     0      |     0      |     1      |     8      |     6      |     0      |     0     
Adult (40- |     0      |     0      |     2      |     1      |     1      |     2      |     0     
Senior (65 |     0      |     0      |     0      |     4      |     2      |     0      |     0     
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
  "evaluation_date": "2025-11-25T16:30:27.324830",
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
