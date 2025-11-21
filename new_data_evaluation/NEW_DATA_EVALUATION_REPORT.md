# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-21T10:31:53.449069
**Model:** models\neural_classifier.pth
**New Data Source:** new_data/real_world_recreation_gov_20251121_102205.csv

---

## 1. Data Validation - Ensuring Data is Truly New

### Data Provenance Checklist

- ✓ **Data Not Used in Training:** Confirmed
- ✓ **Data Not Used in Validation:** Confirmed
- ✓ **Data Not Used in Initial Testing:** Confirmed
- ✓ **Data Did Not Influence Hyperparameters:** Confirmed
- ✓ **Data Collected After Model Training:** Confirmed

**Data Collection Method:** Recreation Programs Dataset - Public recreation activities

**Total New Samples:** 46

### Class Distribution in New Data

| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 11 | 23.9% |
| Preschool (4-6) | 10 | 21.7% |
| Elementary (7-10) | 11 | 23.9% |
| Teen+ (11+) | 14 | 30.4% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.5217 (52.17%)
- **Precision:** 0.4238
- **Recall:** 0.5217
- **F1-Score:** 0.4283

### Prediction Confidence Statistics

- **Mean Confidence:** 0.3133
- **Median Confidence:** 0.2915
- **Std Deviation:** 0.0612
- **Min Confidence:** 0.2677
- **Max Confidence:** 0.5431

---

## 4. Per-Class Performance on New Data

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.0000 | 0.0000 | 0.0000 | 11 |
| Preschool (4-6) | 0.6364 | 0.7000 | 0.6667 | 10 |
| Elementary (7-10) | 0.6000 | 0.2727 | 0.3750 | 11 |
| Teen+ (11+) | 0.4667 | 1.0000 | 0.6364 | 14 |

---

## 5. Detailed Classification Report

```
Toddler (0-3):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 11.0

Preschool (4-6):
  Precision: 0.6364
  Recall: 0.7000
  F1-Score: 0.6667
  Support: 10.0

Elementary (7-10):
  Precision: 0.6000
  Recall: 0.2727
  F1-Score: 0.3750
  Support: 11.0

Teen+ (11+):
  Precision: 0.4667
  Recall: 1.0000
  F1-Score: 0.6364
  Support: 14.0

macro avg:
  Precision: 0.4258
  Recall: 0.4932
  F1-Score: 0.4195
  Support: 46.0

weighted avg:
  Precision: 0.4238
  Recall: 0.5217
  F1-Score: 0.4283
  Support: 46.0

```

---

## 6. Confusion Matrix

See `figures/confusion_matrix_new_data.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     0      |     1      |     0      |     10    
Preschool  |     0      |     7      |     2      |     1     
Elementary |     0      |     3      |     3      |     5     
Teen+ (11+ |     0      |     0      |     0      |     14    
```

---

## 7. Conclusions and Recommendations


---

## 8. Visualizations

The following visualizations have been generated:

1. **Confusion Matrix:** `figures/confusion_matrix_new_data.png`
2. **Performance Comparison:** `figures/baseline_vs_new_comparison.png`
3. **Per-Class Performance:** `figures/per_class_performance_new_data.png`
4. **Confidence Analysis:** `figures/confidence_analysis_new_data.png`

---

## 9. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-21T10:31:53.449069",
  "model_path": "models\\neural_classifier.pth",
  "new_data_source": "new_data/real_world_recreation_gov_20251121_102205.csv",
  "data_collection_method": "Recreation Programs Dataset - Public recreation activities",
  "confirmation_data_is_new": true,
  "samples_used_in_training": false,
  "samples_used_in_validation": false,
  "samples_used_in_testing": false,
  "samples_influenced_hyperparameters": false
}
```

---

*Report generated automatically by the New Data Evaluator*
