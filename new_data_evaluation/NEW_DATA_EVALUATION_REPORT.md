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

## 4. Per-Class Performance on New Data

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.0000 | 0.0000 | 0.0000 | 4 |
| Preschool (4-6) | 0.7500 | 0.1200 | 0.2069 | 50 |
| Elementary (7-10) | 0.1111 | 0.0476 | 0.0667 | 42 |
| Teen+ (11+) | 0.0000 | 0.0000 | 0.0000 | 0 |

---

## 5. Detailed Classification Report

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

## 6. Confusion Matrix

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
