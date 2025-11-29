# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-25T09:08:15.221640
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

**Total New Samples:** 52

### Class Distribution in New Data

| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 9 | 17.3% |
| Preschool (4-6) | 8 | 15.4% |
| Elementary (7-10) | 6 | 11.5% |
| Teen+ (11+) | 29 | 55.8% |

---

## 2. Overall Performance on New Data

### Key Metrics

- **Accuracy:** 0.5962 (59.62%)
- **Precision:** 0.4542
- **Recall:** 0.5962
- **F1-Score:** 0.5065

### Prediction Confidence Statistics

- **Mean Confidence:** 0.9096
- **Median Confidence:** 0.9433
- **Std Deviation:** 0.1103
- **Min Confidence:** 0.5379
- **Max Confidence:** 0.9998

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.0000 | 0.0000 | 0.0000 | 9 |
| Preschool (4-6) | 0.2857 | 0.2500 | 0.2667 | 8 |
| Elementary (7-10) | 0.3333 | 0.1667 | 0.2222 | 6 |
| Teen+ (11+) | 0.6667 | 0.9655 | 0.7887 | 29 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 9.0

Preschool (4-6):
  Precision: 0.2857
  Recall: 0.2500
  F1-Score: 0.2667
  Support: 8.0

Elementary (7-10):
  Precision: 0.3333
  Recall: 0.1667
  F1-Score: 0.2222
  Support: 6.0

Teen+ (11+):
  Precision: 0.6667
  Recall: 0.9655
  F1-Score: 0.7887
  Support: 29.0

macro avg:
  Precision: 0.3214
  Recall: 0.3455
  F1-Score: 0.3194
  Support: 52.0

weighted avg:
  Precision: 0.4542
  Recall: 0.5962
  F1-Score: 0.5065
  Support: 52.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     0      |     4      |     1      |     4     
Preschool  |     0      |     2      |     0      |     6     
Elementary |     0      |     1      |     1      |     4     
Teen+ (11+ |     0      |     0      |     1      |     28    
```

---

## 9. Conclusions and Recommendations


---

## 10. Visualizations

The following visualizations have been generated:

### Neural Network
1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`
2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`
3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`

### Comparisons

---

## 11. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-25T09:08:15.221640",
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
