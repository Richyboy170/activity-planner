# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-25T09:35:18.302007
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

- **Accuracy:** 0.6154 (61.54%)
- **Precision:** 0.6164
- **Recall:** 0.6154
- **F1-Score:** 0.5477

### Prediction Confidence Statistics

- **Mean Confidence:** 0.9007
- **Median Confidence:** 0.9813
- **Std Deviation:** 0.1570
- **Min Confidence:** 0.4376
- **Max Confidence:** 0.9996

---

## 6. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 1.0000 | 0.2222 | 0.3636 | 9 |
| Preschool (4-6) | 0.3333 | 0.1250 | 0.1818 | 8 |
| Elementary (7-10) | 0.2500 | 0.1667 | 0.2000 | 6 |
| Teen+ (11+) | 0.6512 | 0.9655 | 0.7778 | 29 |

---

## 7. Detailed Classification Report (Neural Network)

```
Toddler (0-3):
  Precision: 1.0000
  Recall: 0.2222
  F1-Score: 0.3636
  Support: 9.0

Preschool (4-6):
  Precision: 0.3333
  Recall: 0.1250
  F1-Score: 0.1818
  Support: 8.0

Elementary (7-10):
  Precision: 0.2500
  Recall: 0.1667
  F1-Score: 0.2000
  Support: 6.0

Teen+ (11+):
  Precision: 0.6512
  Recall: 0.9655
  F1-Score: 0.7778
  Support: 29.0

macro avg:
  Precision: 0.5586
  Recall: 0.3699
  F1-Score: 0.3808
  Support: 52.0

weighted avg:
  Precision: 0.6164
  Recall: 0.6154
  F1-Score: 0.5477
  Support: 52.0

```

---

## 8. Confusion Matrix

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     2      |     1      |     1      |     5     
Preschool  |     0      |     1      |     1      |     6     
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
  "evaluation_date": "2025-11-25T09:35:18.302007",
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
