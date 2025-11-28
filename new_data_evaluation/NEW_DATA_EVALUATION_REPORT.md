# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-28T16:44:10.345183
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

## 2. Model Comparison: Neural Network vs Random Forest Baseline

Both models evaluated on the same new data.

---

## 3. Neural Network Performance Details

### Overall Metrics

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

---

## 5. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.5714 | 0.5714 | 0.5714 | 7 |
| Preschool (4-6) | 0.7500 | 0.4286 | 0.5455 | 21 |
| Elementary (7-10) | 0.2727 | 0.4286 | 0.3333 | 7 |
| Teen+ (11+) | 0.8750 | 1.0000 | 0.9333 | 35 |

---

## 6. Detailed Classification Report (Neural Network)

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

## 7. Confusion Matrices

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

## 8. Summary and Recommendations


---

## 9. Visualizations

The following visualizations have been generated:

### Neural Network
1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`
2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`
3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`


---

## 10. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-28T16:44:10.345183",
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
