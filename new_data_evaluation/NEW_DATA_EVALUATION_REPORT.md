# Model Evaluation on New Data - Comprehensive Report

**Evaluation Date:** 2025-11-29T10:12:40.743679
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

## 2. Quantitative Result

### Model Comparison: Neural Network vs Random Forest Baseline

Both models evaluated on the same new data.

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Neural Network** | 0.7286 | 0.7469 | 0.7286 | 0.7208 |
| **Random Forest Baseline** | 0.5857 | 0.4990 | 0.5857 | 0.5124 |

### Model Comparison Analysis

✓ **Neural Network outperforms Random Forest Baseline**
  - Accuracy improvement: +0.1429 (+14.29%)
  - The neural network demonstrates superior performance on new data

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

## 4. Random Forest Baseline Performance Details

### Configuration

- **n_estimators:** 100 trees
- **max_depth:** 20
- **Purpose:** Simple, interpretable baseline with minimal tuning

### Overall Metrics on New Data

- **Accuracy:** 0.5857 (58.57%)
- **Precision:** 0.4990
- **Recall:** 0.5857
- **F1-Score:** 0.5124

---

## 5. Per-Class Performance on New Data (Neural Network)

| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.5714 | 0.5714 | 0.5714 | 7 |
| Preschool (4-6) | 0.7500 | 0.4286 | 0.5455 | 21 |
| Elementary (7-10) | 0.2727 | 0.4286 | 0.3333 | 7 |
| Teen+ (11+) | 0.8750 | 1.0000 | 0.9333 | 35 |

### Side-by-Side Comparison (F1-Score)

| Age Group | Neural Network F1 | Random Forest F1 | Difference |
|-----------|-------------------|------------------|------------|
| Toddler (0-3) | 0.5714 | 0.0000 | +0.5714 |
| Preschool (4-6) | 0.5455 | 0.3226 | +0.2229 |
| Elementary (7-10) | 0.3333 | 0.1333 | +0.2000 |
| Teen+ (11+) | 0.9333 | 0.8046 | +0.1287 |

---

## 6. Detailed Classification Reports

### Neural Network
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

### Random Forest Baseline
```
Toddler (0-3):
  Precision: 0.0000
  Recall: 0.0000
  F1-Score: 0.0000
  Support: 7.0

Preschool (4-6):
  Precision: 0.5000
  Recall: 0.2381
  F1-Score: 0.3226
  Support: 21.0

Elementary (7-10):
  Precision: 0.1250
  Recall: 0.1429
  F1-Score: 0.1333
  Support: 7.0

Teen+ (11+):
  Precision: 0.6731
  Recall: 1.0000
  F1-Score: 0.8046
  Support: 35.0

macro avg:
  Precision: 0.3245
  Recall: 0.3452
  F1-Score: 0.3151
  Support: 70.0

weighted avg:
  Precision: 0.4990
  Recall: 0.5857
  F1-Score: 0.5124
  Support: 70.0

```

---

## 7. Confusion Matrices

### Neural Network
See `figures/confusion_matrix_neural_network.png` for visualization.

### Random Forest Baseline
See `figures/confusion_matrix_random_forest_baseline.png` for visualization.

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

### ✓ Neural Network Outperforms Baseline

The neural network demonstrates superior performance compared to the Random Forest baseline:

- Neural Network Accuracy: 0.7286 (72.86%)
- Random Forest Baseline Accuracy: 0.5857 (58.57%)
- Improvement: +0.1429 (+14.29%)

**Recommendations:**
- The neural network is the recommended model for this task
- Continue monitoring performance on future data batches
- Consider the neural network suitable for production use

---

## 9. Qualitative Result

### Example Predictions

| Activity Text (Snippet) | True Label | Predicted Label | Confidence | Result |
|-------------------------|------------|-----------------|------------|--------|
| Pottery Workshop Pottery Workshop Pottery Workshop... | Teen+ (11+) | Teen+ (11+) | 0.9988 | ✅ Correct |
| Photography Walk Photography Walk Photography Walk... | Teen+ (11+) | Teen+ (11+) | 0.9609 | ✅ Correct |
| Adult Yoga Class Adult Yoga Class Adult Yoga Class... | Teen+ (11+) | Teen+ (11+) | 0.9952 | ✅ Correct |
| Color and Shape Sorting Color and Shape Sorting Co... | Preschool (4-6) | Preschool (4-6) | 0.9995 | ✅ Correct |
| Hula Hoop Walk Through Hula Hoop Walk Through Hula... | Toddler (0-3) | Toddler (0-3) | 0.6580 | ✅ Correct |
| Matching Game Hunt Matching Game Hunt Matching Gam... | Toddler (0-3) | Preschool (4-6) | 0.8677 | ❌ Incorrect |
| Safe Knife Skills Safe Knife Skills Safe Knife Ski... | Preschool (4-6) | Toddler (0-3) | 0.9788 | ❌ Incorrect |
| Infant Tummy Time Infant Tummy Time Infant Tummy T... | Toddler (0-3) | Preschool (4-6) | 0.7783 | ❌ Incorrect |
| Paper City Creation Paper City Creation Paper City... | Preschool (4-6) | Elementary (7-10) | 0.5037 | ❌ Incorrect |
| Comic Book Creation Comic Book Creation Comic Book... | Elementary (7-10) | Teen+ (11+) | 0.5922 | ❌ Incorrect |


---

## 10. Visualizations

The following visualizations have been generated:

### Neural Network
1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`
2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`
3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`

### Random Forest Baseline
4. **Confusion Matrix:** `figures/confusion_matrix_random_forest_baseline.png`
5. **Per-Class Performance:** `figures/per_class_performance_random_forest_baseline.png`
6. **Confidence Analysis:** `figures/confidence_analysis_random_forest_baseline.png`

### Model Comparison
7. **Neural Network vs Random Forest Baseline:** `figures/neural_network_vs_random_forest_baseline.png`


---

## 11. Evaluation Metadata

```json
{
  "evaluation_date": "2025-11-29T10:12:40.743679",
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
