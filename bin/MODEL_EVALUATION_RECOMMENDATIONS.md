# Model Evaluation Recommendations - Neural Classifier Performance Analysis

**Date:** 2025-11-26
**Model:** Neural Classifier (models/neural_classifier.pth)
**Assessment:** INCONSISTENT (4/10) - Requires Immediate Action

---

## Executive Summary

The neural network classifier shows a **severe performance degradation** on new data:
- **Baseline accuracy**: 93.12%
- **New data accuracy**: 57.14%
- **Performance drop**: -38.64% (unacceptable for production)

### Root Causes Identified

1. **Severe Class Imbalance**: Training data heavily biased towards "Teen+ (11+)" class (60.1%)
2. **Distribution Shift**: Evaluation data has 18.2% more Preschool samples than training
3. **Model Bias**: Network defaults to predicting majority class, failing on minority classes

---

## Detailed Performance Analysis

### Per-Class Performance Breakdown

| Age Group | Precision | Recall | F1-Score | Issue |
|-----------|-----------|--------|----------|-------|
| **Toddler (0-3)** | 50.0% | **14.3%** | 22.2% | ❌ Severe recall failure |
| **Preschool (4-6)** | 66.7% | **28.6%** | 40.0% | ❌ Poor recall |
| **Elementary (7-10)** | **10.0%** | **14.3%** | 11.8% | ❌ Critical failure |
| **Teen+ (11+)** | 65.3% | **91.4%** | 76.2% | ✅ Good (but at cost of others) |

**Key Finding**: Model correctly identifies **91.4% of Teen+ samples** but only **14.3% of Toddler and Elementary samples**.

### Confusion Matrix Analysis

From the evaluation report:
```
Predicted Teen+ when actually:
├─ Toddler (0-3):     3/7 samples (42.9%)
├─ Preschool (4-6):   8/21 samples (38.1%)
└─ Elementary (7-10): 6/7 samples (85.7%)
```

**Diagnosis**: Model has learned to predict "Teen+" as a safe default due to its dominance in training (60.1%).

---

## Distribution Shift Analysis

### Training vs Evaluation Distribution

| Age Group | Training % | Evaluation % | Shift |
|-----------|-----------|--------------|-------|
| Toddler (0-3) | 5.5% | 10.0% | +4.5% |
| Preschool (4-6) | 11.8% | **30.0%** | **+18.2%** ⚠️ |
| Elementary (7-10) | 22.6% | 10.0% | -12.6% |
| Teen+ (11+) | **60.1%** | 50.0% | -10.1% |

**Critical Issue**: Preschool class increased from 11.8% to 30.0% in evaluation, but model was undertrained on this class.

---

## Actionable Recommendations

### 🔴 Priority 1: Address Class Imbalance (CRITICAL)

**Current Issue**: Training data severely imbalanced (60% Teen+, only 5.5% Toddler)

**Solutions** (implement in order):

1. **Rebalance Training Data**
   - Use SMOTE (Synthetic Minority Over-sampling Technique) for minority classes
   - Or under-sample majority class (Teen+) to 30-40% of dataset
   - Target distribution: 20-25% per class for balanced learning

2. **Stronger Class Weighting**
   - Current: Soft class weights using `1.0 / sqrt(class_counts)`
   - Recommended: Use inverse class frequency `1.0 / class_counts` or focal loss
   - This forces the model to pay more attention to minority classes

3. **Stratified Data Augmentation**
   - Generate more augmented samples specifically for:
     - Toddler (0-3): Increase from 5.5% to 15-20%
     - Preschool (4-6): Increase from 11.8% to 20-25%
     - Elementary (7-10): Keep or increase slightly to 25%

**Implementation Location**: `train_model.py:564-566` (class weighting)

---

### 🟡 Priority 2: Improve Model Architecture

**Current Architecture**:
```
Input (384) → [Dropout 0.15] → Linear(128) → ReLU → BatchNorm → Dropout(0.3)
           → Linear(64) → ReLU → BatchNorm → Dropout(0.3) → Output(4)
```

**Recommended Changes**:

1. **Increase Model Capacity for Small Dataset**
   - Current: 2 layers [128, 64] - may be too shallow
   - Recommended: 3 layers [256, 128, 64] or [192, 96, 48]
   - Rationale: With 24K samples, we can afford slightly more parameters

2. **Reduce Dropout for Better Learning**
   - Current: 0.3 dropout (30%)
   - Recommended: Start with 0.2 (20%) and monitor
   - Rationale: Current dropout may be too aggressive given class imbalance

3. **Try Focal Loss Instead of CrossEntropyLoss**
   - Focal Loss focuses training on hard-to-classify examples
   - Particularly effective for imbalanced datasets
   - Formula: `FL(p_t) = -α(1-p_t)^γ * log(p_t)`

**Implementation Location**: `train_model.py:442-452` (model architecture)

---

### 🟢 Priority 3: Training Strategy Improvements

1. **Increase Training Data Diversity**
   - Current: 24,026 augmented samples
   - Issue: May contain duplicate patterns
   - Solution: Review augmentation techniques to ensure true diversity
   - Check: `text_augmentation.py` for augmentation quality

2. **Cross-Validation on Balanced Folds**
   - Current: 5-fold CV with stratified split
   - Recommended: Ensure each fold has balanced classes
   - Monitor per-class performance across folds

3. **Early Stopping Modification**
   - Current: Patience = 15 epochs based on validation loss
   - Recommended: Use per-class recall as stopping criterion
   - Stop if minority class recall degrades

**Implementation Location**: `train_model.py:550-630` (training loop)

---

### 🔵 Priority 4: Evaluation and Monitoring

1. **Add Per-Class Validation Monitoring**
   - Track recall for each class during training
   - Alert if any class drops below 60% recall
   - Log per-class performance to identify early bias

2. **Implement Calibrated Predictions**
   - Current: Model confidence is high (mean 85%) but often wrong
   - Solution: Apply temperature scaling or Platt scaling
   - This improves probability calibration

3. **Create Balanced Test Sets**
   - Ensure test/validation sets have representative distributions
   - Current eval dataset has different distribution than train
   - Solution: Create multiple test sets with different distributions

---

## Implementation Priority & Impact

| Priority | Action | Expected Impact | Effort | Timeline |
|----------|--------|-----------------|--------|----------|
| 🔴 **P1** | Rebalance training data (SMOTE/undersample) | +20-30% accuracy | Medium | 1-2 days |
| 🔴 **P1** | Stronger class weighting (inverse freq) | +10-15% on minority classes | Low | 1 hour |
| 🟡 **P2** | Focal Loss implementation | +5-10% on hard examples | Medium | 4 hours |
| 🟡 **P2** | Increase model capacity [256, 128, 64] | +5-10% overall | Low | 30 mins |
| 🟡 **P2** | Reduce dropout to 0.2 | +3-5% on minority classes | Low | 10 mins |
| 🟢 **P3** | Per-class early stopping | Prevent overfitting | Medium | 2 hours |
| 🔵 **P4** | Calibrated predictions | Better confidence scores | Medium | 3 hours |

**Recommended First Steps**:
1. ✅ Implement stronger class weighting (1 hour)
2. ✅ Reduce dropout to 0.2 (10 mins)
3. ✅ Rebalance training data with SMOTE (1-2 days)
4. ✅ Retrain and evaluate

---

## Expected Outcomes After Improvements

**Conservative Estimates** (after P1 + P2):
- Overall accuracy: 57% → **75-80%**
- Toddler (0-3) recall: 14% → **60-70%**
- Preschool (4-6) recall: 29% → **65-75%**
- Elementary (7-10) recall: 14% → **55-65%**
- Teen+ (11+) recall: 91% → **85-90%** (slight decrease acceptable)

**Aggressive Estimates** (after all priorities):
- Overall accuracy: 57% → **85-90%**
- Per-class recall: All classes above **70%**
- Model rubric score: 4/10 → **9-10/10**

---

## Code Changes Required

### 1. Stronger Class Weighting (train_model.py:564-566)

**Current**:
```python
class_weights = 1.0 / torch.sqrt(torch.tensor(self.class_counts, dtype=torch.float32))
class_weights = class_weights / class_weights.sum() * self.num_classes
```

**Recommended**:
```python
# Option A: Inverse frequency (stronger)
class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * self.num_classes

# Option B: Effective number of samples (more sophisticated)
beta = 0.9999  # Hyperparameter
effective_num = 1.0 - torch.pow(beta, torch.tensor(self.class_counts, dtype=torch.float32))
class_weights = (1.0 - beta) / effective_num
class_weights = class_weights / class_weights.sum() * self.num_classes
```

### 2. Reduced Dropout (train_model.py:450-451)

**Current**:
```python
hidden_dims = [128, 64]
dropout_rate = 0.3
```

**Recommended**:
```python
hidden_dims = [256, 128, 64]  # Increased capacity
dropout_rate = 0.2  # Reduced dropout
```

### 3. Data Rebalancing (add to train_model.py before line 365)

```python
from imblearn.over_sampling import SMOTE

def balance_data_smote(embeddings, labels):
    """Apply SMOTE to balance minority classes"""
    smote = SMOTE(
        sampling_strategy='auto',  # Balances all classes to majority class count
        random_state=42,
        k_neighbors=5
    )
    X_resampled, y_resampled = smote.fit_resample(embeddings, labels)
    logger.info(f"SMOTE: {len(labels)} → {len(y_resampled)} samples")
    return X_resampled, y_resampled
```

---

## Validation Plan

After implementing changes:

1. **Retrain Model** with new configuration
2. **Evaluate on Same Eval Set** to ensure improvements
3. **Track Metrics**:
   - Overall accuracy target: >80%
   - Per-class recall target: >60% for all classes
   - F1-score target: >0.70 overall
4. **Cross-Validate** to ensure generalization
5. **Monitor** for overfitting with train/val loss gap

---

## Long-Term Recommendations

1. **Collect More Balanced Data**
   - Actively collect more Toddler and Preschool samples
   - Target: 500+ unique samples per class

2. **Ensemble Methods**
   - Combine Neural Network with Random Forest
   - Use voting or stacking for final predictions

3. **Feature Engineering**
   - Explore additional features beyond embeddings
   - Add explicit age features, duration categories, etc.

4. **Regular Model Retraining**
   - Retrain quarterly with new data
   - Monitor distribution shift over time

---

## Conclusion

The neural classifier shows **severe overfitting to the majority class** due to:
- 60% of training data being "Teen+ (11+)"
- Insufficient representation of Toddler and Preschool classes
- Model defaulting to safe predictions

**Immediate action required**: Implement Priority 1 recommendations to achieve production-ready performance.

**Target**: Increase accuracy from 57% to 80%+ within 1-2 weeks of focused effort.

---

*Generated: 2025-11-26*
*Next Review: After implementing P1 and P2 recommendations*
