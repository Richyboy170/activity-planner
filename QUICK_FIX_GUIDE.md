# Quick Fix Guide - Immediate Performance Improvements

**Estimated Time**: 15 minutes for quick wins, 1-2 days for full fix
**Expected Impact**: 57% → 75-80% accuracy

---

## 🚀 Quick Wins (15 minutes)

### Fix 1: Stronger Class Weighting
**File**: `train_model.py:564`
**Change**: Use inverse frequency instead of sqrt

```python
# Replace lines 564-566
class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)
class_weights = class_weights / class_weights.sum() * self.num_classes
```

### Fix 2: Reduce Dropout
**File**: `train_model.py:450-451`
**Change**: Reduce from 0.3 to 0.2

```python
hidden_dims = [128, 64]
dropout_rate = 0.2  # Changed from 0.3
```

### Fix 3: Increase Model Capacity
**File**: `train_model.py:450`
**Change**: Add more hidden layers

```python
hidden_dims = [256, 128, 64]  # Changed from [128, 64]
dropout_rate = 0.2
```

**Run**: `python3 train_model.py`

---

## 🔧 Full Fix (1-2 days)

### Step 1: Install SMOTE
```bash
pip install imbalanced-learn
```

### Step 2: Add SMOTE Rebalancing
**File**: `train_model.py`
**Location**: Before line 364 (in `prepare_data` method)

```python
from imblearn.over_sampling import SMOTE

# Add this function to NeuralTrainer class
def balance_with_smote(self, X, y):
    """Balance dataset using SMOTE"""
    logger.info("\n[Data Balancing] Applying SMOTE...")

    # Show original distribution
    unique, counts = np.unique(y, return_counts=True)
    label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
    logger.info("  Original distribution:")
    for label, count in zip(unique, counts):
        logger.info(f"    {label_names[label]}: {count}")

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # Show new distribution
    unique, counts = np.unique(y_balanced, return_counts=True)
    logger.info("  After SMOTE:")
    for label, count in zip(unique, counts):
        logger.info(f"    {label_names[label]}: {count}")

    logger.info(f"✓ SMOTE complete: {len(y)} → {len(y_balanced)} samples\n")
    return X_balanced, y_balanced
```

### Step 3: Use SMOTE in Training
**File**: `train_model.py:364`
**Modify the `prepare_data` method**:

```python
# After creating labels (around line 389), add:
labels = np.array(labels)

# Balance the dataset with SMOTE
embeddings, labels = self.balance_with_smote(embeddings, labels)

# Then continue with train/val/test split
```

### Step 4: Retrain
```bash
python3 train_model.py
```

### Step 5: Re-evaluate
```bash
python3 evaluate_new_data.py
```

---

## 📊 Expected Results

### Before (Current)
- Overall Accuracy: 57.14%
- Toddler Recall: 14.3%
- Preschool Recall: 28.6%
- Elementary Recall: 14.3%
- Teen+ Recall: 91.4%

### After Quick Wins (15 min)
- Overall Accuracy: ~65-70%
- Minority Class Recall: ~35-45%
- Reduced bias to Teen+

### After Full Fix (1-2 days)
- Overall Accuracy: ~75-85%
- All Classes Recall: >60%
- Balanced predictions across classes

---

## ✅ Validation Checklist

After retraining:
- [ ] Overall accuracy >75%
- [ ] All class recalls >50%
- [ ] No class with recall <40%
- [ ] Train/val loss gap <0.2
- [ ] Confusion matrix shows balanced predictions
- [ ] F1-score >0.70

---

## 🚨 Troubleshooting

**Issue**: SMOTE fails with "not enough neighbors"
**Solution**: Reduce `k_neighbors=3` to `k_neighbors=2` or `k_neighbors=1`

**Issue**: Training takes too long
**Solution**: Reduce augmented dataset size or use smaller batch size

**Issue**: Overfitting increases
**Solution**: Increase dropout back to 0.25-0.3, add L2 regularization

---

## 📝 Notes

- Quick wins alone should give 10-15% improvement
- Full SMOTE rebalancing gives additional 10-20% improvement
- Monitor training closely to avoid overfitting to synthetic samples
- Consider collecting more real Toddler/Preschool samples for long-term fix

---

*Created: 2025-11-26*
