# Why Is the Model Accuracy So Low on New Data?

## Executive Summary

The trained neural network achieves **92.59% accuracy** on the test set (from the same distribution as training), but drops dramatically to **38-52% accuracy** on new, unseen data. This represents a **40-54% absolute performance drop**, indicating severe overfitting and poor generalization.

**Key Finding:** The model has learned to memorize patterns specific to the training data rather than learning generalizable features for age-appropriate activity classification.

---

## Table of Contents

1. [Performance Comparison](#performance-comparison)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Detailed Investigation](#detailed-investigation)
4. [Why Each Issue Matters](#why-each-issue-matters)
5. [Recommendations](#recommendations)

---

## Performance Comparison

### Test Set Performance (Same Distribution)
- **Accuracy:** 92.59%
- **Precision:** 0.93
- **Recall:** 0.93
- **F1-Score:** 0.93

**Per-Class Performance:**
| Age Group | Precision | Recall | F1-Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.87 | 1.00 | 0.93 |
| Preschool (4-6) | 1.00 | 0.90 | 0.95 |
| Elementary (7-10) | 0.89 | 0.98 | 0.93 |
| Teen+ (11+) | 0.93 | 0.81 | 0.87 |

### New Data Performance (Recreation Gov Dataset - 46 samples)
- **Accuracy:** 52.17%
- **Precision:** 0.42
- **Recall:** 0.52
- **F1-Score:** 0.43

**Per-Class Performance:**
| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 0.00 | 0.00 | 0.00 | 11 |
| Preschool (4-6) | 0.64 | 0.70 | 0.67 | 10 |
| Elementary (7-10) | 0.60 | 0.27 | 0.38 | 11 |
| Teen+ (11+) | 0.47 | 1.00 | 0.64 | 14 |

**Critical Issues:**
- ❌ Complete failure on Toddler class (0% precision/recall)
- ❌ Teen+ class: 100% recall but only 47% precision → **severe overprediction**
- ❌ Elementary class: only 27% recall → **most activities misclassified as Teen+**

### Recent Evaluation (96 samples)
- **Accuracy:** 38.54%
- **Precision:** 0.63
- **Recall:** 0.39
- **F1-Score:** 0.47

**Confusion Matrix Highlights:**
- 38 out of 96 predictions were Teen+ (39.6%)
- But actual Teen+ samples: **0** (0%)
- Model predicted Teen+ for activities that were actually Preschool or Elementary

**Performance Drop:** -40% to -54% absolute accuracy

---

## Root Cause Analysis

### 1. 🎯 Overfitting to Training Distribution

**Problem:** The model learned patterns specific to the training data distribution rather than generalizable age-appropriate features.

**Evidence:**

**Training Data Distribution (1,069 activities):**
```
Toddler (0-3):       143 activities (13.4%)
Preschool (4-6):     379 activities (35.5%)
Elementary (7-10):   337 activities (31.5%)
Teen+ (11+):         210 activities (19.6%)
```

**New Data Distribution (46 samples):**
```
Toddler (0-3):       11 samples (23.9%)
Preschool (4-6):     10 samples (21.7%)
Elementary (7-10):   11 samples (23.9%)
Teen+ (11+):         14 samples (30.4%)
```

**Recent Evaluation (96 samples):**
```
Toddler (0-3):       4 samples (4.2%)
Preschool (4-6):     50 samples (52.1%)
Elementary (7-10):   42 samples (43.8%)
Teen+ (11+):         0 samples (0%)
```

**Why This Matters:**
- Training data has very few Toddler activities (13.4%), so model rarely sees this class
- New data has different distribution, confusing the model
- Model learned "if it looks like 35% of training samples, it's Preschool" rather than "if it has toddler-appropriate features, it's Toddler"

### 2. 📊 Class Imbalance in Training Data

**Problem:** Unequal representation of age groups biases the model toward majority classes.

**Training Distribution:**
- **Dominant class:** Preschool (35.5%)
- **Underrepresented:** Toddler (13.4%)
- **Ratio:** 2.66:1 (Preschool to Toddler)

**Impact on Predictions:**
- Model has 0% precision/recall on Toddler class in new data
- Model defaults to predicting Preschool or Teen+ when uncertain
- Minority classes get "drowned out" during training

**Loss Function Issue:**
```python
criterion = nn.CrossEntropyLoss()  # ❌ No class weighting
```

This treats all classes equally during training, but since Preschool appears 2.66x more often than Toddler, the model sees far more examples of Preschool patterns and optimizes primarily for the majority class.

### 3. 🔤 Feature Mismatch and Text Structure Differences

**Problem:** Training data and new data have fundamentally different text structures.

**Training Data Structure:**
```csv
title,age_min,age_max,duration_mins,tags,cost,indoor_outdoor,season,
materials_needed,how_to_play,players,parent_caution

Animal Walk,2,4,15,"exercise, fun",free,both,all,[],
"['Choose an animal...', 'Encourage them to copy...']",1+,no
```

**New Data Structure:**
```csv
title,description,age_min,age_max,tags,cost,indoor_outdoor,season,
players,duration_mins,source,fetched_at

Toddler Playgroup,"Supervised play session for toddlers with
age-appropriate toys and activities",1,3,"social, play, development",
low,indoor,all,5-10,45,Recreation Programs Dataset,2025-11-21
```

**Key Differences:**

| Feature | Training Data | New Data | Impact |
|---------|---------------|----------|--------|
| **Instruction Field** | `how_to_play` (step-by-step list) | `description` (paragraph) | Different text patterns |
| **Text Style** | Imperative ("Choose...", "Encourage...") | Descriptive ("Supervised play...") | Different vocabulary |
| **Detail Level** | Very detailed, 3-4 steps | Concise, 1-2 sentences | Different embedding density |
| **Fields Missing** | Has `materials_needed`, `parent_caution` | No materials/caution | Less context |

**Example Comparison:**

**Training Data:**
```
Title: Animal Walk
How to Play: ['Choose an animal like a snake, frog, horse, or bear
and show children how it moves.', 'Encourage them to copy the movements,
adding their own sounds and gestures.', 'Take turns picking new animals...']
Tags: exercise, fun
```

**New Data:**
```
Title: Toddler Playgroup
Description: Supervised play session for toddlers with age-appropriate
toys and activities
Tags: social, play, development
```

**Embedding Impact:**
- Training embeddings cluster around "imperative instructions + materials"
- New data embeddings cluster around "descriptive summaries"
- These form **different regions in the embedding space**
- Model's learned decision boundaries don't transfer

### 4. 🧠 Model Architecture Overfitting

**Current Architecture:**
```python
Input (384) → 512 → 512 → 384 → 384 → 256 → 256 → 128 → 128 → 64 → 64 → Output (4)
```

**Total Parameters:** 140,868 (primary model) or similar

**Issues:**

1. **Too Many Parameters for Dataset Size:**
   - Training samples: 1,069
   - Parameters: 140,868
   - Ratio: **132 parameters per training sample**
   - Rule of thumb: Need ~10 samples per parameter
   - **Actual need:** 1.4 million samples!

2. **Deep Architecture:**
   - 10 hidden layers for a simple 4-class classification
   - Each layer can learn dataset-specific patterns
   - More capacity = more overfitting risk

3. **Regularization Insufficient:**
   ```python
   layers.append(nn.Dropout(dropout))  # dropout=0.3
   ```
   - 30% dropout helps, but not enough for this level of overfitting
   - No L2 regularization on weights
   - No early stopping based on validation loss

### 5. 🎲 Synthetic Labeling Based on Single Feature

**Current Labeling Logic** (train_model.py:333-343):
```python
def create_labels(df_activities):
    labels = []
    for idx, row in df_activities.iterrows():
        age_min = row['age_min']
        if age_min <= 3:
            labels.append(0)  # Toddler
        elif age_min <= 6:
            labels.append(1)  # Preschool
        elif age_min <= 10:
            labels.append(2)  # Elementary
        else:
            labels.append(3)  # Teen+
    return labels
```

**Critical Issues:**

1. **Single Feature Decision:**
   - Uses ONLY `age_min` to determine class
   - Ignores `age_max`, content, tags, complexity
   - Example: "Drama Club" with age_min=8, age_max=12 → labeled as Elementary
     - But content suggests older children
     - Model learns "Drama" → Elementary, fails when seeing similar activities for teens

2. **Boundary Ambiguity:**
   - Activity: age_min=10 → Elementary
   - Activity: age_min=11 → Teen+
   - Often same activity type, but different labels
   - Model learns arbitrary cutoffs rather than semantic differences

3. **Ignores Age Range:**
   - "Parent-Child Swim" (1-3): Toddler ✓
   - "Junior Soccer" (7-10): Elementary ✓
   - "Drama Club" (8-12): Elementary ❌ (overlaps with Teen+)
   - Model doesn't learn that wider ranges = different classification

### 6. 🌍 Limited Training Data Diversity

**Training Data Source:**
- Single source: Custom-created dataset
- Consistent naming conventions
- Similar writing style
- Predictable structure

**New Data Sources:**
- Recreation Programs Dataset (public parks & rec)
- UCI Student Performance Dataset (academic research)
- Different organizations, different styles

**Vocabulary Differences:**

| Training Data | New Data |
|---------------|----------|
| "toddler", "kids", "children" | "youth", "participants", "students" |
| "how to play" | "program description" |
| "materials needed" | "equipment required" or missing |
| Casual tone | Professional/formal tone |

**Example:**

**Training:** "Throw Snowballs - Pack soft snow into small balls and demonstrate how to aim and throw."

**New Data:** "Winter Recreation Program - Outdoor winter activities including supervised snowball games with safety protocols."

Same activity, completely different semantic representation!

### 7. 📉 No Cross-Dataset Validation

**Current Training/Test Split:**
```python
# train_model.py:347-356
X_temp, X_test, y_temp, y_test = train_test_split(
    embeddings, labels, test_size=0.10, random_state=42, stratify=labels
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp
)
```

**What's Wrong:**
- All splits from **same dataset**
- Train, validation, and test have **identical data distribution**
- Model learns dataset-specific patterns
- Validation loss goes down, but model isn't actually generalizing

**Why Test Accuracy Was Misleading:**
- 92.59% accuracy on test set
- Seemed excellent!
- But test set had same biases, vocabulary, and structure as training
- **Not a true measure of generalization**

**Better Approach:**
```python
# Use different datasets for validation
train_data = original_dataset
val_data = recreation_gov_sample  # Different source!
test_data = uci_student_sample    # Different source!
```

### 8. 🎭 Embedding Model Limitations

**Current Embedding Model:**
```python
self.model = SentenceTransformer('all-MiniLM-L6-v2')
```

**Specifications:**
- Embedding dimension: 384
- General-purpose sentence embeddings
- Trained on diverse text, but not activity-specific

**Issues:**

1. **Not Domain-Specific:**
   - Embeddings optimized for general semantic similarity
   - Doesn't know that "toddler playgroup" and "baby music class" are similar age-wise
   - Might place "advanced robotics" and "competitive swimming" far apart even though both are teen activities

2. **Single Embedding Per Activity:**
   - Combines title, description, tags, all fields into one embedding
   - Loses structured information
   - Can't separately weight "title is most important for age classification"

3. **Fixed Embeddings:**
   - Embeddings frozen during training
   - Classifier can't adjust embedding space to better separate age groups
   - Must work with whatever semantic space MiniLM provides

---

## Why Each Issue Matters

### Overfitting to Distribution
**Real-World Impact:**
- Model works only on data similar to training
- Any new data source → poor performance
- Can't deploy to production
- Users get wrong activity recommendations

**Technical Impact:**
- Memorization vs. learning
- Model captures noise, not signal
- Decision boundaries too specific

### Class Imbalance
**Real-World Impact:**
- Toddlers get almost no correct recommendations
- Parents of young children can't use the system
- Discriminates against minority classes

**Technical Impact:**
- Gradients dominated by majority class
- Loss function optimized for Preschool/Elementary
- Rare classes never learned properly

### Feature Mismatch
**Real-World Impact:**
- Can't integrate external activity databases
- Limited to proprietary data format
- Manual reformatting required

**Technical Impact:**
- Embedding distributions don't overlap
- Out-of-distribution (OOD) detection impossible
- Transfer learning fails

### Architecture Overfitting
**Real-World Impact:**
- Model too complex for simple task
- Slow training and inference
- Hard to deploy on mobile/edge devices

**Technical Impact:**
- High variance, low bias
- Perfect training accuracy, poor test accuracy
- Unstable to small data changes

### Synthetic Labeling Issues
**Real-World Impact:**
- Labels don't reflect true age appropriateness
- Activities mislabeled at boundaries
- Content contradicts label

**Technical Impact:**
- Noisy labels → confused model
- Multi-modal age distributions
- Inconsistent ground truth

### Limited Diversity
**Real-World Impact:**
- Can't generalize to new activity types
- Fails on different regions/cultures
- Vocabulary drift over time

**Technical Impact:**
- Narrow semantic coverage
- Embedding clusters too tight
- No robustness to paraphrasing

### No Cross-Dataset Validation
**Real-World Impact:**
- False confidence in model quality
- Production failures surprise team
- Wasted development effort

**Technical Impact:**
- Optimistic performance estimates
- Validation doesn't catch overfitting
- Hyperparameters tuned to wrong objective

### Embedding Limitations
**Real-World Impact:**
- Can't capture age-specific semantics
- Similar activities separated incorrectly
- Content understanding too shallow

**Technical Impact:**
- Fixed representation
- No task-specific optimization
- Information bottleneck at 384 dimensions

---

## Detailed Investigation

### Confusion Matrix Analysis (Recreation Gov Data)

```
         | Toddler (0 | Preschool  | Elementary | Teen+ (11+
------------------------------------------------------------
Toddler (0 |     0      |     1      |     0      |     10
Preschool  |     0      |     7      |     2      |     1
Elementary |     0      |     3      |     3      |     5
Teen+ (11+ |     0      |     0      |     0      |     14
```

**Key Observations:**

1. **Toddler Row (11 actual Toddler activities):**
   - Correct predictions: 0
   - Predicted as Teen+: 10 (91%)
   - Predicted as Preschool: 1 (9%)
   - **Catastrophic failure:** Model thinks toddler activities are for teenagers!

2. **Teen+ Row (14 actual Teen+ activities):**
   - Correct predictions: 14
   - **Perfect recall!**
   - But high false positive rate (overpredicts Teen+)

3. **Pattern:**
   - Model defaulting to Teen+ when uncertain
   - Learned bias: "if in doubt, predict Teen+"
   - Likely due to Teen+ activities having more complex vocabulary

### Recent Evaluation Analysis (96 samples, 0 actual Teen+)

**Predictions:**
- Teen+ predicted: 38 times
- Actual Teen+: 0 times
- **38 false positives!**

**This proves:**
- Model sees complex/formal language → predicts Teen+
- New data uses more formal descriptions → triggers Teen+ bias
- Model hasn't learned age-appropriate content, just linguistic complexity

### Prediction Confidence Analysis

**Recreation Gov Data:**
- Mean confidence: 0.31 (31%)
- Median confidence: 0.29 (29%)
- **Very low!** Model is uncertain but still makes predictions

**Recent Evaluation:**
- Mean confidence: 0.76 (76%)
- **Very high!** But accuracy is only 38.54%
- **Model is confidently wrong**

**Interpretation:**
- Low confidence → Model knows it doesn't know (OOD detection working)
- High confidence + low accuracy → Model has false certainty
- Indicates different failure modes for different data sources

---

## Recommendations

### Immediate Actions (High Priority)

#### 1. 🎯 Implement Class Weighting
```python
# Calculate class weights inversely proportional to frequency
class_counts = [143, 379, 337, 210]  # Toddler, Preschool, Elementary, Teen+
class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum()  # Normalize

# Use weighted loss
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
```

**Expected Impact:** +10-15% accuracy on minority classes

#### 2. 📚 Augment Training Data with Diverse Sources
- Add Recreation Gov dataset samples to training (properly labeled)
- Include UCI student activities
- Mix multiple data sources in training
- Ensure each source represented in train/val/test

**Expected Impact:** +20-30% accuracy on new data

#### 3. 🏗️ Simplify Model Architecture
```python
# Before: 10 layers, 140K parameters
# After: 3 layers, much smaller
hidden_dims = [256, 128]  # Much simpler
dropout = 0.5  # Stronger regularization
```

**Expected Impact:** +5-10% generalization, faster training

#### 4. 🎲 Improve Labeling Strategy
```python
def create_labels(row):
    age_min, age_max = row['age_min'], row['age_max']
    age_mid = (age_min + age_max) / 2

    # Use midpoint + consider range
    if age_mid <= 3.5:
        return 0  # Toddler
    elif age_mid <= 7:
        return 1  # Preschool
    elif age_mid <= 11:
        return 2  # Elementary
    else:
        return 3  # Teen+
```

**Expected Impact:** +5-8% accuracy from better ground truth

### Medium-Term Actions (Medium Priority)

#### 5. 🔄 Implement Cross-Dataset Validation
```python
# Split by data source, not randomly
train_data = original_dataset.sample(frac=0.8)
val_data = recreation_gov_dataset  # Different source!
test_data = uci_student_dataset    # Different source!
```

**Expected Impact:** Realistic performance estimates, better hyperparameter tuning

#### 6. 📊 Add Data Augmentation
- Paraphrase activity descriptions
- Synonym replacement
- Back-translation (English → French → English)
- Mix formal and casual language

**Expected Impact:** +10-15% robustness to language variations

#### 7. 🎯 Fine-Tune Embedding Model
```python
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Fine-tune on activity pairs
model = SentenceTransformer('all-MiniLM-L6-v2')
train_loss = losses.SoftmaxLoss(...)
model.fit(train_objectives=[(dataloader, train_loss)])
```

**Expected Impact:** +15-20% accuracy from domain-specific embeddings

### Long-Term Actions (Lower Priority)

#### 8. 🏗️ Multi-Task Learning
- Predict age_min and age_max separately
- Add auxiliary tasks (predict indoor/outdoor, cost, etc.)
- Share representations, multiple heads

**Expected Impact:** Better feature learning, +10-15% accuracy

#### 9. 🌍 Collect More Diverse Data
- Partner with recreation departments
- Scrape public activity databases
- User-contributed activities
- International sources

**Expected Impact:** Foundational improvement, enables real-world deployment

#### 10. 🔍 Ensemble Methods
- Train multiple models with different architectures
- Random Forest + Neural Network + Gradient Boosting
- Average predictions

**Expected Impact:** +5-10% accuracy, more robust

### Evaluation Improvements

#### 11. 📈 Better Metrics
- Track per-class metrics (not just overall accuracy)
- Monitor precision/recall separately
- Use F1-score as primary metric
- Add calibration metrics (Expected Calibration Error)

#### 12. 🎯 Stratified Evaluation
- Test on each age group separately
- Test on each data source separately
- Identify specific failure modes

#### 13. 🚨 Add OOD Detection
```python
# Flag low-confidence predictions
if max(prediction_probs) < 0.5:
    return "UNCERTAIN - manual review needed"
```

**Expected Impact:** Better user experience, safer recommendations

---

## Conclusion

The model's poor performance on new data stems from **systematic overfitting** at multiple levels:

1. **Dataset level:** Overfitting to training distribution and vocabulary
2. **Model level:** Too many parameters for available data
3. **Training level:** No class balancing, insufficient regularization
4. **Evaluation level:** Test set from same distribution, masking issues

The **92.59% test accuracy was misleading** because the test set came from the same distribution. The true measure of model quality is **38-52% accuracy on new data**.

### Priority Order for Fixes

**Immediate (Do First):**
1. Implement class weighting → Quick win for minority classes
2. Simplify architecture → Reduce overfitting
3. Mix training data sources → Learn generalizable patterns

**Medium-Term (Do Next):**
4. Cross-dataset validation → Catch overfitting during development
5. Improve labeling → Better ground truth
6. Data augmentation → More training diversity

**Long-Term (Do Eventually):**
7. Fine-tune embeddings → Domain-specific representations
8. Multi-task learning → Better features
9. Ensemble methods → Robust predictions

### Expected Outcome

With these fixes:
- **Current new data accuracy:** 38-52%
- **Expected accuracy after immediate fixes:** 60-70%
- **Expected accuracy after all fixes:** 75-85%

The model will never reach 92% on new data (that was overfitting), but **75-85% is realistic and usable** for production deployment.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-22
**Author:** Analysis of activity-planner model evaluation results
