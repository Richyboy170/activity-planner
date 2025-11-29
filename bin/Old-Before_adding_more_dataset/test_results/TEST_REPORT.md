# Model Testing Report

**Test Date:** 2025-11-07T16:00:20.046430

---

## Baseline Model: Random Forest

### Architecture

- **Model Type:** Random Forest
- **Number of Estimators:** 100
- **Max Depth:** 20
- **Tuning Required:** Minimal
- **Complexity:** Low to Medium

### Rationale

Random Forest was chosen as the baseline because:
- Simple to implement and requires minimal hyperparameter tuning
- Well-suited for tabular/embedding data
- Provides interpretable feature importance
- Establishes a strong baseline for comparison

### Quantitative Results

- **Accuracy:** 0.6545
- **Precision:** 0.5792
- **Recall:** 0.6545
- **F1 Score:** 0.6106

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7500 | 1.0000 | 0.8571 |
| Preschool (4-6) | 0.0000 | 0.0000 | 0.0000 |
| Elementary (7-10) | 0.7273 | 0.6957 | 0.7111 |
| Teen+ (11+) | 0.5238 | 0.6875 | 0.5946 |

## Primary Model: Neural Network

### Architecture

- **Model Type:** Multi-layer Neural Network
- **Architecture:** 384 → 256 → 128 → 64 → 4
- **Total Parameters:** 140,868
- **Trainable Parameters:** 140,868

**Layer Details:**

- Input: 384 (Sentence-BERT embeddings)
- Hidden 1: Linear(384, 256) + BatchNorm + ReLU + Dropout(0.3)
- Hidden 2: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.3)
- Hidden 3: Linear(128, 64) + BatchNorm + ReLU + Dropout(0.3)
- Output: Linear(64, 4)

**Training Configuration:**

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Regularization:** BatchNorm, Dropout(0.3)

### Rationale

The multi-layer neural network architecture was chosen because:
- Can learn complex non-linear patterns in the embedding space
- Progressive dimensionality reduction (384→256→128→64) allows hierarchical feature learning
- BatchNorm and Dropout provide regularization to prevent overfitting
- Well-suited for high-dimensional embedding inputs

### Quantitative Results

- **Accuracy:** 0.9091
- **Precision:** 0.9177
- **Recall:** 0.9091
- **F1 Score:** 0.9095

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.8889 | 0.8889 | 0.8889 |
| Preschool (4-6) | 0.8571 | 0.8571 | 0.8571 |
| Elementary (7-10) | 1.0000 | 0.8696 | 0.9302 |
| Teen+ (11+) | 0.8421 | 1.0000 | 0.9143 |

## Model Comparison

- **Accuracy Improvement:** 0.2545 (+25.45%)
- **F1 Score Improvement:** 0.2989 (+29.89%)
- **Better Performing Model:** Primary

## Qualitative Analysis

### Prediction Categories

**Both Correct:** 34 samples

**Both Wrong:** 3 samples

**Baseline Correct, Primary Wrong:** 2 samples

**Primary Correct, Baseline Wrong:** 16 samples

## Visualizations

All visualizations are saved in the `figures/` directory:

1. **Confusion Matrices** - `confusion_matrices.png`
2. **Learning Curves** - `learning_curves.png`
3. **Per-Class Performance** - `per_class_performance.png`
4. **Model Comparison** - `model_comparison.png`
5. **Confidence Distribution** - `confidence_distribution.png`

## Challenges and Observations

### Baseline Model Challenges

- Random Forest models can struggle with high-dimensional embedding spaces (384 dimensions)
- Limited ability to learn complex non-linear patterns compared to neural networks
- May overfit if trees are too deep or underfit if too shallow

### Primary Model Challenges

- Requires more training time compared to Random Forest
- Needs careful hyperparameter tuning (learning rate, dropout, batch size)
- Risk of overfitting on small datasets without proper regularization
- Less interpretable than tree-based models

### Dataset Observations

- 80/10/10 train/validation/test split ensures robust evaluation
- Class imbalance may affect per-class performance
- Embedding quality from Sentence-BERT is crucial for both models

---

**Full numerical results are available in `test_report.json`**
