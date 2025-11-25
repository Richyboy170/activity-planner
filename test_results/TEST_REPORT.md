# Model Testing Report

**Test Date:** 2025-11-25T13:41:13.826361

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

- **Accuracy:** 0.6296
- **Precision:** 0.6311
- **Recall:** 0.6296
- **F1 Score:** 0.6196

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7000 | 0.3043 | 0.4242 |
| Preschool (4-6) | 0.5319 | 0.5319 | 0.5319 |
| Elementary (7-10) | 0.5192 | 0.5625 | 0.5400 |
| Teen+ (11+) | 0.7500 | 0.8451 | 0.7947 |

## Primary Model: Neural Network

### Architecture

- **Model Type:** Multi-layer Neural Network
- **Architecture:** 384 → 256 → 128 → 4
- **Total Parameters:** 132,740
- **Trainable Parameters:** 132,740

**Layer Details:**

- Input: 384 (Sentence-BERT embeddings)
- Hidden 1: Linear(384, 256) + BatchNorm + ReLU + Dropout(0.5)
- Hidden 2: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.5)
- Output: Linear(128, 4)

**Training Configuration:**

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Regularization:** BatchNorm, Dropout(0.5)

### Rationale

The multi-layer neural network architecture was chosen because:
- Can learn complex non-linear patterns in the embedding space
- Progressive dimensionality reduction (384→256→128→64) allows hierarchical feature learning
- BatchNorm and Dropout provide regularization to prevent overfitting
- Well-suited for high-dimensional embedding inputs

### Quantitative Results

- **Accuracy:** 0.5926
- **Precision:** 0.5753
- **Recall:** 0.5926
- **F1 Score:** 0.5659

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7647 | 0.5652 | 0.6500 |
| Preschool (4-6) | 0.5185 | 0.2979 | 0.3784 |
| Elementary (7-10) | 0.4000 | 0.3750 | 0.3871 |
| Teen+ (11+) | 0.6700 | 0.9437 | 0.7836 |

## Model Comparison

- **Accuracy Improvement:** -0.0370 (-3.70%)
- **F1 Score Improvement:** -0.0537 (-5.37%)
- **Better Performing Model:** Baseline

## Qualitative Analysis

### Prediction Categories

**Both Correct:** 83 samples

**Both Wrong:** 41 samples

**Baseline Correct, Primary Wrong:** 36 samples

**Primary Correct, Baseline Wrong:** 29 samples

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
