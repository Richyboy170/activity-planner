# Model Testing Report

**Test Date:** 2025-11-22T18:13:25.287657

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

- **Accuracy:** 0.5463
- **Precision:** 0.6248
- **Recall:** 0.5463
- **F1 Score:** 0.5192

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 1.0000 | 0.1429 | 0.2500 |
| Preschool (4-6) | 0.4915 | 0.8788 | 0.6304 |
| Elementary (7-10) | 0.5312 | 0.4595 | 0.4928 |
| Teen+ (11+) | 0.7333 | 0.4583 | 0.5641 |

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

- **Accuracy:** 0.3704
- **Precision:** 0.3997
- **Recall:** 0.3704
- **F1 Score:** 0.3364

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.8333 | 0.3571 | 0.5000 |
| Preschool (4-6) | 0.4286 | 0.1818 | 0.2553 |
| Elementary (7-10) | 0.2000 | 0.1892 | 0.1944 |
| Teen+ (11+) | 0.4151 | 0.9167 | 0.5714 |

## Model Comparison

- **Accuracy Improvement:** -0.1759 (-17.59%)
- **F1 Score Improvement:** -0.1828 (-18.28%)
- **Better Performing Model:** Baseline

## Qualitative Analysis

### Prediction Categories

**Both Correct:** 18 samples

**Both Wrong:** 27 samples

**Baseline Correct, Primary Wrong:** 41 samples

**Primary Correct, Baseline Wrong:** 22 samples

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
