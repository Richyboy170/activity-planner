# Model Testing Report

**Test Date:** 2025-11-25T16:29:32.885328

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

- **Accuracy:** 0.4444
- **Precision:** 0.4375
- **Recall:** 0.4444
- **F1 Score:** 0.4215

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7778 | 0.3043 | 0.4375 |
| Preschool (4-6) | 0.4694 | 0.4894 | 0.4792 |
| Elementary (7-10) | 0.3590 | 0.5833 | 0.4444 |
| Teen (11-17) | 0.4906 | 0.5000 | 0.4952 |
| Young Adult (18-39) | 0.0000 | 0.0000 | 0.0000 |
| Adult (40-64) | 0.0000 | 0.0000 | 0.0000 |
| Senior (65+) | 0.0000 | 0.0000 | 0.0000 |

## Primary Model: Neural Network

### Architecture

- **Model Type:** Multi-layer Neural Network
- **Architecture:** 384 → 256 → 128 → 7
- **Total Parameters:** 133,127
- **Trainable Parameters:** 133,127

**Layer Details:**

- Input: 384 (Sentence-BERT embeddings)
- Hidden 1: Linear(384, 256) + BatchNorm + ReLU + Dropout(0.5)
- Hidden 2: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.5)
- Output: Linear(128, 7)

**Training Configuration:**

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Regularization:** BatchNorm, Dropout(0.5)

### Rationale

The multi-layer neural network architecture was chosen because:
- Can learn complex non-linear patterns in the embedding space
- Progressive dimensionality reduction (384→256→128→7) allows hierarchical feature learning
- BatchNorm and Dropout provide regularization to prevent overfitting
- Well-suited for high-dimensional embedding inputs

### Quantitative Results

- **Accuracy:** 0.7725
- **Precision:** 0.7785
- **Recall:** 0.7725
- **F1 Score:** 0.7731

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7407 | 0.8696 | 0.8000 |
| Preschool (4-6) | 0.7500 | 0.7021 | 0.7253 |
| Elementary (7-10) | 0.7347 | 0.7500 | 0.7423 |
| Teen (11-17) | 0.8913 | 0.7885 | 0.8367 |
| Young Adult (18-39) | 0.7222 | 0.8667 | 0.7879 |
| Adult (40-64) | 0.6000 | 0.7500 | 0.6667 |
| Senior (65+) | 0.0000 | 0.0000 | 0.0000 |

## Model Comparison

- **Accuracy Improvement:** 0.3280 (+32.80%)
- **F1 Score Improvement:** 0.3515 (+35.15%)
- **Better Performing Model:** Primary

## Qualitative Analysis

### Prediction Categories

**Both Correct:** 75 samples

**Both Wrong:** 34 samples

**Baseline Correct, Primary Wrong:** 9 samples

**Primary Correct, Baseline Wrong:** 71 samples

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
