# Model Testing Report

**Test Date:** 2025-11-26T16:42:21.927377

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

- **Accuracy:** 0.6772
- **Precision:** 0.6553
- **Recall:** 0.6772
- **F1 Score:** 0.6017

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 1.0000 | 0.1818 | 0.3077 |
| Preschool (4-6) | 0.5000 | 0.0417 | 0.0769 |
| Elementary (7-10) | 0.5455 | 0.3000 | 0.3871 |
| Teen+ (11+) | 0.6933 | 0.9912 | 0.8159 |

## Primary Model: Neural Network

### Architecture

- **Model Type:** Multi-layer Neural Network
- **Architecture:** 384 → 128 → 64 → 4
- **Total Parameters:** 58,180
- **Trainable Parameters:** 58,180

**Layer Details:**

- Input: 384 (Sentence-BERT embeddings)
- Hidden 1: Linear(384, 128) + BatchNorm + ReLU + Dropout(0.5)
- Hidden 2: Linear(128, 64) + BatchNorm + ReLU + Dropout(0.5)
- Output: Linear(64, 4)

**Training Configuration:**

- **Optimizer:** Adam (lr=0.001)
- **Loss Function:** CrossEntropyLoss
- **Regularization:** BatchNorm, Dropout(0.5)

### Rationale

The multi-layer neural network architecture was chosen because:
- Can learn complex non-linear patterns in the embedding space
- Progressive dimensionality reduction (384→128→64→4) allows hierarchical feature learning
- Reduced architecture (60% fewer parameters than 384→256→128→4) prevents overfitting on small dataset
- BatchNorm and Dropout provide regularization to prevent overfitting
- Well-suited for high-dimensional embedding inputs

### Quantitative Results

- **Accuracy:** 0.9312
- **Precision:** 0.9365
- **Recall:** 0.9312
- **F1 Score:** 0.9317

### Per-Class Performance

| Age Group | Precision | Recall | F1 Score |
|-----------|-----------|--------|----------|
| Toddler (0-3) | 0.7333 | 1.0000 | 0.8462 |
| Preschool (4-6) | 0.9500 | 0.7917 | 0.8636 |
| Elementary (7-10) | 0.8537 | 0.8750 | 0.8642 |
| Teen+ (11+) | 0.9823 | 0.9737 | 0.9780 |

## Model Comparison

- **Accuracy Improvement:** 0.2540 (+25.40%)
- **F1 Score Improvement:** 0.3300 (+33.00%)
- **Better Performing Model:** Primary

## Qualitative Analysis

### Prediction Categories

**Both Correct:** 124 samples

**Both Wrong:** 9 samples

**Baseline Correct, Primary Wrong:** 4 samples

**Primary Correct, Baseline Wrong:** 52 samples

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
