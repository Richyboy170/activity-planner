# Neural Network Architecture

This document describes the architecture of the Neural Network model used for activity age group classification.

## Model Overview

The model is a Feed-Forward Neural Network (Multi-Layer Perceptron) designed to classify activities into one of four age groups based on their text descriptions and numerical features.

### Input Features

The model takes a combined input vector of dimension **387**, consisting of:
1.  **Text Embeddings (384 dimensions):** Generated using `sentence-transformers/all-MiniLM-L6-v2`. The input text is a combination of the activity's `title` (weighted 3x), `tags` (weighted 2x), and `how_to_play` (weighted 2x).
2.  **Numerical Features (3 dimensions):**
    *   `age_min` (Normalized)
    *   `age_max` (Normalized)
    *   `duration_mins` (Normalized)

### Architecture Details

The network consists of an input layer, three hidden layers, and an output layer.

| Layer Type | Input Dimension | Output Dimension | Activation | Normalization | Dropout | Description |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input Dropout** | 387 | 387 | - | - | 0.1 | Applied to input features for robustness. |
| **Hidden Layer 1** | 387 | 256 | ReLU | BatchNorm1d | 0.2 | First dense layer. |
| **Hidden Layer 2** | 256 | 128 | ReLU | BatchNorm1d | 0.2 | Second dense layer. |
| **Hidden Layer 3** | 128 | 64 | ReLU | BatchNorm1d | 0.2 | Third dense layer. |
| **Output Layer** | 64 | 4 | Softmax* | - | - | Output probabilities for 4 classes. |

*\*Note: The raw output is logits; Softmax is applied during inference/evaluation.*

### Output Classes

The model predicts one of the following 4 classes:
0.  **Toddler (0-3)**
1.  **Preschool (4-6)**
2.  **Elementary (7-10)**
3.  **Teen+ (11+)**

### Training Configuration

*   **Loss Function:** CrossEntropyLoss with class weights (inverse frequency) to handle class imbalance.
*   **Optimizer:** Adam
*   **Learning Rate:** 0.001
*   **Weight Decay (L2 Regularization):** 5e-5
*   **Batch Size:** 32
*   **Data Balancing:** SMOTE is used on the training set to balance class distribution.
