# Random Forest Baseline Architecture

This document describes the architecture and configuration of the Random Forest baseline model implemented in `train_random_forest_baseline.py`.

## Model Overview

The Random Forest model serves as a baseline to evaluate the performance of the Neural Network model. It uses the same input features but applies a traditional machine learning approach (ensemble of decision trees) for classification.

### Input Features

The model uses a combined feature vector of dimension **387**, identical to the Neural Network model:

1.  **Text Embeddings (384 dimensions):**
    *   Generated using the `sentence-transformers/all-MiniLM-L6-v2` model.
    *   Input text combines `title`, `tags`, and `how_to_play` fields.
    *   Embeddings are generated via the `DenseEmbedder` class.

2.  **Numerical Features (3 dimensions):**
    *   `age_min`
    *   `age_max`
    *   `duration_mins`
    *   Extracted via the `ActivityDataProcessor` class.

### Model Configuration

The model is a **Random Forest Classifier** from `scikit-learn` with the following hyperparameters:

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **n_estimators** | 50 | Number of trees in the forest. |
| **max_depth** | 5 | Maximum depth of the tree. |
| **random_state** | 42 | Seed for reproducibility. |
| **n_jobs** | -1 | Uses all available processors for training. |

### Data Processing & Training Pipeline

1.  **Data Loading**: Loads dataset from `dataset/dataset.csv`.
2.  **Feature Engineering**:
    *   Text embeddings are generated.
    *   Numerical features are extracted.
    *   Features are concatenated into a single vector.
3.  **Label Generation**:
    *   Activities are classified into 4 age groups based on the midpoint of `age_min` and `age_max`.
    *   **Classes**:
        *   0: Toddler (0-3)
        *   1: Preschool (4-6)
        *   2: Elementary (7-10)
        *   3: Teen+ (11+)
4.  **Data Splitting**:
    *   **Train**: 80%
    *   **Validation**: 10%
    *   **Test**: 10%
    *   Split is randomized with `random_state=42`.

### Outputs

The script produces the following artifacts in the `models/` directory:

*   `random_forest_baseline.pkl`: The trained Random Forest model.
*   `train_embeddings.npy`: Training data embeddings (for consistency in future evaluations).
*   `train_labels.npy`: Training data labels.

### Usage

The baseline model is used by:
*   `evaluate_new_data.py`: To provide a baseline comparison when evaluating new data.
*   `test_models.py`: To compare against the Neural Network model.
