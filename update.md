# Activity Planner Updates

## Feature Selection for Neural Network Classifier

### Overview
Implemented feature selection in `train_model.py` to train the neural network classifier using only the most relevant features: **title**, **tags**, **age_min**, **age_max**, and **duration_mins**.

### Changes Made

#### 1. Text Feature Selection (Lines 143-167)
- **Modified**: `create_text_representations()` method
- **Change**: Now uses only **title** and **tags** for text features
- **Removed**: All other text fields (how_to_play, indoor_outdoor, season, etc.)
- **Implementation**:
  - Title: Repeated 3 times for higher weight
  - Tags: Repeated 2 times for importance
  - Creates text representations that focus on the most descriptive features

#### 2. Numerical Feature Extraction (Lines 169-203)
- **Added**: New `extract_numerical_features()` method
- **Features Extracted**: age_min, age_max, duration_mins
- **Normalization**: Min-max normalization applied to all numerical features
- **Missing Values**: Handled by defaulting to 0
- **Output**: Returns normalized numpy array of shape (n_samples, 3)

#### 3. Feature Combination (Lines 422-433)
- **Modified**: `prepare_data()` method signature
- **Change**: Now accepts both embeddings and numerical_features parameters
- **Implementation**: Concatenates text embeddings (384-dim) with numerical features (3-dim)
- **Result**: Combined feature vector of 387 dimensions

#### 4. Cross-Validation Update (Lines 695-748)
- **Modified**: `cross_validate()` method signature
- **Change**: Now accepts numerical_features parameter
- **Implementation**: Combines features before K-fold splitting
- **Benefit**: Ensures proper evaluation with all selected features

#### 5. Model Architecture (Lines 504-564)
- **Modified**: `build_model()` logging
- **Input Dimension**: Now 387 (384 embeddings + 3 numerical features)
- **Architecture**: 387 → 256 → 128 → 64 → 4 classes
- **Features Used**: Clearly logged during model building

#### 6. Training Pipeline (Lines 880-987)
- **Added**: Numerical feature extraction step (Lines 912-917)
- **Modified**: All training calls to pass numerical features
- **Updated**: Input dimension calculation for combined features
- **Flow**:
  1. Load dataset
  2. Create text representations (title + tags only)
  3. Extract numerical features (age_min, age_max, duration_mins)
  4. Generate embeddings for text
  5. Combine embeddings with numerical features
  6. Train neural network on combined features

### Technical Details

#### Feature Dimensions
- **Text Embeddings**: 384 dimensions (from all-MiniLM-L6-v2)
- **Numerical Features**: 3 dimensions (age_min, age_max, duration_mins)
- **Combined**: 387 dimensions total

#### Normalization
```python
# Min-max normalization for numerical features
for i in range(numerical_features.shape[1]):
    col = numerical_features[:, i]
    col_min = col.min()
    col_max = col.max()
    if col_max > col_min:
        numerical_features[:, i] = (col - col_min) / (col_max - col_min)
```

#### Feature Selection Rationale
- **title**: Most descriptive field for activity identification
- **tags**: Captures key characteristics and categories
- **age_min/age_max**: Critical for age group classification
- **duration_mins**: Important temporal feature

### Benefits

1. **Focused Learning**: Model concentrates on most relevant features
2. **Reduced Noise**: Eliminates less important fields that may confuse the model
3. **Better Performance**: Cleaner signal for age group classification
4. **Interpretability**: Easier to understand what the model is learning from

### Usage

Run training with feature selection:
```bash
python train_model.py
```

The model will automatically:
1. Extract only title and tags for text features
2. Generate embeddings from these text features
3. Extract and normalize age_min, age_max, duration_mins
4. Combine all features into a 387-dimensional vector
5. Train the neural network classifier

### Commit Details

**Branch**: `claude/feature-selection-neural-network-01EZL3hjr4gqvh1z3iGB7RH3`

**Commit**: `4730ba5` - "Implement feature selection for neural network classifier"

**Files Modified**:
- `train_model.py` (+97 lines, -33 lines)

### Testing

The implementation includes:
- Syntax validation (passed)
- Feature dimension logging
- Combined feature shape verification
- Proper error handling for missing values
