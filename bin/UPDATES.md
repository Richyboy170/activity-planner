# Updates Log

## November 22, 2025

### Balanced Epoch Distribution with Dataset Shuffling
- **Implemented balanced age group distribution across all epochs**
  - Dataset rows are now shuffled during loading for data augmentation
  - Stratified train/validation/test split ensures consistent distribution
  - Each epoch sees the same proportion of activities from all age groups:
    - Toddler (0-3)
    - Preschool (4-6)
    - Elementary (7-10)
    - Teen+ (11+)
  - Added detailed logging of age group distribution in each dataset split
  - Reason: Improve model generalization and prevent overfitting to specific age groups
  - Files updated:
    - `train_model.py` - Enhanced data loading and splitting logic

### Impact
- Better training stability with balanced representation of all age groups
- Improved model performance across different age categories
- More robust generalization to new data
- Detailed distribution metrics help monitor data balance

---

### Model Architecture Enhancement
- **Increased neural network layers from 3 to 5 layers**
  - Previous architecture: `[256, 128, 64]` (3 hidden layers)
  - New architecture: `[256, 128, 64, 32, 16]` (5 hidden layers)
  - Reason: Improve model performance and learning capacity
  - Files updated:
    - `train_model.py` - Main training script
    - `evaluate_new_data.py` - Evaluation script
    - `run_evaluation.py` - Simple evaluation script

### Impact
- Enhanced model depth should improve feature learning capabilities
- May require retraining the model to see performance improvements
- Architecture now has more gradual dimension reduction for better feature transformation
