# Neural Network Training for Activity Planner

## Overview
This implementation adds a neural network classifier that can be trained on activity embeddings with proper train/validation/test splits and batch-wise loss tracking.

## New Features

### 1. Neural Network Training (`train_model.py`)
- **Train/Validation/Test Split**: Data is split into 70% train, 15% validation, 15% test
- **Batch Loss Display**: Shows loss for each batch during training with progress bars
- **Multi-layer Classifier**:
  - Input: Sentence-BERT embeddings (384 dimensions)
  - Architecture: 384 → 256 → 128 → 64 → 4 classes
  - Uses ReLU activation, BatchNorm, and Dropout for regularization
- **Age-based Classification**: Activities are classified into 4 age groups:
  - Toddler (0-3 years)
  - Preschool (4-6 years)
  - Elementary (7-10 years)
  - Teen+ (11+ years)

### 2. Activity Filtering (`planner.html`)
- **Filter by Location**: Indoor, Outdoor, Both, or All
- **Filter by Cost**: Free, Low, Medium, or All
- **Filter by Duration**: Short (0-15 min), Medium (15-45 min), Long (45+ min), or All
- **Filter by Age Range**: Toddler, Preschool, Elementary, Teen+, or All Ages
- **Clear All Filters**: Reset all filters to default (All)
- Filters appear after search results are displayed
- Real-time filtering without re-querying the server

## Installation Requirements

### PyTorch Installation
```bash
# For CPU-only (recommended for development)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Other Dependencies (should already be installed)
```bash
pip install scikit-learn tqdm pandas numpy
```

## Usage

### Training the Model
```bash
# Basic training with default parameters
python train_model.py

# Custom parameters
python train_model.py --dataset dataset/dataset.csv --output-dir models --model all-MiniLM-L6-v2

# Training will now include:
# - Step 6/7: Neural network training with batch loss display
# - Final test accuracy and loss metrics
```

### Training Output
```
[Neural Network] Preparing train/validation/test split...
✓ Train set: 70.0% of data
✓ Validation set: 15.0% of data
✓ Test set: 15.0% of data

[Neural Network] Starting training for 10 epochs...
Epoch 1/10: 100%|████████| batch_loss: 1.2345, avg_loss: 1.2500
  Train Loss: 1.2500
  Val Loss: 1.1200
  Val Accuracy: 45.23%

...

[Neural Network] Training complete!
✓ Test Loss: 0.8234
✓ Test Accuracy: 72.50%
```

### Generated Files
After training, the following files will be created in the `models/` directory:
- `neural_classifier.pth` - Trained neural network model
- `training_history.json` - Training and validation loss per epoch
- `embeddings.npy` - Sentence-BERT embeddings
- `faiss_index.bin` - FAISS similarity index
- `bm25_docs.pkl` - BM25 keyword index

## Filter Buttons Usage

### In the Web Interface
1. Search for activities using the search box
2. Filter buttons will appear automatically after results are loaded
3. Click any filter button to narrow down results
4. Multiple filters can be active at once
5. Click "Clear Filters" to reset all filters

### Filter Logic
- **Location**: Matches exact indoor_outdoor field value
- **Cost**: Matches cost level (free/low/medium)
- **Duration**: Filters by duration_mins ranges
- **Age**: Filters activities that overlap with the selected age range

## Notes
- The neural network classifier is optional and runs after embedding generation
- Training typically takes 2-5 minutes depending on dataset size and hardware
- Filter buttons work client-side for instant filtering without server requests
- The model uses cross-entropy loss and Adam optimizer with learning rate 0.001
