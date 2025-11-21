# Model Testing Suite - README

## Overview

This directory contains a comprehensive testing suite for evaluating the Activity Planner's classification models according to academic model testing guidelines.

## Files Created

### 1. `test_models.py`
**Purpose**: Main testing script that evaluates both baseline and primary models

**Features**:
- Trains and tests Random Forest baseline model
- Tests neural network primary model
- Generates quantitative metrics (accuracy, precision, recall, F1)
- Creates qualitative analysis (learning curves, confidence distributions)
- Produces professional visualizations
- Generates comprehensive reports

**Usage**:
```bash
# Basic usage
python test_models.py

# With custom dataset
python test_models.py --data path/to/dataset.csv

# With custom output directory
python test_models.py --output my_results
```

### 2. `MODEL_TESTING_GUIDELINES.md`
**Purpose**: Comprehensive documentation of testing methodology

**Contents**:
- Model architecture diagrams
- Rationale for model choices
- Evaluation metrics explanation
- Visual element specifications
- Challenge documentation
- Academic standards compliance checklist

### 3. `setup_test_environment.sh`
**Purpose**: Setup script to verify and install dependencies

**Usage**:
```bash
chmod +x setup_test_environment.sh
./setup_test_environment.sh
```

## Testing Methodology

### Models Tested

#### Baseline Model: Random Forest
- **Type**: Ensemble tree-based classifier
- **Parameters**: 100 estimators, max depth 20
- **Rationale**: Simple, effective baseline requiring minimal tuning
- **Training Time**: < 10 seconds

#### Primary Model: Neural Network (ActivityClassifier)
- **Architecture**: 384 → 256 → 128 → 64 → 4
- **Parameters**: ~170,000 trainable
- **Regularization**: BatchNorm + Dropout(0.3)
- **Training Time**: 2-5 minutes

### Classification Task

Both models classify activities into age-appropriate groups:
1. **Toddler (0-3 years)**
2. **Preschool (4-6 years)**
3. **Elementary (7-10 years)**
4. **Teen+ (11+ years)**

### Data Split

- **Training**: 80% of data
- **Validation**: 10% of data (for neural network tuning)
- **Testing**: 10% of data (final evaluation)
- **Random Seed**: 42 (for reproducibility)

## Output Structure

After running `test_models.py`, the following files are generated:

```
test_results/
├── test_report.json              # Complete numerical results
├── TEST_REPORT.md                # Human-readable markdown report
└── figures/
    ├── confusion_matrices.png    # Side-by-side confusion matrices
    ├── learning_curves.png       # Training/validation curves
    ├── per_class_performance.png # Per-class metrics comparison
    ├── model_comparison.png      # Overall metrics comparison
    └── confidence_distribution.png # Prediction confidence histograms
```

## Evaluation Metrics

### Quantitative Metrics

1. **Accuracy**: Overall classification correctness
2. **Precision**: Per-class and weighted average
3. **Recall**: Per-class and weighted average
4. **F1 Score**: Harmonic mean of precision and recall
5. **Confusion Matrix**: Detailed error analysis
6. **Loss Curves** (Neural Network): Training and validation loss

### Qualitative Analysis

1. **Learning Curves**: How model performance improves over epochs
2. **Confidence Distribution**: Model certainty analysis
3. **Error Categories**:
   - Both models correct
   - Both models wrong
   - Baseline correct, primary wrong
   - Primary correct, baseline wrong
4. **Per-Class Performance**: Which age groups are easier to classify
5. **Sample Predictions**: Detailed examination of interesting cases

## Visualizations

All visualizations are:
- **High resolution**: 300 DPI for publication quality
- **Professional styling**: Clean, readable plots
- **Color coded**: Blue for baseline, green for primary
- **Well-labeled**: Clear titles, axes, and legends
- **Comparison-focused**: Side-by-side layouts

### Figure Descriptions

#### 1. Confusion Matrices (`confusion_matrices.png`)
Shows true vs predicted labels for both models side-by-side. Diagonal values indicate correct predictions, off-diagonal show misclassifications.

#### 2. Learning Curves (`learning_curves.png`)
Two plots:
- **Loss curves**: Training and validation loss over epochs
- **Accuracy curve**: Validation accuracy over epochs

Helps identify overfitting (diverging curves) or underfitting (both curves plateau at poor performance).

#### 3. Per-Class Performance (`per_class_performance.png`)
Three bar charts comparing baseline vs primary:
- **Precision by age group**
- **Recall by age group**
- **F1 Score by age group**

Shows which model performs better on specific age groups.

#### 4. Model Comparison (`model_comparison.png`)
Overall metric comparison with value labels. Clearly shows which model wins on each metric.

#### 5. Confidence Distribution (`confidence_distribution.png`)
Histograms showing the distribution of prediction confidence (max probability) for both models. Higher confidence indicates more certain predictions.

## Reports

### JSON Report (`test_report.json`)

Complete structured data including:
- All numerical metrics
- Confusion matrices
- Training history
- Sample predictions with probabilities
- Model architecture details
- Comparison statistics

### Markdown Report (`TEST_REPORT.md`)

Human-readable report with:
- Model architecture descriptions
- Rationale for design choices
- Quantitative results tables
- Per-class performance breakdown
- Model comparison analysis
- Qualitative observations
- Challenges encountered
- References to visualizations

## Academic Standards Compliance

### Baseline Model Requirements ✅

- [x] Reasonable choice for the problem (Random Forest for embeddings)
- [x] Easy to implement and minimal tuning (default parameters work well)
- [x] Professional, concise visual elements (high-quality figures)
- [x] Quantitative results provided (accuracy, precision, recall, F1)
- [x] Qualitative results provided (confidence analysis, error categories)
- [x] Challenges documented (high dimensionality, non-linearity limits)

### Primary Model Requirements ✅

- [x] Architecture makes sense for problem (progressive compression of embeddings)
- [x] Neural network implementation (PyTorch with proper layers)
- [x] Professional, concise visual elements (learning curves, confusion matrix)
- [x] Detailed architecture description (layer-by-layer breakdown)
- [x] Model complexity specified (170K parameters, 4 layers)
- [x] Reproducible (fixed seed, saved checkpoints, documented hyperparameters)
- [x] Quantitative results (accuracy, loss curves, all metrics)
- [x] Qualitative results (learning curves, sample analysis)
- [x] Challenges documented (overfitting risk, tuning requirements, training time)

## Prerequisites

### Required Python Packages

```
torch>=2.1.0
scikit-learn>=1.3.2
sentence-transformers>=3.0.0
matplotlib
seaborn
pandas>=2.1.0
numpy>=1.26.0
rank-bm25
faiss-cpu
tqdm
```

### Installation

```bash
# Option 1: Install from requirements.txt
pip install -r requirements.txt

# Option 2: Run setup script
./setup_test_environment.sh

# Option 3: Install manually
pip install torch scikit-learn sentence-transformers matplotlib seaborn
```

## Running Tests

### Full Test Suite

```bash
python test_models.py
```

This will:
1. Load dataset and generate embeddings (~30 seconds)
2. Train and test baseline model (~10 seconds)
3. Load or train neural network (~2-5 minutes if training)
4. Generate all visualizations (~10 seconds)
5. Create comprehensive reports (~5 seconds)

**Total time**: 3-7 minutes depending on whether neural network needs training

### Expected Console Output

```
==========================================================
ACTIVITY CLASSIFIER MODEL TESTING SUITE
==========================================================

Loading and preparing data...
✓ Data loaded: Train=XXX, Val=XXX, Test=XXX

==========================================================
TESTING BASELINE MODEL (Random Forest)
==========================================================
Training Random Forest baseline...
✓ Baseline training complete

✓ Baseline Model Results:
  Accuracy:  0.XXXX
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1 Score:  0.XXXX

==========================================================
TESTING PRIMARY MODEL (Neural Network)
==========================================================
✓ Loading model from models/neural_classifier.pth

✓ Primary Model Results:
  Architecture: 384 → 256 → 128 → 64 → 4
  Total Parameters: XXX,XXX
  Accuracy:  0.XXXX
  Precision: 0.XXXX
  Recall:    0.XXXX
  F1 Score:  0.XXXX

==========================================================
GENERATING VISUALIZATIONS
==========================================================
  ✓ Confusion matrices saved
  ✓ Learning curves saved
  ✓ Per-class performance plots saved
  ✓ Model comparison plot saved
  ✓ Confidence distribution plots saved

==========================================================
QUALITATIVE ANALYSIS: Sample Predictions
==========================================================
Prediction Category Breakdown:
  Both Correct: XX samples
  Both Wrong: XX samples
  Baseline Correct, Primary Wrong: XX samples
  Primary Correct, Baseline Wrong: XX samples

==========================================================
GENERATING COMPREHENSIVE REPORT
==========================================================
✓ JSON report saved to test_results/test_report.json
✓ Markdown report saved to test_results/TEST_REPORT.md

==========================================================
TESTING COMPLETE!
==========================================================

Results saved to: test_results
  - JSON Report: test_report.json
  - Markdown Report: TEST_REPORT.md
  - Visualizations: figures/
```

## Interpreting Results

### Accuracy Comparison

- **Improvement > 5%**: Primary model significantly better
- **Improvement 1-5%**: Primary model marginally better
- **Improvement < 1%**: Models perform similarly
- **Improvement < 0%**: Baseline might be sufficient

### Confusion Matrix Analysis

- **Diagonal values**: Correct predictions (higher is better)
- **Off-diagonal values**: Misclassifications
- **Row patterns**: Which true classes get confused
- **Column patterns**: Which predicted classes are over-predicted

### Learning Curves

- **Converging curves**: Good learning, no overfitting
- **Diverging curves**: Overfitting (train continues improving, val plateaus/worsens)
- **Both plateau high**: Good generalization
- **Both plateau low**: Underfitting (need more capacity or features)

### Confidence Distribution

- **High confidence peak (>0.8)**: Model is certain about predictions
- **Spread out**: Model is uncertain
- **Bimodal**: Model separates easy and hard examples

## Troubleshooting

### Issue: Module Not Found Errors

**Solution**: Install missing packages
```bash
pip install [package-name]
```

### Issue: CUDA Out of Memory

**Solution**: The code automatically falls back to CPU. If issues persist, reduce batch size in train_model.py

### Issue: Model Not Found

**Solution**: The script will automatically train a new model if `models/neural_classifier.pth` doesn't exist

### Issue: Poor Performance

**Possible causes**:
1. Small dataset size
2. Class imbalance
3. Poor embedding quality
4. Need hyperparameter tuning

## Extending the Tests

### Adding More Metrics

Edit `test_models.py` and add metrics in the `test_baseline_model()` or `test_primary_model()` functions:

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
results["matthews_correlation"] = float(mcc)
```

### Adding More Visualizations

Add new plotting functions in the `ModelTester` class:

```python
def _plot_new_visualization(self, fig_dir: Path):
    # Your plotting code here
    plt.savefig(fig_dir / 'new_plot.png', dpi=300)
    plt.close()
```

Then call it in `generate_visualizations()`.

### Testing Different Baseline Models

Modify the `BaselineModel` class to use different algorithms:

```python
from sklearn.svm import SVC
self.model = SVC(kernel='rbf', C=1.0)
```

## Integration with Main Project

These test results can inform:

1. **Production Deployment**: Choose which model to use in `app.py`
2. **Confidence Thresholds**: Set minimum confidence for predictions
3. **Error Handling**: Know which cases need human review
4. **Feature Engineering**: Identify areas for improvement
5. **Data Collection**: Understand which age groups need more examples

## Citation

If using this testing methodology in academic work, please cite the relevant papers:

- **Sentence-BERT**: Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.
- **Random Forests**: Breiman, L. (2001). Random forests. Machine learning.
- **Evaluation Metrics**: Scikit-learn library (Pedregosa et al., 2011)

## Support

For questions or issues:
1. Check `MODEL_TESTING_GUIDELINES.md` for detailed methodology
2. Review code comments in `test_models.py`
3. Examine existing test results in `test_results/`

---

**Created**: 2025-11-07
**Last Updated**: 2025-11-07
**Version**: 1.0
**Status**: Ready for use
