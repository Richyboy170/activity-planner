"""
Simple Evaluation Script for New Data
Evaluates the trained model on evaluation_dataset.csv
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Neural Network Model (matches train_model.py architecture)
class NeuralClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dims=[512, 512, 384, 384, 256, 256, 128, 128, 64, 64], num_classes=4, dropout=0.3):
        super(NeuralClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Age group mapping
age_groups = {
    0: 'Toddler (0-3)',
    1: 'Preschool (4-6)',
    2: 'Elementary (7-10)',
    3: 'Teen+ (11+)'
}

print("="*80)
print("EVALUATION ON NEW DATA")
print("="*80)

# Load the evaluation dataset
print("\n[1/5] Loading evaluation dataset...")
df = pd.read_csv('dataset/evaluation_dataset.csv')
print(f"✓ Loaded {len(df)} samples")

# Create activity text representations
print("\n[2/5] Creating activity representations...")
activity_texts = []
for _, row in df.iterrows():
    text_parts = []
    if pd.notna(row.get('title')):
        text_parts.extend([str(row['title'])] * 3)
    if pd.notna(row.get('tags')):
        text_parts.extend([str(row['tags'])] * 2)
    if pd.notna(row.get('description')):
        text_parts.append(str(row['description']))
    for field in ['cost', 'indoor_outdoor', 'season', 'players']:
        if field in df.columns and pd.notna(row.get(field)):
            text_parts.append(f"{field}: {row[field]}")
    activity_texts.append(' '.join(text_parts))

# Derive labels from age ranges
labels = []
for _, row in df.iterrows():
    age_min = row['age_min']
    if age_min <= 3:
        labels.append(0)
    elif age_min <= 6:
        labels.append(1)
    elif age_min <= 10:
        labels.append(2)
    else:
        labels.append(3)
labels = np.array(labels)

# Class distribution
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:")
for label, count in zip(unique, counts):
    print(f"  {age_groups[label]}: {count} samples ({count/len(labels)*100:.1f}%)")

# Load the model checkpoint first to determine input dimension
print("\n[3/5] Loading trained model checkpoint...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

checkpoint = torch.load('models/neural_classifier.pth', map_location=device)

# Handle different checkpoint formats
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Detect input dimension from checkpoint
first_layer_key = 'model.0.weight'
if first_layer_key in state_dict:
    input_dim = state_dict[first_layer_key].shape[1]
    print(f"✓ Detected input dimension from checkpoint: {input_dim}")
else:
    print("⚠ Could not detect input dimension, using default 384")
    input_dim = 384

# Choose appropriate embedding model based on dimension
if input_dim == 768:
    embedding_model_name = 'all-mpnet-base-v2'
elif input_dim == 384:
    embedding_model_name = 'all-MiniLM-L6-v2'
else:
    print(f"⚠ Unusual input dimension {input_dim}, using all-MiniLM-L6-v2")
    embedding_model_name = 'all-MiniLM-L6-v2'

print(f"Using embedding model: {embedding_model_name}")

# Generate embeddings
print("\n[4/5] Generating embeddings...")
embedding_model = SentenceTransformer(embedding_model_name)
embeddings = embedding_model.encode(activity_texts, show_progress_bar=True)
print(f"✓ Generated embeddings: {embeddings.shape}")

# Create model with correct input dimension
print(f"\nCreating model with input_dim={input_dim}...")
model = NeuralClassifier(input_dim=input_dim, hidden_dims=[512, 512, 384, 384, 256, 256, 128, 128, 64, 64], num_classes=4, dropout=0.3)

# Load state dict
model.load_state_dict(state_dict)
model.to(device)
model.eval()
print("✓ Model loaded successfully")

# Evaluate
print("\n[5/5] Running evaluation...")
X = torch.FloatTensor(embeddings).to(device)

with torch.no_grad():
    outputs = model(X)
    probabilities = torch.softmax(outputs, dim=1)
    predictions = outputs.argmax(dim=1).cpu().numpy()
    confidences = probabilities.max(dim=1)[0].cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(labels, predictions)
precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)

print("\n" + "="*80)
print("EVALUATION RESULTS")
print("="*80)
print(f"\nOverall Metrics:")
print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\nPrediction Confidence:")
print(f"  Mean: {np.mean(confidences):.4f}")
print(f"  Min:  {np.min(confidences):.4f}")
print(f"  Max:  {np.max(confidences):.4f}")

# Per-class metrics
print(f"\nPer-Class Performance:")
all_labels = list(range(4))
precision_per_class, recall_per_class, f1_per_class, support_per_class = \
    precision_recall_fscore_support(labels, predictions, labels=all_labels, average=None, zero_division=0)

for i in range(4):
    print(f"  {age_groups[i]}:")
    print(f"    Precision: {precision_per_class[i]:.4f}")
    print(f"    Recall:    {recall_per_class[i]:.4f}")
    print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    print(f"    Support:   {support_per_class[i]}")

# Confusion Matrix
print(f"\nConfusion Matrix:")
conf_matrix = confusion_matrix(labels, predictions, labels=all_labels)
print("         | " + " | ".join([f"{age_groups[i][:10]:^10}" for i in range(4)]))
print("-" * 60)
for i in range(4):
    row = f"{age_groups[i][:10]:^10} | " + " | ".join([f"{conf_matrix[i][j]:^10}" for j in range(4)])
    print(row)

# Load baseline and compare
try:
    with open('test_results/test_report.json', 'r') as f:
        baseline = json.load(f).get('neural_network', {})
        baseline_acc = baseline.get('accuracy', 0)
        print(f"\nComparison with Baseline:")
        print(f"  Baseline Test Accuracy: {baseline_acc:.4f}")
        print(f"  New Data Accuracy:      {accuracy:.4f}")
        print(f"  Difference:             {accuracy - baseline_acc:+.4f}")
except:
    print("\nBaseline comparison not available")

# Save results
results = {
    'overall_metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'num_samples': len(labels)
    },
    'per_class_metrics': {
        age_groups[i]: {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
        for i in range(4)
    },
    'confusion_matrix': conf_matrix.tolist(),
    'confidence_stats': {
        'mean': float(np.mean(confidences)),
        'std': float(np.std(confidences)),
        'min': float(np.min(confidences)),
        'max': float(np.max(confidences))
    }
}

Path('evaluation_results').mkdir(exist_ok=True)
with open('evaluation_results/results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create markdown summary
md_content = f"""# Evaluation Results

## Dataset Information
- **Total Samples**: {len(labels)}
- **Embedding Model**: {embedding_model_name}
- **Input Dimension**: {input_dim}

## Class Distribution
| Age Group | Count | Percentage |
|-----------|-------|------------|
"""

for label, count in zip(unique, counts):
    md_content += f"| {age_groups[label]} | {count} | {count/len(labels)*100:.1f}% |\n"

md_content += f"""
## Overall Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | {accuracy:.4f} ({accuracy*100:.2f}%) |
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
| **F1-Score** | {f1:.4f} |

## Per-Class Performance
| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
"""

for i in range(4):
    md_content += f"| {age_groups[i]} | {precision_per_class[i]:.4f} | {recall_per_class[i]:.4f} | {f1_per_class[i]:.4f} | {support_per_class[i]} |\n"

md_content += f"""
## Confusion Matrix
|  | {age_groups[0]} | {age_groups[1]} | {age_groups[2]} | {age_groups[3]} |
|--|{"--|".join(['-'*len(age_groups[i]) for i in range(4)])}--|
"""

for i in range(4):
    row_values = " | ".join([str(conf_matrix[i][j]) for j in range(4)])
    md_content += f"| **{age_groups[i]}** | {row_values} |\n"

md_content += f"""
## Prediction Confidence
| Statistic | Value |
|-----------|-------|
| Mean | {np.mean(confidences):.4f} |
| Std Dev | {np.std(confidences):.4f} |
| Min | {np.min(confidences):.4f} |
| Max | {np.max(confidences):.4f} |
"""

# Add baseline comparison if available
try:
    with open('test_results/test_report.json', 'r') as f:
        baseline = json.load(f).get('neural_network', {})
        baseline_acc = baseline.get('accuracy', 0)
        diff = accuracy - baseline_acc
        md_content += f"""
## Baseline Comparison
| Metric | Value |
|--------|-------|
| Baseline Test Accuracy | {baseline_acc:.4f} |
| New Data Accuracy | {accuracy:.4f} |
| Difference | {diff:+.4f} |
"""
except:
    md_content += "\n## Baseline Comparison\nBaseline comparison not available.\n"

with open('evaluation_results/summary.md', 'w') as f:
    f.write(md_content)

print("\n" + "="*80)
print(f"✓ Results saved to evaluation_results/results.json")
print(f"✓ Summary saved to evaluation_results/summary.md")
print("="*80)
