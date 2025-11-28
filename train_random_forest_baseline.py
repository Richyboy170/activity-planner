"""
Train and Save Random Forest Baseline Model

This script trains a Random Forest baseline model and saves it for use in evaluations.
This ensures that the evaluation scripts can compare the neural network against
the Random Forest baseline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path

# Import from train_model.py
from train_model import (
    ActivityDataProcessor,
    DenseEmbedder,
    TrainingConfig
)


def train_and_save_random_forest_baseline():
    """Train Random Forest baseline and save for future evaluations."""
    print("="*80)
    print("TRAINING RANDOM FOREST BASELINE MODEL")
    print("="*80)

    # Create config
    config = TrainingConfig(dataset_path='dataset/dataset.csv')

    # Load dataset
    print("\nLoading dataset...")
    processor = ActivityDataProcessor(config)
    df = processor.load_dataset()

    # Generate embeddings
    print("\nGenerating embeddings...")
    embedder = DenseEmbedder(config)
    embedder.load_model()
    texts = processor.create_text_representations()
    embeddings = embedder.generate_embeddings(texts)

    # Create age group labels (matching train_model-unaugmented.py)
    print("\nCreating labels...")
    def get_age_group(row) -> int:
        age_min = row['age_min']
        age_max = row['age_max']
        age_mid = (age_min + age_max) / 2

        if age_mid <= 3.5:
            return 0  # Toddler
        elif age_mid <= 7:
            return 1  # Preschool
        elif age_mid <= 11:
            return 2  # Elementary
        else:
            return 3  # Teen+

    labels = df.apply(get_age_group, axis=1).values

    # Split data (80/10/10 - matching test_models.py)
    print("\nSplitting data...")
    n = len(embeddings)
    train_size = int(0.8 * n)
    val_size = int(0.1 * n)

    indices = np.random.RandomState(42).permutation(n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    X_train = embeddings[train_idx]
    y_train = labels[train_idx]
    X_val = embeddings[val_idx]
    y_val = labels[val_idx]
    X_test = embeddings[test_idx]
    y_test = labels[test_idx]

    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")

    # Train Random Forest
    print("\n" + "="*80)
    print("Training Random Forest (this may take a minute)...")
    print("="*80)

    rf_classifier = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    rf_classifier.fit(X_train, y_train)

    # Evaluate on test set
    print("\n" + "="*80)
    print("Evaluating Random Forest...")
    print("="*80)

    y_pred = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("\nClassification Report:")
    age_groups = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
    print(classification_report(y_test, y_pred, target_names=age_groups))

    # Save the model
    output_dir = Path('models')
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / 'random_forest_baseline.pkl'
    joblib.dump(rf_classifier, model_path)
    print(f"\n✓ Random Forest model saved to: {model_path}")

    # Save training data for future use
    train_embeddings_path = output_dir / 'train_embeddings.npy'
    train_labels_path = output_dir / 'train_labels.npy'

    np.save(train_embeddings_path, X_train)
    np.save(train_labels_path, y_train)
    print(f"✓ Training embeddings saved to: {train_embeddings_path}")
    print(f"✓ Training labels saved to: {train_labels_path}")

    print("\n" + "="*80)
    print("RANDOM FOREST BASELINE TRAINING COMPLETE!")
    print("="*80)
    print("\nThe Random Forest baseline is now available for:")
    print("  - evaluate_new_data.py (new data evaluation)")
    print("  - test_models.py (model testing)")
    print("\nNext steps:")
    print("  1. Run: python evaluate_new_data.py --new-data dataset/evaluation_dataset.csv")
    print("  2. Check the report for Random Forest baseline comparison")


if __name__ == '__main__':
    train_and_save_random_forest_baseline()
