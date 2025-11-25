"""
Model Testing and Evaluation Suite
===================================
This script tests and compares the baseline and primary neural network models
according to the model testing guidelines.

Baseline Model: Random Forest Classifier
Primary Model: Multi-layer Neural Network (ActivityClassifier)

Both models classify activities into age groups:
- Toddler (0-3 years)
- Preschool (4-6 years)
- Elementary (7-10 years)
- Teen+ (11+ years)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.model_selection import learning_curve
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import sys
from datetime import datetime

# Import from train_model.py
from train_model import (
    ActivityClassifier,
    ActivityDataProcessor,
    DenseEmbedder,
    ActivityDataset,
    TrainingConfig
)


class BaselineModel:
    """Random Forest baseline classifier for activity age classification."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 20, random_state: int = 42):
        """
        Initialize Random Forest baseline model.

        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            random_state: Random seed for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the Random Forest model."""
        print("Training Random Forest baseline...")
        self.model.fit(X_train, y_train)
        print("✓ Baseline training complete")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        return self.model.predict_proba(X)

    def get_model_info(self) -> Dict[str, Any]:
        """Get model architecture information."""
        return {
            "model_type": "Random Forest",
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "total_parameters": "N/A (tree-based)",
            "tuning_required": "Minimal",
            "complexity": "Low to Medium"
        }


class ModelTester:
    """Comprehensive testing suite for baseline and primary models."""

    def __init__(self, data_path: str, output_dir: str = "test_results"):
        """
        Initialize the model tester.

        Args:
            data_path: Path to the dataset CSV
            output_dir: Directory to save test results
        """
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Class labels
        self.age_groups = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']

        # Results storage
        self.baseline_results = {}
        self.primary_results = {}

    def load_and_prepare_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray, np.ndarray]:
        """Load dataset and prepare train/val/test splits with embeddings."""
        print("Loading and preparing data...")

        # Create config with the dataset path
        config = TrainingConfig(dataset_path=self.data_path)

        # Load dataset
        processor = ActivityDataProcessor(config)
        df = processor.load_dataset()

        # Generate embeddings
        embedder = DenseEmbedder(config)
        embedder.load_model()
        texts = processor.create_text_representations()
        embeddings = embedder.generate_embeddings(texts)

        # Create age group labels (matching train_model-unaugmented.py)
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

        # Split data (80/10/10)
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

        print(f"✓ Data loaded: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return X_train, y_train, X_val, y_val, X_test, y_test

    def test_baseline_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Test the baseline Random Forest model.

        Returns:
            Dictionary containing all test results
        """
        print("\n" + "="*60)
        print("TESTING BASELINE MODEL (Random Forest)")
        print("="*60)

        # Train model
        baseline = BaselineModel(n_estimators=100, max_depth=20)
        baseline.train(X_train, y_train)

        # Make predictions
        y_pred = baseline.predict(X_test)
        y_proba = baseline.predict_proba(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # Confusion matrix (ensure all 4 classes are included)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(4))

        # Per-class metrics (ensure all 4 classes are included)
        per_class_metrics = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=np.arange(4)
        )

        results = {
            "model_info": baseline.get_model_info(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "per_class_precision": per_class_metrics[0].tolist(),
            "per_class_recall": per_class_metrics[1].tolist(),
            "per_class_f1": per_class_metrics[2].tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist()
        }

        # Print results
        print(f"\n✓ Baseline Model Results:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        self.baseline_results = results
        return results

    def test_primary_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          model_path: str = "models/neural_classifier.pth") -> Dict[str, Any]:
        """
        Test the primary neural network model.

        Returns:
            Dictionary containing all test results
        """
        print("\n" + "="*60)
        print("TESTING PRIMARY MODEL (Neural Network)")
        print("="*60)

        # Model architecture (must match training)
        hidden_dims = [256, 128]

        # Check if model exists
        model_path = Path(model_path)
        if not model_path.exists():
            print(f"⚠ Model not found at {model_path}")
            print("Training new model...")
            model, history = self._train_neural_network(X_train, y_train, X_val, y_val)
        else:
            print(f"✓ Loading model from {model_path}")

            # Load checkpoint first to inspect architecture
            checkpoint = torch.load(model_path, map_location='cpu')

            # Extract state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Detect number of classes from the output layer
            # The output layer is the last linear layer (model.9.weight for hidden_dims=[256,128])
            output_layer_key = None
            for key in state_dict.keys():
                if 'weight' in key and len(state_dict[key].shape) == 2:
                    output_layer_key = key

            if output_layer_key:
                num_classes_in_checkpoint = state_dict[output_layer_key].shape[0]
                expected_num_classes = 4

                if num_classes_in_checkpoint != expected_num_classes:
                    print(f"⚠ WARNING: Checkpoint has {num_classes_in_checkpoint} output classes, but current code expects {expected_num_classes}")
                    print(f"⚠ Using checkpoint architecture ({num_classes_in_checkpoint} classes). Please retrain the model for {expected_num_classes} classes.")

                # Create model with the architecture from the checkpoint
                model = ActivityClassifier(input_dim=384, hidden_dims=hidden_dims, num_classes=num_classes_in_checkpoint)
            else:
                # Fallback: use expected architecture
                print("⚠ Could not detect architecture from checkpoint, using default (4 classes)")
                model = ActivityClassifier(input_dim=384, hidden_dims=hidden_dims, num_classes=4)

            # Load the state dict
            model.load_state_dict(state_dict)
            model.eval()

            # Try to load history
            history_path = model_path.parent / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            else:
                history = None

        # Make predictions
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        X_test_tensor = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            outputs = model(X_test_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

        y_pred = predictions.cpu().numpy()
        y_proba = probabilities.cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # Confusion matrix (ensure all 4 classes are included)
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(4))

        # Per-class metrics (ensure all 4 classes are included)
        per_class_metrics = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0, labels=np.arange(4)
        )

        # Get model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Get the actual number of output classes from the model
        num_output_classes = list(model.parameters())[-1].shape[0]
        architecture_str = f"384 → 256 → 128 → {num_output_classes}"

        results = {
            "model_info": {
                "model_type": "Multi-layer Neural Network",
                "architecture": architecture_str,
                "layers": [
                    "Input: 384 (Sentence-BERT embeddings)",
                    "Hidden 1: Linear(384, 256) + BatchNorm + ReLU + Dropout(0.5)",
                    "Hidden 2: Linear(256, 128) + BatchNorm + ReLU + Dropout(0.5)",
                    f"Output: Linear(128, {num_output_classes})"
                ],
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "optimizer": "Adam (lr=0.001)",
                "loss_function": "CrossEntropyLoss",
                "regularization": ["BatchNorm", "Dropout(0.5)"]
            },
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "per_class_precision": per_class_metrics[0].tolist(),
            "per_class_recall": per_class_metrics[1].tolist(),
            "per_class_f1": per_class_metrics[2].tolist(),
            "predictions": y_pred.tolist(),
            "probabilities": y_proba.tolist(),
            "training_history": history
        }

        # Print results
        print(f"\n✓ Primary Model Results:")
        print(f"  Architecture: {architecture_str}")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        self.primary_results = results
        return results

    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             epochs: int = 50, batch_size: int = 32) -> Tuple[nn.Module, Dict]:
        """Train the neural network model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        # Model architecture (must match training)
        hidden_dims = [256, 128]

        # Create model
        model = ActivityClassifier(input_dim=384, hidden_dims=hidden_dims, num_classes=4).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Prepare data
        train_dataset = ActivityDataset(X_train, y_train)
        val_dataset = ActivityDataset(X_val, y_val)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

        # Training history (use plural keys to match train_model.py)
        history = {
            "train_losses": [],
            "val_losses": [],
            "val_accuracies": []
        }

        # Training loop
        print("Training neural network...")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()

            val_loss /= len(val_loader)
            val_accuracy = correct / total

            history["train_losses"].append(train_loss)
            history["val_losses"].append(val_loss)
            history["val_accuracies"].append(val_accuracy)

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val Acc={val_accuracy:.4f}")

        # Save model
        model_path = Path("models/neural_classifier.pth")
        model_path.parent.mkdir(exist_ok=True)

        # Save in the same format as train_model.py
        torch.save({
            'model_state_dict': model.state_dict(),
            'train_losses': history["train_losses"],
            'val_losses': history["val_losses"],
        }, model_path)

        history_path = model_path.parent / "training_history.json"
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        print(f"✓ Model saved to {model_path}")

        return model, history

    def generate_visualizations(self, X_test: np.ndarray, y_test: np.ndarray):
        """Generate all visualization plots."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        # Create figure directory
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # 1. Confusion Matrices
        self._plot_confusion_matrices(fig_dir)

        # 2. Learning Curves (if available)
        if self.primary_results.get("training_history"):
            self._plot_learning_curves(fig_dir)

        # 3. Per-class Performance
        self._plot_per_class_performance(fig_dir)

        # 4. Model Comparison
        self._plot_model_comparison(fig_dir)

        # 5. Prediction Confidence Distribution
        self._plot_confidence_distribution(fig_dir)

        print(f"✓ Visualizations saved to {fig_dir}")

    def _plot_confusion_matrices(self, fig_dir: Path):
        """Plot confusion matrices for both models."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Baseline
        cm_baseline = np.array(self.baseline_results["confusion_matrix"])
        sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.age_groups, yticklabels=self.age_groups,
                   ax=axes[0])
        axes[0].set_title('Baseline Model (Random Forest)\nConfusion Matrix')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')

        # Primary
        cm_primary = np.array(self.primary_results["confusion_matrix"])
        sns.heatmap(cm_primary, annot=True, fmt='d', cmap='Greens',
                   xticklabels=self.age_groups, yticklabels=self.age_groups,
                   ax=axes[1])
        axes[1].set_title('Primary Model (Neural Network)\nConfusion Matrix')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')

        plt.tight_layout()
        plt.savefig(fig_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Confusion matrices saved")

    def _plot_learning_curves(self, fig_dir: Path):
        """Plot learning curves for neural network."""
        history = self.primary_results["training_history"]

        # Check if history has the required keys
        # Handle both singular and plural key names for backward compatibility
        train_loss_key = None
        val_loss_key = None

        if "train_loss" in history:
            train_loss_key = "train_loss"
        elif "train_losses" in history:
            train_loss_key = "train_losses"

        if "val_loss" in history:
            val_loss_key = "val_loss"
        elif "val_losses" in history:
            val_loss_key = "val_losses"

        if not train_loss_key or not val_loss_key:
            print("  ⚠ Training history missing required keys, skipping learning curves")
            return

        # Check if we have accuracy data
        has_accuracy = "val_accuracy" in history or "val_accuracies" in history

        if has_accuracy:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        else:
            fig, axes = plt.subplots(1, 1, figsize=(10, 5))
            axes = [axes]  # Make it a list for consistent indexing

        # Loss curves
        epochs = range(1, len(history[train_loss_key]) + 1)
        axes[0].plot(epochs, history[train_loss_key], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history[val_loss_key], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Accuracy curve (only if available)
        if has_accuracy:
            acc_key = "val_accuracy" if "val_accuracy" in history else "val_accuracies"
            axes[1].plot(epochs, history[acc_key], 'g-', label='Validation Accuracy', linewidth=2)
            axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Accuracy', fontsize=12)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / 'learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Learning curves saved")

    def _plot_per_class_performance(self, fig_dir: Path):
        """Plot per-class performance metrics."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['precision', 'recall', 'f1']
        titles = ['Precision by Age Group', 'Recall by Age Group', 'F1 Score by Age Group']

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            baseline_values = self.baseline_results[f"per_class_{metric}"]
            primary_values = self.primary_results[f"per_class_{metric}"]

            x = np.arange(len(self.age_groups))
            width = 0.35

            axes[idx].bar(x - width/2, baseline_values, width, label='Baseline (RF)',
                         color='steelblue', alpha=0.8)
            axes[idx].bar(x + width/2, primary_values, width, label='Primary (NN)',
                         color='forestgreen', alpha=0.8)

            axes[idx].set_title(title, fontsize=12, fontweight='bold')
            axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
            axes[idx].set_xticks(x)
            axes[idx].set_xticklabels(self.age_groups, rotation=15, ha='right')
            axes[idx].legend(fontsize=9)
            axes[idx].grid(True, axis='y', alpha=0.3)
            axes[idx].set_ylim([0, 1.0])

        plt.tight_layout()
        plt.savefig(fig_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Per-class performance plots saved")

    def _plot_model_comparison(self, fig_dir: Path):
        """Plot overall model comparison."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        baseline_values = [self.baseline_results[m] for m in metrics]
        primary_values = [self.primary_results[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        bars1 = ax.bar(x - width/2, baseline_values, width, label='Baseline (Random Forest)',
                      color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, primary_values, width, label='Primary (Neural Network)',
                      color='forestgreen', alpha=0.8)

        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(fig_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Model comparison plot saved")

    def _plot_confidence_distribution(self, fig_dir: Path):
        """Plot prediction confidence distributions."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Baseline
        baseline_proba = np.array(self.baseline_results["probabilities"])
        baseline_confidence = np.max(baseline_proba, axis=1)
        axes[0].hist(baseline_confidence, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].set_title('Baseline Model\nPrediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Confidence (Max Probability)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].axvline(baseline_confidence.mean(), color='red', linestyle='--',
                       label=f'Mean: {baseline_confidence.mean():.3f}')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Primary
        primary_proba = np.array(self.primary_results["probabilities"])
        primary_confidence = np.max(primary_proba, axis=1)
        axes[1].hist(primary_confidence, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1].set_title('Primary Model\nPrediction Confidence Distribution', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Confidence (Max Probability)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].axvline(primary_confidence.mean(), color='red', linestyle='--',
                       label=f'Mean: {primary_confidence.mean():.3f}')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(fig_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("  ✓ Confidence distribution plots saved")

    def analyze_sample_predictions(self, X_test: np.ndarray, y_test: np.ndarray,
                                   num_samples: int = 5):
        """Analyze predictions on selected samples."""
        print("\n" + "="*60)
        print("QUALITATIVE ANALYSIS: Sample Predictions")
        print("="*60)

        baseline_pred = np.array(self.baseline_results["predictions"])
        primary_pred = np.array(self.primary_results["predictions"])
        baseline_proba = np.array(self.baseline_results["probabilities"])
        primary_proba = np.array(self.primary_results["probabilities"])

        # Find interesting samples
        samples = {
            "Both Correct": [],
            "Both Wrong": [],
            "Baseline Correct, Primary Wrong": [],
            "Primary Correct, Baseline Wrong": []
        }

        for i in range(len(y_test)):
            baseline_correct = (baseline_pred[i] == y_test[i])
            primary_correct = (primary_pred[i] == y_test[i])

            if baseline_correct and primary_correct:
                samples["Both Correct"].append(i)
            elif not baseline_correct and not primary_correct:
                samples["Both Wrong"].append(i)
            elif baseline_correct and not primary_correct:
                samples["Baseline Correct, Primary Wrong"].append(i)
            elif primary_correct and not baseline_correct:
                samples["Primary Correct, Baseline Wrong"].append(i)

        analysis = {}

        for category, indices in samples.items():
            analysis[category] = {
                "count": len(indices),
                "samples": []
            }

            # Select a few samples from each category
            selected = np.random.choice(indices, size=min(num_samples, len(indices)), replace=False)

            for idx in selected:
                analysis[category]["samples"].append({
                    "index": int(idx),
                    "true_label": self.age_groups[y_test[idx]],
                    "baseline_prediction": self.age_groups[baseline_pred[idx]],
                    "baseline_confidence": float(baseline_proba[idx][baseline_pred[idx]]),
                    "primary_prediction": self.age_groups[primary_pred[idx]],
                    "primary_confidence": float(primary_proba[idx][primary_pred[idx]])
                })

        # Print summary
        print("\nPrediction Category Breakdown:")
        for category, data in analysis.items():
            print(f"  {category}: {data['count']} samples")

        return analysis

    def generate_report(self, sample_analysis: Dict):
        """Generate comprehensive testing report."""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)

        report = {
            "metadata": {
                "test_date": datetime.now().isoformat(),
                "dataset": str(self.data_path),
                "age_groups": self.age_groups
            },
            "baseline_model": self.baseline_results,
            "primary_model": self.primary_results,
            "sample_analysis": sample_analysis,
            "comparison": {
                "accuracy_improvement": self.primary_results["accuracy"] - self.baseline_results["accuracy"],
                "f1_improvement": self.primary_results["f1_score"] - self.baseline_results["f1_score"],
                "better_model": "Primary" if self.primary_results["accuracy"] > self.baseline_results["accuracy"] else "Baseline"
            }
        }

        # Save JSON report
        report_path = self.output_dir / "test_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"✓ JSON report saved to {report_path}")

        # Generate markdown report
        self._generate_markdown_report(report)

        return report

    def _generate_markdown_report(self, report: Dict):
        """Generate human-readable markdown report."""
        md_path = self.output_dir / "TEST_REPORT.md"

        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Model Testing Report\n\n")
            f.write(f"**Test Date:** {report['metadata']['test_date']}\n\n")
            f.write("---\n\n")

            # Baseline Model
            f.write("## Baseline Model: Random Forest\n\n")
            f.write("### Architecture\n\n")
            baseline_info = report['baseline_model']['model_info']
            f.write(f"- **Model Type:** {baseline_info['model_type']}\n")
            f.write(f"- **Number of Estimators:** {baseline_info['n_estimators']}\n")
            f.write(f"- **Max Depth:** {baseline_info['max_depth']}\n")
            f.write(f"- **Tuning Required:** {baseline_info['tuning_required']}\n")
            f.write(f"- **Complexity:** {baseline_info['complexity']}\n\n")

            f.write("### Rationale\n\n")
            f.write("Random Forest was chosen as the baseline because:\n")
            f.write("- Simple to implement and requires minimal hyperparameter tuning\n")
            f.write("- Well-suited for tabular/embedding data\n")
            f.write("- Provides interpretable feature importance\n")
            f.write("- Establishes a strong baseline for comparison\n\n")

            f.write("### Quantitative Results\n\n")
            baseline = report['baseline_model']
            f.write(f"- **Accuracy:** {baseline['accuracy']:.4f}\n")
            f.write(f"- **Precision:** {baseline['precision']:.4f}\n")
            f.write(f"- **Recall:** {baseline['recall']:.4f}\n")
            f.write(f"- **F1 Score:** {baseline['f1_score']:.4f}\n\n")

            f.write("### Per-Class Performance\n\n")
            f.write("| Age Group | Precision | Recall | F1 Score |\n")
            f.write("|-----------|-----------|--------|----------|\n")
            for i, group in enumerate(self.age_groups):
                f.write(f"| {group} | {baseline['per_class_precision'][i]:.4f} | "
                       f"{baseline['per_class_recall'][i]:.4f} | "
                       f"{baseline['per_class_f1'][i]:.4f} |\n")
            f.write("\n")

            # Primary Model
            f.write("## Primary Model: Neural Network\n\n")
            f.write("### Architecture\n\n")
            primary_info = report['primary_model']['model_info']
            f.write(f"- **Model Type:** {primary_info['model_type']}\n")
            f.write(f"- **Architecture:** {primary_info['architecture']}\n")
            f.write(f"- **Total Parameters:** {primary_info['total_parameters']:,}\n")
            f.write(f"- **Trainable Parameters:** {primary_info['trainable_parameters']:,}\n\n")

            f.write("**Layer Details:**\n\n")
            for layer in primary_info['layers']:
                f.write(f"- {layer}\n")
            f.write("\n")

            f.write("**Training Configuration:**\n\n")
            f.write(f"- **Optimizer:** {primary_info['optimizer']}\n")
            f.write(f"- **Loss Function:** {primary_info['loss_function']}\n")
            f.write(f"- **Regularization:** {', '.join(primary_info['regularization'])}\n\n")

            f.write("### Rationale\n\n")
            f.write("The multi-layer neural network architecture was chosen because:\n")
            f.write("- Can learn complex non-linear patterns in the embedding space\n")
            f.write("- Progressive dimensionality reduction (384→256→128→7) allows hierarchical feature learning\n")
            f.write("- BatchNorm and Dropout provide regularization to prevent overfitting\n")
            f.write("- Well-suited for high-dimensional embedding inputs\n\n")

            f.write("### Quantitative Results\n\n")
            primary = report['primary_model']
            f.write(f"- **Accuracy:** {primary['accuracy']:.4f}\n")
            f.write(f"- **Precision:** {primary['precision']:.4f}\n")
            f.write(f"- **Recall:** {primary['recall']:.4f}\n")
            f.write(f"- **F1 Score:** {primary['f1_score']:.4f}\n\n")

            f.write("### Per-Class Performance\n\n")
            f.write("| Age Group | Precision | Recall | F1 Score |\n")
            f.write("|-----------|-----------|--------|----------|\n")
            for i, group in enumerate(self.age_groups):
                f.write(f"| {group} | {primary['per_class_precision'][i]:.4f} | "
                       f"{primary['per_class_recall'][i]:.4f} | "
                       f"{primary['per_class_f1'][i]:.4f} |\n")
            f.write("\n")

            # Model Comparison
            f.write("## Model Comparison\n\n")
            comp = report['comparison']
            f.write(f"- **Accuracy Improvement:** {comp['accuracy_improvement']:.4f} "
                   f"({comp['accuracy_improvement']*100:+.2f}%)\n")
            f.write(f"- **F1 Score Improvement:** {comp['f1_improvement']:.4f} "
                   f"({comp['f1_improvement']*100:+.2f}%)\n")
            f.write(f"- **Better Performing Model:** {comp['better_model']}\n\n")

            # Qualitative Analysis
            f.write("## Qualitative Analysis\n\n")
            f.write("### Prediction Categories\n\n")
            for category, data in report['sample_analysis'].items():
                f.write(f"**{category}:** {data['count']} samples\n\n")

            # Visualizations
            f.write("## Visualizations\n\n")
            f.write("All visualizations are saved in the `figures/` directory:\n\n")
            f.write("1. **Confusion Matrices** - `confusion_matrices.png`\n")
            f.write("2. **Learning Curves** - `learning_curves.png`\n")
            f.write("3. **Per-Class Performance** - `per_class_performance.png`\n")
            f.write("4. **Model Comparison** - `model_comparison.png`\n")
            f.write("5. **Confidence Distribution** - `confidence_distribution.png`\n\n")

            # Challenges
            f.write("## Challenges and Observations\n\n")
            f.write("### Baseline Model Challenges\n\n")
            f.write("- Random Forest models can struggle with high-dimensional embedding spaces (384 dimensions)\n")
            f.write("- Limited ability to learn complex non-linear patterns compared to neural networks\n")
            f.write("- May overfit if trees are too deep or underfit if too shallow\n\n")

            f.write("### Primary Model Challenges\n\n")
            f.write("- Requires more training time compared to Random Forest\n")
            f.write("- Needs careful hyperparameter tuning (learning rate, dropout, batch size)\n")
            f.write("- Risk of overfitting on small datasets without proper regularization\n")
            f.write("- Less interpretable than tree-based models\n\n")

            f.write("### Dataset Observations\n\n")
            f.write("- 80/10/10 train/validation/test split ensures robust evaluation\n")
            f.write("- Class imbalance may affect per-class performance\n")
            f.write("- Embedding quality from Sentence-BERT is crucial for both models\n\n")

            f.write("---\n\n")
            f.write("**Full numerical results are available in `test_report.json`**\n")

        print(f"✓ Markdown report saved to {md_path}")

    def run_full_test_suite(self):
        """Run the complete testing pipeline."""
        print("\n" + "="*60)
        print("ACTIVITY CLASSIFIER MODEL TESTING SUITE")
        print("="*60)
        print()

        # Load data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_prepare_data()

        # Test baseline model
        self.test_baseline_model(X_train, y_train, X_test, y_test)

        # Test primary model
        self.test_primary_model(X_train, y_train, X_val, y_val, X_test, y_test)

        # Generate visualizations
        self.generate_visualizations(X_test, y_test)

        # Analyze samples
        sample_analysis = self.analyze_sample_predictions(X_test, y_test)

        # Generate final report
        report = self.generate_report(sample_analysis)

        print("\n" + "="*60)
        print("TESTING COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {self.output_dir}")
        print(f"  - JSON Report: test_report.json")
        print(f"  - Markdown Report: TEST_REPORT.md")
        print(f"  - Visualizations: figures/")
        print()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Test baseline and primary models')
    parser.add_argument('--data', type=str, default='dataset/dataset.csv',
                       help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='test_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # Run test suite
    tester = ModelTester(data_path=args.data, output_dir=args.output)
    tester.run_full_test_suite()


if __name__ == "__main__":
    main()
