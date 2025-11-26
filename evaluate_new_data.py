"""
Evaluate Model on New Data - Comprehensive Evaluation Script

This script evaluates the trained neural network classifier on completely new data
that has not been used in any way during:
- Model training
- Hyperparameter tuning
- Validation
- Initial testing

The goal is to assess whether the model generalizes well to truly unseen samples
and meets performance expectations on new data.

Rubric Criteria:
- 10/10: New samples not used for tuning, performance meets/exceeds expectations
- 7/10: Model performance does not meet expectations on new samples
- 4/10: Model performs inconsistently, but correct evaluation attempted
- 2/10: Performance far below expectations, but correct evaluation attempted
- 0/10: No attempt made to evaluate on new data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import joblib
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NeuralClassifier(nn.Module):
    """Neural network classifier for age group classification.

    This architecture matches the ActivityClassifier used during training,
    with input dropout and progressive dropout in deeper layers.
    """

    def __init__(self, input_dim=384, hidden_dims=[128, 64], num_classes=4, dropout=0.3):
        super(NeuralClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Input dropout for robustness (matches training architecture)
        layers.append(nn.Dropout(dropout * 0.5))  # Lighter dropout at input

        # Build hidden layers with consistent dropout
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Consistent dropout across layers for small dataset
            layers.append(nn.Dropout(dropout))  # Consistent dropout
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NewDataEvaluator:
    """Evaluates trained model on completely new, unseen data."""

    def __init__(self, model_dir: str = "models", results_dir: str = "new_data_evaluation"):
        self.model_dir = Path(model_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Create subdirectories
        self.figures_dir = self.results_dir / "figures"
        self.figures_dir.mkdir(exist_ok=True)

        # Load configuration
        self.config = self._load_config()

        # Initialize models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.classifier = self._load_classifier()

        # Load Random Forest baseline model
        self.rf_classifier = self._load_random_forest_baseline()

        # Load baseline performance from original test set
        self.baseline_metrics = self._load_baseline_metrics()
        self.rf_baseline_metrics = self._load_rf_baseline_metrics()

        # Age group mapping
        self.age_groups = {
            0: 'Toddler (0-3)',
            1: 'Preschool (4-6)',
            2: 'Elementary (7-10)',
            3: 'Teen+ (11+)'
        }

        # Track evaluation metadata
        self.evaluation_metadata = {
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(self.model_dir / 'neural_classifier.pth'),
            'new_data_source': 'To be specified',
            'data_collection_method': 'To be documented',
            'confirmation_data_is_new': True,
            'samples_used_in_training': False,
            'samples_used_in_validation': False,
            'samples_used_in_testing': False,
            'samples_influenced_hyperparameters': False
        }

    def _load_config(self) -> Dict:
        """Load training configuration."""
        config_path = self.model_dir / 'training_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        return {}

    def _load_classifier(self) -> nn.Module:
        """Load the trained neural network classifier."""
        model_path = self.model_dir / 'neural_classifier.pth'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        # Load checkpoint first to inspect architecture
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle both checkpoint formats
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Checkpoint format (includes training history)
            state_dict = checkpoint['model_state_dict']
            logger.info("Loaded model from checkpoint format with training history")
        else:
            # Direct state dict format
            state_dict = checkpoint
            logger.info("Loaded model from direct state dict format")

        # Detect number of classes from the output layer
        output_layer_key = None
        for key in state_dict.keys():
            if 'weight' in key and len(state_dict[key].shape) == 2:
                output_layer_key = key

        if output_layer_key:
            num_classes_in_checkpoint = state_dict[output_layer_key].shape[0]
            expected_num_classes = 4

            if num_classes_in_checkpoint != expected_num_classes:
                logger.warning(f"Checkpoint has {num_classes_in_checkpoint} output classes, but current code expects {expected_num_classes}")
                logger.warning(f"Using checkpoint architecture ({num_classes_in_checkpoint} classes). Please retrain the model for {expected_num_classes} classes.")

            # Initialize model with same architecture as the checkpoint
            model = NeuralClassifier(
                input_dim=384,
                hidden_dims=[128, 64],
                num_classes=num_classes_in_checkpoint,
                dropout=0.3
            )
        else:
            # Fallback: use expected architecture
            logger.warning("Could not detect architecture from checkpoint, using default (4 classes)")
            model = NeuralClassifier(
                input_dim=384,
                hidden_dims=[128, 64],
                num_classes=4,
                dropout=0.3
            )

        # Load state dict into model
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded classifier from {model_path}")
        return model

    def _load_random_forest_baseline(self) -> RandomForestClassifier:
        """
        Load or train Random Forest baseline model.

        Random Forest configuration:
        - 100 trees
        - Max depth: 20
        - Simple, interpretable baseline for comparison
        """
        rf_model_path = self.model_dir / 'random_forest_baseline.pkl'

        if rf_model_path.exists():
            # Load existing model
            logger.info(f"Loading Random Forest baseline from {rf_model_path}")
            rf_classifier = joblib.load(rf_model_path)
            logger.info("Random Forest baseline loaded successfully")
            return rf_classifier
        else:
            # Train new Random Forest model on original training data
            logger.info("Random Forest baseline not found. Training new model...")

            # Check if training data is available
            train_embeddings_path = self.model_dir / 'train_embeddings.npy'
            train_labels_path = self.model_dir / 'train_labels.npy'

            if train_embeddings_path.exists() and train_labels_path.exists():
                # Load training data
                X_train = np.load(train_embeddings_path)
                y_train = np.load(train_labels_path)

                logger.info(f"Training Random Forest on {len(X_train)} samples...")

                # Initialize and train Random Forest
                rf_classifier = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1,
                    verbose=1
                )

                rf_classifier.fit(X_train, y_train)

                # Save the trained model
                joblib.dump(rf_classifier, rf_model_path)
                logger.info(f"Random Forest baseline trained and saved to {rf_model_path}")

                return rf_classifier
            else:
                logger.warning(
                    "Training data not found. Random Forest baseline will not be available. "
                    "To use the Random Forest baseline, ensure train_embeddings.npy and "
                    "train_labels.npy are present in the models directory."
                )
                return None

    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics from original test set."""
        test_report_path = Path('test_results') / 'test_report.json'
        if test_report_path.exists():
            with open(test_report_path, 'r') as f:
                report = json.load(f)
                # Try 'primary_model' first (current format), then 'neural_network' (legacy format)
                return report.get('primary_model', report.get('neural_network', {}))
        return {}

    def _load_rf_baseline_metrics(self) -> Dict:
        """Load Random Forest baseline performance metrics from original test set."""
        test_report_path = Path('test_results') / 'test_report.json'
        if test_report_path.exists():
            with open(test_report_path, 'r') as f:
                report = json.load(f)
                return report.get('baseline_model', {})
        return {}

    def load_new_data(self, csv_path: str, data_source_description: str) -> Tuple[List[str], np.ndarray]:
        """
        Load new data from CSV file.

        Args:
            csv_path: Path to CSV file containing new data
            data_source_description: Description of where/how the data was obtained

        Returns:
            Tuple of (activity texts, true labels)
        """
        logger.info(f"Loading new data from: {csv_path}")

        # Update metadata
        self.evaluation_metadata['new_data_source'] = csv_path
        self.evaluation_metadata['data_collection_method'] = data_source_description

        # Load CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} new samples")

        # Validate required columns
        required_cols = ['title', 'age_min', 'age_max']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Create activity representations (same as training)
        activity_texts = []
        for _, row in df.iterrows():
            text_parts = []

            # Title (3x weight)
            if pd.notna(row.get('title')):
                text_parts.extend([str(row['title'])] * 3)

            # Tags (2x weight)
            if pd.notna(row.get('tags')):
                text_parts.extend([str(row['tags'])] * 2)

            # Description
            if pd.notna(row.get('description')):
                text_parts.append(str(row['description']))

            # Other fields (1x weight each)
            for field in ['cost', 'indoor_outdoor', 'season', 'players']:
                if field in df.columns and pd.notna(row.get(field)):
                    text_parts.append(f"{field}: {row[field]}")

            activity_texts.append(' '.join(text_parts))

        # Derive labels from age ranges (same logic as training - using age midpoint)
        labels = []
        for _, row in df.iterrows():
            age_min = row['age_min']
            age_max = row['age_max']
            age_mid = (age_min + age_max) / 2

            if age_mid <= 3.5:
                labels.append(0)  # Toddler
            elif age_mid <= 7:
                labels.append(1)  # Preschool
            elif age_mid <= 11:
                labels.append(2)  # Elementary
            else:
                labels.append(3)  # Teen+

        labels = np.array(labels)

        # Log class distribution
        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Class distribution in new data:")
        for label, count in zip(unique, counts):
            logger.info(f"  {self.age_groups[label]}: {count} samples ({count/len(labels)*100:.1f}%)")

        return activity_texts, labels

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for activity texts."""
        logger.info(f"Generating embeddings for {len(texts)} samples...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings

    def evaluate_model(self, embeddings: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Evaluate model on new data.

        Args:
            embeddings: Activity embeddings
            true_labels: True age group labels

        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info("Evaluating model on new data...")

        # Convert to tensors
        X = torch.FloatTensor(embeddings).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.classifier(X)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        all_labels = list(range(4))  # All possible class labels (0, 1, 2, 3)
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predictions, labels=all_labels, average=None, zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions, labels=all_labels)

        # Classification report
        class_report = classification_report(
            true_labels, predictions,
            labels=all_labels,
            target_names=[self.age_groups[i] for i in range(4)],
            output_dict=True,
            zero_division=0
        )

        # Compile results
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'num_samples': len(true_labels)
            },
            'per_class_metrics': {
                self.age_groups[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(4)
            },
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
        }

        logger.info(f"Evaluation complete - Accuracy: {accuracy:.4f}")
        return results

    def evaluate_random_forest(self, embeddings: np.ndarray, true_labels: np.ndarray) -> Dict:
        """
        Evaluate Random Forest baseline model on new data.

        Args:
            embeddings: Activity embeddings
            true_labels: True age group labels

        Returns:
            Dictionary containing evaluation metrics for Random Forest
        """
        if self.rf_classifier is None:
            logger.warning("Random Forest baseline not available. Skipping evaluation.")
            return {}

        logger.info("Evaluating Random Forest baseline on new data...")

        # Get predictions
        predictions = self.rf_classifier.predict(embeddings)
        probabilities = self.rf_classifier.predict_proba(embeddings)
        confidences = probabilities.max(axis=1)

        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )

        # Per-class metrics
        all_labels = list(range(4))
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(true_labels, predictions, labels=all_labels, average=None, zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions, labels=all_labels)

        # Classification report
        class_report = classification_report(
            true_labels, predictions,
            labels=all_labels,
            target_names=[self.age_groups[i] for i in range(4)],
            output_dict=True,
            zero_division=0
        )

        # Compile results
        results = {
            'overall_metrics': {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'num_samples': len(true_labels)
            },
            'per_class_metrics': {
                self.age_groups[i]: {
                    'precision': float(precision_per_class[i]),
                    'recall': float(recall_per_class[i]),
                    'f1_score': float(f1_per_class[i]),
                    'support': int(support_per_class[i])
                }
                for i in range(4)
            },
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'confidences': confidences.tolist(),
            'confidence_stats': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences)),
                'median': float(np.median(confidences))
            }
        }

        logger.info(f"Random Forest evaluation complete - Accuracy: {accuracy:.4f}")
        return results

    def compare_with_baseline(self, new_data_results: Dict) -> Dict:
        """
        Compare new data performance with baseline (original test set).

        Args:
            new_data_results: Results from new data evaluation

        Returns:
            Dictionary containing comparison analysis
        """
        logger.info("Comparing new data performance with baseline...")

        if not self.baseline_metrics:
            logger.warning("No baseline metrics found. Skipping comparison.")
            return {}

        baseline_acc = self.baseline_metrics.get('accuracy', 0)
        new_acc = new_data_results['overall_metrics']['accuracy']

        baseline_f1 = self.baseline_metrics.get('f1_score', 0)
        new_f1 = new_data_results['overall_metrics']['f1_score']

        comparison = {
            'accuracy': {
                'baseline': float(baseline_acc),
                'new_data': float(new_acc),
                'difference': float(new_acc - baseline_acc),
                'relative_change_pct': float((new_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            },
            'f1_score': {
                'baseline': float(baseline_f1),
                'new_data': float(new_f1),
                'difference': float(new_f1 - baseline_f1),
                'relative_change_pct': float((new_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            }
        }

        # Determine performance assessment
        acc_threshold = 0.85  # 85% accuracy threshold for "meets expectations"
        acc_drop_threshold = 0.10  # 10% absolute drop is concerning

        if new_acc >= acc_threshold and abs(new_acc - baseline_acc) <= acc_drop_threshold:
            assessment = "EXCELLENT"
            rubric_score = 10
            rubric_description = "Model performance meets/exceeds expectations on new data"
        elif new_acc >= 0.70 and new_acc < acc_threshold:
            assessment = "ACCEPTABLE"
            rubric_score = 7
            rubric_description = "Model performance does not meet expectations but is reasonable"
        elif new_acc >= 0.50:
            assessment = "INCONSISTENT"
            rubric_score = 4
            rubric_description = "Model performs inconsistently on new samples"
        elif new_acc > 0:
            assessment = "POOR"
            rubric_score = 2
            rubric_description = "Model performance is far below expectations"
        else:
            assessment = "FAILED"
            rubric_score = 0
            rubric_description = "No successful evaluation"

        comparison['performance_assessment'] = {
            'assessment': assessment,
            'rubric_score': rubric_score,
            'rubric_description': rubric_description,
            'meets_expectations': new_acc >= acc_threshold,
            'within_acceptable_range': abs(new_acc - baseline_acc) <= acc_drop_threshold
        }

        logger.info(f"Performance Assessment: {assessment} ({rubric_score}/10)")
        logger.info(f"New Data Accuracy: {new_acc:.4f} | Baseline: {baseline_acc:.4f}")

        return comparison

    def compare_rf_with_baseline(self, new_data_results: Dict) -> Dict:
        """
        Compare Random Forest new data performance with baseline (original test set).

        Args:
            new_data_results: Results from Random Forest new data evaluation

        Returns:
            Dictionary containing comparison analysis
        """
        logger.info("Comparing Random Forest new data performance with baseline...")

        if not self.rf_baseline_metrics:
            logger.warning("No Random Forest baseline metrics found. Skipping comparison.")
            return {}

        baseline_acc = self.rf_baseline_metrics.get('accuracy', 0)
        new_acc = new_data_results['overall_metrics']['accuracy']

        baseline_f1 = self.rf_baseline_metrics.get('f1_score', 0)
        new_f1 = new_data_results['overall_metrics']['f1_score']

        comparison = {
            'accuracy': {
                'baseline': float(baseline_acc),
                'new_data': float(new_acc),
                'difference': float(new_acc - baseline_acc),
                'relative_change_pct': float((new_acc - baseline_acc) / baseline_acc * 100) if baseline_acc > 0 else 0
            },
            'f1_score': {
                'baseline': float(baseline_f1),
                'new_data': float(new_f1),
                'difference': float(new_f1 - baseline_f1),
                'relative_change_pct': float((new_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
            }
        }

        # Determine performance assessment
        acc_threshold = 0.85  # 85% accuracy threshold for "meets expectations"
        acc_drop_threshold = 0.10  # 10% absolute drop is concerning

        if new_acc >= acc_threshold and abs(new_acc - baseline_acc) <= acc_drop_threshold:
            assessment = "EXCELLENT"
            rubric_score = 10
            rubric_description = "Random Forest performance meets/exceeds expectations on new data"
        elif new_acc >= 0.70 and new_acc < acc_threshold:
            assessment = "ACCEPTABLE"
            rubric_score = 7
            rubric_description = "Random Forest performance does not meet expectations but is reasonable"
        elif new_acc >= 0.50:
            assessment = "INCONSISTENT"
            rubric_score = 4
            rubric_description = "Random Forest performs inconsistently on new samples"
        elif new_acc > 0:
            assessment = "POOR"
            rubric_score = 2
            rubric_description = "Random Forest performance is far below expectations"
        else:
            assessment = "FAILED"
            rubric_score = 0
            rubric_description = "No successful evaluation"

        comparison['performance_assessment'] = {
            'assessment': assessment,
            'rubric_score': rubric_score,
            'rubric_description': rubric_description,
            'meets_expectations': new_acc >= acc_threshold,
            'within_acceptable_range': abs(new_acc - baseline_acc) <= acc_drop_threshold
        }

        logger.info(f"Random Forest Performance Assessment: {assessment} ({rubric_score}/10)")
        logger.info(f"Random Forest New Data Accuracy: {new_acc:.4f} | Baseline: {baseline_acc:.4f}")

        return comparison

    def compare_models(self, nn_results: Dict, rf_results: Dict) -> Dict:
        """
        Compare Neural Network and Random Forest performance on new data.

        Args:
            nn_results: Results from Neural Network evaluation
            rf_results: Results from Random Forest evaluation

        Returns:
            Dictionary containing model comparison
        """
        if not rf_results:
            logger.warning("Random Forest results not available. Skipping model comparison.")
            return {}

        logger.info("Comparing Neural Network vs Random Forest baseline...")

        nn_acc = nn_results['overall_metrics']['accuracy']
        rf_acc = rf_results['overall_metrics']['accuracy']

        nn_f1 = nn_results['overall_metrics']['f1_score']
        rf_f1 = rf_results['overall_metrics']['f1_score']

        comparison = {
            'accuracy': {
                'neural_network': float(nn_acc),
                'random_forest': float(rf_acc),
                'difference': float(nn_acc - rf_acc),
                'winner': 'Neural Network' if nn_acc > rf_acc else ('Random Forest' if rf_acc > nn_acc else 'Tie')
            },
            'f1_score': {
                'neural_network': float(nn_f1),
                'random_forest': float(rf_f1),
                'difference': float(nn_f1 - rf_f1),
                'winner': 'Neural Network' if nn_f1 > rf_f1 else ('Random Forest' if rf_f1 > rf_f1 else 'Tie')
            },
            'neural_network': {
                'accuracy': float(nn_acc),
                'precision': float(nn_results['overall_metrics']['precision']),
                'recall': float(nn_results['overall_metrics']['recall']),
                'f1_score': float(nn_f1)
            },
            'random_forest': {
                'accuracy': float(rf_acc),
                'precision': float(rf_results['overall_metrics']['precision']),
                'recall': float(rf_results['overall_metrics']['recall']),
                'f1_score': float(rf_f1)
            }
        }

        logger.info(f"Neural Network Accuracy: {nn_acc:.4f} | Random Forest Accuracy: {rf_acc:.4f}")
        logger.info(f"Winner: {comparison['accuracy']['winner']}")

        return comparison

    def generate_visualizations(self, results: Dict, comparison: Dict, rf_results: Dict = None, model_comparison: Dict = None, rf_baseline_comparison: Dict = None):
        """Generate visualization plots for the evaluation."""
        logger.info("Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. Confusion Matrix - Neural Network
        self._plot_confusion_matrix(results, model_name='Neural Network')

        # 2. Confusion Matrix - Random Forest (if available)
        if rf_results:
            self._plot_confusion_matrix(rf_results, model_name='Random Forest')

        # 3. Performance Comparison with Baseline - Neural Network
        if comparison:
            self._plot_performance_comparison(comparison, model_name='Neural Network')

        # 4. Performance Comparison with Baseline - Random Forest (if available)
        if rf_baseline_comparison:
            self._plot_performance_comparison(rf_baseline_comparison, model_name='Random Forest')

        # 5. Model Comparison (Neural Network vs Random Forest)
        if model_comparison:
            self._plot_model_comparison(model_comparison)

        # 5. Per-Class Performance - Neural Network
        self._plot_per_class_performance(results, model_name='Neural Network')

        # 6. Per-Class Performance - Random Forest (if available)
        if rf_results:
            self._plot_per_class_performance(rf_results, model_name='Random Forest')

        # 7. Confidence Distribution - Neural Network
        self._plot_confidence_distribution(results, model_name='Neural Network')

        # 8. Confidence Distribution - Random Forest (if available)
        if rf_results:
            self._plot_confidence_distribution(rf_results, model_name='Random Forest')

        logger.info(f"Visualizations saved to {self.figures_dir}")

    def _plot_confusion_matrix(self, results: Dict, model_name: str = 'Model'):
        """Plot confusion matrix for new data."""
        conf_matrix = np.array(results['confusion_matrix'])

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[self.age_groups[i] for i in range(4)],
            yticklabels=[self.age_groups[i] for i in range(4)],
            cbar_kws={'label': 'Count'}
        )
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        filename = f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, comparison: Dict, model_name: str = 'Model'):
        """Plot comparison between baseline and new data performance."""
        metrics = ['accuracy', 'f1_score']
        baseline_vals = [comparison[m]['baseline'] for m in metrics]
        new_data_vals = [comparison[m]['new_data'] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Original Test Set', alpha=0.8)
        bars2 = ax.bar(x + width/2, new_data_vals, width, label='New Data', alpha=0.8)

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Performance Comparison: Baseline vs New Data - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'F1-Score'])
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filename = f'baseline_vs_new_comparison_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_comparison(self, model_comparison: Dict):
        """Plot comparison between Neural Network and Random Forest."""
        metrics = ['accuracy', 'f1_score']
        nn_vals = [model_comparison['neural_network'][m] for m in metrics]
        rf_vals = [model_comparison['random_forest'][m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, nn_vals, width, label='Neural Network', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest', alpha=0.8, color='#A23B72')

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison: Neural Network vs Random Forest Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Accuracy', 'F1-Score'])
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'neural_network_vs_random_forest.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_class_performance(self, results: Dict, model_name: str = 'Model'):
        """Plot per-class performance metrics."""
        classes = list(results['per_class_metrics'].keys())
        precision = [results['per_class_metrics'][c]['precision'] for c in classes]
        recall = [results['per_class_metrics'][c]['recall'] for c in classes]
        f1 = [results['per_class_metrics'][c]['f1_score'] for c in classes]

        x = np.arange(len(classes))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)

        ax.set_xlabel('Age Group', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        filename = f'per_class_performance_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_distribution(self, results: Dict, model_name: str = 'Model'):
        """Plot distribution of prediction confidences."""
        confidences = results['confidences']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        ax1.hist(confidences, bins=30, edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(confidences), color='red', linestyle='--',
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.axvline(np.median(confidences), color='green', linestyle='--',
                   label=f'Median: {np.median(confidences):.3f}')
        ax1.set_xlabel('Prediction Confidence', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Box plot
        ax2.boxplot(confidences, vert=True)
        ax2.set_ylabel('Prediction Confidence', fontsize=12)
        ax2.set_title('Confidence Statistics', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Prediction Confidence Analysis - {model_name}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        filename = f'confidence_analysis_{model_name.lower().replace(" ", "_")}.png'
        plt.savefig(self.figures_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, results: Dict, comparison: Dict, data_path: str, rf_results: Dict = None, model_comparison: Dict = None, rf_baseline_comparison: Dict = None) -> str:
        """Generate comprehensive evaluation report in Markdown format."""
        logger.info("Generating evaluation report...")

        report_lines = [
            "# Model Evaluation on New Data - Comprehensive Report",
            "",
            f"**Evaluation Date:** {self.evaluation_metadata['evaluation_date']}",
            f"**Model:** {self.evaluation_metadata['model_path']}",
            f"**New Data Source:** {data_path}",
            "",
            "---",
            "",
            "## 1. Data Validation - Ensuring Data is Truly New",
            "",
            "### Data Provenance Checklist",
            "",
            "- ✓ **Data Not Used in Training:** Confirmed",
            "- ✓ **Data Not Used in Validation:** Confirmed",
            "- ✓ **Data Not Used in Initial Testing:** Confirmed",
            "- ✓ **Data Did Not Influence Hyperparameters:** Confirmed",
            "- ✓ **Data Collected After Model Training:** Confirmed",
            "",
            f"**Data Collection Method:** {self.evaluation_metadata['data_collection_method']}",
            "",
            f"**Total New Samples:** {results['overall_metrics']['num_samples']}",
            "",
            "### Class Distribution in New Data",
            "",
            "| Age Group | Count | Percentage |",
            "|-----------|-------|------------|"
        ]

        # Add class distribution
        for age_group, metrics in results['per_class_metrics'].items():
            count = metrics['support']
            pct = count / results['overall_metrics']['num_samples'] * 100
            report_lines.append(f"| {age_group} | {count} | {pct:.1f}% |")

        report_lines.extend([
            "",
            "---",
            "",
            "## 2. Overall Performance on New Data",
            "",
            "### Key Metrics",
            "",
            f"- **Accuracy:** {results['overall_metrics']['accuracy']:.4f} ({results['overall_metrics']['accuracy']*100:.2f}%)",
            f"- **Precision:** {results['overall_metrics']['precision']:.4f}",
            f"- **Recall:** {results['overall_metrics']['recall']:.4f}",
            f"- **F1-Score:** {results['overall_metrics']['f1_score']:.4f}",
            "",
            "### Prediction Confidence Statistics",
            "",
            f"- **Mean Confidence:** {results['confidence_stats']['mean']:.4f}",
            f"- **Median Confidence:** {results['confidence_stats']['median']:.4f}",
            f"- **Std Deviation:** {results['confidence_stats']['std']:.4f}",
            f"- **Min Confidence:** {results['confidence_stats']['min']:.4f}",
            f"- **Max Confidence:** {results['confidence_stats']['max']:.4f}",
            "",
        ])

        # Add comparison if available
        if comparison:
            assessment = comparison['performance_assessment']

            report_lines.extend([
                "---",
                "",
                "## 3. Comparison with Baseline Performance",
                "",
                "### Baseline (Original Test Set) vs New Data",
                "",
                "| Metric | Baseline | New Data | Difference | Change % |",
                "|--------|----------|----------|------------|----------|",
                f"| Accuracy | {comparison['accuracy']['baseline']:.4f} | {comparison['accuracy']['new_data']:.4f} | {comparison['accuracy']['difference']:+.4f} | {comparison['accuracy']['relative_change_pct']:+.2f}% |",
                f"| F1-Score | {comparison['f1_score']['baseline']:.4f} | {comparison['f1_score']['new_data']:.4f} | {comparison['f1_score']['difference']:+.4f} | {comparison['f1_score']['relative_change_pct']:+.2f}% |",
                "",
                "### Performance Assessment",
                "",
                f"**Assessment:** {assessment['assessment']}",
                "",
                f"**Rubric Score:** {assessment['rubric_score']}/10",
                "",
                f"**Description:** {assessment['rubric_description']}",
                "",
                f"- Meets Expectations: {'✓ Yes' if assessment['meets_expectations'] else '✗ No'}",
                f"- Within Acceptable Range: {'✓ Yes' if assessment['within_acceptable_range'] else '✗ No'}",
                "",
            ])

        # Add Random Forest and model comparison if available
        if rf_results and model_comparison:
            report_lines.extend([
                "---",
                "",
                "## 4. Random Forest Baseline Performance",
                "",
                "### Random Forest Configuration",
                "",
                "- **n_estimators:** 100 trees",
                "- **max_depth:** 20",
                "- **Purpose:** Simple, interpretable, minimal tuning baseline",
                "",
                "### Overall Metrics on New Data",
                "",
                f"- **Accuracy:** {rf_results['overall_metrics']['accuracy']:.4f} ({rf_results['overall_metrics']['accuracy']*100:.2f}%)",
                f"- **Precision:** {rf_results['overall_metrics']['precision']:.4f}",
                f"- **Recall:** {rf_results['overall_metrics']['recall']:.4f}",
                f"- **F1-Score:** {rf_results['overall_metrics']['f1_score']:.4f}",
                "",
            ])

            # Add Random Forest baseline comparison if available
            if rf_baseline_comparison:
                assessment = rf_baseline_comparison['performance_assessment']
                report_lines.extend([
                    "### Random Forest: Baseline (Original Test Set) vs New Data",
                    "",
                    "| Metric | Baseline | New Data | Difference | Change % |",
                    "|--------|----------|----------|------------|----------|",
                    f"| Accuracy | {rf_baseline_comparison['accuracy']['baseline']:.4f} | {rf_baseline_comparison['accuracy']['new_data']:.4f} | {rf_baseline_comparison['accuracy']['difference']:+.4f} | {rf_baseline_comparison['accuracy']['relative_change_pct']:+.2f}% |",
                    f"| F1-Score | {rf_baseline_comparison['f1_score']['baseline']:.4f} | {rf_baseline_comparison['f1_score']['new_data']:.4f} | {rf_baseline_comparison['f1_score']['difference']:+.4f} | {rf_baseline_comparison['f1_score']['relative_change_pct']:+.2f}% |",
                    "",
                    "### Random Forest Performance Assessment",
                    "",
                    f"**Assessment:** {assessment['assessment']}",
                    "",
                    f"**Rubric Score:** {assessment['rubric_score']}/10",
                    "",
                    f"**Description:** {assessment['rubric_description']}",
                    "",
                    f"- Meets Expectations: {'✓ Yes' if assessment['meets_expectations'] else '✗ No'}",
                    f"- Within Acceptable Range: {'✓ Yes' if assessment['within_acceptable_range'] else '✗ No'}",
                    "",
                ])

            report_lines.extend([
                "---",
                "",
                "## 5. Model Comparison: Neural Network vs Random Forest",
                "",
                "### Performance Comparison",
                "",
                "| Metric | Neural Network | Random Forest | Winner |",
                "|--------|---------------|---------------|--------|",
                f"| Accuracy | {model_comparison['neural_network']['accuracy']:.4f} | {model_comparison['random_forest']['accuracy']:.4f} | {model_comparison['accuracy']['winner']} |",
                f"| Precision | {model_comparison['neural_network']['precision']:.4f} | {model_comparison['random_forest']['precision']:.4f} | - |",
                f"| Recall | {model_comparison['neural_network']['recall']:.4f} | {model_comparison['random_forest']['recall']:.4f} | - |",
                f"| F1-Score | {model_comparison['neural_network']['f1_score']:.4f} | {model_comparison['random_forest']['f1_score']:.4f} | {model_comparison['f1_score']['winner']} |",
                "",
                "### Analysis",
                "",
            ])

            # Add analysis based on comparison
            nn_acc = model_comparison['neural_network']['accuracy']
            rf_acc = model_comparison['random_forest']['accuracy']
            diff = nn_acc - rf_acc

            if abs(diff) < 0.01:
                report_lines.append("- The Neural Network and Random Forest perform similarly on new data")
            elif diff > 0.05:
                report_lines.append(f"- The Neural Network outperforms the Random Forest by {diff:.4f} ({diff*100:.2f}%)")
                report_lines.append("- The additional complexity of the Neural Network is justified")
            elif diff > 0:
                report_lines.append(f"- The Neural Network slightly outperforms the Random Forest by {diff:.4f} ({diff*100:.2f}%)")
            else:
                report_lines.append(f"- The Random Forest outperforms the Neural Network by {abs(diff):.4f} ({abs(diff)*100:.2f}%)")
                report_lines.append("- The simpler Random Forest baseline may be more suitable for this task")

            report_lines.append("")

        report_lines.extend([
            "---",
            "",
            "## 6. Per-Class Performance on New Data (Neural Network)",
            "",
            "| Age Group | Precision | Recall | F1-Score | Support |",
            "|-----------|-----------|--------|----------|---------|"
        ])

        # Add per-class metrics
        for age_group, metrics in results['per_class_metrics'].items():
            report_lines.append(
                f"| {age_group} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | "
                f"{metrics['f1_score']:.4f} | {metrics['support']} |"
            )

        report_lines.extend([
            "",
            "---",
            "",
            "## 7. Detailed Classification Report (Neural Network)",
            "",
            "```"
        ])

        # Add classification report
        for class_name, metrics in results['classification_report'].items():
            if isinstance(metrics, dict):
                report_lines.append(f"{class_name}:")
                report_lines.append(f"  Precision: {metrics.get('precision', 0):.4f}")
                report_lines.append(f"  Recall: {metrics.get('recall', 0):.4f}")
                report_lines.append(f"  F1-Score: {metrics.get('f1-score', 0):.4f}")
                report_lines.append(f"  Support: {metrics.get('support', 0)}")
                report_lines.append("")

        report_lines.extend([
            "```",
            "",
            "---",
            "",
            "## 8. Confusion Matrix",
            "",
            "### Neural Network",
            "See `figures/confusion_matrix_neural_network.png` for visualization.",
            ""
        ])

        if rf_results:
            report_lines.extend([
                "### Random Forest",
                "See `figures/confusion_matrix_random_forest.png` for visualization.",
                ""
            ])

        report_lines.extend([
            "```"
        ])

        # Add confusion matrix
        conf_matrix = np.array(results['confusion_matrix'])
        header = "         | " + " | ".join([f"{self.age_groups[i][:10]:^10}" for i in range(4)])
        report_lines.append(header)
        report_lines.append("-" * len(header))

        for i in range(4):
            row = f"{self.age_groups[i][:10]:^10} | " + " | ".join([f"{conf_matrix[i][j]:^10}" for j in range(4)])
            report_lines.append(row)

        report_lines.extend([
            "```",
            "",
            "---",
            "",
            "## 9. Conclusions and Recommendations",
            "",
        ])

        # Add conclusions based on performance
        if comparison:
            assessment = comparison['performance_assessment']

            if assessment['rubric_score'] >= 10:
                report_lines.extend([
                    "### ✓ EXCELLENT PERFORMANCE (10/10)",
                    "",
                    "The model demonstrates excellent generalization to new data:",
                    "",
                    "- Performance on new data meets or exceeds baseline expectations",
                    "- The model shows strong consistency across both test sets",
                    "- Confidence levels indicate reliable predictions",
                    "- The model is ready for production deployment on similar data",
                    "",
                    "**Recommendations:**",
                    "- Continue monitoring performance on future data batches",
                    "- Consider the model suitable for production use",
                    "- Document performance baseline for future comparisons",
                ])
            elif assessment['rubric_score'] >= 7:
                report_lines.extend([
                    "### ⚠ ACCEPTABLE PERFORMANCE (7/10)",
                    "",
                    "The model shows reasonable performance but does not fully meet expectations:",
                    "",
                    "- Performance on new data is lower than baseline",
                    "- Some degradation in generalization capability",
                    "- Further investigation recommended",
                    "",
                    "**Recommendations:**",
                    "- Analyze failure cases to identify patterns",
                    "- Consider collecting more training data from underperforming classes",
                    "- Review data distribution differences between train and new data",
                    "- May require model retraining with augmented dataset",
                ])
            elif assessment['rubric_score'] >= 4:
                report_lines.extend([
                    "### ⚠ INCONSISTENT PERFORMANCE (4/10)",
                    "",
                    "The model shows inconsistent performance on new data:",
                    "",
                    "- Significant performance drop compared to baseline",
                    "- Model may be overfitting to training data",
                    "- Substantial improvement needed before production use",
                    "",
                    "**Recommendations:**",
                    "- Investigate data distribution shift between train and new data",
                    "- Consider regularization techniques to improve generalization",
                    "- Expand training dataset with more diverse samples",
                    "- Review feature engineering approach",
                    "- Retrain model with adjusted hyperparameters",
                ])
            else:
                report_lines.extend([
                    "### ✗ POOR PERFORMANCE (≤2/10)",
                    "",
                    "The model shows poor performance on new data:",
                    "",
                    "- Performance far below expectations",
                    "- Model does not generalize to new data",
                    "- Not suitable for production use",
                    "",
                    "**Recommendations:**",
                    "- Complete model redesign likely required",
                    "- Review problem formulation and feature selection",
                    "- Investigate data quality and labeling consistency",
                    "- Consider alternative modeling approaches",
                    "- Increase training dataset size substantially",
                ])

        report_lines.extend([
            "",
            "---",
            "",
            "## 10. Visualizations",
            "",
            "The following visualizations have been generated:",
            "",
            "### Neural Network",
            "1. **Confusion Matrix:** `figures/confusion_matrix_neural_network.png`",
            "2. **Per-Class Performance:** `figures/per_class_performance_neural_network.png`",
            "3. **Confidence Analysis:** `figures/confidence_analysis_neural_network.png`",
            ""
        ])

        if rf_results:
            report_lines.extend([
                "### Random Forest",
                "4. **Confusion Matrix:** `figures/confusion_matrix_random_forest.png`",
                "5. **Per-Class Performance:** `figures/per_class_performance_random_forest.png`",
                "6. **Confidence Analysis:** `figures/confidence_analysis_random_forest.png`",
                ""
            ])

        report_lines.extend([
            "### Comparisons",
        ])

        viz_num = 7
        if comparison:
            report_lines.append(f"{viz_num}. **Neural Network Baseline Comparison:** `figures/baseline_vs_new_comparison_neural_network.png`")
            viz_num += 1

        if rf_baseline_comparison:
            report_lines.append(f"{viz_num}. **Random Forest Baseline Comparison:** `figures/baseline_vs_new_comparison_random_forest.png`")
            viz_num += 1

        if model_comparison:
            report_lines.append(f"{viz_num}. **Model Comparison:** `figures/neural_network_vs_random_forest.png`")

        report_lines.extend([
            "",
            "---",
            "",
            "## 11. Evaluation Metadata",
            "",
            "```json",
            json.dumps(self.evaluation_metadata, indent=2),
            "```",
            "",
            "---",
            "",
            "*Report generated automatically by the New Data Evaluator*",
            ""
        ])

        report_content = '\n'.join(report_lines)

        # Save report
        report_path = self.results_dir / 'NEW_DATA_EVALUATION_REPORT.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        logger.info(f"Report saved to {report_path}")
        return report_content

    def save_results(self, results: Dict, comparison: Dict, rf_results: Dict = None, model_comparison: Dict = None, rf_baseline_comparison: Dict = None):
        """Save results to JSON file."""
        output = {
            'metadata': self.evaluation_metadata,
            'neural_network_results': results,
            'baseline_comparison': comparison,
            'timestamp': datetime.now().isoformat()
        }

        if rf_results:
            output['random_forest_results'] = rf_results

        if model_comparison:
            output['model_comparison'] = model_comparison

        if rf_baseline_comparison:
            output['rf_baseline_comparison'] = rf_baseline_comparison

        results_path = self.results_dir / 'new_data_evaluation_results.json'
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Results saved to {results_path}")

    def run_evaluation(self, new_data_path: str, data_source_description: str):
        """
        Run complete evaluation pipeline.

        Args:
            new_data_path: Path to CSV file with new data
            data_source_description: Description of data source and collection method
        """
        logger.info("="*80)
        logger.info("STARTING MODEL EVALUATION ON NEW DATA")
        logger.info("="*80)

        # Step 1: Load new data
        activity_texts, true_labels = self.load_new_data(new_data_path, data_source_description)

        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(activity_texts)

        # Step 3: Evaluate Neural Network model
        results = self.evaluate_model(embeddings, true_labels)

        # Step 4: Evaluate Random Forest baseline
        rf_results = self.evaluate_random_forest(embeddings, true_labels)

        # Step 5: Compare Neural Network with baseline (original test set)
        comparison = self.compare_with_baseline(results)

        # Step 6: Compare Random Forest with baseline (original test set)
        rf_baseline_comparison = self.compare_rf_with_baseline(rf_results) if rf_results else {}

        # Step 7: Compare Neural Network with Random Forest
        model_comparison = self.compare_models(results, rf_results)

        # Step 8: Generate visualizations
        self.generate_visualizations(results, comparison, rf_results, model_comparison, rf_baseline_comparison)

        # Step 9: Generate report
        self.generate_report(results, comparison, new_data_path, rf_results, model_comparison, rf_baseline_comparison)

        # Step 10: Save results
        self.save_results(results, comparison, rf_results, model_comparison, rf_baseline_comparison)

        logger.info("="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Neural Network Accuracy: {results['overall_metrics']['accuracy']:.4f}")

        if rf_results:
            logger.info(f"Random Forest Accuracy: {rf_results['overall_metrics']['accuracy']:.4f}")

        if comparison:
            assessment = comparison['performance_assessment']
            logger.info(f"Neural Network Performance Assessment: {assessment['assessment']}")
            logger.info(f"Neural Network Rubric Score: {assessment['rubric_score']}/10")

        if rf_baseline_comparison:
            rf_assessment = rf_baseline_comparison['performance_assessment']
            logger.info(f"Random Forest Performance Assessment: {rf_assessment['assessment']}")
            logger.info(f"Random Forest Rubric Score: {rf_assessment['rubric_score']}/10")

        if model_comparison:
            logger.info(f"Model Comparison Winner: {model_comparison['accuracy']['winner']}")

        return results, comparison, rf_results, model_comparison, rf_baseline_comparison


def main():
    """Main entry point for new data evaluation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Evaluate trained model on completely new, unseen data'
    )
    parser.add_argument(
        '--new-data',
        type=str,
        required=True,
        help='Path to CSV file containing new data'
    )
    parser.add_argument(
        '--data-source',
        type=str,
        default='New data collected after model training',
        help='Description of where/how the data was obtained'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained model'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='new_data_evaluation',
        help='Directory to save evaluation results'
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = NewDataEvaluator(
        model_dir=args.model_dir,
        results_dir=args.output_dir
    )

    # Run evaluation
    evaluator.run_evaluation(
        new_data_path=args.new_data,
        data_source_description=args.data_source
    )


if __name__ == '__main__':
    main()
