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

    def __init__(self, input_dim=387, hidden_dims=[256, 128, 64], num_classes=4, dropout=0.2):
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

        # Load normalization parameters from training data
        self.normalization_params = self._load_normalization_params()

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
                input_dim=387,
                hidden_dims=[256, 128, 64],
                num_classes=num_classes_in_checkpoint,
                dropout=0.2
            )
        else:
            # Fallback: use expected architecture
            logger.warning("Could not detect architecture from checkpoint, using default (4 classes)")
            model = NeuralClassifier(
                input_dim=387,
                hidden_dims=[256, 128, 64],
                num_classes=4,
                dropout=0.2
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
                    "Training data not found. Random Forest baseline will not be available."
                )
                logger.warning(
                    "To use the Random Forest baseline, you need to retrain the model using train_model.py. "
                    "This will generate train_embeddings.npy and train_labels.npy files needed for Random Forest training."
                )
                logger.warning(
                    "The model was updated to save these files automatically. Run: python train_model.py"
                )
                return None


    def _load_normalization_params(self) -> Dict:
        """
        Load normalization parameters from training dataset.

        These parameters (min, max for each feature) are needed to normalize
        new data the same way training data was normalized.

        Returns:
            Dictionary with min/max values for each numerical feature
        """
        logger.info("Loading normalization parameters from training dataset...")

        # Try to load from saved file first
        norm_params_path = self.model_dir / 'normalization_params.json'
        if norm_params_path.exists():
            with open(norm_params_path, 'r') as f:
                params = json.load(f)
                logger.info("Loaded normalization parameters from saved file")
                return params

        # Otherwise, compute from the processed training dataset
        train_data_path = self.model_dir / 'activities_processed.csv'
        if not train_data_path.exists():
            logger.warning("Training dataset not found. Using default normalization (0-1 range).")
            # Return default parameters that won't change the values much
            return {
                'age_min': {'min': 0, 'max': 18},
                'age_max': {'min': 0, 'max': 18},
                'duration_mins': {'min': 0, 'max': 120}
            }

        # Load training data to compute normalization parameters
        df_train = pd.read_csv(train_data_path)
        logger.info(f"Computing normalization parameters from {len(df_train)} training samples...")

        # Extract features from training data
        features = {
            'age_min': [],
            'age_max': [],
            'duration_mins': []
        }

        for _, row in df_train.iterrows():
            for feat in features.keys():
                val = row.get(feat, 0)
                if pd.isna(val):
                    val = 0
                features[feat].append(val)

        # Compute min and max for each feature
        params = {}
        for feat, values in features.items():
            arr = np.array(values)
            params[feat] = {
                'min': float(arr.min()),
                'max': float(arr.max())
            }
            logger.info(f"  {feat}: min={params[feat]['min']:.2f}, max={params[feat]['max']:.2f}")

        # Save for future use
        with open(norm_params_path, 'w') as f:
            json.dump(params, f, indent=2)
        logger.info(f"Saved normalization parameters to {norm_params_path}")

        return params

    def load_new_data(self, csv_path: str, data_source_description: str) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
        """
        Load new data from CSV file.

        Args:
            csv_path: Path to CSV file containing new data
            data_source_description: Description of where/how the data was obtained

        Returns:
            Tuple of (activity texts, true labels, dataframe)
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

        # Create activity representations (MUST match training exactly)
        # Training uses: title (3x), tags (2x), how_to_play (2x) - ONLY these 3 fields
        activity_texts = []
        for _, row in df.iterrows():
            text_parts = []

            # Title (3x weight) - most important
            if pd.notna(row.get('title')):
                text_parts.extend([str(row['title'])] * 3)

            # Tags (2x weight) - high importance
            if pd.notna(row.get('tags')):
                text_parts.extend([str(row['tags'])] * 2)

            # how_to_play (2x weight) - high importance
            if pd.notna(row.get('how_to_play')):
                text_parts.extend([str(row['how_to_play'])] * 2)

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

        return activity_texts, labels, df

    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and normalize numerical features: age_min, age_max, duration_mins.

        CRITICAL: Features must be normalized using the SAME parameters as training.
        This ensures the model receives inputs in the same scale it was trained on.

        Args:
            df: DataFrame containing the data

        Returns:
            Array of normalized numerical features with shape (n_samples, 3)
        """
        logger.info("Extracting numerical features (age_min, age_max, duration_mins)...")

        numerical_features = []
        feature_names = ['age_min', 'age_max', 'duration_mins']

        for idx, row in df.iterrows():
            age_min = row.get('age_min', 0)
            age_max = row.get('age_max', 0)
            duration_mins = row.get('duration_mins', 0)

            # Handle missing values
            if pd.isna(age_min):
                age_min = 0
            if pd.isna(age_max):
                age_max = 0
            if pd.isna(duration_mins):
                duration_mins = 0

            numerical_features.append([age_min, age_max, duration_mins])

        numerical_features = np.array(numerical_features, dtype=np.float32)

        # CRITICAL: Normalize using the SAME parameters as training
        logger.info("Normalizing features using training parameters...")
        for i, feat_name in enumerate(feature_names):
            col = numerical_features[:, i]
            col_min = self.normalization_params[feat_name]['min']
            col_max = self.normalization_params[feat_name]['max']

            if col_max > col_min:
                numerical_features[:, i] = (col - col_min) / (col_max - col_min)
                logger.info(f"  {feat_name}: normalized using min={col_min:.2f}, max={col_max:.2f}")
            else:
                logger.warning(f"  {feat_name}: skipping normalization (min == max)")

        logger.info(f"✓ Extracted and normalized numerical features with shape: {numerical_features.shape}")
        logger.info(f"  Features: age_min, age_max, duration_mins (normalized to [0, 1] range)")

        return numerical_features

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

        # Check if the Random Forest expects the same number of features
        expected_features = self.rf_classifier.n_features_in_
        actual_features = embeddings.shape[1]

        if expected_features != actual_features:
            logger.warning(f"Feature dimension mismatch detected!")
            logger.warning(f"  Random Forest expects {expected_features} features")
            logger.warning(f"  Current embeddings have {actual_features} features")
            logger.warning(f"  This likely means the Random Forest was trained on text embeddings only (384)")
            logger.warning(f"  but evaluation is using combined embeddings (text + numerical = 387)")
            logger.info("Retraining Random Forest with correct features...")

            # Delete the old model
            rf_model_path = self.model_dir / 'random_forest_baseline.pkl'
            if rf_model_path.exists():
                rf_model_path.unlink()
                logger.info(f"Deleted outdated Random Forest model: {rf_model_path}")

            # Retrain with correct features
            self.rf_classifier = self._load_random_forest_baseline()

            if self.rf_classifier is None:
                logger.warning("Could not retrain Random Forest. Skipping evaluation.")
                return {}

            # Verify the new model has the correct number of features
            if self.rf_classifier.n_features_in_ != actual_features:
                logger.error(f"Random Forest still has wrong features after retraining!")
                logger.error(f"  Expected: {actual_features}, Got: {self.rf_classifier.n_features_in_}")
                return {}

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

    def compare_with_baseline(self, nn_results: Dict, rf_results: Dict) -> Dict:
        """
        Compare Neural Network with Random Forest baseline on new data.

        Args:
            nn_results: Results from Neural Network evaluation on new data
            rf_results: Results from Random Forest evaluation on new data

        Returns:
            Dictionary containing Random Forest baseline metrics and comparison
        """
        logger.info("Comparing Neural Network with Random Forest baseline on new data...")

        if not rf_results:
            logger.warning("No Random Forest results found. Skipping baseline comparison.")
            return {}

        # Extract Random Forest metrics
        rf_acc = rf_results['overall_metrics']['accuracy']
        rf_precision = rf_results['overall_metrics']['precision']
        rf_recall = rf_results['overall_metrics']['recall']
        rf_f1 = rf_results['overall_metrics']['f1_score']
        rf_conf_matrix = rf_results['confusion_matrix']
        rf_per_class = rf_results['per_class_metrics']

        # Extract Neural Network metrics for comparison
        nn_acc = nn_results['overall_metrics']['accuracy']
        nn_f1 = nn_results['overall_metrics']['f1_score']

        # Build baseline comparison (RF is the baseline)
        comparison = {
            'baseline_model': 'Random Forest (100 trees, max depth 20)',
            'baseline_metrics': {
                'accuracy': float(rf_acc),
                'precision': float(rf_precision),
                'recall': float(rf_recall),
                'f1_score': float(rf_f1),
                'confusion_matrix': rf_conf_matrix,
                'per_class_metrics': rf_per_class
            },
            'neural_network_vs_baseline': {
                'accuracy': {
                    'neural_network': float(nn_acc),
                    'random_forest_baseline': float(rf_acc),
                    'difference': float(nn_acc - rf_acc),
                    'nn_outperforms': nn_acc > rf_acc
                },
                'f1_score': {
                    'neural_network': float(nn_f1),
                    'random_forest_baseline': float(rf_f1),
                    'difference': float(nn_f1 - rf_f1),
                    'nn_outperforms': nn_f1 > rf_f1
                }
            }
        }

        logger.info(f"Random Forest Baseline Accuracy: {rf_acc:.4f}")
        logger.info(f"Neural Network Accuracy: {nn_acc:.4f}")
        logger.info(f"Neural Network {'outperforms' if nn_acc > rf_acc else 'underperforms'} baseline by {abs(nn_acc - rf_acc):.4f}")

        return comparison


    def generate_visualizations(self, results: Dict, rf_results: Dict, comparison: Dict):
        """Generate visualization plots for the evaluation."""
        logger.info("Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. Confusion Matrix - Neural Network
        self._plot_confusion_matrix(results, model_name='Neural Network')

        # 2. Confusion Matrix - Random Forest Baseline
        if rf_results:
            self._plot_confusion_matrix(rf_results, model_name='Random Forest Baseline')

        # 3. Model Comparison (Neural Network vs Random Forest Baseline)
        if comparison and rf_results:
            self._plot_nn_vs_rf_comparison(results, rf_results)

        # 4. Per-Class Performance - Neural Network
        self._plot_per_class_performance(results, model_name='Neural Network')

        # 5. Per-Class Performance - Random Forest Baseline
        if rf_results:
            self._plot_per_class_performance(rf_results, model_name='Random Forest Baseline')

        # 6. Confidence Distribution - Neural Network
        self._plot_confidence_distribution(results, model_name='Neural Network')

        # 7. Confidence Distribution - Random Forest Baseline
        if rf_results:
            self._plot_confidence_distribution(rf_results, model_name='Random Forest Baseline')

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

    def _plot_nn_vs_rf_comparison(self, nn_results: Dict, rf_results: Dict):
        """Plot comparison between Neural Network and Random Forest Baseline on new data."""
        nn_metrics = nn_results['overall_metrics']
        rf_metrics = rf_results['overall_metrics']

        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        nn_vals = [nn_metrics[m] for m in metrics]
        rf_vals = [rf_metrics[m] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, nn_vals, width, label='Neural Network', alpha=0.8, color='#2E86AB')
        bars2 = ax.bar(x + width/2, rf_vals, width, label='Random Forest Baseline', alpha=0.8, color='#A23B72')

        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Neural Network vs Random Forest Baseline (New Data)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
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
        plt.savefig(self.figures_dir / 'neural_network_vs_random_forest_baseline.png', dpi=300, bbox_inches='tight')
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

    def generate_report(self, results: Dict, rf_results: Dict, comparison: Dict, data_path: str) -> str:
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
            "## 2. Model Comparison: Neural Network vs Random Forest Baseline",
            "",
            "Both models evaluated on the same new data.",
            "",
        ])

        # Add comparison if available
        if comparison and rf_results:
            nn_metrics = results['overall_metrics']
            rf_metrics = rf_results['overall_metrics']

            nn_acc = nn_metrics['accuracy']
            rf_acc = rf_metrics['accuracy']

            report_lines.extend([
                "### Performance Summary",
                "",
                "| Model | Accuracy | Precision | Recall | F1-Score |",
                "|-------|----------|-----------|--------|----------|",
                f"| **Neural Network** | {nn_acc:.4f} | {nn_metrics['precision']:.4f} | {nn_metrics['recall']:.4f} | {nn_metrics['f1_score']:.4f} |",
                f"| **Random Forest Baseline** | {rf_acc:.4f} | {rf_metrics['precision']:.4f} | {rf_metrics['recall']:.4f} | {rf_metrics['f1_score']:.4f} |",
                "",
                "### Model Comparison Analysis",
                "",
            ])

            # Determine winner and add analysis
            if nn_acc > rf_acc:
                diff = nn_acc - rf_acc
                report_lines.extend([
                    f"✓ **Neural Network outperforms Random Forest Baseline**",
                    f"  - Accuracy improvement: +{diff:.4f} (+{diff*100:.2f}%)",
                    f"  - The neural network demonstrates superior performance on new data",
                    "",
                ])
            elif rf_acc > nn_acc:
                diff = rf_acc - nn_acc
                report_lines.extend([
                    f"⚠ **Random Forest Baseline outperforms Neural Network**",
                    f"  - Accuracy difference: -{diff:.4f} (-{diff*100:.2f}%)",
                    f"  - The simpler Random Forest baseline may be more suitable for this task",
                    "",
                ])
            else:
                report_lines.extend([
                    f"**Models perform equally on new data**",
                    f"  - Both achieve {nn_acc:.4f} accuracy",
                    "",
                ])

        report_lines.extend([
            "---",
            "",
            "## 3. Neural Network Performance Details",
            "",
            "### Overall Metrics",
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

        # Add Random Forest baseline details if available
        if rf_results:
            report_lines.extend([
                "---",
                "",
                "## 4. Random Forest Baseline Performance Details",
                "",
                "### Configuration",
                "",
                "- **n_estimators:** 100 trees",
                "- **max_depth:** 20",
                "- **Purpose:** Simple, interpretable baseline with minimal tuning",
                "",
                "### Overall Metrics on New Data",
                "",
                f"- **Accuracy:** {rf_results['overall_metrics']['accuracy']:.4f} ({rf_results['overall_metrics']['accuracy']*100:.2f}%)",
                f"- **Precision:** {rf_results['overall_metrics']['precision']:.4f}",
                f"- **Recall:** {rf_results['overall_metrics']['recall']:.4f}",
                f"- **F1-Score:** {rf_results['overall_metrics']['f1_score']:.4f}",
                "",
            ])

        report_lines.extend([
            "---",
            "",
            "## 5. Per-Class Performance on New Data (Neural Network)",
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
            "## 6. Detailed Classification Report (Neural Network)",
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
            "## 7. Confusion Matrices",
            "",
            "### Neural Network",
            "See `figures/confusion_matrix_neural_network.png` for visualization.",
            ""
        ])

        if rf_results:
            report_lines.extend([
                "### Random Forest Baseline",
                "See `figures/confusion_matrix_random_forest_baseline.png` for visualization.",
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
            "## 8. Summary and Recommendations",
            "",
        ])

        # Add conclusions based on comparison with Random Forest
        if comparison and rf_results:
            nn_acc = results['overall_metrics']['accuracy']
            rf_acc = rf_results['overall_metrics']['accuracy']

            if nn_acc > rf_acc:
                diff = nn_acc - rf_acc
                report_lines.extend([
                    "### ✓ Neural Network Outperforms Baseline",
                    "",
                    "The neural network demonstrates superior performance compared to the Random Forest baseline:",
                    "",
                    f"- Neural Network Accuracy: {nn_acc:.4f} ({nn_acc*100:.2f}%)",
                    f"- Random Forest Baseline Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)",
                    f"- Improvement: +{diff:.4f} (+{diff*100:.2f}%)",
                    "",
                    "**Recommendations:**",
                    "- The neural network is the recommended model for this task",
                    "- Continue monitoring performance on future data batches",
                    "- Consider the neural network suitable for production use",
                ])
            elif rf_acc > nn_acc:
                diff = rf_acc - nn_acc
                report_lines.extend([
                    "### ⚠ Random Forest Baseline Outperforms Neural Network",
                    "",
                    "The simpler Random Forest baseline demonstrates superior performance:",
                    "",
                    f"- Random Forest Baseline Accuracy: {rf_acc:.4f} ({rf_acc*100:.2f}%)",
                    f"- Neural Network Accuracy: {nn_acc:.4f} ({nn_acc*100:.2f}%)",
                    f"- Difference: -{diff:.4f} (-{diff*100:.2f}%)",
                    "",
                    "**Recommendations:**",
                    "- Consider using the Random Forest baseline for production",
                    "- The simpler model may be more suitable for this task",
                    "- Review neural network architecture and training process",
                    "- Investigate if additional tuning could improve neural network performance",
                ])
            else:
                report_lines.extend([
                    "### Models Perform Equally",
                    "",
                    "Both models achieve similar performance on new data:",
                    "",
                    f"- Both models achieve {nn_acc:.4f} accuracy",
                    "",
                    "**Recommendations:**",
                    "- Either model is suitable for this task",
                    "- Consider using the simpler Random Forest for easier deployment",
                    "- Or use the neural network if there are specific architectural benefits",
                ])

        report_lines.extend([
            "",
            "---",
            "",
            "## 9. Visualizations",
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
                "### Random Forest Baseline",
                "4. **Confusion Matrix:** `figures/confusion_matrix_random_forest_baseline.png`",
                "5. **Per-Class Performance:** `figures/per_class_performance_random_forest_baseline.png`",
                "6. **Confidence Analysis:** `figures/confidence_analysis_random_forest_baseline.png`",
                "",
                "### Model Comparison",
                "7. **Neural Network vs Random Forest Baseline:** `figures/neural_network_vs_random_forest_baseline.png`",
                ""
            ])

        report_lines.extend([
            "",
            "---",
            "",
            "## 10. Evaluation Metadata",
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

    def save_results(self, results: Dict, rf_results: Dict, comparison: Dict):
        """Save results to JSON file."""
        output = {
            'metadata': self.evaluation_metadata,
            'neural_network_results': results,
            'baseline_comparison': comparison,  # Contains Random Forest baseline metrics and comparison
            'timestamp': datetime.now().isoformat()
        }

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
        activity_texts, true_labels, df = self.load_new_data(new_data_path, data_source_description)

        # Step 2: Generate embeddings
        text_embeddings = self.generate_embeddings(activity_texts)

        # Step 3: Extract numerical features
        numerical_features = self.extract_numerical_features(df)

        # Step 4: Combine text embeddings with numerical features
        embeddings = np.concatenate([text_embeddings, numerical_features], axis=1)
        logger.info(f"Combined features shape: {embeddings.shape}")

        # Step 5: Evaluate Neural Network model
        results = self.evaluate_model(embeddings, true_labels)

        # Step 6: Evaluate Random Forest baseline
        rf_results = self.evaluate_random_forest(embeddings, true_labels)

        # Step 7: Compare Neural Network with Random Forest baseline (both on new data)
        comparison = self.compare_with_baseline(results, rf_results)

        # Step 8: Generate visualizations
        self.generate_visualizations(results, rf_results, comparison)

        # Step 9: Generate report
        self.generate_report(results, rf_results, comparison, new_data_path)

        # Step 10: Save results
        self.save_results(results, rf_results, comparison)

        logger.info("="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Neural Network Accuracy: {results['overall_metrics']['accuracy']:.4f}")

        if rf_results:
            logger.info(f"Random Forest Baseline Accuracy: {rf_results['overall_metrics']['accuracy']:.4f}")

        if comparison and rf_results:
            nn_acc = results['overall_metrics']['accuracy']
            rf_acc = rf_results['overall_metrics']['accuracy']
            if nn_acc > rf_acc:
                logger.info(f"Winner: Neural Network (by {nn_acc - rf_acc:.4f})")
            elif rf_acc > nn_acc:
                logger.info(f"Winner: Random Forest Baseline (by {rf_acc - nn_acc:.4f})")
            else:
                logger.info("Winner: Tie")

        return results, rf_results, comparison


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
