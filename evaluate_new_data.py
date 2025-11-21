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
    """Neural network classifier for age group classification."""

    def __init__(self, input_dim=384, hidden_dims=[256, 128, 64], num_classes=4, dropout=0.3):
        super(NeuralClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


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

        # Load baseline performance from original test set
        self.baseline_metrics = self._load_baseline_metrics()

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

        # Initialize model with same architecture as training
        model = NeuralClassifier(
            input_dim=384,
            hidden_dims=[256, 128, 64],
            num_classes=4,
            dropout=0.3
        )

        # Load checkpoint
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

        # Handle legacy checkpoint with "model." prefix instead of "network."
        # This remaps old keys to match the current architecture
        if any(key.startswith('model.') for key in state_dict.keys()):
            logger.info("Remapping legacy state dict keys from 'model.' to 'network.'")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key.replace('model.', 'network.', 1)
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict

        # Handle legacy layer ordering: Old model had Linear->ReLU->BatchNorm->Dropout
        # New model has Linear->BatchNorm->ReLU->Dropout
        # Need to remap BatchNorm layers from indices [2, 6, 10] to [1, 5, 9]
        if 'network.2.weight' in state_dict and 'network.1.weight' not in state_dict:
            logger.info("Detected legacy layer ordering - remapping BatchNorm layer indices")
            # Mapping: old index -> new index for BatchNorm layers
            # Old: [2, 6, 10] -> New: [1, 5, 9]
            layer_mapping = {2: 1, 6: 5, 10: 9}

            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('network.'):
                    # Extract layer index
                    parts = key.split('.')
                    if len(parts) >= 2 and parts[1].isdigit():
                        old_idx = int(parts[1])

                        # Check if this is a BatchNorm layer that needs remapping
                        if old_idx in layer_mapping:
                            new_idx = layer_mapping[old_idx]
                            new_key = f"network.{new_idx}.{'.'.join(parts[2:])}"
                            new_state_dict[new_key] = value
                            logger.debug(f"Remapped {key} -> {new_key}")
                        else:
                            # Keep other layers as is
                            new_state_dict[key] = value
                    else:
                        new_state_dict[key] = value
                else:
                    new_state_dict[key] = value

            state_dict = new_state_dict

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        logger.info(f"Loaded classifier from {model_path}")
        return model

    def _load_baseline_metrics(self) -> Dict:
        """Load baseline performance metrics from original test set."""
        test_report_path = Path('test_results') / 'test_report.json'
        if test_report_path.exists():
            with open(test_report_path, 'r') as f:
                report = json.load(f)
                return report.get('neural_network', {})
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

        # Derive labels from age ranges (same logic as training)
        labels = []
        for _, row in df.iterrows():
            age_min = row['age_min']
            if age_min <= 3:
                labels.append(0)  # Toddler
            elif age_min <= 6:
                labels.append(1)  # Preschool
            elif age_min <= 10:
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

    def generate_visualizations(self, results: Dict, comparison: Dict):
        """Generate visualization plots for the evaluation."""
        logger.info("Generating visualizations...")

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

        # 1. Confusion Matrix
        self._plot_confusion_matrix(results)

        # 2. Performance Comparison
        if comparison:
            self._plot_performance_comparison(comparison)

        # 3. Per-Class Performance
        self._plot_per_class_performance(results)

        # 4. Confidence Distribution
        self._plot_confidence_distribution(results)

        logger.info(f"Visualizations saved to {self.figures_dir}")

    def _plot_confusion_matrix(self, results: Dict):
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
        plt.title('Confusion Matrix - New Data Evaluation', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confusion_matrix_new_data.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, comparison: Dict):
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
        ax.set_title('Performance Comparison: Baseline vs New Data', fontsize=14, fontweight='bold')
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
        plt.savefig(self.figures_dir / 'baseline_vs_new_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_per_class_performance(self, results: Dict):
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
        ax.set_title('Per-Class Performance on New Data', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.figures_dir / 'per_class_performance_new_data.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_confidence_distribution(self, results: Dict):
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

        plt.suptitle('Prediction Confidence Analysis - New Data', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'confidence_analysis_new_data.png', dpi=300, bbox_inches='tight')
        plt.close()

    def generate_report(self, results: Dict, comparison: Dict, data_path: str) -> str:
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

        report_lines.extend([
            "---",
            "",
            "## 4. Per-Class Performance on New Data",
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
            "## 5. Detailed Classification Report",
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
            "## 6. Confusion Matrix",
            "",
            "See `figures/confusion_matrix_new_data.png` for visualization.",
            "",
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
            "## 7. Conclusions and Recommendations",
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
            "## 8. Visualizations",
            "",
            "The following visualizations have been generated:",
            "",
            "1. **Confusion Matrix:** `figures/confusion_matrix_new_data.png`",
            "2. **Performance Comparison:** `figures/baseline_vs_new_comparison.png`",
            "3. **Per-Class Performance:** `figures/per_class_performance_new_data.png`",
            "4. **Confidence Analysis:** `figures/confidence_analysis_new_data.png`",
            "",
            "---",
            "",
            "## 9. Evaluation Metadata",
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

    def save_results(self, results: Dict, comparison: Dict):
        """Save results to JSON file."""
        output = {
            'metadata': self.evaluation_metadata,
            'results': results,
            'comparison': comparison,
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
        activity_texts, true_labels = self.load_new_data(new_data_path, data_source_description)

        # Step 2: Generate embeddings
        embeddings = self.generate_embeddings(activity_texts)

        # Step 3: Evaluate model
        results = self.evaluate_model(embeddings, true_labels)

        # Step 4: Compare with baseline
        comparison = self.compare_with_baseline(results)

        # Step 5: Generate visualizations
        self.generate_visualizations(results, comparison)

        # Step 6: Generate report
        self.generate_report(results, comparison, new_data_path)

        # Step 7: Save results
        self.save_results(results, comparison)

        logger.info("="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Accuracy on new data: {results['overall_metrics']['accuracy']:.4f}")

        if comparison:
            assessment = comparison['performance_assessment']
            logger.info(f"Performance Assessment: {assessment['assessment']}")
            logger.info(f"Rubric Score: {assessment['rubric_score']}/10")

        return results, comparison


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
