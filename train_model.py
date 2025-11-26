#!/usr/bin/env python3
"""
🚀 Activity Planner - Local Model Training Script
==================================================
Train Sentence-BERT embeddings for activity recommendation using hybrid search.
Includes neural network training with train/validation/test split and batch loss tracking.

This script replaces creating_colab.py and runs locally with adjustable parameters.
"""

import pandas as pd
import numpy as np
import pickle
import json
import os
import logging
import time
import psutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, KFold
from imblearn.over_sampling import SMOTE
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(context: str = ""):
    """Log current memory usage with context"""
    mem_mb = get_memory_usage()
    logger.info(f"  💾 Memory usage{f' ({context})' if context else ''}: {mem_mb:.2f} MB")


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


@dataclass
class TrainingConfig:
    """Configuration for model training with adjustable parameters"""

    # Dataset parameters
    dataset_path: str = 'dataset/dataset_augmented.csv'
    output_dir: str = 'models'

    # Model parameters
    sentence_bert_model: str = 'all-MiniLM-L6-v2'  # Options: all-MiniLM-L6-v2, all-mpnet-base-v2, etc.

    # BM25 parameters
    bm25_k1: float = 1.5  # Term frequency saturation parameter
    bm25_b: float = 0.75  # Document length normalization

    # Embedding parameters
    embedding_batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True

    # FAISS index parameters
    use_gpu: bool = False
    faiss_metric: str = 'L2'  # Options: 'L2' (Euclidean), 'IP' (Inner Product)

    # Hybrid search parameters
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Neural re-ranker weights (adjustable for fine-tuning)
    weight_retrieval: float = 0.4
    weight_age_fit: float = 0.3
    weight_preference: float = 0.3

    def save(self, filepath: str):
        """Save configuration to JSON"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Configuration saved to {filepath}")

    @classmethod
    def load(cls, filepath: str):
        """Load configuration from JSON"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)


class ActivityDataProcessor:
    """Process activity dataset for embedding generation"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.df_activities = None
        self.activity_texts = None

    def load_dataset(self) -> pd.DataFrame:
        """Load and validate dataset with shuffling for data augmentation"""
        logger.info(f"Loading dataset from: {self.config.dataset_path}")

        if not os.path.exists(self.config.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {self.config.dataset_path}")

        self.df_activities = pd.read_csv(self.config.dataset_path)

        # Shuffle the dataset rows for data augmentation
        logger.info(f"✓ Loaded {len(self.df_activities)} activities")
        logger.info("  Shuffling dataset rows for augmentation...")
        self.df_activities = self.df_activities.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info("✓ Dataset shuffled")
        logger.info(f"  Columns: {list(self.df_activities.columns)}")

        return self.df_activities

    def create_text_representations(self) -> List[str]:
        """Create rich text representations for each activity"""
        logger.info("Creating text representations for activities...")

        def create_activity_text(row):
            """Combine all relevant fields with weighted importance"""
            parts = []

            # Title (most important - repeat 3 times for weight)
            if pd.notna(row.get('title')):
                title = str(row['title'])
                parts.extend([title, title, title])

            # Tags and description fields (high importance - repeat 2 times)
            for col in ['tags', 'how_to_play', 'indoor_outdoor', 'season']:
                if col in row.index and pd.notna(row[col]):
                    value = str(row[col])
                    parts.extend([value, value])

            # Other metadata (normal importance)
            for col in row.index:
                if col not in ['title', 'tags', 'how_to_play', 'indoor_outdoor', 'season']:
                    if pd.notna(row[col]):
                        parts.append(str(row[col]))

            return ' '.join(parts)

        self.activity_texts = self.df_activities.apply(create_activity_text, axis=1).tolist()
        logger.info(f"✓ Created {len(self.activity_texts)} text representations")
        logger.info(f"  Sample (first 150 chars): {self.activity_texts[0][:150]}...")

        return self.activity_texts


class BM25Indexer:
    """Build and save BM25 keyword search index"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenized_docs = None
        self.bm25 = None

    def build_index(self, documents: List[str]):
        """Build BM25 index with custom parameters"""
        logger.info(f"Building BM25 index (k1={self.config.bm25_k1}, b={self.config.bm25_b})...")

        # Tokenize documents
        self.tokenized_docs = [doc.lower().split() for doc in tqdm(
            documents,
            desc="Tokenizing",
            disable=not self.config.show_progress
        )]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_docs,
            k1=self.config.bm25_k1,
            b=self.config.bm25_b
        )

        logger.info(f"✓ BM25 index created with {len(self.tokenized_docs)} documents")

    def save(self, output_dir: str):
        """Save tokenized documents for BM25"""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, 'bm25_docs.pkl')

        with open(filepath, 'wb') as f:
            pickle.dump(self.tokenized_docs, f)

        file_size = os.path.getsize(filepath) / 1024
        logger.info(f"✓ BM25 index saved to {filepath} ({file_size:.2f} KB)")

    def test_retrieval(self, query: str, top_k: int = 5) -> List[tuple]:
        """Test BM25 retrieval"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]


class DenseEmbedder:
    """Generate and save Sentence-BERT embeddings with FAISS index"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.embeddings = None
        self.faiss_index = None

    def load_model(self):
        """Load Sentence-BERT model"""
        logger.info(f"Loading Sentence-BERT model: {self.config.sentence_bert_model}")
        self.model = SentenceTransformer(self.config.sentence_bert_model)
        logger.info(f"✓ Model loaded (embedding dimension: {self.model.get_sentence_embedding_dimension()})")

    def generate_embeddings(self, documents: List[str]):
        """Generate embeddings for all documents"""
        logger.info(f"Generating embeddings for {len(documents)} documents...")

        self.embeddings = self.model.encode(
            documents,
            batch_size=self.config.embedding_batch_size,
            show_progress_bar=self.config.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings
        )

        logger.info(f"✓ Embeddings generated")
        logger.info(f"  Shape: {self.embeddings.shape}")
        logger.info(f"  Memory: {self.embeddings.nbytes / 1024 / 1024:.2f} MB")

        return self.embeddings

    def build_faiss_index(self):
        """Build FAISS index for fast similarity search"""
        logger.info(f"Building FAISS index (metric: {self.config.faiss_metric})...")

        dimension = self.embeddings.shape[1]

        # Choose index type based on metric
        if self.config.faiss_metric == 'L2':
            self.faiss_index = faiss.IndexFlatL2(dimension)
        elif self.config.faiss_metric == 'IP':
            self.faiss_index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown FAISS metric: {self.config.faiss_metric}")

        # Add embeddings to index
        self.faiss_index.add(self.embeddings.astype('float32'))

        logger.info(f"✓ FAISS index created")
        logger.info(f"  Total vectors: {self.faiss_index.ntotal}")
        logger.info(f"  Dimension: {dimension}")

    def save(self, output_dir: str):
        """Save embeddings and FAISS index"""
        os.makedirs(output_dir, exist_ok=True)

        # Save embeddings
        emb_path = os.path.join(output_dir, 'embeddings.npy')
        np.save(emb_path, self.embeddings)
        emb_size = os.path.getsize(emb_path) / 1024 / 1024
        logger.info(f"✓ Embeddings saved to {emb_path} ({emb_size:.2f} MB)")

        # Save FAISS index
        faiss_path = os.path.join(output_dir, 'faiss_index.bin')
        faiss.write_index(self.faiss_index, faiss_path)
        faiss_size = os.path.getsize(faiss_path) / 1024 / 1024
        logger.info(f"✓ FAISS index saved to {faiss_path} ({faiss_size:.2f} MB)")

    def test_retrieval(self, query: str, top_k: int = 5) -> List[tuple]:
        """Test dense retrieval"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)

        # Convert distances to similarity scores
        if self.config.faiss_metric == 'L2':
            scores = 1.0 / (1.0 + distances[0])
        else:  # IP
            scores = distances[0]

        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]


class ActivityDataset(Dataset):
    """PyTorch Dataset for activity classification"""

    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class ActivityClassifier(nn.Module):
    """Neural network for activity classification/ranking with heavy regularization"""

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.5):
        super(ActivityClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Input dropout for robustness (moderate for small dataset)
        layers.append(nn.Dropout(dropout * 0.5))  # 0.15 with dropout=0.3

        # Build hidden layers with aggressive dropout
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            # Consistent dropout across layers for small dataset
            layers.append(nn.Dropout(dropout))  # Consistent 0.3 dropout
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class NeuralTrainer:
    """Train neural network classifier on activity data"""

    def __init__(self, config: 'TrainingConfig'):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu else 'cpu')
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.train_losses = []
        self.val_losses = []
        logger.info(f"Neural trainer initialized on device: {self.device}")

    def balance_with_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Balance dataset using SMOTE to address class imbalance

        Args:
            X: Feature embeddings
            y: Labels

        Returns:
            Tuple of balanced (X, y)
        """
        logger.info("\n[Data Balancing] Applying SMOTE to address class imbalance...")

        # Show original distribution
        unique, counts = np.unique(y, return_counts=True)
        label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']

        logger.info("  Original distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"    {label_names[label]}: {count} ({count/len(y)*100:.1f}%)")

        # Determine k_neighbors based on smallest class
        min_samples = min(counts)
        k_neighbors = min(5, min_samples - 1)  # Ensure we don't exceed available neighbors

        if k_neighbors < 1:
            logger.warning(f"  ⚠ Smallest class has only {min_samples} samples. SMOTE requires at least 2.")
            logger.warning("  Skipping SMOTE balancing. Consider collecting more data.")
            return X, y

        logger.info(f"  Using k_neighbors={k_neighbors} (based on smallest class size: {min_samples})")

        try:
            # Apply SMOTE with automatic balancing
            smote = SMOTE(
                sampling_strategy='auto',  # Balance all minority classes to match majority
                random_state=42,
                k_neighbors=k_neighbors
            )
            X_balanced, y_balanced = smote.fit_resample(X, y)

            # Show new distribution
            unique, counts = np.unique(y_balanced, return_counts=True)
            logger.info("\n  After SMOTE:")
            for label, count in zip(unique, counts):
                logger.info(f"    {label_names[label]}: {count} ({count/len(y_balanced)*100:.1f}%)")

            logger.info(f"\n✓ SMOTE complete: {len(y)} → {len(y_balanced)} samples")
            logger.info(f"  Increase: +{len(y_balanced) - len(y)} synthetic samples ({(len(y_balanced)/len(y) - 1)*100:.1f}%)\n")

            return X_balanced, y_balanced

        except Exception as e:
            logger.error(f"  ✗ SMOTE failed: {e}")
            logger.warning("  Continuing with original unbalanced data")
            return X, y

    def prepare_data(self, embeddings: np.ndarray, df_activities: pd.DataFrame):
        """Split data into train/validation/test sets with balanced age group distribution"""
        logger.info("\n[Neural Network] Preparing train/validation/test split with balanced distribution...")

        # Create labels based on age groups using age midpoint for better accuracy
        # Toddler (0-3): 0, Preschool (4-6): 1, Elementary (7-10): 2, Teen+ (11+): 3
        labels = []
        label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']

        for idx, row in df_activities.iterrows():
            # Use midpoint of age range for more accurate classification
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

        # Display distribution in full dataset
        logger.info("\n  Age Group Distribution in Full Dataset:")
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"    {label_names[label]}: {count} ({count/len(labels)*100:.1f}%)")

        # Stratified split to ensure balanced distribution across all sets
        # First split: separate test set (10%)
        X_temp, X_test, y_temp, y_test = train_test_split(
            embeddings, labels, test_size=0.10, random_state=42, stratify=labels
        )

        # Second split: separate train and validation (80% train, 10% val from remaining)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.111, random_state=42, stratify=y_temp  # 0.111 * 0.90 ≈ 0.10
        )

        # Apply SMOTE to training set only to balance classes
        # This prevents data leakage from validation/test sets
        X_train, y_train = self.balance_with_smote(X_train, y_train)

        # Display distribution in each split
        logger.info(f"\n✓ Train set: {len(X_train)} samples ({len(X_train)/len(embeddings)*100:.1f}%)")
        unique, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"    {label_names[label]}: {count} ({count/len(y_train)*100:.1f}%)")

        logger.info(f"\n✓ Validation set: {len(X_val)} samples ({len(X_val)/len(embeddings)*100:.1f}%)")
        unique, counts = np.unique(y_val, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"    {label_names[label]}: {count} ({count/len(y_val)*100:.1f}%)")

        logger.info(f"\n✓ Test set: {len(X_test)} samples ({len(X_test)/len(embeddings)*100:.1f}%)")
        unique, counts = np.unique(y_test, return_counts=True)
        for label, count in zip(unique, counts):
            logger.info(f"    {label_names[label]}: {count} ({count/len(y_test)*100:.1f}%)")

        # Create datasets
        train_dataset = ActivityDataset(X_train, y_train)
        val_dataset = ActivityDataset(X_val, y_val)
        test_dataset = ActivityDataset(X_test, y_test)

        # Create data loaders with shuffling to ensure each epoch sees varied order
        # but maintains the same distribution due to stratified split
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        logger.info("\n✓ Balanced distribution ensured: Each epoch will see the same proportion of age groups")

        # Store class counts for weighted loss calculation
        self.class_counts = counts.copy()
        self.num_classes = len(np.unique(labels))

        return len(np.unique(labels))

    def build_model(self, input_dim: int, num_classes: int):
        """Build neural network model with heavy regularization"""
        logger.info(f"\n[Neural Network] Building model...")
        logger.info(f"  Input dimension: {input_dim}")
        logger.info(f"  Number of classes: {num_classes}")

        # Improved architecture for SMOTE-balanced dataset
        # Using 3 hidden layers with reduced dropout for better learning
        hidden_dims = [256, 128, 64]  # Increased capacity for balanced data
        dropout_rate = 0.2       # Reduced from 0.3 for better minority class learning
        self.model = ActivityClassifier(input_dim, hidden_dims, num_classes, dropout=dropout_rate)
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"✓ Model built with {trainable_params:,} trainable parameters")
        logger.info(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
        logger.info(f"  Dropout: {dropout_rate} (input: {dropout_rate * 0.5}, hidden: {dropout_rate})")
        logger.info(f"  Regularization: L2 weight decay (5e-5), inverse frequency class weighting, early stopping")
        logger.info(f"  Optimized for SMOTE-balanced dataset with improved capacity")

    def train_epoch(self, optimizer, criterion, epoch: int, num_epochs: int):
        """Train for one epoch with batch-wise loss display"""
        self.model.train()
        epoch_loss = 0.0
        batch_losses = []

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Track loss
            batch_loss = loss.item()
            batch_losses.append(batch_loss)
            epoch_loss += batch_loss

            # Update progress bar with current batch loss
            progress_bar.set_postfix({
                'batch_loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })

        avg_epoch_loss = epoch_loss / len(self.train_loader)
        self.train_losses.append(avg_epoch_loss)

        return avg_epoch_loss, batch_losses

    def validate(self, criterion):
        """Validate model on validation set"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = 100 * correct / total
        self.val_losses.append(avg_val_loss)

        return avg_val_loss, accuracy

    def test(self, criterion):
        """Evaluate model on test set"""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_test_loss = test_loss / len(self.test_loader)
        accuracy = 100 * correct / total

        return avg_test_loss, accuracy

    def train(self, num_epochs: int = 100, learning_rate: float = 0.001, patience: int = 15):
        """Full training loop with train/validation/test evaluation and early stopping"""
        train_start_time = time.time()

        logger.info(f"\n[Neural Network] Starting training for {num_epochs} epochs...")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  L2 Regularization (weight_decay): 5e-5 (optimized for small dataset)")
        logger.info(f"  Early stopping patience: {patience} epochs")
        logger.info(f"  Device: {self.device}")
        log_memory_usage("before training")
        logger.info("="*70)

        # Use stronger class weights with SMOTE-balanced data
        # With SMOTE balancing training data, we still weight by original class counts
        # to ensure the model respects true class importance
        class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * self.num_classes  # Normalize
        class_weights = class_weights.to(self.device)

        logger.info(f"\n  Class Weights (inverse frequency for SMOTE-balanced data):")
        label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
        for i, weight in enumerate(class_weights):
            logger.info(f"    {label_names[i]}: {weight:.4f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # Moderate weight_decay optimized for small dataset (1800 samples)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=5e-5)

        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            # Train
            train_loss, batch_losses = self.train_epoch(optimizer, criterion, epoch, num_epochs)

            # Validate
            val_loss, val_acc = self.validate(criterion)

            # Log epoch summary
            logger.info(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Val Loss: {val_loss:.4f}")
            logger.info(f"  Val Accuracy: {val_acc:.2f}%")
            logger.info(f"  Gap (Train-Val Loss): {abs(train_loss - val_loss):.4f}")

            # Early stopping and model checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                epochs_without_improvement = 0
                logger.info(f"  ✓ New best validation loss! Model checkpoint saved.")
            else:
                epochs_without_improvement += 1
                logger.info(f"  No improvement for {epochs_without_improvement} epoch(s)")

            # Early stopping check
            if epochs_without_improvement >= patience:
                logger.info(f"\n⚠ Early stopping triggered after {epoch+1} epochs (patience={patience})")
                logger.info(f"  Best validation loss: {best_val_loss:.4f}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            logger.info("\n✓ Restored best model from checkpoint")

        train_total_time = time.time() - train_start_time

        logger.info("\n" + "="*70)
        logger.info("[Neural Network] Training complete!")
        logger.info("="*70)
        logger.info(f"⏱ Training Time: {format_time(train_total_time)}")
        log_memory_usage("after training")

        # Final test evaluation
        logger.info("\n[Neural Network] Evaluating on test set...")
        test_loss, test_acc = self.test(criterion)
        logger.info(f"✓ Test Loss: {test_loss:.4f}")
        logger.info(f"✓ Test Accuracy: {test_acc:.2f}%")

        return test_loss, test_acc

    def cross_validate(self, embeddings: np.ndarray, labels: np.ndarray, k_folds: int = 5,
                      num_epochs: int = 100, learning_rate: float = 0.001):
        """Perform K-fold cross-validation to evaluate model generalization"""
        logger.info(f"\n[K-Fold Cross-Validation] Starting {k_folds}-fold cross-validation...")
        logger.info(f"  Dataset size: {len(embeddings)}")
        logger.info(f"  Number of folds: {k_folds}")
        logger.info(f"  Max epochs per fold: {num_epochs}")
        logger.info("="*70)

        cv_start_time = time.time()
        log_memory_usage("before cross-validation")

        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_results = []

        label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(embeddings)):
            logger.info(f"\n{'='*70}")
            logger.info(f"Fold {fold_idx + 1}/{k_folds}")
            logger.info(f"{'='*70}")

            # Split data
            X_train_fold = embeddings[train_idx]
            y_train_fold = labels[train_idx]
            X_val_fold = embeddings[val_idx]
            y_val_fold = labels[val_idx]

            logger.info(f"  Train size: {len(X_train_fold)} | Validation size: {len(X_val_fold)}")

            # Create datasets
            train_dataset = ActivityDataset(X_train_fold, y_train_fold)
            val_dataset = ActivityDataset(X_val_fold, y_val_fold)

            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

            # Rebuild model for this fold (using updated architecture)
            num_classes = len(np.unique(labels))
            input_dim = embeddings.shape[1]
            hidden_dims = [256, 128, 64]  # Match main training architecture

            model = ActivityClassifier(input_dim, hidden_dims, num_classes, dropout=0.2)
            model.to(self.device)

            # Calculate stronger class weights for this fold (inverse frequency)
            unique, counts = np.unique(y_train_fold, return_counts=True)
            class_weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
            class_weights = class_weights / class_weights.sum() * num_classes
            class_weights = class_weights.to(self.device)

            criterion = nn.CrossEntropyLoss(weight=class_weights)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)

            # Train this fold with early stopping
            best_val_loss = float('inf')
            patience = 10
            epochs_without_improvement = 0

            fold_start_time = time.time()

            # Create progress bar for epochs
            epoch_pbar = tqdm(range(num_epochs), desc=f"  Training Fold {fold_idx + 1}",
                            unit="epoch", leave=False, position=0)

            for epoch in epoch_pbar:
                # Train
                model.train()
                train_loss = 0.0
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                avg_train_loss = train_loss / len(train_loader)

                # Validate
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1)
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()

                avg_val_loss = val_loss / len(val_loader)
                val_acc = 100 * correct / total

                # Update progress bar
                epoch_pbar.set_postfix({
                    'train_loss': f'{avg_train_loss:.4f}',
                    'val_loss': f'{avg_val_loss:.4f}',
                    'val_acc': f'{val_acc:.1f}%',
                    'no_improve': epochs_without_improvement
                })

                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_acc = val_acc
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    epoch_pbar.close()
                    logger.info(f"  ⚠ Early stopping at epoch {epoch+1} (patience reached)")
                    break

            epoch_pbar.close()
            fold_time = time.time() - fold_start_time
            logger.info(f"  ⏱ Fold {fold_idx + 1} completed in {format_time(fold_time)}")

            logger.info(f"\n  Fold {fold_idx + 1} Results:")
            logger.info(f"    Best Val Loss: {best_val_loss:.4f}")
            logger.info(f"    Best Val Accuracy: {best_val_acc:.2f}%")

            fold_results.append({
                'fold': fold_idx + 1,
                'val_loss': best_val_loss,
                'val_accuracy': best_val_acc
            })

        # Summary statistics
        cv_total_time = time.time() - cv_start_time

        logger.info("\n" + "="*70)
        logger.info("[Cross-Validation Summary]")
        logger.info("="*70)

        avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
        std_val_loss = np.std([r['val_loss'] for r in fold_results])
        avg_val_acc = np.mean([r['val_accuracy'] for r in fold_results])
        std_val_acc = np.std([r['val_accuracy'] for r in fold_results])

        logger.info(f"  Average Validation Loss: {avg_val_loss:.4f} ± {std_val_loss:.4f}")
        logger.info(f"  Average Validation Accuracy: {avg_val_acc:.2f}% ± {std_val_acc:.2f}%")
        logger.info(f"  ⏱ Total CV Time: {format_time(cv_total_time)}")
        log_memory_usage("after cross-validation")
        logger.info("\n  Per-Fold Results:")
        for result in fold_results:
            logger.info(f"    Fold {result['fold']}: Loss={result['val_loss']:.4f}, Acc={result['val_accuracy']:.2f}%")

        return fold_results, avg_val_loss, avg_val_acc

    def save(self, output_dir: str):
        """Save trained model and training history"""
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(output_dir, 'neural_classifier.pth')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, model_path)

        logger.info(f"✓ Neural model saved to {model_path}")

        # Save training history
        history_path = os.path.join(output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump({
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, f, indent=2)

        logger.info(f"✓ Training history saved to {history_path}")


class ModelTrainer:
    """Main training pipeline orchestrator"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_processor = ActivityDataProcessor(config)
        self.bm25_indexer = BM25Indexer(config)
        self.dense_embedder = DenseEmbedder(config)
        self.neural_trainer = NeuralTrainer(config)
        self.df_activities = None
        self.activity_texts = None

    def train(self):
        """Execute full training pipeline"""
        pipeline_start_time = time.time()

        logger.info("="*70)
        logger.info("🚀 ACTIVITY PLANNER - MODEL TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_memory_usage("start")

        # Step 1: Load and process data
        logger.info("\n[1/8] Loading dataset...")
        step_start = time.time()
        self.df_activities = self.data_processor.load_dataset()
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")
        log_memory_usage()

        logger.info("\n[2/8] Creating text representations...")
        step_start = time.time()
        self.activity_texts = self.data_processor.create_text_representations()
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")
        log_memory_usage()

        # Step 2: Build BM25 index
        logger.info("\n[3/8] Building BM25 keyword index...")
        step_start = time.time()
        self.bm25_indexer.build_index(self.activity_texts)
        self.bm25_indexer.save(self.config.output_dir)
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")
        log_memory_usage()

        # Step 3: Generate embeddings and build FAISS
        logger.info("\n[4/8] Generating Sentence-BERT embeddings...")
        step_start = time.time()
        self.dense_embedder.load_model()
        self.dense_embedder.generate_embeddings(self.activity_texts)
        self.dense_embedder.build_faiss_index()
        self.dense_embedder.save(self.config.output_dir)
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")
        log_memory_usage()

        # Step 4: Save dataset and config
        logger.info("\n[5/8] Saving dataset and configuration...")
        step_start = time.time()
        self._save_dataset()
        self.config.save(os.path.join(self.config.output_dir, 'training_config.json'))
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")

        # Step 5: Perform K-fold cross-validation for model evaluation
        logger.info("\n[6/8] Running K-fold cross-validation...")
        step_start = time.time()

        # Create labels for cross-validation
        labels = []
        for idx, row in self.df_activities.iterrows():
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

        # Run cross-validation
        cv_results, avg_cv_loss, avg_cv_acc = self.neural_trainer.cross_validate(
            self.dense_embedder.embeddings,
            labels,
            k_folds=5,
            num_epochs=100,
            learning_rate=0.001
        )
        logger.info(f"  ⏱ Cross-validation completed in {format_time(time.time() - step_start)}")

        # Step 6: Train final neural network on full train/val/test split
        logger.info("\n[7/8] Training final neural network classifier...")
        step_start = time.time()
        num_classes = self.neural_trainer.prepare_data(self.dense_embedder.embeddings, self.df_activities)
        self.neural_trainer.build_model(
            input_dim=self.dense_embedder.embeddings.shape[1],
            num_classes=num_classes
        )
        test_loss, test_acc = self.neural_trainer.train(num_epochs=100, learning_rate=0.001, patience=15)
        self.neural_trainer.save(self.config.output_dir)
        logger.info(f"  ⏱ Final training completed in {format_time(time.time() - step_start)}")
        log_memory_usage()

        # Step 7: Test pipeline
        logger.info("\n[8/8] Testing hybrid retrieval pipeline...")
        step_start = time.time()
        self._test_pipeline()
        logger.info(f"  ⏱ Step completed in {format_time(time.time() - step_start)}")

        pipeline_total_time = time.time() - pipeline_start_time

        logger.info("\n" + "="*70)
        logger.info("✅ MODEL TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱ Total Pipeline Time: {format_time(pipeline_total_time)}")
        log_memory_usage("final")
        logger.info(f"\nModels saved in: {self.config.output_dir}/")
        logger.info("\nGenerated files:")
        logger.info("  ✓ bm25_docs.pkl - BM25 keyword index")
        logger.info("  ✓ embeddings.npy - Sentence-BERT embeddings")
        logger.info("  ✓ faiss_index.bin - FAISS similarity index")
        logger.info("  ✓ neural_classifier.pth - Neural network classifier")
        logger.info("  ✓ training_history.json - Training loss history")
        logger.info("  ✓ activities_processed.csv - Processed dataset")
        logger.info("  ✓ training_config.json - Training parameters")
        logger.info(f"\nCross-Validation Results ({len(cv_results)}-Fold):")
        logger.info(f"  Average CV Loss: {avg_cv_loss:.4f}")
        logger.info(f"  Average CV Accuracy: {avg_cv_acc:.2f}%")
        logger.info(f"\nFinal Test Set Performance:")
        logger.info(f"  Test Loss: {test_loss:.4f}")
        logger.info(f"  Test Accuracy: {test_acc:.2f}%")
        logger.info("\nNext steps:")
        logger.info("  1. Run: python app_optimized.py")
        logger.info("  2. Open: http://localhost:5000")
        logger.info("  3. Start searching for activities! 🎉")

    def _save_dataset(self):
        """Save processed dataset"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        output_path = os.path.join(self.config.output_dir, 'activities_processed.csv')
        self.df_activities.to_csv(output_path, index=False)

        file_size = os.path.getsize(output_path) / 1024
        logger.info(f"✓ Dataset saved to {output_path} ({file_size:.2f} KB)")

    def _test_pipeline(self):
        """Test the complete hybrid search pipeline"""
        test_query = "fun outdoor activities for kids"
        logger.info(f"Test query: '{test_query}'")

        # Test BM25
        bm25_results = self.bm25_indexer.test_retrieval(test_query, top_k=5)
        logger.info("\nBM25 Results (Keyword Search):")
        for rank, (idx, score) in enumerate(bm25_results, 1):
            title = self.df_activities.iloc[idx].get('title', f'Activity {idx}')
            logger.info(f"  {rank}. {title} (score: {score:.4f})")

        # Test Dense
        dense_results = self.dense_embedder.test_retrieval(test_query, top_k=5)
        logger.info("\nDense Results (Semantic Search):")
        for rank, (idx, score) in enumerate(dense_results, 1):
            title = self.df_activities.iloc[idx].get('title', f'Activity {idx}')
            logger.info(f"  {rank}. {title} (score: {score:.4f})")

        # Test RRF fusion
        logger.info("\nHybrid Results (RRF Fusion):")
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)
        for rank, (idx, score) in enumerate(fused[:5], 1):
            title = self.df_activities.iloc[idx].get('title', f'Activity {idx}')
            logger.info(f"  {rank}. {title} (score: {score:.4f})")

    def _reciprocal_rank_fusion(self, sparse_results, dense_results):
        """Merge BM25 and dense results using RRF"""
        rrf_scores = {}

        for rank, (idx, _) in enumerate(sparse_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.config.rrf_k + rank)

        for rank, (idx, _) in enumerate(dense_results, 1):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.config.rrf_k + rank)

        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description="Train Activity Planner models with Sentence-BERT embeddings"
    )

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='dataset/dataset_augmented.csv',
                        help='Path to activity dataset CSV')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')

    # Model arguments
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-BERT model name')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for embedding generation')

    # BM25 arguments
    parser.add_argument('--bm25-k1', type=float, default=1.5,
                        help='BM25 k1 parameter (term frequency saturation)')
    parser.add_argument('--bm25-b', type=float, default=0.75,
                        help='BM25 b parameter (document length normalization)')

    # FAISS arguments
    parser.add_argument('--faiss-metric', type=str, default='L2', choices=['L2', 'IP'],
                        help='FAISS distance metric')

    # RRF arguments
    parser.add_argument('--rrf-k', type=int, default=60,
                        help='Reciprocal Rank Fusion constant')

    # Neural re-ranker weights
    parser.add_argument('--weight-retrieval', type=float, default=0.4,
                        help='Weight for retrieval score in re-ranking')
    parser.add_argument('--weight-age', type=float, default=0.3,
                        help='Weight for age fit in re-ranking')
    parser.add_argument('--weight-preference', type=float, default=0.3,
                        help='Weight for preference match in re-ranking')

    # Other
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable progress bars')

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        sentence_bert_model=args.model,
        embedding_batch_size=args.batch_size,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        faiss_metric=args.faiss_metric,
        rrf_k=args.rrf_k,
        weight_retrieval=args.weight_retrieval,
        weight_age_fit=args.weight_age,
        weight_preference=args.weight_preference,
        show_progress=not args.no_progress
    )

    # Train models
    trainer = ModelTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
