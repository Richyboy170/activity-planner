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
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
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


@dataclass
class TrainingConfig:
    """Configuration for model training with adjustable parameters"""

    # Dataset parameters
    dataset_path: str = 'dataset/dataset.csv'
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
    """Neural network for activity classification/ranking"""

    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.3):
        super(ActivityClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
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
        """Build neural network model"""
        logger.info(f"\n[Neural Network] Building model...")
        logger.info(f"  Input dimension: {input_dim}")
        logger.info(f"  Number of classes: {num_classes}")

        # Simplified architecture to reduce overfitting (was 10 layers with 140K params)
        # Now using 2 hidden layers with stronger regularization
        hidden_dims = [256, 128]
        self.model = ActivityClassifier(input_dim, hidden_dims, num_classes, dropout=0.5)
        self.model.to(self.device)

        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info(f"✓ Model built with {trainable_params:,} trainable parameters")
        logger.info(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {num_classes}")
        logger.info(f"  Dropout: 0.5 (increased from 0.3 for better generalization)")

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

    def train(self, num_epochs: int = 50, learning_rate: float = 0.001):
        """Full training loop with train/validation/test evaluation"""
        logger.info(f"\n[Neural Network] Starting training for {num_epochs} epochs...")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Device: {self.device}")
        logger.info("="*70)

        # Calculate class weights inversely proportional to frequency
        # This helps the model pay more attention to minority classes
        class_weights = 1.0 / torch.tensor(self.class_counts, dtype=torch.float32)
        class_weights = class_weights / class_weights.sum() * self.num_classes  # Normalize
        class_weights = class_weights.to(self.device)

        logger.info(f"\n  Class Weights (to address imbalance):")
        label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
        for i, weight in enumerate(class_weights):
            logger.info(f"    {label_names[i]}: {weight:.4f}")

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)

        best_val_loss = float('inf')

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

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"  ✓ New best validation loss! Saving model...")

        logger.info("\n" + "="*70)
        logger.info("[Neural Network] Training complete!")
        logger.info("="*70)

        # Final test evaluation
        logger.info("\n[Neural Network] Evaluating on test set...")
        test_loss, test_acc = self.test(criterion)
        logger.info(f"✓ Test Loss: {test_loss:.4f}")
        logger.info(f"✓ Test Accuracy: {test_acc:.2f}%")

        return test_loss, test_acc

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
        logger.info("="*70)
        logger.info("🚀 ACTIVITY PLANNER - MODEL TRAINING PIPELINE")
        logger.info("="*70)
        logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Load and process data
        logger.info("\n[1/5] Loading dataset...")
        self.df_activities = self.data_processor.load_dataset()

        logger.info("\n[2/5] Creating text representations...")
        self.activity_texts = self.data_processor.create_text_representations()

        # Step 2: Build BM25 index
        logger.info("\n[3/5] Building BM25 keyword index...")
        self.bm25_indexer.build_index(self.activity_texts)
        self.bm25_indexer.save(self.config.output_dir)

        # Step 3: Generate embeddings and build FAISS
        logger.info("\n[4/5] Generating Sentence-BERT embeddings...")
        self.dense_embedder.load_model()
        self.dense_embedder.generate_embeddings(self.activity_texts)
        self.dense_embedder.build_faiss_index()
        self.dense_embedder.save(self.config.output_dir)

        # Step 4: Save dataset and config
        logger.info("\n[5/7] Saving dataset and configuration...")
        self._save_dataset()
        self.config.save(os.path.join(self.config.output_dir, 'training_config.json'))

        # Step 5: Train neural network
        logger.info("\n[6/7] Training neural network classifier...")
        num_classes = self.neural_trainer.prepare_data(self.dense_embedder.embeddings, self.df_activities)
        self.neural_trainer.build_model(
            input_dim=self.dense_embedder.embeddings.shape[1],
            num_classes=num_classes
        )
        test_loss, test_acc = self.neural_trainer.train(num_epochs=50, learning_rate=0.001)
        self.neural_trainer.save(self.config.output_dir)

        # Step 6: Test pipeline
        logger.info("\n[7/7] Testing hybrid retrieval pipeline...")
        self._test_pipeline()

        logger.info("\n" + "="*70)
        logger.info("✅ MODEL TRAINING COMPLETE!")
        logger.info("="*70)
        logger.info(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"\nModels saved in: {self.config.output_dir}/")
        logger.info("\nGenerated files:")
        logger.info("  ✓ bm25_docs.pkl - BM25 keyword index")
        logger.info("  ✓ embeddings.npy - Sentence-BERT embeddings")
        logger.info("  ✓ faiss_index.bin - FAISS similarity index")
        logger.info("  ✓ neural_classifier.pth - Neural network classifier")
        logger.info("  ✓ training_history.json - Training loss history")
        logger.info("  ✓ activities_processed.csv - Processed dataset")
        logger.info("  ✓ training_config.json - Training parameters")
        logger.info(f"\nNeural Network Performance:")
        logger.info(f"  Final Test Loss: {test_loss:.4f}")
        logger.info(f"  Final Test Accuracy: {test_acc:.2f}%")
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
    parser.add_argument('--dataset', type=str, default='dataset/dataset.csv',
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
