# ============================================================================
# FILE 1: GOOGLE COLAB NOTEBOOK CODE
# Copy this entire code into a new Google Colab cell and run it
# ============================================================================

# STEP 1: Mount Google Drive and Load Dataset
from google.colab import drive
import os

drive.mount('/content/drive')

dataset_path = '/content/drive/MyDrive/APS360-Applied Fundamentals of Deep Learning/activity-website/dataset/dataset.csv'
print("Dataset exists:", os.path.exists(dataset_path))

# ============================================================================
# STEP 2: Install Dependencies
# Run this in a separate cell:
# !pip install -q sentence-transformers faiss-cpu rank-bm25 scikit-learn pandas numpy scipy torch

# ============================================================================
# STEP 3: Import Libraries
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

print("✓ All libraries imported successfully!")

# ============================================================================
# STEP 4: Load Your Dataset
df_activities = pd.read_csv(dataset_path)
print(f'✓ Loaded {len(df_activities)} activities')
print(f'Columns: {df_activities.columns.tolist()}')
print(f'\nFirst activity:\n{df_activities.iloc[0]}')

# ============================================================================
# STEP 5: Create Text Representations for Each Activity
def create_activity_text(row):
    """Combine all text fields for each activity"""
    parts = []
    for col in row.index:
        if pd.notna(row[col]):
            parts.append(str(row[col]))
    return ' '.join(parts)

activity_texts = df_activities.apply(create_activity_text, axis=1).tolist()
print(f'✓ Created {len(activity_texts)} text representations')
print(f'Sample text: {activity_texts[0][:200]}...')

# ============================================================================
# STEP 6: Build BM25 Index (Keyword-based Retrieval)
class BM25Retriever:
    """Keyword-based search using BM25"""
    def __init__(self, documents):
        self.tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)
        self.documents = documents
    
    def retrieve(self, query, top_k=10):
        """Get top-k keyword matches"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

bm25_retriever = BM25Retriever(activity_texts)
print('✓ BM25 retriever initialized')

# Test BM25
test_results = bm25_retriever.retrieve('outdoor activities', top_k=3)
print(f'\nBM25 Test - "outdoor activities":')
for idx, score in test_results:
    print(f'  - {df_activities.iloc[idx].get("title", "Activity")}: {score:.4f}')

# ============================================================================
# STEP 7: Build Dense Retrieval Index (Semantic Search)
class DenseRetriever:
    """Semantic search using Sentence-BERT + FAISS"""
    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):
        print(f'Loading {model_name}...')
        self.model = SentenceTransformer(model_name)
        print('Encoding documents...')
        self.embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings.astype('float32'))
        self.documents = documents
        print(f'✓ FAISS index created with {len(documents)} documents')
    
    def retrieve(self, query, top_k=10):
        """Get top-k semantic matches"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        scores = 1.0 / (1.0 + distances[0])
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]

dense_retriever = DenseRetriever(activity_texts)
print('✓ Dense retriever initialized')

# Test dense retrieval
test_results = dense_retriever.retrieve('outdoor activities', top_k=3)
print(f'\nDense Test - "outdoor activities":')
for idx, score in test_results:
    print(f'  - {df_activities.iloc[idx].get("title", "Activity")}: {score:.4f}')

# ============================================================================
# STEP 8: Reciprocal Rank Fusion (Merge BM25 + Dense)
def reciprocal_rank_fusion(sparse_results, dense_results, k=60):
    """Merge keyword and semantic results"""
    rrf_scores = {}
    
    for rank, (idx, _) in enumerate(sparse_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
    
    for rank, (idx, _) in enumerate(dense_results, 1):
        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)
    
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused

print('✓ RRF fusion ready')

# ============================================================================
# STEP 9: Save All Models to Colab
print('\n' + '='*70)
print('SAVING MODELS FOR LOCAL USE')
print('='*70)

# Save BM25 tokenized docs
with open('bm25_docs.pkl', 'wb') as f:
    pickle.dump(bm25_retriever.tokenized_docs, f)
print('✓ Saved bm25_docs.pkl')

# Save embeddings
np.save('embeddings.npy', dense_retriever.embeddings)
print('✓ Saved embeddings.npy')

# Save FAISS index
faiss.write_index(dense_retriever.index, 'faiss_index.bin')
print('✓ Saved faiss_index.bin')

# Save dataset
df_activities.to_csv('activities_processed.csv', index=False)
print('✓ Saved activities_processed.csv')

# ============================================================================
# STEP 10: Create ZIP and Download
print('\nCreating ZIP file for download...')
import zipfile
with zipfile.ZipFile('model_artifacts.zip', 'w') as zipf:
    zipf.write('bm25_docs.pkl')
    zipf.write('embeddings.npy')
    zipf.write('faiss_index.bin')
    zipf.write('activities_processed.csv')

print('✓ Created model_artifacts.zip')

# Download from Colab
from google.colab import files
files.download('model_artifacts.zip')
print('\n✓ Download started! Save the ZIP file to your computer.')

# ============================================================================
# STEP 11: Quick Test of Full Pipeline
print('\n' + '='*70)
print('FULL PIPELINE TEST')
print('='*70)

query = 'fun outdoor activities for kids'
bm25_res = bm25_retriever.retrieve(query, top_k=5)
dense_res = dense_retriever.retrieve(query, top_k=5)
fused = reciprocal_rank_fusion(bm25_res, dense_res)

print(f'\nQuery: "{query}"')
print('\nTop 5 Results:')
for rank, (idx, score) in enumerate(fused[:5], 1):
    title = df_activities.iloc[idx].get('title', 'Activity')
    print(f'{rank}. {title} (score: {score:.4f})')

print('\n✓ Pipeline test complete!')
print('✓ Ready to download models and deploy locally!')
