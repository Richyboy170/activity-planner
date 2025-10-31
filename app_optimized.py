# ============================================================================
# FILE 2: FLASK BACKEND - app_optimized.py
# Run this locally in VSCode after downloading models from Colab
# ============================================================================

import pandas as pd
import numpy as np
import pickle
import logging
import os
from datetime import datetime
from typing import List, Dict, Tuple

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD PRE-TRAINED MODELS FROM COLAB
# ============================================================================

class OptimizedRecommenderAPI:
    """Load pre-trained models and serve recommendations"""
    
    def __init__(self, dataset_path: str, models_dir: str = 'models'):
        logger.info("="*70)
        logger.info("Loading Pre-trained Models from Colab")
        logger.info("="*70)
        
        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        self.df_activities = pd.read_csv(dataset_path)
        logger.info(f"✓ Loaded {len(self.df_activities)} activities")
        
        # Create activity texts (same as Colab)
        self.activity_texts = self._create_activity_texts()
        
        # Load BM25
        logger.info("Loading BM25...")
        with open(f'{models_dir}/bm25_docs.pkl', 'rb') as f:
            tokenized_docs = pickle.load(f)
        self.bm25_retriever = BM25Okapi(tokenized_docs)
        logger.info(f"✓ BM25 ready ({len(tokenized_docs)} docs)")
        
        # Load embeddings
        logger.info("Loading embeddings...")
        self.embeddings = np.load(f'{models_dir}/embeddings.npy')
        logger.info(f"✓ Embeddings ready ({self.embeddings.shape})")
        
        # Load FAISS index
        logger.info("Loading FAISS index...")
        self.faiss_index = faiss.read_index(f'{models_dir}/faiss_index.bin')
        logger.info(f"✓ FAISS ready ({self.faiss_index.ntotal} vectors)")
        
        # Load Sentence-BERT model for query encoding
        logger.info("Loading Sentence-BERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Model ready")
        
        logger.info("="*70)
        logger.info("✓ ALL MODELS LOADED - READY FOR RECOMMENDATIONS!")
        logger.info("="*70)
    
    def _create_activity_texts(self) -> List[str]:
        """Create text for each activity (same as Colab)"""
        def create_text(row):
            parts = []
            for col in row.index:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
            return ' '.join(parts)
        
        return self.df_activities.apply(create_text, axis=1).tolist()
    
    def bm25_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_retriever.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def dense_retrieve(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Semantic search using FAISS"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        scores = 1.0 / (1.0 + distances[0])
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]
    
    def recommend(
        self,
        query: str,
        group_members: List[Dict] = None,
        preferences: List[str] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Get recommendations for group members
        
        query: Search string (e.g., "outdoor activities")
        group_members: List of members with ages
        preferences: List of preferred tags
        top_k: Number of results
        """
        if group_members is None:
            group_members = []
        if preferences is None:
            preferences = []
        
        try:
            # STAGE 1: Hybrid Retrieval (BM25 + Dense)
            logger.info(f"Query: {query}")
            bm25_results = self.bm25_retrieve(query, top_k=20)
            dense_results = self.dense_retrieve(query, top_k=20)
            
            # STAGE 2: Reciprocal Rank Fusion
            rrf_scores = {}
            for rank, (idx, _) in enumerate(bm25_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
            for rank, (idx, _) in enumerate(dense_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
            
            fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            
            # STAGE 3: Score by Group Member Details
            scored_results = []
            for idx, retrieval_score in fused:
                activity = self.df_activities.iloc[idx]
                
                # Calculate age fit for this group
                if group_members:
                    group_ages = [m.get('age', 10) for m in group_members]
                    activity_age_min = activity.get('age_min', 0) if 'age_min' in activity.index else 0
                    activity_age_max = activity.get('age_max', 100) if 'age_max' in activity.index else 100
                    
                    age_fits = []
                    for age in group_ages:
                        if activity_age_min <= age <= activity_age_max:
                            age_fits.append(1.0)
                        elif age < activity_age_min:
                            age_fits.append(max(0, 1 - (activity_age_min - age) / 10))
                        else:
                            age_fits.append(max(0, 1 - (age - activity_age_max) / 10))
                    age_fit = np.mean(age_fits) if age_fits else 0.5
                else:
                    age_fit = 0.5
                
                # Calculate preference match
                if preferences:
                    activity_text = str(activity).lower()
                    pref_matches = sum(1 for pref in preferences if pref.lower() in activity_text)
                    pref_score = min(pref_matches / len(preferences), 1.0)
                else:
                    pref_score = 0.5
                
                # FINAL SCORE: Combine retrieval + age + preference
                final_score = (
                    0.4 * min(retrieval_score, 1.0) +
                    0.3 * age_fit +
                    0.3 * pref_score
                )
                
                scored_results.append((idx, final_score))
            
            # Sort by final score
            scored_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
            
            # Format recommendations
            recommendations = []
            for rank, (idx, score) in enumerate(scored_results[:top_k], 1):
                try:
                    activity = self.df_activities.iloc[idx].to_dict()
                    activity['rank'] = rank
                    activity['recommendation_score'] = float(score)
                    
                    # Handle JSON serialization
                    for col in activity:
                        if isinstance(activity[col], list):
                            activity[col] = ', '.join(map(str, activity[col]))
                        elif pd.isna(activity[col]):
                            activity[col] = None
                    
                    recommendations.append(activity)
                except Exception as e:
                    logger.warning(f"Error with activity {idx}: {e}")
            
            return {
                'status': 'success',
                'query': query,
                'recommendations': recommendations,
                'count': len(recommendations),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'recommendations': [],
                'count': 0
            }


# ============================================================================
# FLASK APP
# ============================================================================

def create_app(dataset_path='dataset/dataset.csv', models_dir='models'):
    """Create Flask app"""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    CORS(app)
    
    try:
        api = OptimizedRecommenderAPI(dataset_path, models_dir)
    except Exception as e:
        logger.error(f"Failed to init API: {e}")
        api = None
    
    # Serve frontend
    @app.route('/')
    def index():
        return send_from_directory('templates', 'index.html')
    
    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory('static', path)
    
    # API: Health check
    @app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status': 'healthy',
            'activities_loaded': len(api.df_activities) if api else 0
        }), 200
    
    # API: Main recommendation endpoint
    @app.route('/api/recommend', methods=['POST'])
    def recommend():
        if not api:
            return jsonify({'error': 'API not initialized'}), 500
        
        try:
            data = request.get_json()
            query = data.get('query', '')
            group_members = data.get('group_members', [])
            preferences = data.get('preferences', [])
            top_k = data.get('top_k', 5)
            
            if not query:
                return jsonify({'error': 'Query required'}), 400
            
            result = api.recommend(query, group_members, preferences, top_k)
            return jsonify(result), 200 if result['status'] == 'success' else 500
        
        except Exception as e:
            logger.error(f"Error: {e}")
            return jsonify({'error': str(e)}), 500
    
    return app


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/dataset.csv')
    parser.add_argument('--models', default='models')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    
    app = create_app(args.dataset, args.models)
    logger.info(f"Starting on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=False)
