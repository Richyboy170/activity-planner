"""
Activity Planner - Flask Web Application
Integrated AI search with Sentence-BERT embeddings and group member linkage scoring
"""

import pandas as pd
import numpy as np
import pickle
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss

from database import ActivityDatabase, initialize_database

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActivityRecommenderWithLinkage:
    """
    Advanced recommender with Sentence-BERT embeddings and linkage scoring
    Calculates linkage between search query, group members, and activities
    """

    def __init__(self, dataset_path: str, models_dir: str = 'models',
                 db_path: str = 'activity_planner.db'):
        logger.info("="*70)
        logger.info("🚀 Activity Recommender with AI Linkage Scoring")
        logger.info("="*70)

        # Initialize database
        self.db = ActivityDatabase(db_path)
        self.db.load_activities_from_csv(dataset_path)

        # Load dataset
        logger.info(f"Loading dataset: {dataset_path}")
        self.df_activities = pd.read_csv(dataset_path)
        logger.info(f"✓ Loaded {len(self.df_activities)} activities")

        # Create activity texts
        self.activity_texts = self._create_activity_texts()

        # Load BM25
        logger.info("Loading BM25 index...")
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

        # Load Sentence-BERT model
        logger.info("Loading Sentence-BERT model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✓ Model ready")

        logger.info("="*70)
        logger.info("✅ RECOMMENDER INITIALIZED - READY FOR AI SEARCH!")
        logger.info("="*70)

    def _create_activity_texts(self) -> List[str]:
        """Create text for each activity"""
        def create_text(row):
            parts = []
            for col in row.index:
                if pd.notna(row[col]):
                    parts.append(str(row[col]))
            return ' '.join(parts)

        return self.df_activities.apply(create_text, axis=1).tolist()

    def bm25_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """BM25 keyword search"""
        tokenized_query = query.lower().split()
        scores = self.bm25_retriever.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def dense_retrieve(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Semantic search using FAISS"""
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.faiss_index.search(query_embedding.astype('float32'), top_k)
        scores = 1.0 / (1.0 + distances[0])
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]

    def calculate_linkage_score(
        self,
        query: str,
        activity_idx: int,
        group_members: List[Dict],
        preferences: List[str]
    ) -> Dict[str, float]:
        """
        Calculate linkage scores between query, group members, and activity
        Returns detailed breakdown of scores
        """
        activity = self.df_activities.iloc[activity_idx]
        activity_text = self.activity_texts[activity_idx]

        scores = {}

        # 1. Query-Activity Semantic Linkage (using embeddings)
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        activity_embedding = self.embeddings[activity_idx:activity_idx+1]
        semantic_similarity = float(np.dot(query_embedding, activity_embedding.T)[0][0])
        scores['semantic_linkage'] = max(0.0, min(1.0, semantic_similarity))

        # 2. Age Fit Linkage (group members age compatibility)
        if group_members:
            group_ages = [m.get('age', 10) for m in group_members]
            activity_age_min = activity.get('age_min', 0) if 'age_min' in activity.index else 0
            activity_age_max = activity.get('age_max', 100) if 'age_max' in activity.index else 100

            age_fits = []
            for member_age in group_ages:
                if activity_age_min <= member_age <= activity_age_max:
                    # Perfect fit
                    age_fits.append(1.0)
                elif member_age < activity_age_min:
                    # Too young - decay based on gap
                    gap = activity_age_min - member_age
                    age_fits.append(max(0, 1 - (gap / 10)))
                else:
                    # Too old - decay based on gap
                    gap = member_age - activity_age_max
                    age_fits.append(max(0, 1 - (gap / 10)))

            scores['age_fit_linkage'] = float(np.mean(age_fits))
        else:
            scores['age_fit_linkage'] = 0.5

        # 3. Preference Linkage (tags, keywords match)
        if preferences:
            activity_text_lower = activity_text.lower()
            matches = 0
            for pref in preferences:
                if pref.lower() in activity_text_lower:
                    matches += 1

            scores['preference_linkage'] = min(1.0, matches / len(preferences))
        else:
            scores['preference_linkage'] = 0.5

        # 4. Group Size Compatibility Linkage
        if group_members:
            group_size = len(group_members)
            players_info = str(activity.get('players', '1+')).lower()

            if '1+' in players_info or 'any' in players_info:
                scores['group_size_linkage'] = 1.0
            elif '+' in players_info:
                # Extract minimum number
                try:
                    min_players = int(players_info.split('+')[0])
                    if group_size >= min_players:
                        scores['group_size_linkage'] = 1.0
                    else:
                        scores['group_size_linkage'] = group_size / min_players
                except:
                    scores['group_size_linkage'] = 0.5
            else:
                scores['group_size_linkage'] = 0.5
        else:
            scores['group_size_linkage'] = 0.5

        # 5. Context Linkage (indoor/outdoor, season, etc.)
        context_score = 0.5  # Default neutral
        # This can be extended with user's current context (location, season, etc.)
        scores['context_linkage'] = context_score

        # Calculate overall linkage score (weighted combination)
        overall_linkage = (
            0.35 * scores['semantic_linkage'] +
            0.25 * scores['age_fit_linkage'] +
            0.20 * scores['preference_linkage'] +
            0.10 * scores['group_size_linkage'] +
            0.10 * scores['context_linkage']
        )
        scores['overall_linkage'] = overall_linkage

        return scores

    def search_with_linkage(
        self,
        query: str,
        session_id: str,
        group_members: Optional[List[Dict]] = None,
        preferences: Optional[List[str]] = None,
        top_k: int = 10
    ) -> Dict:
        """
        AI-powered search with linkage scoring
        Combines hybrid retrieval + group member linkage scoring
        """
        if group_members is None:
            group_members = []
        if preferences is None:
            preferences = []

        try:
            logger.info(f"🔍 Search query: '{query}' for session {session_id}")

            # Stage 1: Hybrid Retrieval (BM25 + Dense)
            bm25_results = self.bm25_retrieve(query, top_k=30)
            dense_results = self.dense_retrieve(query, top_k=30)

            # Stage 2: Reciprocal Rank Fusion
            rrf_scores = {}
            for rank, (idx, _) in enumerate(bm25_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)
            for rank, (idx, _) in enumerate(dense_results, 1):
                rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (60 + rank)

            fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

            # Stage 3: Calculate Linkage Scores
            ranked_results = []
            for idx, retrieval_score in fused[:top_k * 2]:  # Consider more candidates
                linkage_scores = self.calculate_linkage_score(
                    query, idx, group_members, preferences
                )

                # Combine retrieval score with linkage score
                final_score = (
                    0.3 * min(retrieval_score, 1.0) +  # Retrieval relevance
                    0.7 * linkage_scores['overall_linkage']  # Group linkage
                )

                ranked_results.append({
                    'activity_idx': int(idx),
                    'retrieval_score': float(retrieval_score),
                    'linkage_scores': linkage_scores,
                    'final_score': float(final_score)
                })

            # Sort by final score
            ranked_results = sorted(ranked_results, key=lambda x: x['final_score'], reverse=True)

            # Stage 4: Format top-k results with full activity details
            recommendations = []
            for rank, result in enumerate(ranked_results[:top_k], 1):
                idx = result['activity_idx']
                activity = self.df_activities.iloc[idx].to_dict()

                # Clean up for JSON serialization
                for col in activity:
                    if pd.isna(activity[col]):
                        activity[col] = None
                    elif isinstance(activity[col], (list, dict)):
                        activity[col] = str(activity[col])

                activity['id'] = idx + 1  # DB IDs start from 1
                activity['rank'] = rank
                activity['scores'] = {
                    'final_score': result['final_score'],
                    'retrieval_score': result['retrieval_score'],
                    'semantic_linkage': result['linkage_scores']['semantic_linkage'],
                    'age_fit_linkage': result['linkage_scores']['age_fit_linkage'],
                    'preference_linkage': result['linkage_scores']['preference_linkage'],
                    'group_size_linkage': result['linkage_scores']['group_size_linkage'],
                    'context_linkage': result['linkage_scores']['context_linkage'],
                    'overall_linkage': result['linkage_scores']['overall_linkage']
                }

                recommendations.append(activity)

                # Save to database
                self.db.save_activity_ranking(
                    session_id=session_id,
                    activity_id=idx + 1,  # DB IDs start from 1
                    query=query,
                    retrieval_score=result['retrieval_score'],
                    age_fit_score=result['linkage_scores']['age_fit_linkage'],
                    preference_score=result['linkage_scores']['preference_linkage'],
                    final_score=result['final_score'],
                    rank_position=rank
                )

            # Save search to database
            self.db.save_search(session_id, query, recommendations)

            return {
                'status': 'success',
                'query': query,
                'recommendations': recommendations,
                'count': len(recommendations),
                'session_id': session_id,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Search error: {e}", exc_info=True)
            return {
                'status': 'error',
                'message': str(e),
                'recommendations': [],
                'count': 0
            }


# ============================================================================
# FLASK APP
# ============================================================================

def create_app(dataset_path='dataset/dataset.csv', models_dir='models',
               db_path='activity_planner.db'):
    """Create Flask app with integrated AI search"""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.secret_key = os.urandom(24)
    CORS(app)

    try:
        api = ActivityRecommenderWithLinkage(dataset_path, models_dir, db_path)
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        api = None

    @app.route('/')
    def index():
        return send_from_directory('templates', 'index.html')

    @app.route('/static/<path:path>')
    def serve_static(path):
        return send_from_directory('static', path)

    @app.route('/api/health', methods=['GET'])
    def health():
        stats = api.db.get_activity_stats() if api else {}
        return jsonify({
            'status': 'healthy',
            'activities_loaded': len(api.df_activities) if api else 0,
            'database_stats': stats
        }), 200

    @app.route('/api/session/create', methods=['POST'])
    def create_session():
        """Create a new group session"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            data = request.get_json()
            members = data.get('members', [])
            group_name = data.get('group_name', 'My Group')
            preferences = data.get('preferences', [])

            if not members:
                return jsonify({'error': 'At least one group member required'}), 400

            # Generate session ID
            session_id = str(uuid.uuid4())

            # Save to database
            api.db.create_group_session(session_id, members, group_name, preferences)

            # Store in Flask session
            session['session_id'] = session_id

            logger.info(f"✓ Created session {session_id} with {len(members)} members")

            return jsonify({
                'status': 'success',
                'session_id': session_id,
                'group_name': group_name,
                'members': members,
                'preferences': preferences
            }), 200

        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/session/<session_id>', methods=['GET'])
    def get_session(session_id):
        """Retrieve session information"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            session_data = api.db.get_group_session(session_id)
            if not session_data:
                return jsonify({'error': 'Session not found'}), 404

            return jsonify({
                'status': 'success',
                'session': session_data
            }), 200

        except Exception as e:
            logger.error(f"Error retrieving session: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/recommend', methods=['POST'])
    def recommend():
        """Get personalized recommendations without requiring a session"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            data = request.get_json()
            query = data.get('query', '')
            group_members = data.get('group_members', [])
            preferences = data.get('preferences', [])
            top_k = data.get('top_k', 10)

            if not query:
                return jsonify({'error': 'Search query required'}), 400

            # Generate a temporary session ID for this request
            session_id = str(uuid.uuid4())

            # Perform AI search with linkage scoring
            result = api.search_with_linkage(
                query=query,
                session_id=session_id,
                group_members=group_members,
                preferences=preferences,
                top_k=top_k
            )

            # Format response to match expected structure
            if result['status'] == 'success':
                # Transform to expected format
                formatted_recommendations = []
                for rec in result['recommendations']:
                    formatted_rec = {
                        'id': rec.get('id'),  # Include activity ID for details view
                        'title': rec.get('title', 'Activity'),
                        'description': rec.get('description', ''),
                        'duration_mins': rec.get('duration_mins'),
                        'cost': rec.get('cost'),
                        'location': rec.get('location'),
                        'recommendation_score': rec['scores']['final_score'] if 'scores' in rec else 0.5
                    }
                    formatted_recommendations.append(formatted_rec)

                return jsonify({
                    'status': 'success',
                    'recommendations': formatted_recommendations
                }), 200
            else:
                return jsonify(result), 500

        except Exception as e:
            logger.error(f"Recommendation error: {e}", exc_info=True)
            return jsonify({'error': str(e), 'status': 'error'}), 500

    @app.route('/api/search', methods=['POST'])
    def search():
        """AI-powered activity search with linkage scoring"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            data = request.get_json()
            query = data.get('query', '')
            session_id = data.get('session_id', '')
            top_k = data.get('top_k', 10)

            if not query:
                return jsonify({'error': 'Search query required'}), 400

            if not session_id:
                return jsonify({'error': 'Session ID required'}), 400

            # Get group information
            session_data = api.db.get_group_session(session_id)
            if not session_data:
                return jsonify({'error': 'Invalid session'}), 404

            group_members = session_data['members']
            preferences = session_data.get('preferences', [])

            # Perform AI search with linkage scoring
            result = api.search_with_linkage(
                query=query,
                session_id=session_id,
                group_members=group_members,
                preferences=preferences,
                top_k=top_k
            )

            return jsonify(result), 200 if result['status'] == 'success' else 500

        except Exception as e:
            logger.error(f"Search error: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @app.route('/api/activities', methods=['GET'])
    def get_all_activities():
        """Get all activities from database"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            activities = api.db.get_all_activities()
            return jsonify({
                'status': 'success',
                'activities': activities,
                'count': len(activities)
            }), 200

        except Exception as e:
            logger.error(f"Error getting activities: {e}")
            return jsonify({'error': str(e)}), 500

    @app.route('/api/activity/<int:activity_id>', methods=['GET'])
    def get_activity(activity_id):
        """Get single activity details"""
        if not api:
            return jsonify({'error': 'API not initialized'}), 500

        try:
            activity = api.db.get_activity_by_id(activity_id)
            if not activity:
                return jsonify({'error': 'Activity not found'}), 404

            return jsonify({
                'status': 'success',
                'activity': activity
            }), 200

        except Exception as e:
            logger.error(f"Error getting activity: {e}")
            return jsonify({'error': str(e)}), 500

    return app


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/dataset.csv')
    parser.add_argument('--models', default='models')
    parser.add_argument('--db', default='activity_planner.db')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()

    app = create_app(args.dataset, args.models, args.db)
    logger.info(f"🚀 Starting Activity Planner on http://localhost:{args.port}")
    app.run(host='0.0.0.0', port=args.port, debug=True)
