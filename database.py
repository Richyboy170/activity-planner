"""
Database module for Activity Planner
Stores activities and group information for embedding calculations
"""

import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ActivityDatabase:
    """SQLite database for storing activities and search sessions"""

    def __init__(self, db_path: str = 'activity_planner.db'):
        self.db_path = db_path
        self.conn = None
        self._init_database()

    def _init_database(self):
        """Initialize database schema"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        cursor = self.conn.cursor()

        # Activities table - stores all activity details for embedding calculations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                age_min INTEGER,
                age_max INTEGER,
                duration_mins INTEGER,
                tags TEXT,
                cost TEXT,
                indoor_outdoor TEXT,
                season TEXT,
                materials_needed TEXT,
                how_to_play TEXT,
                players TEXT,
                parent_caution TEXT,
                full_text TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Group sessions table - stores user's group member information
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                group_name TEXT,
                members_json TEXT NOT NULL,
                preferences_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Search history table - tracks searches for analytics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                query TEXT NOT NULL,
                results_json TEXT,
                num_results INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Activity rankings table - stores linkage scores
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS activity_rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                activity_id INTEGER NOT NULL,
                query TEXT,
                retrieval_score REAL,
                age_fit_score REAL,
                preference_score REAL,
                final_score REAL,
                rank_position INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (activity_id) REFERENCES activities (id)
            )
        ''')

        # Create indexes for faster queries
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session_id
            ON search_history (session_id)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_activity_title
            ON activities (title)
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_activity_rankings_session
            ON activity_rankings (session_id, final_score DESC)
        ''')

        self.conn.commit()
        logger.info(f"✓ Database initialized: {self.db_path}")

    def load_activities_from_csv(self, csv_path: str, force_reload: bool = False):
        """Load activities from CSV into database"""
        cursor = self.conn.cursor()

        # Check if activities already loaded
        cursor.execute('SELECT COUNT(*) FROM activities')
        count = cursor.fetchone()[0]

        if count > 0 and not force_reload:
            logger.info(f"✓ Database already contains {count} activities (use force_reload=True to reload)")
            return count

        # Clear existing activities if force reload
        if force_reload:
            cursor.execute('DELETE FROM activities')
            logger.info("Cleared existing activities for reload")

        # Load CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            logger.warning(f"UTF-8 decode failed for {csv_path}, trying latin1")
            try:
                df = pd.read_csv(csv_path, encoding='latin1')
            except UnicodeDecodeError:
                logger.warning(f"Latin1 decode failed for {csv_path}, trying cp1252")
                df = pd.read_csv(csv_path, encoding='cp1252')
        
        logger.info(f"Loading {len(df)} activities from {csv_path}...")

        # Insert activities
        inserted = 0
        for _, row in df.iterrows():
            # Create full text representation
            full_text_parts = []
            for col in df.columns:
                if pd.notna(row[col]):
                    full_text_parts.append(str(row[col]))
            full_text = ' '.join(full_text_parts)

            cursor.execute('''
                INSERT INTO activities (
                    title, age_min, age_max, duration_mins, tags, cost,
                    indoor_outdoor, season, materials_needed, how_to_play,
                    players, parent_caution, full_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                row.get('title'),
                row.get('age_min'),
                row.get('age_max'),
                row.get('duration_mins'),
                row.get('tags'),
                row.get('cost'),
                row.get('indoor_outdoor'),
                row.get('season'),
                str(row.get('materials_needed', '')),
                str(row.get('how_to_play', '')),
                row.get('players'),
                row.get('parent_caution'),
                full_text
            ))
            inserted += 1

        self.conn.commit()
        logger.info(f"✓ Loaded {inserted} activities into database")
        return inserted

    def get_all_activities(self) -> List[Dict]:
        """Retrieve all activities"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM activities ORDER BY id')
        rows = cursor.fetchall()

        return [dict(row) for row in rows]

    def get_activity_by_id(self, activity_id: int) -> Optional[Dict]:
        """Get single activity by ID"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM activities WHERE id = ?', (activity_id,))
        row = cursor.fetchone()

        return dict(row) if row else None

    def create_group_session(self, session_id: str, members: List[Dict],
                           group_name: Optional[str] = None,
                           preferences: Optional[List[str]] = None) -> str:
        """Create a new group session"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO group_sessions (session_id, group_name, members_json, preferences_json)
            VALUES (?, ?, ?, ?)
        ''', (
            session_id,
            group_name,
            json.dumps(members),
            json.dumps(preferences) if preferences else None
        ))

        self.conn.commit()
        logger.info(f"✓ Created group session: {session_id}")
        return session_id

    def get_group_session(self, session_id: str) -> Optional[Dict]:
        """Retrieve group session"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM group_sessions WHERE session_id = ?', (session_id,))
        row = cursor.fetchone()

        if row:
            session = dict(row)
            session['members'] = json.loads(session['members_json'])
            if session['preferences_json']:
                session['preferences'] = json.loads(session['preferences_json'])
            else:
                session['preferences'] = []
            return session

        return None

    def save_search(self, session_id: str, query: str,
                   results: List[Dict]) -> int:
        """Save search query and results"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO search_history (session_id, query, results_json, num_results)
            VALUES (?, ?, ?, ?)
        ''', (
            session_id,
            query,
            json.dumps(results),
            len(results)
        ))

        self.conn.commit()
        return cursor.lastrowid

    def save_activity_ranking(self, session_id: str, activity_id: int,
                            query: str, retrieval_score: float,
                            age_fit_score: float, preference_score: float,
                            final_score: float, rank_position: int):
        """Save detailed ranking scores for an activity"""
        cursor = self.conn.cursor()

        cursor.execute('''
            INSERT INTO activity_rankings (
                session_id, activity_id, query, retrieval_score,
                age_fit_score, preference_score, final_score, rank_position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            session_id, activity_id, query, retrieval_score,
            age_fit_score, preference_score, final_score, rank_position
        ))

        self.conn.commit()

    def get_search_history(self, session_id: str) -> List[Dict]:
        """Get search history for a session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM search_history
            WHERE session_id = ?
            ORDER BY created_at DESC
        ''', (session_id,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_top_activities_for_session(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get top-ranked activities for a session"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT
                a.*,
                ar.final_score,
                ar.rank_position,
                ar.retrieval_score,
                ar.age_fit_score,
                ar.preference_score
            FROM activities a
            JOIN activity_rankings ar ON a.id = ar.activity_id
            WHERE ar.session_id = ?
            ORDER BY ar.final_score DESC
            LIMIT ?
        ''', (session_id, limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_activity_stats(self) -> Dict:
        """Get database statistics"""
        cursor = self.conn.cursor()

        # Activity count
        cursor.execute('SELECT COUNT(*) FROM activities')
        activity_count = cursor.fetchone()[0]

        # Session count
        cursor.execute('SELECT COUNT(*) FROM group_sessions')
        session_count = cursor.fetchone()[0]

        # Search count
        cursor.execute('SELECT COUNT(*) FROM search_history')
        search_count = cursor.fetchone()[0]

        # Average age range
        cursor.execute('SELECT AVG(age_min), AVG(age_max) FROM activities')
        avg_ages = cursor.fetchone()

        return {
            'total_activities': activity_count,
            'total_sessions': session_count,
            'total_searches': search_count,
            'avg_age_min': avg_ages[0] if avg_ages[0] else 0,
            'avg_age_max': avg_ages[1] if avg_ages[1] else 0
        }

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("✓ Database connection closed")


def initialize_database(csv_path: str = 'dataset/dataset.csv',
                       db_path: str = 'activity_planner.db',
                       force_reload: bool = False) -> ActivityDatabase:
    """Initialize database and load activities"""
    db = ActivityDatabase(db_path)
    db.load_activities_from_csv(csv_path, force_reload=force_reload)
    return db


if __name__ == '__main__':
    # Test database initialization
    logging.basicConfig(level=logging.INFO)

    print("Initializing Activity Planner database...")
    db = initialize_database(force_reload=True)

    # Show stats
    stats = db.get_activity_stats()
    print("\nDatabase Statistics:")
    print(f"  Total Activities: {stats['total_activities']}")
    print(f"  Total Sessions: {stats['total_sessions']}")
    print(f"  Total Searches: {stats['total_searches']}")

    # Get sample activity
    activities = db.get_all_activities()
    if activities:
        print(f"\nSample Activity:")
        sample = activities[0]
        print(f"  Title: {sample['title']}")
        print(f"  Age Range: {sample['age_min']}-{sample['age_max']}")
        print(f"  Duration: {sample['duration_mins']} mins")

    db.close()
    print("\n✓ Database test complete!")
