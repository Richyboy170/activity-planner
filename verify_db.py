from database import ActivityDatabase
import pandas as pd
import os

print("Testing Database Initialization...")
try:
    db = ActivityDatabase('test_db.db')
    print("✓ Database initialized")
    
    print("Testing CSV Loading...")
    db.load_activities_from_csv('dataset/dataset.csv', force_reload=True)
    print("✓ CSV Loaded")
    
    stats = db.get_activity_stats()
    print(f"Stats: {stats}")
    
    db.close()
    os.remove('test_db.db')
    print("✓ Cleanup complete")
    
except Exception as e:
    print(f"❌ Database failed: {e}")
