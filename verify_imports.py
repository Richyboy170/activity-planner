try:
    print("Importing flask...")
    import flask
    print("✓ flask imported")
    
    print("Importing pandas...")
    import pandas
    print("✓ pandas imported")
    
    print("Importing numpy...")
    import numpy
    print("✓ numpy imported")
    
    print("Importing onnxruntime...")
    import onnxruntime
    print("✓ onnxruntime imported")
    
    print("Importing transformers...")
    import transformers
    print("✓ transformers imported")
    
    print("Importing rank_bm25...")
    import rank_bm25
    print("✓ rank_bm25 imported")
    
    print("Importing faiss...")
    import faiss
    print("✓ faiss imported")
    
    print("Importing app...")
    import app
    print("✓ app imported")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
