import pickle
import numpy as np
import faiss
import onnxruntime as ort
from transformers import AutoTokenizer
import os

models_dir = 'models'

print("1. Testing BM25...")
try:
    with open(f'{models_dir}/bm25_docs.pkl', 'rb') as f:
        tokenized_docs = pickle.load(f)
    print(f"✓ BM25 loaded. Docs: {len(tokenized_docs)}")
except Exception as e:
    print(f"❌ BM25 failed: {e}")

print("\n2. Testing Embeddings...")
try:
    embeddings = np.load(f'{models_dir}/embeddings.npy')
    print(f"✓ Embeddings loaded. Shape: {embeddings.shape}")
except Exception as e:
    print(f"❌ Embeddings failed: {e}")

print("\n3. Testing FAISS...")
try:
    faiss_index = faiss.read_index(f'{models_dir}/faiss_index.bin')
    print(f"✓ FAISS loaded. Total: {faiss_index.ntotal}")
except Exception as e:
    print(f"❌ FAISS failed: {e}")

print("\n4. Testing ONNX Tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(f'{models_dir}/onnx')
    print("✓ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer failed: {e}")

print("\n5. Testing ONNX Model...")
try:
    ort_session = ort.InferenceSession(f'{models_dir}/onnx/model_quantized.onnx')
    print("✓ ONNX Model loaded")
except Exception as e:
    print(f"❌ ONNX Model failed: {e}")
