import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 Activity Ranking Model - Complete Training Pipeline\n",
                "## Google Colab - Generate All Model Artifacts\n",
                "\n",
                "This notebook will:\n",
                "1. ✅ Mount Google Drive\n",
                "2. ✅ Load your dataset\n",
                "3. ✅ Build BM25 index\n",
                "4. ✅ Generate Sentence-BERT embeddings (GPU accelerated)\n",
                "5. ✅ Create FAISS similarity index\n",
                "6. ✅ Save all models\n",
                "7. ✅ **Download model_artifacts.zip**\n",
                "\n",
                "**Output files:**\n",
                "- `bm25_docs.pkl` - Keyword index\n",
                "- `embeddings.npy` - Semantic embeddings\n",
                "- `faiss_index.bin` - Similarity search\n",
                "- `activities_processed.csv` - Dataset copy"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 1: Mount Google Drive"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "import os\n",
                "\n",
                "print('Mounting Google Drive...')\n",
                "drive.mount('/content/drive')\n",
                "print('✓ Drive mounted!')\n",
                "\n",
                "# Verify dataset\n",
                "dataset_path = '/content/drive/MyDrive/APS360-Applied Fundamentals of Deep Learning/activity-website/dataset/dataset.csv'\n",
                "print(f'Looking for: {dataset_path}')\n",
                "print(f'Exists: {os.path.exists(dataset_path)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 2: Install All Dependencies"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Installing dependencies...')\n",
                "!pip install -q sentence-transformers faiss-cpu rank-bm25 scikit-learn pandas numpy scipy torch\n",
                "print('✓ Dependencies installed!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 3: Import Everything"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import pickle\n",
                "import json\n",
                "import os\n",
                "import zipfile\n",
                "from datetime import datetime\n",
                "from typing import List, Dict, Tuple\n",
                "\n",
                "from rank_bm25 import BM25Okapi\n",
                "from sentence_transformers import SentenceTransformer\n",
                "import faiss\n",
                "from sklearn.metrics.pairwise import cosine_similarity\n",
                "\n",
                "print('✓ All imports successful!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 4: Load Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "dataset_path = '/content/drive/MyDrive/APS360-Applied Fundamentals of Deep Learning/activity-website/dataset/dataset.csv'\n",
                "\n",
                "print(f'Loading dataset from: {dataset_path}')\n",
                "df_activities = pd.read_csv(dataset_path)\n",
                "\n",
                "print(f'✓ Successfully loaded {len(df_activities)} activities')\n",
                "print(f'\\nDataset info:')\n",
                "print(f'  Shape: {df_activities.shape}')\n",
                "print(f'  Columns: {list(df_activities.columns)}')\n",
                "print(f'\\nFirst activity:')\n",
                "print(df_activities.iloc[0])"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 5: Create Text Representations"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def create_activity_text(row):\n",
                "    \"\"\"Combine all columns into one text for retrieval\"\"\"\n",
                "    parts = []\n",
                "    for col in row.index:\n",
                "        if pd.notna(row[col]):\n",
                "            parts.append(str(row[col]))\n",
                "    return ' '.join(parts)\n",
                "\n",
                "print('Creating text representations...')\n",
                "activity_texts = df_activities.apply(create_activity_text, axis=1).tolist()\n",
                "\n",
                "print(f'✓ Created {len(activity_texts)} text representations')\n",
                "print(f'\\nSample text (Activity 1):')\n",
                "print(f'  {activity_texts[0][:200]}...')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 6: Build BM25 Index (Keyword Search)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class BM25Retriever:\n",
                "    \"\"\"BM25 keyword-based retrieval\"\"\"\n",
                "    def __init__(self, documents):\n",
                "        print(f'  Tokenizing {len(documents)} documents...')\n",
                "        self.tokenized_docs = [doc.lower().split() for doc in documents]\n",
                "        print(f'  Building BM25 index...')\n",
                "        self.bm25 = BM25Okapi(self.tokenized_docs)\n",
                "        self.documents = documents\n",
                "        print(f'  ✓ BM25 ready!')\n",
                "    \n",
                "    def retrieve(self, query, top_k=10):\n",
                "        tokenized_query = query.lower().split()\n",
                "        scores = self.bm25.get_scores(tokenized_query)\n",
                "        top_indices = np.argsort(scores)[::-1][:top_k]\n",
                "        return [(int(idx), float(scores[idx])) for idx in top_indices]\n",
                "\n",
                "print('Building BM25 Retriever...')\n",
                "bm25_retriever = BM25Retriever(activity_texts)\n",
                "print('✓ BM25 index created!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 7: Build Dense Retriever (Semantic Search + FAISS)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "class DenseRetriever:\n",
                "    \"\"\"Sentence-BERT + FAISS semantic search\"\"\"\n",
                "    def __init__(self, documents, model_name='all-MiniLM-L6-v2'):\n",
                "        print(f'  Loading Sentence-BERT model: {model_name}')\n",
                "        self.model = SentenceTransformer(model_name)\n",
                "        \n",
                "        print(f'  Encoding {len(documents)} documents to embeddings...')\n",
                "        self.embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=True)\n",
                "        print(f'  Embeddings shape: {self.embeddings.shape}')\n",
                "        \n",
                "        print(f'  Building FAISS index...')\n",
                "        dimension = self.embeddings.shape[1]\n",
                "        self.index = faiss.IndexFlatL2(dimension)\n",
                "        self.index.add(self.embeddings.astype('float32'))\n",
                "        self.documents = documents\n",
                "        print(f'  ✓ FAISS index ready!')\n",
                "    \n",
                "    def retrieve(self, query, top_k=10):\n",
                "        query_embedding = self.model.encode([query], convert_to_numpy=True)\n",
                "        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)\n",
                "        scores = 1.0 / (1.0 + distances[0])\n",
                "        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores)]\n",
                "\n",
                "print('Building Dense Retriever (this may take 1-2 minutes)...')\n",
                "dense_retriever = DenseRetriever(activity_texts)\n",
                "print('✓ Dense retriever ready!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 8: Test Full Hybrid Pipeline"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def reciprocal_rank_fusion(sparse_results, dense_results, k=60):\n",
                "    \"\"\"Merge BM25 and dense results\"\"\"\n",
                "    rrf_scores = {}\n",
                "    for rank, (idx, _) in enumerate(sparse_results, 1):\n",
                "        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)\n",
                "    for rank, (idx, _) in enumerate(dense_results, 1):\n",
                "        rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (k + rank)\n",
                "    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)\n",
                "\n",
                "print('Testing hybrid pipeline...')\n",
                "query = 'outdoor activities for kids'\n",
                "\n",
                "bm25_res = bm25_retriever.retrieve(query, top_k=5)\n",
                "dense_res = dense_retriever.retrieve(query, top_k=5)\n",
                "fused = reciprocal_rank_fusion(bm25_res, dense_res)\n",
                "\n",
                "title_col = 'title' if 'title' in df_activities.columns else df_activities.columns[0]\n",
                "\n",
                "print(f'\\nQuery: \"{query}\"')\n",
                "print('\\nTop 5 Results:')\n",
                "for rank, (idx, score) in enumerate(fused[:5], 1):\n",
                "    activity_title = df_activities.iloc[idx][title_col]\n",
                "    print(f'  {rank}. {activity_title} (Score: {score:.4f})')\n",
                "\n",
                "print('\\n✓ Pipeline test successful!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 9: Save All Model Artifacts"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('\\n' + '='*70)\n",
                "print('SAVING ALL MODEL ARTIFACTS')\n",
                "print('='*70)\n",
                "\n",
                "# Create working directory\n",
                "os.makedirs('model_artifacts_dir', exist_ok=True)\n",
                "os.chdir('model_artifacts_dir')\n",
                "\n",
                "# 1. Save BM25 tokenized documents\n",
                "print('\\n1. Saving BM25 tokenized documents...')\n",
                "with open('bm25_docs.pkl', 'wb') as f:\n",
                "    pickle.dump(bm25_retriever.tokenized_docs, f)\n",
                "print('   ✓ bm25_docs.pkl saved')\n",
                "print(f'   Size: {os.path.getsize(\"bm25_docs.pkl\") / 1024:.2f} KB')\n",
                "\n",
                "# 2. Save embeddings\n",
                "print('\\n2. Saving Sentence-BERT embeddings...')\n",
                "np.save('embeddings.npy', dense_retriever.embeddings)\n",
                "print('   ✓ embeddings.npy saved')\n",
                "print(f'   Size: {os.path.getsize(\"embeddings.npy\") / 1024 / 1024:.2f} MB')\n",
                "print(f'   Shape: {dense_retriever.embeddings.shape}')\n",
                "\n",
                "# 3. Save FAISS index\n",
                "print('\\n3. Saving FAISS index...')\n",
                "faiss.write_index(dense_retriever.index, 'faiss_index.bin')\n",
                "print('   ✓ faiss_index.bin saved')\n",
                "print(f'   Size: {os.path.getsize(\"faiss_index.bin\") / 1024 / 1024:.2f} MB')\n",
                "\n",
                "# 4. Save dataset\n",
                "print('\\n4. Saving activity dataset...')\n",
                "df_activities.to_csv('activities_processed.csv', index=False)\n",
                "print('   ✓ activities_processed.csv saved')\n",
                "print(f'   Size: {os.path.getsize(\"activities_processed.csv\") / 1024:.2f} KB')\n",
                "\n",
                "print('\\n' + '='*70)\n",
                "print('✓ ALL MODELS SAVED SUCCESSFULLY!')\n",
                "print('='*70)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 10: Create ZIP File"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('\\nCreating model_artifacts.zip...')\n",
                "\n",
                "zip_path = '/content/model_artifacts.zip'\n",
                "with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:\n",
                "    zipf.write('bm25_docs.pkl', 'bm25_docs.pkl')\n",
                "    zipf.write('embeddings.npy', 'embeddings.npy')\n",
                "    zipf.write('faiss_index.bin', 'faiss_index.bin')\n",
                "    zipf.write('activities_processed.csv', 'activities_processed.csv')\n",
                "    print('  Added: bm25_docs.pkl')\n",
                "    print('  Added: embeddings.npy')\n",
                "    print('  Added: faiss_index.bin')\n",
                "    print('  Added: activities_processed.csv')\n",
                "\n",
                "print(f'\\n✓ ZIP created!')\n",
                "print(f'  Location: {zip_path}')\n",
                "print(f'  Size: {os.path.getsize(zip_path) / 1024 / 1024:.2f} MB')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 11: Download ZIP File"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('\\n' + '='*70)\n",
                "print('DOWNLOADING MODEL ARTIFACTS')\n",
                "print('='*70)\n",
                "\n",
                "from google.colab import files\n",
                "\n",
                "print('\\nStarting download of model_artifacts.zip...')\n",
                "files.download('/content/model_artifacts.zip')\n",
                "\n",
                "print('\\n' + '='*70)\n",
                "print('✓ DOWNLOAD COMPLETE!')\n",
                "print('='*70)\n",
                "print('\\nYour models are ready!')\n",
                "print('\\nNext steps:')\n",
                "print('1. Extract model_artifacts.zip')\n",
                "print('2. Place files in family-activity-planner/models/')\n",
                "print('3. Run: python app_optimized.py')\n",
                "print('4. Open: http://localhost:5000')\n",
                "print('\\n🎉 Done!')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## STEP 12: Verify Files (Optional)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('Verifying saved files...')\n",
                "print('\\nFiles in /content/model_artifacts_dir:')\n",
                "os.chdir('model_artifacts_dir')\n",
                "for file in os.listdir('.'):\n",
                "    size_mb = os.path.getsize(file) / 1024 / 1024\n",
                "    print(f'  ✓ {file} ({size_mb:.2f} MB)')\n",
                "\n",
                "print('\\nFiles in /content/:')\n",
                "os.chdir('/content')\n",
                "if os.path.exists('model_artifacts.zip'):\n",
                "    size_mb = os.path.getsize('model_artifacts.zip') / 1024 / 1024\n",
                "    print(f'  ✓ model_artifacts.zip ({size_mb:.2f} MB)')\n",
                "else:\n",
                "    print('  ✗ model_artifacts.zip NOT FOUND')\n",
                "\n",
                "print('\\n✓ Verification complete!')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebook
with open('activity_ranking_complete.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✅ COMPLETE NOTEBOOK GENERATED!")
print("\nFile: activity_ranking_complete.ipynb")
print("\n" + "="*70)
print("WHAT'S INCLUDED:")
print("="*70)
print("\n✅ 12 cells with all functions:")
print("  1. Mount Drive")
print("  2. Install dependencies")
print("  3. Import all libraries")
print("  4. Load dataset from Drive")
print("  5. Create text representations")
print("  6. Build BM25 keyword index")
print("  7. Build dense retriever with Sentence-BERT + FAISS")
print("  8. Test hybrid pipeline")
print("  9. Save all model artifacts (pkl, npy, bin, csv)")
print("  10. Create ZIP file")
print("  11. Download ZIP file (automatic)")
print("  12. Verify all files")
print("\n✅ File outputs:")
print("  - bm25_docs.pkl")
print("  - embeddings.npy")
print("  - faiss_index.bin")
print("  - activities_processed.csv")
print("  - model_artifacts.zip (ready to download)")
print("\n" + "="*70)
print("HOW TO USE:")
print("="*70)
print("\n1. Download: activity_ranking_complete.ipynb")
print("2. Upload to Google Colab")
print("3. Run all cells (top to bottom)")
print("4. Watch for download of model_artifacts.zip")
print("5. Extract and place in VSCode project")
print("\nThat's it! 🚀")