# ============================================================================
# QUICK SETUP GUIDE - COMPLETE WORKFLOW
# ============================================================================

## PHASE 1: GOOGLE COLAB (Train Models - 30 minutes)

### Step 1: Create New Colab Notebook
1. Go to https://colab.research.google.com/
2. Click "New Notebook"
3. Rename it to "Activity_Ranking_Model"

### Step 2: Copy & Run Colab Code
1. Open file: `1_COLAB_NOTEBOOK.py`
2. Copy ALL the code
3. Paste into your Colab notebook
4. **IMPORTANT: Split into cells as follows:**

   **Cell 1:** Mount Drive (lines 1-12)
   ```python
   from google.colab import drive
   import os
   drive.mount('/content/drive')
   dataset_path = '/content/drive/MyDrive/APS360-Applied Fundamentals of Deep Learning/activity-website/dataset/dataset.csv'
   print("Dataset exists:", os.path.exists(dataset_path))
   ```

   **Cell 2:** Install (lines 14-15) - UNCOMMENT and run
   ```python
   !pip install -q sentence-transformers faiss-cpu rank-bm25 scikit-learn pandas numpy scipy torch
   ```

   **Cell 3:** Run the rest of the code (lines 17-end)

### Step 3: Wait for Training
- BM25: ~1 second
- Sentence-BERT + embeddings: ~30-60 seconds (depends on dataset size)
- FAISS: ~5 seconds
- Total: ~1-2 minutes

### Step 4: Download Models
- Bottom of Colab output: "model_artifacts.zip" will download
- Save it to your computer
- Extract the ZIP file

---

## PHASE 2: VSCODE LOCAL SETUP (30 minutes)

### Step 1: Create Project Folder
```bash
mkdir family-activity-planner
cd family-activity-planner
```

### Step 2: Create Subdirectories
```bash
mkdir -p models dataset static templates
```

### Step 3: Place Files
Extract the model_artifacts.zip you downloaded:
```
models/
  ├── bm25_docs.pkl           ← From ZIP
  ├── embeddings.npy          ← From ZIP
  ├── faiss_index.bin         ← From ZIP
  └── activities_processed.csv ← From ZIP

dataset/
  └── dataset.csv             ← Your original dataset

static/
  └── api-integration.js       ← Copy file 3_API_INTEGRATION.js here

templates/
  └── index.html              ← Copy your existing HTML here

app_optimized.py              ← Copy file 2_FLASK_BACKEND.py here (rename to app_optimized.py)
requirements.txt              ← Copy requirements.txt here
```

### Step 4: Create Virtual Environment
```bash
# macOS/Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate
```

### Step 5: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 6: Update HTML (Your index.html)
Add this **before closing </body> tag**:
```html
<script src="/static/api-integration.js"></script>
```

---

## PHASE 3: RUN LOCALLY (5 minutes)

### Step 1: Start Backend
```bash
python app_optimized.py
```

### Expected Output:
```
======================================================================
Loading Pre-trained Models from Colab
======================================================================
✓ Loaded 150 activities
✓ BM25 ready (150 docs)
✓ Embeddings ready ((150, 384))
✓ FAISS ready (150 vectors)
✓ Model loaded
======================================================================
✓ ALL MODELS LOADED - READY FOR RECOMMENDATIONS!
======================================================================
Starting on http://localhost:5000
```

### Step 2: Open Browser
```
http://localhost:5000
```

### Step 3: Check Console
- Press F12 (Developer Tools)
- Go to Console tab
- Should show: `✓ Backend connected. Activities loaded: XXX`

### Step 4: Test Search
- Type in search bar: "outdoor activities"
- Press Enter
- Should get results in < 1 second

### Step 5: Test Group Recommendations
- Add group members (ages, interests)
- Click "Get Recommendations"
- Should get personalized results

---

## FILE LOCATIONS CHECKLIST

```
family-activity-planner/
│
├── models/                           ✓ Extract ZIP here
│   ├── bm25_docs.pkl
│   ├── embeddings.npy
│   ├── faiss_index.bin
│   └── activities_processed.csv
│
├── dataset/                          ✓ Your data
│   └── dataset.csv
│
├── static/                           ✓ Frontend assets
│   └── api-integration.js            (Rename from 3_API_INTEGRATION.js)
│
├── templates/                        ✓ HTML
│   └── index.html                    (With <script> tag added)
│
├── app_optimized.py                  ✓ Backend Flask app
│   (Rename from 2_FLASK_BACKEND.py)
│
├── requirements.txt                  ✓ Dependencies
│
└── venv/                             ✓ Virtual environment
    (Created with python -m venv venv)
```

---

## HOW IT WORKS

### Data Flow:

User searches for "outdoor activities" with group members (Alice 6yr, Bob 9yr)
    ↓
Frontend sends POST to http://localhost:5000/api/recommend
    ↓
Backend receives:
  - query: "outdoor activities"
  - group_members: [{"age": 6}, {"age": 9}]
  - preferences: []
    ↓
Backend processes:
  1. BM25 searches for keyword matches
  2. Dense search finds semantic matches
  3. Reciprocal Rank Fusion merges results
  4. Score each activity:
     - Retrieval score (40%)
     - Age fit for group (30%)
     - Preference match (30%)
  5. Sort by final score
    ↓
Backend returns JSON with top recommendations:
  [
    {"rank": 1, "title": "Nature Scavenger Hunt", "score": 0.87, ...},
    {"rank": 2, "title": "Bike Riding", "score": 0.84, ...},
    ...
  ]
    ↓
Frontend displays recommendations with scores

---

## TROUBLESHOOTING

### "models/embeddings.npy not found"
→ Extract model_artifacts.zip to models/ directory
→ Verify files: ls models/

### "Backend not available" (in browser console)
→ Check: python app_optimized.py is running
→ Check: http://localhost:5000/api/health responds
→ Check: API_BASE_URL in api-integration.js = 'http://localhost:5000/api'

### "ModuleNotFoundError: No module named 'sentence_transformers'"
→ Activate virtual environment: source venv/bin/activate
→ Reinstall: pip install -r requirements.txt

### Slow startup (60+ seconds)
→ You're probably running app.py instead of app_optimized.py
→ Use: python app_optimized.py (which loads pre-trained models)
→ Should start in 2-3 seconds

### Search returns no results
→ Check: dataset.csv exists and is in dataset/ folder
→ Check: CSV has proper columns
→ Test: http://localhost:5000/api/health (should show activity count)

### Browser shows "Cannot GET /"
→ Check: index.html is in templates/ folder
→ Check: Flask app is running (see terminal)
→ Try: curl http://localhost:5000/

---

## WHAT EACH FILE DOES

### 1_COLAB_NOTEBOOK.py
- Loads your dataset from Google Drive
- Builds BM25 keyword index
- Generates Sentence-BERT embeddings
- Creates FAISS similarity index
- Exports all models as ZIP
- **Run this in Google Colab** (split into cells as shown)

### 2_FLASK_BACKEND.py (rename to app_optimized.py)
- Loads pre-trained models from disk
- Listens for API requests
- Implements scoring logic:
  1. Hybrid retrieval (BM25 + Dense)
  2. RRF fusion
  3. Score by group member age/preferences
- Returns JSON recommendations
- **Run this locally in VSCode**

### 3_API_INTEGRATION.js (copy to static/)
- Listens for user searches/clicks
- Sends requests to backend API
- Receives recommendations
- Displays results in browser
- **Add to your HTML**

### requirements.txt
- Lists all Python packages needed
- Install with: pip install -r requirements.txt

---

## COMMAND QUICK REFERENCE

### Google Colab:
```python
# Cell 1: Mount and verify
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install
!pip install -q sentence-transformers faiss-cpu rank-bm25 scikit-learn pandas numpy scipy torch

# Cell 3: Run training (paste all code from 1_COLAB_NOTEBOOK.py)
```

### VSCode Terminal:
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python app_optimized.py

# Test
curl http://localhost:5000/api/health
```

### Browser:
```
Open: http://localhost:5000
Press F12: Check console for "✓ Backend connected"
Search: Type and press Enter
```

---

## EXPECTED PERFORMANCE

| Operation | Time |
|-----------|------|
| Backend startup | 2-3 seconds |
| First API call | 0.5-1 second |
| Subsequent calls | 0.2-0.8 seconds |
| Browser response | < 1 second |

---

## SUCCESS CHECKLIST

- [ ] Colab notebook runs without errors
- [ ] Models downloaded as model_artifacts.zip
- [ ] Files extracted to correct folders
- [ ] Virtual environment created and activated
- [ ] Dependencies installed (pip install -r requirements.txt)
- [ ] Backend starts: python app_optimized.py
- [ ] Browser shows frontend: http://localhost:5000
- [ ] Console shows: "✓ Backend connected"
- [ ] Search returns results in < 1 second
- [ ] Group recommendations work correctly
- [ ] No errors in terminal or console

---

You're ready to go! Follow this guide step-by-step and you'll have a working system! 🚀
