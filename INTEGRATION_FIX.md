# Model Integration Fix

## Problem Identified

The recommendation, search engine, and planner models were not being used because a **mock function in index.html was overriding the real API integration**.

### Specific Issues Fixed

1. **Function Override (CRITICAL)**
   - **Location:** `templates/index.html` lines 3149-3180
   - **Issue:** A mock `getPersonalizedRecommendations()` function was showing hardcoded alert messages instead of calling the actual ML backend
   - **Fix:** Removed the mock function to allow the real API integration from `api-integration.js` to execute

## Architecture Overview

The application uses a sophisticated 3-stage ML recommendation system:

### Backend Components (`app_optimized.py`)

1. **BM25 Retriever** - Keyword-based search using rank-bm25
2. **Dense Retriever** - Semantic search using Sentence-BERT (all-MiniLM-L6-v2) + FAISS
3. **Hybrid Ranking** - Combines both using Reciprocal Rank Fusion (RRF)

### Model Artifacts (in `models/` directory)

- `bm25_docs.pkl` - Tokenized documents for BM25
- `embeddings.npy` - 384-dimensional Sentence-BERT embeddings
- `faiss_index.bin` - FAISS index for fast similarity search
- `activities_processed.csv` - Activity dataset with 12 columns

### API Flow

```
User clicks "Get Recommendations" button
    ↓
api-integration.js → getPersonalizedRecommendations()
    ↓
POST /api/recommend with group member data
    ↓
app_optimized.py → OptimizedRecommenderAPI.recommend()
    ├→ BM25 keyword search
    ├→ Dense semantic search
    ├→ Reciprocal Rank Fusion
    ├→ Age-based filtering (40% weight)
    └→ Preference matching (30% weight)
    ↓
JSON response with ranked recommendations
    ↓
showResultsPage() displays results
```

## Changes Made

### File: `templates/index.html`

**Removed lines 3149-3180:**
- Deleted mock `getPersonalizedRecommendations()` function that showed alerts
- Replaced with comment explaining function is now in api-integration.js

**Why this fixes the issue:**
- The button event listener (line 3709) calls `getPersonalizedRecommendations()`
- Without the mock, this resolves to `window.getPersonalizedRecommendations` from api-integration.js
- The real function makes proper API calls to the ML backend

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend Server

```bash
python app_optimized.py --dataset dataset/dataset.csv --models models --port 5000
```

The backend will:
- Load 12+ activities from dataset
- Initialize BM25, FAISS, and Sentence-BERT models
- Start Flask server on http://localhost:5000

### 3. Open Frontend

Navigate to http://localhost:5000 in your browser

### 4. Use the Application

1. Click "My Group" in navigation
2. Add group members using:
   - "Quick Add Templates" buttons, OR
   - "Add Member" form
3. Click "Get Recommendations" button
4. View personalized activity recommendations based on:
   - Group age ranges
   - Member interests
   - Special needs and allergies
   - Activity preferences

## Verification

The integration is working correctly if:

1. Backend logs show:
   ```
   ✓ ALL MODELS LOADED - READY FOR RECOMMENDATIONS!
   ```

2. Browser console shows (F12 → Console):
   ```
   ✓ Backend connected
   === GET RECOMMENDATIONS CLICKED ===
   ✓ Got recommendations: 15
   ```

3. Results page displays with:
   - Group profile summary
   - Activity cards with match scores
   - Ranked recommendations (1-15)

## Technical Details

### API Endpoint: POST /api/recommend

**Request:**
```json
{
  "query": "outdoor sports",
  "group_members": [
    {
      "age": 10,
      "name": "Child",
      "overall_ability": 5,
      "interests": ["sports", "outdoor"],
      "special_needs": [],
      "allergies": ""
    }
  ],
  "preferences": ["sports", "outdoor"],
  "top_k": 15
}
```

**Response:**
```json
{
  "status": "success",
  "query": "outdoor sports",
  "recommendations": [
    {
      "rank": 1,
      "recommendation_score": 0.85,
      "title": "Activity Name",
      "description": "...",
      "duration_mins": 30,
      "cost": "low",
      "location": "outdoor",
      "age_min": 5,
      "age_max": 12
    }
  ],
  "count": 15,
  "timestamp": "2025-11-06T..."
}
```

### Scoring Formula

Final score = (0.4 × retrieval_score) + (0.3 × age_fit) + (0.3 × preference_match)

Where:
- **retrieval_score**: RRF score from BM25 + Dense retrieval
- **age_fit**: How well activity age range matches group ages
- **preference_match**: Overlap between activity tags and group interests

## Files Modified

- `templates/index.html` - Removed mock function override

## Files Verified (No changes needed)

- `static/api-integration.js` - Real API integration (working correctly)
- `app_optimized.py` - Backend ML models (working correctly)
- `models/*` - Pre-trained model artifacts (present and valid)
- `dataset/dataset.csv` - Activity dataset (present and valid)

## Summary

The fix was simple but critical: **removing a single mock function** that was preventing the real ML models from being used. The entire backend infrastructure (BM25, Sentence-BERT, FAISS) was already properly implemented and ready to use - it just needed the frontend to actually call it instead of showing a fake alert message.
