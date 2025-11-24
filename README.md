# 🎯 Activity Planner - AI-Powered Family Activity Search

## Overview

Activity Planner is an intelligent recommendation system that uses **Sentence-BERT embeddings** and **hybrid search** to help families find perfect activities based on group member profiles and preferences.

### Key Features

- **🤖 AI-Powered Search**: Uses Sentence-BERT embeddings for semantic understanding
- **🔗 Linkage Scoring**: Calculates connections between search queries, group members, and activities
- **🎯 Hybrid Retrieval**: Combines BM25 keyword search with dense semantic search
- **👥 Group-Aware**: Considers ages, preferences, and group size
- **📊 Transparent Scoring**: Shows detailed breakdown of why activities match
- **💾 Database Integration**: Stores activities and search history

## Architecture

### Search Pipeline

```
Query + Group Members
         ↓
[1] Hybrid Retrieval (BM25 + Sentence-BERT + FAISS)
         ↓
[2] Reciprocal Rank Fusion (RRF)
         ↓
[3] Linkage Score Calculation
    - Semantic Linkage (query ↔ activity)
    - Age Fit Linkage (members ↔ activity age range)
    - Preference Linkage (tags ↔ activity)
    - Group Size Linkage (players ↔ group size)
    - Context Linkage (season, location, etc.)
         ↓
[4] Final Ranking & Display
```

### Linkage Scoring Formula

```python
Overall Linkage = 0.35 × Semantic + 0.25 × Age Fit +
                  0.20 × Preference + 0.10 × Group Size +
                  0.10 × Context

Final Score = 0.30 × Retrieval Score + 0.70 × Overall Linkage
```

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Models

Run the training script to generate embeddings and indexes:

```bash
python train_model.py --dataset dataset/dataset.csv --output-dir models
```

**Adjustable Parameters:**

```bash
python train_model.py \
  --dataset dataset/dataset.csv \
  --output-dir models \
  --model all-MiniLM-L6-v2 \
  --batch-size 32 \
  --bm25-k1 1.5 \
  --bm25-b 0.75 \
  --rrf-k 60 \
  --weight-retrieval 0.4 \
  --weight-age 0.3 \
  --weight-preference 0.3
```

**Available Sentence-BERT Models:**
- `all-MiniLM-L6-v2` (fast, 384 dim)
- `all-mpnet-base-v2` (slower, better quality, 768 dim)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

### 3. Initialize Database

The database will be automatically initialized on first run, or manually:

```bash
python database.py
```

### 4. Run Application

```bash
python app.py --dataset dataset/dataset.csv --models models --port 5000
```

Visit: http://localhost:5000

## Usage

### 1. Add Group Members

- Enter group name
- Add each member with their name and age
- Optionally add preference tags (e.g., "outdoor", "creative", "sports")

### 2. Search Activities

- Enter a natural language query (e.g., "fun outdoor activities for kids")
- The AI will:
  - Understand semantic meaning
  - Calculate linkage scores with group members
  - Rank activities by overall match

### 3. View Results

Each activity shows:
- **Rank** and **Title**
- **Metadata**: players, duration, age range, cost, location type
- **Match Scores**:
  - Overall Match (final score)
  - Semantic Linkage (query relevance)
  - Age Fit (age compatibility)
  - Preference Match (tag alignment)

## API Endpoints

### Create Group Session

```bash
POST /api/session/create
{
  "group_name": "Smith Family",
  "members": [
    {"name": "Sarah", "age": 7},
    {"name": "Tom", "age": 10}
  ],
  "preferences": ["outdoor", "creative"]
}
```

### Search Activities

```bash
POST /api/search
{
  "session_id": "uuid-here",
  "query": "fun outdoor activities",
  "top_k": 10
}
```

### Get All Activities

```bash
GET /api/activities
```

### Health Check

```bash
GET /api/health
```

## Database Schema

### Activities Table

Stores all activity details for embedding calculations:

- `id`, `title`, `age_min`, `age_max`, `duration_mins`
- `tags`, `cost`, `indoor_outdoor`, `season`
- `materials_needed`, `how_to_play`, `players`, `parent_caution`
- `full_text` (combined for embeddings)

### Group Sessions Table

Stores user group information:

- `session_id`, `group_name`, `members_json`, `preferences_json`

### Search History Table

Tracks all searches for analytics:

- `session_id`, `query`, `results_json`, `num_results`

### Activity Rankings Table

Stores detailed linkage scores:

- `session_id`, `activity_id`, `query`
- `retrieval_score`, `age_fit_score`, `preference_score`, `final_score`
- `rank_position`

## Training Configuration

All training parameters are configurable via `training_config.json`:

```json
{
  "dataset_path": "dataset/dataset.csv",
  "output_dir": "models",
  "sentence_bert_model": "all-MiniLM-L6-v2",
  "bm25_k1": 1.5,
  "bm25_b": 0.75,
  "embedding_batch_size": 32,
  "normalize_embeddings": true,
  "faiss_metric": "L2",
  "rrf_k": 60,
  "weight_retrieval": 0.4,
  "weight_age_fit": 0.3,
  "weight_preference": 0.3
}
```

## Model Files

After training, the following files are generated in `models/`:

- `bm25_docs.pkl` - Tokenized documents for BM25 keyword search
- `embeddings.npy` - Sentence-BERT activity embeddings (numpy array)
- `faiss_index.bin` - FAISS similarity search index
- `activities_processed.csv` - Processed dataset copy
- `training_config.json` - Training parameters used

## Adjusting Parameters

### BM25 Parameters

- `k1` (1.0-2.0): Controls term frequency saturation
  - Higher = more weight on term frequency
  - Default: 1.5

- `b` (0.0-1.0): Controls document length normalization
  - 0 = no normalization
  - 1 = full normalization
  - Default: 0.75

### RRF Parameter

- `k` (30-100): Reciprocal Rank Fusion constant
  - Higher = more conservative fusion
  - Default: 60

### Linkage Weights

Adjust in code or via parameters:

```python
weight_retrieval = 0.3   # How much to trust retrieval scores
weight_linkage = 0.7     # How much to trust linkage scores

# Linkage component weights:
semantic = 0.35    # Query-activity semantic similarity
age_fit = 0.25     # Age compatibility
preference = 0.20  # Tag/preference match
group_size = 0.10  # Group size compatibility
context = 0.10     # Contextual factors
```

## Performance

- **Dataset**: 100+ activities
- **Embedding Generation**: ~30 seconds
- **Search Latency**: <100ms per query
- **Memory**: ~50MB for models

## Project Structure

```
activity-planner/
├── dataset/
│   └── dataset.csv           # Activity dataset
├── models/
│   ├── bm25_docs.pkl         # BM25 index
│   ├── embeddings.npy        # Sentence-BERT embeddings
│   ├── faiss_index.bin       # FAISS index
│   └── training_config.json  # Training parameters
├── templates/
│   └── planner.html          # Frontend interface
├── train_model.py            # Training script with adjustable parameters
├── database.py               # Database management
├── app.py                    # Flask web application
├── requirements.txt          # Python dependencies
└── README_NEW.md             # This file
```

## Example Queries

- "fun outdoor activities for kids"
- "creative indoor games for rainy days"
- "educational activities for toddlers"
- "high energy sports for teenagers"
- "quiet activities for evening time"

## Technologies Used

- **Sentence-BERT**: Semantic embeddings (sentence-transformers)
- **BM25**: Keyword-based search (rank-bm25)
- **FAISS**: Fast similarity search (faiss-cpu)
- **Flask**: Web framework
- **SQLite**: Database storage
- **NumPy/Pandas**: Data processing

## Model Evaluation & Baseline Comparison

### Evaluating on New Data

The project includes a comprehensive evaluation script (`evaluate_new_data.py`) that tests both the **Neural Network classifier** and a **Random Forest baseline** on completely new, unseen data.

#### Random Forest Baseline

**Configuration:**
- **100 trees** (n_estimators=100)
- **Max depth: 20**
- **Purpose:** Simple, interpretable, minimal tuning baseline for comparison

The Random Forest baseline provides:
- Fast training and inference
- Interpretable feature importance
- Strong performance on tabular data
- Minimal hyperparameter tuning required

#### Running Evaluation

```bash
python evaluate_new_data.py \
  --new-data path/to/new_data.csv \
  --data-source "Description of data source" \
  --model-dir models \
  --output-dir new_data_evaluation
```

#### What Gets Evaluated

1. **Neural Network Performance**
   - Accuracy, Precision, Recall, F1-Score
   - Per-class metrics
   - Confusion matrix
   - Prediction confidence statistics

2. **Random Forest Baseline Performance**
   - Same metrics as Neural Network
   - Direct comparison with Neural Network

3. **Comparison Analysis**
   - Neural Network vs Random Forest
   - New data vs Original test set baseline
   - Performance assessment with rubric scoring

#### Evaluation Outputs

After running evaluation, you'll find in `new_data_evaluation/`:

**Reports:**
- `NEW_DATA_EVALUATION_REPORT.md` - Comprehensive markdown report with:
  - Overall performance metrics (both models)
  - Model comparison (Neural Network vs Random Forest)
  - Baseline comparison (new data vs original test set)
  - Per-class performance breakdown
  - Performance assessment and recommendations
  - Rubric scoring (0-10)

**Visualizations** in `figures/`:
- `confusion_matrix_neural_network.png`
- `confusion_matrix_random_forest.png`
- `per_class_performance_neural_network.png`
- `per_class_performance_random_forest.png`
- `confidence_analysis_neural_network.png`
- `confidence_analysis_random_forest.png`
- `baseline_vs_new_comparison.png`
- `neural_network_vs_random_forest.png`

**JSON Results:**
- `new_data_evaluation_results.json` - Raw metrics for both models

#### Interpretation

The evaluation provides a **rubric score (0-10)** based on performance:

- **10/10**: Performance meets/exceeds expectations
- **7/10**: Reasonable but below expectations
- **4/10**: Inconsistent performance
- **2/10**: Poor performance
- **0/10**: Failed evaluation

The comparison between Neural Network and Random Forest helps determine:
- Whether the additional complexity of the Neural Network is justified
- If a simpler baseline model (Random Forest) is sufficient
- Which model generalizes better to new data

## Future Enhancements

- [ ] Add calendar integration
- [ ] Export to PDF/CSV/ICS
- [ ] Multi-language support
- [ ] Image/video content for activities
- [ ] User feedback loop for model improvement
- [ ] Advanced filters (location, weather, equipment)
- [ ] Social sharing features

## References

- **Sentence-BERT**: Reimers & Gurevych, 2019
- **BM25**: Robertson & Zaragoza, 2009
- **FAISS**: Johnson et al., 2017
- **RRF**: Cormack et al., 2009

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.

## Support

For issues or questions, please open a GitHub issue.

---

**Built with ❤️ for families seeking quality activities together**
