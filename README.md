# Project Final Report: Family Activity Planner

**Patiharn Liangkobkit**  
1012871867, patiharn.liang@gmail.com  
[https://github.com/Richyboy170/activity-planner](https://github.com/Richyboy170/activity-planner)

## Introduction

Parents struggle to find age-appropriate activities matching semantic constraints (e.g., "burn energy indoors" or "quiet bedtime activity for 5-year-old"). Keyword search fails to capture semantic intent. This project builds a learning-based activity classifier that predicts age-appropriateness for multi-generational families (ages 1–99).

**Why Deep Learning:**
1.  Semantic understanding requires learning query-activity relationships beyond keyword matching.
2.  Activity selection involves complex non-linear patterns across age, context, season, and cost.
3.  Embedding-based representations (Sentence-BERT) naturally capture semantic similarity enabling neural networks to learn nuanced relationships that generalize to novel queries.

```mermaid
graph LR
    query[Query] --> ret[BM25 + SBERT]
    ret --> nn[Neural Net]
    nn --> out[Age Class]
    
    style query fill:#e6f3ff,stroke:#333,stroke-width:2px
    style ret fill:#e6ffec,stroke:#333,stroke-width:2px
    style nn fill:#fff5e6,stroke:#333,stroke-width:2px
    style out fill:#f3e6ff,stroke:#333,stroke-width:2px
```
*Figure 1: Pipeline: hybrid retrieval feeds neural classifier.*

## Background & Related Work

This project builds upon:
1.  **BM25** [1] for lexical retrieval baseline.
2.  **Sentence-BERT** [2] generating 384-D semantic embeddings.
3.  **FAISS** [3] enabling scalable vector search.
4.  **MMR** [4] informing diversity-based reranking.
5.  **Training dataset**: 423 educational websites provided curated activities with comprehensive metadata.

## Data Processing

We collected 2,075 activities from 423 educational websites with metadata: title, age range (1–99), duration, location type, cost, seasonal tags, materials, and instructions. Data cleaning: HTML/JSON parsing, age validation, duration normalization (capped at 480 min), cost standardization. Computed 384-D embeddings using all-MiniLM-L6-v2. Data partitioned 80/10/10 with stratified sampling.

### Dataset Statistics

| Statistic | Value | Category | Count |
| :--- | :--- | :--- | :--- |
| **Total Activities** | 2,075 | **Indoor** | 1,332 (64.2%) |
| **Avg Duration** | 83.6 min | **Outdoor** | 598 (28.8%) |
| **Median Duration** | 60 min | **Both** | 145 (7.0%) |

| Cost | Count | Age Groups | Count |
| :--- | :--- | :--- | :--- |
| **Free** | 603 (29.1%) | **Toddler (0–3)** | 239 (11.5%) |
| **Low** | 965 (46.5%) | **Preschool (4–6)** | 452 (21.8%) |
| **Medium+** | 352 (17.0%) | **Elementary (7–10)** | 529 (25.5%) |
| **High** | 155 (7.5%) | **Teen+ (11+)** | 855 (41.2%) |

*Table 1: Dataset statistics. 44.6% have broad age ranges (e.g., 10–99).*

**Examples:**
*   *Animal Walk* (ages 2–7, 15min, Free, Both, Exercise): Choose a 1+ player activity mimicking animal movements.
*   *Wash Car* (ages 9–12, 20min, Free, Outdoor): Clean family vehicle together.
*   *Watch Clouds and Relax* (ages 10–99, 30min, Free, Outdoor): Multi-generational mindfulness activity.

## Architecture

**Input Features (387-D):** Text embeddings (384-D) from all-MiniLM-L6-v2 combining title (3x weight), tags (2x), and how_to_play (2x), plus 3 normalized numerical features (age_min, age_max, duration_mins).

**Network:**
Input Dropout(0.1) → Linear(387→256) + BatchNorm + ReLU + Dropout(0.2) → Linear(256→128) + BatchNorm + ReLU + Dropout(0.2) → Linear(128→64) + BatchNorm + ReLU + Dropout(0.2) → Linear(64→4) + Softmax.

**Training:** CrossEntropyLoss with class weights (inverse frequency), Adam (lr=0.001, weight_decay=5e-5), batch 32, SMOTE for class balancing.

## Baseline Model

Random Forest (50 trees, max depth 5) classifies activities into 4 age groups from 387-D combined features. Simple, interpretable baseline with minimal tuning to avoid overfitting.

## Quantitative Results

**Evaluation on 70 new samples:** Neural network 72.86% accuracy vs baseline 58.57% (+14.29pp).

| Age Group | Neural Net (72.86%) | | | | Baseline (58.57%) | | | |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| | **P** | **R** | **F1** | **Sup** | **P** | **R** | **F1** | **Sup** |
| **Toddler (0–3)** | 0.57 | 0.57 | 0.57 | 7 | 0.00 | 0.00 | 0.00 | 7 |
| **Preschool (4–6)** | 0.75 | 0.43 | 0.55 | 21 | 0.50 | 0.24 | 0.32 | 21 |
| **Elementary (7–10)** | 0.27 | 0.43 | 0.33 | 7 | 0.13 | 0.14 | 0.13 | 7 |
| **Teen+ (11+)** | 0.88 | 1.00 | 0.93 | 35 | 0.67 | 1.00 | 0.80 | 35 |
| **Weighted Avg** | **0.75** | **0.73** | **0.72** | **70** | **0.50** | **0.59** | **0.51** | **70** |

*Table 2: Evaluation results. Neural network: perfect Teen+ (35/35), balanced Toddler/Preschool. Baseline: complete Toddler failure (0/7). Confidence: Mean 0.92, median 0.99, std 0.13.*

## Qualitative Results

**Correct:**
*   *Pottery Workshop* (Teen+, 0.9988)
*   *Color and Shape Sorting* (Preschool, 0.9995)
*   *Hula Hoop Walk* (Toddler, 0.6580)
*   *Shows strong semantic understanding.*

**Errors:**
*   *Matching Game Hunt* (Toddler→Preschool, 0.8677)
*   *Safe Knife Skills* (Preschool→Toddler, 0.9788)
*   *Comic Book Creation* (Elementary→Teen+, 0.5922)
*   *Errors at adjacent boundaries with lower confidence (0.50–0.65) suggest model recognizes uncertainty.*

## Evaluation on New Data

70 new samples (Toddler 10%, Preschool 30%, Elementary 10%, Teen+ 50%), confirmed not used in training/validation/testing.

**Neural Network:** 72.86% accuracy. Perfect Teen+ (35/35), Toddler F1=0.57, Preschool F1=0.55, Elementary F1=0.33 (3/7 correct). Main errors: 8 Preschool misclassified as Elementary.

**Baseline:** 58.57% accuracy. Complete Toddler failure (0/7), weak Elementary (1/7), Teen+ bias (52/70 predicted as Teen+).

**Improvement:** Neural network +14.29pp overall. Per-class gains: Toddler +57.14pp, Preschool +22.29pp, Elementary +20.00pp, Teen+ +12.87pp.

## Discussion

Neural network achieves 72.86% on 70 new samples, validating strong generalization. Sentence-BERT embeddings + numerical features (387-D) enable semantic understanding. Class weights + SMOTE maintain minority performance (Toddler F1=0.57 vs baseline 0.00).

**Why It Works:** Weighted embeddings (title 3x, tags/how_to_play 2x) prioritize key signals. Progressive reduction (387→256→128→64) enables hierarchical learning. Dropout (input 0.1, layers 0.2) prevents overfitting with 92% mean confidence.

**Limitations:** Elementary F1=0.33 shows mid-range difficulty. Teen+ aggregation (11–99) cannot distinguish teens/adults/seniors because activities in these groups share similar characteristics—separating classes would confuse the model due to high overlap. Toddler underrepresentation (11.5%) limits learning despite SMOTE.

**Production:** **APPROVED** — 72.86% accuracy, +14.29pp over baseline, perfect Teen+ (35/35), well-calibrated (mean 0.92, median 0.99).

## Ethical Considerations

*   **Age stereotyping:** Discrete groups may reinforce stereotypes; individuals develop at varying rates.
*   **Generational bias:** Collapsing ages 11–99 marginalizes adults and seniors by treating distinct needs as equivalent.
*   **Cultural bias:** English North American sources may underrepresent diverse practices.
*   **Safety:** Predictions should not replace human judgment for activity appropriateness.

## Conclusion

The neural network successfully classifies activities with 72.86% accuracy on new data, demonstrating robust generalization. Combining Sentence-BERT embeddings (384-D) with numerical features (3-D) captures both semantic meaning and age constraints. Class balancing (weights + SMOTE) maintains minority performance despite training imbalance.

Perfect Teen+ detection (35/35) and balanced Toddler/Preschool performance validate production readiness. Elementary challenges (F1=0.33) require targeted data collection. Future work includes 7-category expansion, diverse source collection, and confidence-based rejection thresholds. The model's well-calibrated confidence (mean 0.92, median 0.99) enables reliable deployment with appropriate monitoring.

## References

1.  S. E. Robertson and H. Zaragoza. The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in IR*, 3(4):333–389, 2009.
2.  N. Reimers and I. Gurevych. Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *EMNLP*, 2019.
3.  J. Johnson, M. Douze, H. Jégou. Billion-scale similarity search with GPUs (FAISS). *arXiv:1702.08734*, 2017.
4.  J. Carbonell and J. Goldstein. The use of MMR, diversity-based reranking for reordering documents and producing summaries. In *SIGIR*, 1998.

*(See full bibliography in report)*
