# Evaluation Results

## Dataset Information
- **Total Samples**: 96
- **Embedding Model**: all-mpnet-base-v2
- **Input Dimension**: 768

## Class Distribution
| Age Group | Count | Percentage |
|-----------|-------|------------|
| Toddler (0-3) | 4 | 4.2% |
| Preschool (4-6) | 50 | 52.1% |
| Elementary (7-10) | 42 | 43.8% |

## Overall Metrics
| Metric | Value |
|--------|-------|
| **Accuracy** | 0.3854 (38.54%) |
| **Precision** | 0.6324 |
| **Recall** | 0.3854 |
| **F1-Score** | 0.4734 |

## Per-Class Performance
| Age Group | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Toddler (0-3) | 1.0000 | 0.2500 | 0.4000 | 4 |
| Preschool (4-6) | 0.7143 | 0.5000 | 0.5882 | 50 |
| Elementary (7-10) | 0.5000 | 0.2619 | 0.3438 | 42 |
| Teen+ (11+) | 0.0000 | 0.0000 | 0.0000 | 0 |

## Confusion Matrix
|  | Toddler (0-3) | Preschool (4-6) | Elementary (7-10) | Teen+ (11+) |
|--|---------------|-----------------|-------------------|-------------|
| **Toddler (0-3)** | 1 | 3 | 0 | 0 |
| **Preschool (4-6)** | 0 | 25 | 11 | 14 |
| **Elementary (7-10)** | 0 | 7 | 11 | 24 |
| **Teen+ (11+)** | 0 | 0 | 0 | 0 |

## Prediction Confidence
| Statistic | Value |
|-----------|-------|
| Mean | 0.7551 |
| Std Dev | 0.1718 |
| Min | 0.3904 |
| Max | 0.9985 |

## Baseline Comparison
| Metric | Value |
|--------|-------|
| Baseline Test Accuracy | 0.0000 |
| New Data Accuracy | 0.3854 |
| Difference | +0.3854 |
