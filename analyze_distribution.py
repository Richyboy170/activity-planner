#!/usr/bin/env python3
"""
Analyze distribution differences between training and evaluation datasets
"""

import csv
from collections import Counter

def get_label(age_min, age_max):
    """Classify activity into age group based on midpoint"""
    age_mid = (age_min + age_max) / 2
    if age_mid <= 3.5:
        return 0, 'Toddler (0-3)'
    elif age_mid <= 7:
        return 1, 'Preschool (4-6)'
    elif age_mid <= 11:
        return 2, 'Elementary (7-10)'
    else:
        return 3, 'Teen+ (11+)'

def analyze_dataset(filepath):
    """Analyze a dataset and return statistics"""
    labels = []
    age_mids = []

    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            age_min = float(row['age_min'])
            age_max = float(row['age_max'])
            age_mid = (age_min + age_max) / 2

            label_id, label_name = get_label(age_min, age_max)
            labels.append(label_id)
            age_mids.append(age_mid)

    # Calculate statistics
    label_counts = Counter(labels)
    total = len(labels)

    return {
        'total': total,
        'labels': label_counts,
        'age_mid_mean': sum(age_mids) / len(age_mids) if age_mids else 0,
        'age_mid_min': min(age_mids) if age_mids else 0,
        'age_mid_max': max(age_mids) if age_mids else 0
    }

def print_distribution(name, stats):
    """Print distribution statistics"""
    print(f'\n=== {name} ===')
    print(f'Total samples: {stats["total"]}')
    print(f'\nClass Distribution:')

    label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
    for label_id in range(4):
        count = stats['labels'].get(label_id, 0)
        pct = (count / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f'  {label_names[label_id]}: {count} ({pct:.1f}%)')

    print(f'\nAge Midpoint Statistics:')
    print(f'  Mean: {stats["age_mid_mean"]:.2f}')
    print(f'  Range: {stats["age_mid_min"]:.2f} - {stats["age_mid_max"]:.2f}')

if __name__ == '__main__':
    print('='*70)
    print('DATA DISTRIBUTION ANALYSIS')
    print('='*70)

    # Analyze training data
    train_stats = analyze_dataset('dataset/dataset_augmented.csv')
    print_distribution('TRAINING DATA', train_stats)

    # Analyze evaluation data
    eval_stats = analyze_dataset('dataset/evaluation_dataset.csv')
    print_distribution('EVALUATION DATA', eval_stats)

    # Calculate distribution shift
    print('\n' + '='*70)
    print('DISTRIBUTION SHIFT ANALYSIS')
    print('='*70)

    label_names = ['Toddler (0-3)', 'Preschool (4-6)', 'Elementary (7-10)', 'Teen+ (11+)']
    print('\nClass Distribution Differences:')
    for label_id in range(4):
        train_pct = (train_stats['labels'].get(label_id, 0) / train_stats['total'] * 100)
        eval_pct = (eval_stats['labels'].get(label_id, 0) / eval_stats['total'] * 100)
        diff = eval_pct - train_pct
        print(f'  {label_names[label_id]}:')
        print(f'    Train: {train_pct:.1f}% | Eval: {eval_pct:.1f}% | Diff: {diff:+.1f}%')

    print(f'\nAge Midpoint Mean Shift: {eval_stats["age_mid_mean"] - train_stats["age_mid_mean"]:+.2f} years')
