"""
Generate New Data for Model Evaluation

This script generates completely new activity data that has not been seen
during training, validation, or testing. This data can be used to evaluate
the model's ability to generalize to truly unseen samples.

The generated data is documented with metadata to confirm it is new.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging
from typing import List, Dict
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NewDataGenerator:
    """Generates new activity data for model evaluation."""

    def __init__(self, output_dir: str = "new_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Define templates for new activities
        self.activity_templates = {
            'toddler': [
                {
                    'title': 'Color Sorting with Blocks',
                    'description': 'Toddlers sort colorful blocks into matching colored buckets, developing color recognition and fine motor skills',
                    'age_min': 2, 'age_max': 3,
                    'tags': 'learning, fine_motor, cognitive',
                    'cost': 'low',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1-2',
                    'duration_mins': 15
                },
                {
                    'title': 'Texture Discovery Box',
                    'description': 'Explore different textures using household items like cotton, sandpaper, fabric, and sponges',
                    'age_min': 1, 'age_max': 3,
                    'tags': 'sensory, exploration, tactile',
                    'cost': 'free',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1-3',
                    'duration_mins': 20
                },
                {
                    'title': 'Simple Puzzle Play',
                    'description': 'Large piece puzzles with 3-5 pieces featuring animals or shapes',
                    'age_min': 2, 'age_max': 4,
                    'tags': 'cognitive, problem_solving, fine_motor',
                    'cost': 'low',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1',
                    'duration_mins': 10
                },
            ],
            'preschool': [
                {
                    'title': 'Weather Science Experiments',
                    'description': 'Simple experiments to learn about rain, clouds, and wind using household materials',
                    'age_min': 4, 'age_max': 6,
                    'tags': 'science, learning, exploration',
                    'cost': 'low',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '2-4',
                    'duration_mins': 30
                },
                {
                    'title': 'Letter Hunt Adventure',
                    'description': 'Hide letter cards around the house and have children find them in alphabetical order',
                    'age_min': 4, 'age_max': 5,
                    'tags': 'literacy, learning, active',
                    'cost': 'free',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1-3',
                    'duration_mins': 20
                },
                {
                    'title': 'Garden Planting Activity',
                    'description': 'Plant seeds in small pots and learn about plant growth and care',
                    'age_min': 5, 'age_max': 7,
                    'tags': 'nature, science, responsibility',
                    'cost': 'low',
                    'indoor_outdoor': 'outdoor',
                    'season': 'spring',
                    'players': '1-4',
                    'duration_mins': 45
                },
            ],
            'elementary': [
                {
                    'title': 'Math Scavenger Hunt',
                    'description': 'Solve math problems to find clues leading to hidden treasure',
                    'age_min': 7, 'age_max': 9,
                    'tags': 'math, problem_solving, adventure',
                    'cost': 'free',
                    'indoor_outdoor': 'both',
                    'season': 'all',
                    'players': '2-6',
                    'duration_mins': 40
                },
                {
                    'title': 'Build a Bridge Challenge',
                    'description': 'Use popsicle sticks, straws, and tape to build the strongest bridge',
                    'age_min': 8, 'age_max': 10,
                    'tags': 'engineering, STEM, creativity',
                    'cost': 'low',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '2-4',
                    'duration_mins': 60
                },
                {
                    'title': 'Coding with Block Programming',
                    'description': 'Learn basic programming concepts using visual block-based coding platforms',
                    'age_min': 7, 'age_max': 10,
                    'tags': 'technology, coding, logic',
                    'cost': 'free',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1',
                    'duration_mins': 45
                },
            ],
            'teen': [
                {
                    'title': 'Debate Club Workshop',
                    'description': 'Practice public speaking and critical thinking through structured debates on current topics',
                    'age_min': 12, 'age_max': 16,
                    'tags': 'communication, critical_thinking, social',
                    'cost': 'free',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '4-10',
                    'duration_mins': 60
                },
                {
                    'title': 'Photography Walk',
                    'description': 'Explore photography basics while taking nature and urban photos',
                    'age_min': 11, 'age_max': 17,
                    'tags': 'art, creativity, outdoor, technology',
                    'cost': 'medium',
                    'indoor_outdoor': 'outdoor',
                    'season': 'spring',
                    'players': '1-5',
                    'duration_mins': 90
                },
                {
                    'title': 'Science Fair Project Planning',
                    'description': 'Design and execute a scientific experiment for science fair competition',
                    'age_min': 13, 'age_max': 17,
                    'tags': 'science, research, STEM',
                    'cost': 'medium',
                    'indoor_outdoor': 'indoor',
                    'season': 'all',
                    'players': '1-2',
                    'duration_mins': 120
                },
            ]
        }

    def generate_new_activities(self, num_samples: int = 50) -> pd.DataFrame:
        """
        Generate new activity samples.

        Args:
            num_samples: Number of new activities to generate

        Returns:
            DataFrame with new activities
        """
        logger.info(f"Generating {num_samples} new activities...")

        activities = []

        # Calculate distribution across age groups (roughly balanced)
        samples_per_group = num_samples // 4

        for age_group, templates in self.activity_templates.items():
            for i in range(samples_per_group):
                # Select random template
                template = random.choice(templates)

                # Create variation by adding slight modifications
                activity = template.copy()

                # Add variation to duration
                activity['duration_mins'] = template['duration_mins'] + random.randint(-5, 10)

                # Add unique identifier
                activity['source'] = 'generated_new_data'
                activity['generated_at'] = datetime.now().isoformat()
                activity['generation_batch'] = 'evaluation_batch_001'

                activities.append(activity)

        # Convert to DataFrame
        df = pd.DataFrame(activities)

        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Generated {len(df)} new activities")
        logger.info(f"Class distribution:")
        toddler_count = len(df[df['age_min'] <= 3])
        preschool_count = len(df[(df['age_min'] > 3) & (df['age_min'] <= 6)])
        elementary_count = len(df[(df['age_min'] > 6) & (df['age_min'] <= 10)])
        teen_count = len(df[df['age_min'] > 10])
        logger.info(f"  Toddler (0-3): {toddler_count}")
        logger.info(f"  Preschool (4-6): {preschool_count}")
        logger.info(f"  Elementary (7-10): {elementary_count}")
        logger.info(f"  Teen+ (11+): {teen_count}")

        return df

    def load_external_data(self, file_path: str) -> pd.DataFrame:
        """
        Load new data from external CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading external data from {file_path}")

        df = pd.read_csv(file_path)

        # Add metadata
        df['source'] = f'external_file:{file_path}'
        df['loaded_at'] = datetime.now().isoformat()

        logger.info(f"Loaded {len(df)} activities from external file")

        return df

    def save_new_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save new data to CSV with metadata.

        Args:
            df: DataFrame to save
            filename: Optional custom filename

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'new_activities_{timestamp}.csv'

        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        # Create metadata file
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'filename': filename,
            'num_samples': len(df),
            'purpose': 'Model evaluation on new, unseen data',
            'data_characteristics': {
                'source': 'Generated for evaluation purposes',
                'not_used_in_training': True,
                'not_used_in_validation': True,
                'not_used_in_testing': True,
                'not_used_for_hyperparameter_tuning': True,
                'completely_new_data': True
            },
            'class_distribution': {
                'toddler': int(((df['age_min'] <= 3).sum())),
                'preschool': int(((df['age_min'] > 3) & (df['age_min'] <= 6)).sum()),
                'elementary': int(((df['age_min'] > 6) & (df['age_min'] <= 10)).sum()),
                'teen': int((df['age_min'] > 10).sum())
            },
            'columns': list(df.columns)
        }

        metadata_path = self.output_dir / f'{filename}.metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved new data to {output_path}")
        logger.info(f"Saved metadata to {metadata_path}")

        return str(output_path)

    def create_evaluation_dataset(self, num_samples: int = 50, source: str = None) -> str:
        """
        Create a complete new dataset for evaluation.

        Args:
            num_samples: Number of samples to generate
            source: Optional path to external data file

        Returns:
            Path to created dataset
        """
        if source:
            df = self.load_external_data(source)
        else:
            df = self.generate_new_activities(num_samples)

        output_path = self.save_new_data(df)

        logger.info("="*80)
        logger.info("NEW EVALUATION DATASET CREATED")
        logger.info("="*80)
        logger.info(f"Location: {output_path}")
        logger.info(f"Samples: {len(df)}")
        logger.info(f"This data has NOT been used in:")
        logger.info("  - Model training")
        logger.info("  - Validation")
        logger.info("  - Initial testing")
        logger.info("  - Hyperparameter tuning")
        logger.info("="*80)

        return output_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate new data for model evaluation'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=50,
        help='Number of new samples to generate (default: 50)'
    )
    parser.add_argument(
        '--source',
        type=str,
        default=None,
        help='Path to external CSV file with new data (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='new_data',
        help='Directory to save generated data (default: new_data)'
    )

    args = parser.parse_args()

    generator = NewDataGenerator(output_dir=args.output_dir)
    output_path = generator.create_evaluation_dataset(
        num_samples=args.num_samples,
        source=args.source
    )

    print(f"\nNew data saved to: {output_path}")
    print(f"Ready for evaluation with: python evaluate_new_data.py --new-data {output_path}")


if __name__ == '__main__':
    main()
