"""
Text Data Augmentation Script
Implements multiple augmentation techniques for NLP datasets:
- Synonym Replacement
- Random Substitution  
- Random Deletion
- Random Swap
- Back Translation

For: Activities Dataset (title, tags, materials_needed, how_to_play)
"""

import pandas as pd
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from typing import List, Tuple
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextAugmenter:
    """
    Handles all text augmentation operations.
    Designed for NLP preprocessing pipelines in ML/AI systems.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize augmenter with seed for reproducibility."""
        random.seed(seed)
        self.seed = seed
    
    # ============= TECHNIQUE 1: SYNONYM REPLACEMENT =============
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """
        Replace random words with their synonyms using WordNet.
        
        Args:
            text: Input text string
            n: Number of words to replace
            
        Returns:
            Augmented text with synonyms
            
        Algorithm:
        1. Tokenize input text into words
        2. For each word, query WordNet for synonyms
        3. Randomly select n words and replace with synonyms
        """
        words = word_tokenize(text.lower())
        random_word_list = list(set([word for word in words 
                                     if word.isalnum() and len(word) > 3]))
        random.shuffle(random_word_list)
        num_replaced = 0
        
        for random_word in random_word_list:
            synonyms = self._get_synonyms(random_word)
            if len(synonyms) >= 1:
                synonym = random.choice(synonyms)
                text = text.replace(random_word, synonym)
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return text
    
    @staticmethod
    def _get_synonyms(word: str) -> List[str]:
        """Extract synonyms from WordNet."""
        synonyms = set()
        for synset in wordnet.synsets(word):
            for lemma in synset.lemmas():
                if lemma.name() != word:
                    synonyms.add(lemma.name().replace('_', ' '))
        return list(synonyms)
    
    # ============= TECHNIQUE 2: RANDOM SUBSTITUTION =============
    def random_substitution(self, text: str, p: float = 0.1) -> str:
        """
        Replace random words with random other words from vocab.
        
        Args:
            text: Input text string
            p: Probability of word substitution (0.0 - 1.0)
            
        Returns:
            Augmented text with random substitutions
            
        Algorithm:
        1. Tokenize text
        2. Build vocabulary from input
        3. For each word, randomly decide if substitution occurs
        4. Replace with random vocabulary word
        """
        words = word_tokenize(text.lower())
        vocab = list(set([w for w in words if w.isalnum()]))
        
        new_words = []
        for word in words:
            if random.uniform(0, 1) < p and word.isalnum() and len(vocab) > 0:
                new_words.append(random.choice(vocab))
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    # ============= TECHNIQUE 3: RANDOM DELETION =============
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """
        Randomly delete words from text with probability p.
        
        Args:
            text: Input text string
            p: Probability of word deletion (0.0 - 1.0)
            
        Returns:
            Augmented text with random deletions
            
        Algorithm:
        1. Tokenize text into words
        2. For each word, randomly decide if deletion occurs with probability p
        3. Filter out deleted words
        4. Rejoin tokens into string
        
        Note: If all words are deleted, return original text
        """
        if len(text.split()) == 1:
            return text
        
        new_words = []
        for word in text.split():
            r = random.uniform(0, 1)
            if r > p:  # Keep word if random > p
                new_words.append(word)
        
        if len(new_words) == 0:
            return text
        
        return ' '.join(new_words)
    
    # ============= TECHNIQUE 4: RANDOM SWAP =============
    def random_swap(self, text: str, n: int = 2) -> str:
        """
        Randomly swap words in text.
        
        Args:
            text: Input text string
            n: Number of random swaps to perform
            
        Returns:
            Augmented text with swapped words
            
        Algorithm:
        1. Tokenize text into words
        2. For n iterations:
           a. Randomly select two different word indices
           b. Swap the words at those positions
        3. Rejoin tokens into string
        """
        words = text.split()
        for _ in range(n):
            if len(words) >= 2:
                idx1, idx2 = random.sample(range(len(words)), 2)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    # ============= TECHNIQUE 5: BACK TRANSLATION =============
    def back_translation(self, text: str, 
                        intermediate_lang: str = 'de') -> str:
        """
        Simulate back translation (translate to intermediate language, then back).
        Uses nlpaug library with translation models.
        
        Args:
            text: Input text string
            intermediate_lang: Language code (e.g., 'de' for German, 'fr' for French)
            
        Returns:
            Back-translated text (simulated via contextual replacement)
            
        Note: For production use, implement with actual translation APIs
        (Google Translate, DeepL) or transformers-based models like MarianMT
        
        Current implementation uses contextual word embeddings as proxy
        for true back-translation without requiring API keys.
        """
        try:
            # Try to use contextual embeddings as approximation
            aug = naw.ContextualWordEmbsAug(
                model_path="bert-base-uncased",
                action="insert",
                device="cpu"
            )
            return aug.augment(text)
        except Exception as e:
            print(f"Back-translation fallback: {e}")
            # Fallback: Use synonym replacement as approximation
            return self.synonym_replacement(text, n=2)


class AugmentationPipeline:
    """
    Orchestrates data augmentation workflow for ML training datasets.
    Converts single samples into multiple augmented variants.
    """
    
    def __init__(self, augmenter: TextAugmenter = None):
        """Initialize pipeline with augmenter instance."""
        self.augmenter = augmenter or TextAugmenter()
    
    def augment_text_multi_technique(self, text: str) -> dict:
        """
        Apply all five augmentation techniques to a single text sample.
        
        Args:
            text: Input text to augment
            
        Returns:
            Dictionary with augmented versions from each technique
        """
        return {
            'original': text,
            'synonym_replacement': self.augmenter.synonym_replacement(text, n=2),
            'random_substitution': self.augmenter.random_substitution(text, p=0.15),
            'random_deletion': self.augmenter.random_deletion(text, p=0.1),
            'random_swap': self.augmenter.random_swap(text, n=2),
            'back_translation': self.augmenter.back_translation(text)
        }
    
    def augment_dataset(self, df: pd.DataFrame, 
                       text_columns: List[str],
                       augmentation_factor: int = 2) -> pd.DataFrame:
        """
        Augment entire dataset across specified text columns.
        
        Args:
            df: Input DataFrame
            text_columns: List of column names to augment
            augmentation_factor: Number of augmented copies per sample (max 5 for all techniques)
            
        Returns:
            DataFrame with original + augmented samples
            
        Algorithm:
        1. For each row in dataset
        2. For each text column
        3. Apply all augmentation techniques
        4. Create new rows with augmented text
        5. Concatenate with original DataFrame
        """
        augmented_rows = []
        techniques = ['synonym_replacement', 'random_substitution', 
                     'random_deletion', 'random_swap', 'back_translation']
        
        # Limit factor to max techniques
        factor = min(augmentation_factor, len(techniques))
        techniques = techniques[:factor]
        
        for idx, row in df.iterrows():
            for text_col in text_columns:
                text = str(row[text_col])
                
                # Skip if text is empty or too short
                if len(text.split()) < 2:
                    continue
                
                augmentations = self.augment_text_multi_technique(text)
                
                # Create new rows for each technique
                for technique in techniques:
                    new_row = row.copy()
                    new_row[text_col] = augmentations[technique]
                    new_row['augmentation_technique'] = technique
                    augmented_rows.append(new_row)
        
        augmented_df = pd.DataFrame(augmented_rows)
        
        # Add marker for original data
        df_copy = df.copy()
        df_copy['augmentation_technique'] = 'original'
        
        # Combine and reset index
        result_df = pd.concat([df_copy, augmented_df], ignore_index=True)
        return result_df


def main():
    """
    Main execution pipeline.
    Loads dataset, applies augmentation, and saves output.
    """
    
    print("=" * 70)
    print("TEXT DATA AUGMENTATION PIPELINE")
    print("=" * 70)
    
    # 1. Load dataset
    print("\n[1] Loading dataset...")
    df = pd.read_csv('dataset.csv')
    print(f"    Loaded {len(df)} samples with {len(df.columns)} features")
    
    # 2. Initialize augmentation pipeline
    print("\n[2] Initializing augmentation pipeline...")
    augmenter = TextAugmenter(seed=42)
    pipeline = AugmentationPipeline(augmenter)
    
    # 3. Define text columns to augment
    text_columns = ['title', 'tags', 'materials_needed']
    print(f"    Text columns to augment: {text_columns}")
    
    # 4. Apply augmentation (factor=5 uses all techniques)
    print("\n[3] Applying augmentation with all 5 techniques...")
    print("    - Synonym Replacement")
    print("    - Random Substitution")
    print("    - Random Deletion")
    print("    - Random Swap")
    print("    - Back Translation")
    
    augmented_df = pipeline.augment_dataset(
        df, 
        text_columns=text_columns,
        augmentation_factor=5
    )
    
    print(f"\n[4] Augmentation complete!")
    print(f"    Original samples: {len(df)}")
    print(f"    Total augmented samples: {len(augmented_df)}")
    print(f"    Augmentation ratio: {len(augmented_df) / len(df):.2f}x")
    
    # 5. Display augmentation technique distribution
    print("\n[5] Augmentation technique distribution:")
    technique_counts = augmented_df['augmentation_technique'].value_counts()
    for technique, count in technique_counts.items():
        print(f"    {technique}: {count} samples ({count/len(augmented_df)*100:.1f}%)")
    
    # 6. Save augmented dataset
    output_file = 'dataset_augmented.csv'
    augmented_df.to_csv(output_file, index=False)
    print(f"\n[6] Saved augmented dataset to: {output_file}")
    
    # 7. Display sample augmentations
    print("\n[7] Sample augmentations from first row:")
    sample_text = df['title'].iloc[0]
    print(f"\n    Original: {sample_text}")
    
    augmentations = pipeline.augment_text_multi_technique(sample_text)
    for technique, augmented_text in augmentations.items():
        if technique != 'original':
            print(f"    {technique}: {augmented_text}")
    
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
