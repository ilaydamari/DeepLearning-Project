import re
import string
import numpy as np
from collections import Counter
import pickle
from typing import List, Dict, Tuple
import gensim.downloader as api
from gensim.models import Word2Vec

class TextPreprocessor:
    """
    Text preprocessing utility for lyrics data.
    Handles tokenization, vocabulary building, and sequence preparation.
    """
    
    def __init__(self, min_word_freq=2):
        self.min_word_freq = min_word_freq
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_size = 0
        self.word2vec_model = None
        self.embedding_dim = 300
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Keep only letters, numbers, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.,!?\'\-]', '', text)
        
        # Replace & with and (common in lyrics data)
        text = text.replace('&', 'and')
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): Text to tokenize
            
        Returns:
            List[str]: List of tokens
        """
        # Clean text first
        text = self.clean_text(text)
        
        # Simple word tokenization
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build vocabulary from list of texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        word_counts = Counter()
        
        # Count words in all texts
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # Filter words by frequency
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_word_freq]
        
        # Add special tokens first
        self.word_to_idx = self.special_tokens.copy()
        
        # Add regular words
        for word in sorted(filtered_words):
            if word not in self.word_to_idx:
                self.word_to_idx[word] = len(self.word_to_idx)
        
        # Create reverse mapping
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.word_to_idx)
        
        print(f"Built vocabulary with {self.vocab_size} words")
        print(f"Most common words: {list(word_counts.most_common(10))}")
    
    def load_word2vec_embeddings(self, model_name='word2vec-google-news-300'):
        """
        Load pre-trained Word2Vec embeddings (300 entries per term as required).
        
        Args:
            model_name (str): Name of the pre-trained model to download
        """
        print(f"Loading Word2Vec model: {model_name}")
        print("This may take a few minutes for the first download...")
        try:
            # Download and load the 300-dimensional Word2Vec model
            self.word2vec_model = api.load(model_name)
            print(f"Word2Vec model loaded successfully")
            print(f"Model vocabulary size: {len(self.word2vec_model)}")
            print(f"Embedding dimension: {self.word2vec_model.vector_size}")
            
            # Verify it's 300 dimensions as required
            if self.word2vec_model.vector_size != 300:
                print(f"Warning: Model has {self.word2vec_model.vector_size} dimensions, expected 300")
                
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Please ensure internet connection for downloading the model.")
            print("Alternatively, you can train a custom Word2Vec model.")
            raise e
            
    def get_embedding_matrix(self) -> np.ndarray:
        """
        Create embedding matrix for the vocabulary using Word2Vec.
        
        Returns:
            np.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
        """
        if self.word2vec_model is None:
            raise ValueError("Word2Vec model not loaded. Call load_word2vec_embeddings() first.")
        
        embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        
        found_words = 0
        for word, idx in self.word_to_idx.items():
            if word in self.special_tokens:
                # Initialize special tokens randomly
                embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)
            else:
                try:
                    if word in self.word2vec_model:
                        embedding_matrix[idx] = self.word2vec_model[word]
                        found_words += 1
                    else:
                        # Initialize unknown words randomly
                        embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)
                except KeyError:
                    embedding_matrix[idx] = np.random.normal(0, 0.1, self.embedding_dim)
        
        print(f"Found pre-trained embeddings for {found_words}/{self.vocab_size} words")
        return embedding_matrix
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of token indices.
        
        Args:
            text (str): Input text
            
        Returns:
            List[int]: Sequence of token indices
        """
        tokens = self.tokenize(text)
        sequence = [self.special_tokens['<START>']]
        
        for token in tokens:
            if token in self.word_to_idx:
                sequence.append(self.word_to_idx[token])
            else:
                sequence.append(self.special_tokens['<UNK>'])
        
        sequence.append(self.special_tokens['<END>'])
        return sequence
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert sequence of indices back to text.
        
        Args:
            sequence (List[int]): Sequence of token indices
            
        Returns:
            str: Reconstructed text
        """
        words = []
        for idx in sequence:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                if word not in ['<PAD>', '<START>', '<END>']:
                    words.append(word)
        
        return ' '.join(words)
    
    def prepare_sequences(self, texts: List[str], max_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input/target sequences for training.
        Each training step receives one word at a time as specified in the assignment.
        
        Args:
            texts (List[str]): List of text documents
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input sequences and target sequences
        """
        sequences = []
        
        for text in texts:
            seq = self.text_to_sequence(text)
            sequences.append(seq)
        
        if max_length is None:
            max_length = min(50, max(len(seq) for seq in sequences))  # Reasonable default
        
        # Prepare input and target sequences for word-by-word training
        input_sequences = []
        target_sequences = []
        
        for seq in sequences:
            if len(seq) > 2:  # Must have at least START and END tokens
                # Create sequences where each step predicts the next word
                # This implements the requirement: "receive as input one word of the lyrics"
                for i in range(1, min(len(seq), max_length)):
                    input_seq = seq[:i]  # Previous words up to current position
                    target_word = seq[i]  # Next word to predict
                    
                    # Pad input sequence to max_length-1 (leave room for next word)
                    if len(input_seq) < max_length - 1:
                        padded_input = input_seq + [self.special_tokens['<PAD>']] * (max_length - 1 - len(input_seq))
                    else:
                        padded_input = input_seq[:max_length-1]
                    
                    input_sequences.append(padded_input)
                    target_sequences.append(target_word)
        
        print(f"Created {len(input_sequences)} training sequences")
        return np.array(input_sequences), np.array(target_sequences)
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state to file."""
        state = {
            'word_to_idx': self.word_to_idx,
            'idx_to_word': self.idx_to_word,
            'vocab_size': self.vocab_size,
            'min_word_freq': self.min_word_freq,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state from file."""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.word_to_idx = state['word_to_idx']
        self.idx_to_word = state['idx_to_word']
        self.vocab_size = state['vocab_size']
        self.min_word_freq = state['min_word_freq']
        self.special_tokens = state['special_tokens']
        
        print(f"Preprocessor loaded from {filepath}")

def parse_lyrics_csv(csv_path: str) -> List[str]:
    """
    Parse lyrics from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        List[str]: List of lyrics texts
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(csv_path, header=None, names=['artist', 'song', 'lyrics', 'extra1', 'extra2', 'extra3', 'extra4'])
        lyrics = df['lyrics'].dropna().tolist()
        
        # Filter out very short lyrics
        lyrics = [lyric for lyric in lyrics if len(lyric.split()) > 10]
        
        print(f"Loaded {len(lyrics)} lyrics from {csv_path}")
        return lyrics
    
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []
