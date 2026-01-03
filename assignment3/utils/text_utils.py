"""
Text preprocessing utilities for lyrics generation using Word2Vec embeddings.
Following the style from Deep Learning practical sessions.
"""

import pandas as pd
import numpy as np
import pickle
import gensim.downloader as api
from typing import List, Tuple, Dict, Optional
import torch
import re
from collections import Counter


def parse_lyrics_csv(csv_path: str) -> List[str]:
    """
    Parse lyrics from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        
    Returns:
        List[str]: List of lyrics texts
    """
    try:
        df = pd.read_csv(csv_path)
        # Assuming lyrics are in the 3rd column (index 2)
        lyrics_column = df.columns[2]  
        lyrics_list = df[lyrics_column].dropna().tolist()
        
        # Clean lyrics
        cleaned_lyrics = []
        for lyric in lyrics_list:
            if isinstance(lyric, str):
                # Remove extra characters and normalize
                lyric = lyric.replace('&', '\n').replace(',,,,', '')
                lyric = re.sub(r'\s+', ' ', lyric).strip()
                if len(lyric) > 10:  # Filter very short lyrics
                    cleaned_lyrics.append(lyric.lower())
        
        print(f"Loaded {len(cleaned_lyrics)} lyrics from {csv_path}")
        return cleaned_lyrics
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return []


class TextPreprocessor:
    """Text preprocessor with Word2Vec embeddings following course style."""
    
    def __init__(self, min_word_freq: int = 2):
        """
        Initialize preprocessor.
        
        Args:
            min_word_freq (int): Minimum word frequency for vocabulary
        """
        self.min_word_freq = min_word_freq
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.embedding_matrix = None
        self.word2vec_model = None
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        
        # Token indices
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.SOS_IDX = 2
        self.EOS_IDX = 3
        
    def clean_text(self, text: str) -> List[str]:
        """
        Clean and tokenize text following course preprocessing.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of cleaned tokens
        """
        # Basic text cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
        
        # Simple tokenization
        tokens = text.split()
        
        return tokens
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from texts.
        
        Args:
            texts (List[str]): List of text documents
        """
        print("Building vocabulary...")
        
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            tokens = self.clean_text(text)
            word_counts.update(tokens)
        
        # Create word to index mapping
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.SOS_TOKEN: self.SOS_IDX,
            self.EOS_TOKEN: self.EOS_IDX
        }
        
        # Add words with sufficient frequency
        for word, count in word_counts.items():
            if count >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        
        # Create reverse mapping
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Most common words: {list(word_counts.most_common(10))}")
    
    def load_word2vec_embeddings(self, model_name: str = 'word2vec-google-news-300') -> None:
        """
        Load pre-trained Word2Vec embeddings.
        
        Args:
            model_name (str): Name of the Word2Vec model
        """
        print(f"Loading Word2Vec model: {model_name}")
        try:
            self.word2vec_model = api.load(model_name)
            print("Word2Vec model loaded successfully")
            self._create_embedding_matrix()
        except Exception as e:
            print(f"Error loading Word2Vec model: {e}")
            print("Using random embeddings instead")
            self._create_random_embedding_matrix()
    
    def _create_embedding_matrix(self, embedding_dim: int = 300) -> None:
        """Create embedding matrix from Word2Vec model."""
        print("Creating embedding matrix...")
        
        self.embedding_matrix = np.random.normal(0, 0.1, (self.vocab_size, embedding_dim))
        
        # Set special token embeddings
        self.embedding_matrix[self.PAD_IDX] = np.zeros(embedding_dim)  # PAD token
        
        # Fill embeddings for vocabulary words
        found_words = 0
        for word, idx in self.word2idx.items():
            if word in self.word2vec_model:
                self.embedding_matrix[idx] = self.word2vec_model[word]
                found_words += 1
        
        print(f"Found Word2Vec embeddings for {found_words}/{self.vocab_size} words")
    
    def _create_random_embedding_matrix(self, embedding_dim: int = 300) -> None:
        """Create random embedding matrix as fallback."""
        print("Creating random embedding matrix...")
        self.embedding_matrix = np.random.normal(0, 0.1, (self.vocab_size, embedding_dim))
        self.embedding_matrix[self.PAD_IDX] = np.zeros(embedding_dim)  # PAD token
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert text to sequence of token indices.
        
        Args:
            text (str): Input text
            
        Returns:
            List[int]: Sequence of token indices
        """
        tokens = self.clean_text(text)
        sequence = [self.SOS_IDX]  # Start token
        
        for token in tokens:
            if token in self.word2idx:
                sequence.append(self.word2idx[token])
            else:
                sequence.append(self.UNK_IDX)
        
        sequence.append(self.EOS_IDX)  # End token
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
            if idx in self.idx2word:
                word = self.idx2word[idx]
                if word not in [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN]:
                    words.append(word)
        
        return ' '.join(words)
    
    def prepare_sequences(self, texts: List[str], max_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare input-output sequences for training following course approach.
        
        Args:
            texts (List[str]): List of texts
            max_length (int): Maximum sequence length
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Input and target sequences
        """
        print(f"Preparing sequences with max_length={max_length}")
        
        input_sequences = []
        target_sequences = []
        
        for text in texts:
            sequence = self.text_to_sequence(text)
            
            # Create sliding window sequences as in the course
            for i in range(1, len(sequence)):
                # Input: sequence up to position i
                input_seq = sequence[:i]
                # Target: next word
                target = sequence[i]
                
                # Pad input sequence
                if len(input_seq) > max_length:
                    input_seq = input_seq[-max_length:]  # Take last max_length tokens
                else:
                    # Pad with PAD tokens
                    input_seq = [self.PAD_IDX] * (max_length - len(input_seq)) + input_seq
                
                input_sequences.append(input_seq)
                target_sequences.append(target)
        
        print(f"Generated {len(input_sequences)} training sequences")
        
        return np.array(input_sequences), np.array(target_sequences)
    
    def get_embedding_matrix(self) -> Optional[torch.Tensor]:
        """
        Get embedding matrix as PyTorch tensor.
        
        Returns:
            Optional[torch.Tensor]: Embedding matrix
        """
        if self.embedding_matrix is not None:
            return torch.FloatTensor(self.embedding_matrix)
        return None
    
    def save_preprocessor(self, path: str) -> None:
        """Save preprocessor state."""
        state = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'vocab_size': self.vocab_size,
            'min_word_freq': self.min_word_freq
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        print(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str) -> None:
        """Load preprocessor state."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.word2idx = state['word2idx']
        self.idx2word = state['idx2word']
        self.vocab_size = state['vocab_size']
        self.min_word_freq = state['min_word_freq']
        
        print(f"Preprocessor loaded from {path}")


# Utility function following course style
def create_data_loaders(sequences: np.ndarray, targets: np.ndarray, 
                       batch_size: int = 32, validation_split: float = 0.1) -> Tuple:
    """
    Create train/validation data loaders.
    
    Args:
        sequences (np.ndarray): Input sequences
        targets (np.ndarray): Target sequences
        batch_size (int): Batch size
        validation_split (float): Validation split ratio
        
    Returns:
        Tuple: Train and validation data loaders
    """
    from torch.utils.data import DataLoader, TensorDataset, random_split
    
    # Convert to tensors
    sequences_tensor = torch.LongTensor(sequences)
    targets_tensor = torch.LongTensor(targets)
    
    # Create dataset
    dataset = TensorDataset(sequences_tensor, targets_tensor)
    
    # Split dataset
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
