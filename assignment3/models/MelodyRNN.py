"""
Melody-Conditioned RNN Models - Two Approaches for MIDI Integration
================================================================
Implements two distinct approaches for integrating melody information into lyrics generation:
- Approach A: Concatenation - Melody features concatenated at each timestep
- Approach B: Initial Conditioning - Melody as initial hidden state

Following assignment specifications for melody-conditioned text generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List
from models.RNN_baseline import LyricsRNN, LyricsRNNTrainer


####################################### APPROACH A - MELODY CONCATENATION ##################################
# Concatenate melody features with word embeddings at each RNN timestep
# Direct integration of temporal melody information throughout generation

class MelodyConcatenationRNN(LyricsRNN):
    """
    Approach A: Melody Concatenation Model
    
    Architecture:
    - Word embeddings (300D) + Melody features (84D) → 384D input per timestep
    - RNN processes combined input at each timestep
    - Direct temporal alignment between melody and lyrics
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        melody_feature_dim: int = 84,  # MIDI feature dimension
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'LSTM',
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize Melody Concatenation RNN (Approach A).
        
        Args:
            vocab_size (int): Vocabulary size
            embedding_dim (int): Word embedding dimension
            melody_feature_dim (int): Melody feature dimension per timestep
            hidden_size (int): RNN hidden size
            num_layers (int): Number of RNN layers
            rnn_type (str): 'LSTM' or 'GRU'
            dropout (float): Dropout probability
            pretrained_embeddings (Optional[torch.Tensor]): Pre-trained word embeddings
        """
        # Initialize parent without calling parent's __init__ completely
        nn.Module.__init__(self)
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.melody_feature_dim = melody_feature_dim
        self.combined_input_dim = embedding_dim + melody_feature_dim  # 300 + 84 = 384
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        
        ####### EMBEDDING LAYER - Word Representation ########################
        # Standard word embeddings, same as baseline model
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            print(f"Initialized embeddings with pre-trained Word2Vec vectors")
        
        ####### MELODY PROJECTION - Feature Processing ###################
        # Optional projection layer to process melody features before concatenation
        self.melody_projection = nn.Linear(melody_feature_dim, melody_feature_dim)
        self.melody_dropout = nn.Dropout(dropout)
        
        ####### RNN ARCHITECTURE - Combined Input Processing ##############
        # RNN processes concatenated word+melody features
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                self.combined_input_dim,  # 384D input
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                self.combined_input_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        ####### OUTPUT LAYERS - Text Generation ############################
        self.dropout_layer = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()
        
        print(f"Melody Concatenation RNN (Approach A) initialized:")
        print(f"  - Word embeddings: {embedding_dim}D")
        print(f"  - Melody features: {melody_feature_dim}D") 
        print(f"  - Combined input: {self.combined_input_dim}D")
        print(f"  - RNN type: {rnn_type}")
    
    def forward(
        self,
        input_sequences: torch.Tensor,
        melody_features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with melody concatenation.
        
        Args:
            input_sequences (torch.Tensor): Word sequences [batch_size, seq_len]
            melody_features (torch.Tensor): Melody features [batch_size, seq_len, melody_dim]
            hidden (Optional): Initial hidden states
            
        Returns:
            Tuple: (output_logits, hidden_states)
        """
        batch_size, seq_len = input_sequences.shape
        
        # Initialize hidden states if needed
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_sequences.device)
        
        ####### EMBEDDING PROCESSING - Convert Words to Vectors ##########
        # Standard word embeddings
        embedded = self.embedding(input_sequences)  # [batch, seq_len, embed_dim]
        
        ####### MELODY PROCESSING - Process Musical Features #############
        # Project and process melody features
        melody_processed = self.melody_projection(melody_features)  # [batch, seq_len, melody_dim]
        melody_processed = torch.tanh(melody_processed)  # Activation
        melody_processed = self.melody_dropout(melody_processed)
        
        ####### FEATURE CONCATENATION - Combine Word and Melody ##########
        # Concatenate word embeddings with melody features
        combined_input = torch.cat([embedded, melody_processed], dim=2)  # [batch, seq_len, combined_dim]
        
        ####### RNN PROCESSING - Sequential Modeling #####################
        # Process combined features through RNN
        rnn_output, hidden = self.rnn(combined_input, hidden)  # [batch, seq_len, hidden_size]
        
        ####### OUTPUT GENERATION - Vocabulary Predictions ###############
        # Apply dropout and generate vocabulary predictions
        rnn_output = self.dropout_layer(rnn_output)
        output_logits = self.fc_out(rnn_output)  # [batch, seq_len, vocab_size]
        
        return output_logits, hidden
    
    def generate_text(
        self,
        seed_sequence: torch.Tensor,
        melody_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate text conditioned on melody features.
        
        Args:
            seed_sequence (torch.Tensor): Seed words [1, seed_len]
            melody_features (torch.Tensor): Melody features [1, total_len, melody_dim]
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            device (torch.device): Device
            
        Returns:
            torch.Tensor: Generated sequence
        """
        self.eval()
        with torch.no_grad():
            current_sequence = seed_sequence.clone()
            generated_sequence = seed_sequence.clone()
            hidden = None
            
            for step in range(max_length):
                # Get current melody features (align with current timestep)
                current_pos = min(step, melody_features.size(1) - 1)
                current_melody = melody_features[:, current_pos:current_pos+1, :]  # [1, 1, melody_dim]
                
                # Forward pass with current melody
                output_logits, hidden = self.forward(current_sequence, current_melody, hidden)
                
                # Get last timestep predictions
                next_word_logits = output_logits[0, -1, :]  # [vocab_size]
                
                # Apply temperature and top-k sampling
                if temperature != 1.0:
                    next_word_logits = next_word_logits / temperature
                
                if top_k > 0:
                    vocab_size = next_word_logits.size(0)
                    top_k = min(top_k, vocab_size)
                    top_k_logits, top_k_indices = torch.topk(next_word_logits, top_k)
                    next_word_logits = torch.full_like(next_word_logits, -float('inf'))
                    next_word_logits[top_k_indices] = top_k_logits
                
                # Sample next word
                probabilities = F.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                # Stop if EOS token
                if next_word.item() == 3:  # EOS token
                    break
                
                # Append to sequences
                generated_sequence = torch.cat([generated_sequence, next_word.unsqueeze(0)], dim=1)
                current_sequence = next_word.unsqueeze(0).unsqueeze(0)  # [1, 1]
            
            return generated_sequence


####################################### APPROACH B - MELODY CONDITIONING ################################
# Use melody features to initialize or condition RNN hidden states
# Global melody influence on text generation process

class MelodyConditioningRNN(LyricsRNN):
    """
    Approach B: Melody Conditioning Model
    
    Architecture:
    - Melody features → Global conditioning vector → Initial hidden state
    - Standard word embeddings (300D) as RNN input
    - Melody influences generation through initial hidden state conditioning
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        melody_feature_dim: int = 84,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'LSTM',
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        conditioning_method: str = 'projection'  # 'projection' or 'attention'
    ):
        """
        Initialize Melody Conditioning RNN (Approach B).
        
        Args:
            vocab_size (int): Vocabulary size
            embedding_dim (int): Word embedding dimension  
            melody_feature_dim (int): Melody feature dimension
            hidden_size (int): RNN hidden size
            num_layers (int): Number of RNN layers
            rnn_type (str): 'LSTM' or 'GRU'
            dropout (float): Dropout probability
            pretrained_embeddings (Optional[torch.Tensor]): Pre-trained embeddings
            conditioning_method (str): Method for creating conditioning vector
        """
        # Call parent constructor with standard parameters
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        self.melody_feature_dim = melody_feature_dim
        self.conditioning_method = conditioning_method
        
        ####### MELODY CONDITIONING NETWORK - Global Feature Processing ###
        # Process melody sequence into conditioning vectors for hidden state initialization
        
        if conditioning_method == 'projection':
            # Simple projection approach
            self.melody_encoder = nn.Sequential(
                nn.Linear(melody_feature_dim, hidden_size * 2),  # Expand
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh()
            )
        elif conditioning_method == 'attention':
            # Attention-based melody encoding
            self.melody_attention = nn.MultiheadAttention(
                embed_dim=melody_feature_dim,
                num_heads=4,
                dropout=dropout
            )
            self.melody_encoder = nn.Sequential(
                nn.Linear(melody_feature_dim, hidden_size),
                nn.Tanh()
            )
        else:
            raise ValueError(f"Unknown conditioning method: {conditioning_method}")
        
        # Hidden state initialization networks
        if rnn_type == 'LSTM':
            # Need both h_0 and c_0 for LSTM
            self.h0_projection = nn.Linear(hidden_size, hidden_size * num_layers)
            self.c0_projection = nn.Linear(hidden_size, hidden_size * num_layers)
        else:  # GRU
            # Only h_0 for GRU
            self.h0_projection = nn.Linear(hidden_size, hidden_size * num_layers)
        
        print(f"Melody Conditioning RNN (Approach B) initialized:")
        print(f"  - Conditioning method: {conditioning_method}")
        print(f"  - Melody feature dim: {melody_feature_dim}D")
        print(f"  - Standard word embeddings: {embedding_dim}D")
        print(f"  - RNN type: {rnn_type}")
    
    def encode_melody_sequence(self, melody_features: torch.Tensor) -> torch.Tensor:
        """
        Encode melody sequence into global conditioning vector.
        
        Args:
            melody_features (torch.Tensor): Melody sequence [batch, seq_len, melody_dim]
            
        Returns:
            torch.Tensor: Global melody conditioning vector [batch, hidden_size]
        """
        if self.conditioning_method == 'projection':
            # Simple mean pooling + projection
            melody_mean = torch.mean(melody_features, dim=1)  # [batch, melody_dim]
            conditioning_vector = self.melody_encoder(melody_mean)  # [batch, hidden_size]
            
        elif self.conditioning_method == 'attention':
            # Self-attention over melody sequence
            # Transpose for attention: [seq_len, batch, melody_dim]
            melody_t = melody_features.transpose(0, 1)
            
            # Self-attention to find important melody parts
            attended_melody, _ = self.melody_attention(melody_t, melody_t, melody_t)
            
            # Mean pooling over time + projection
            melody_summary = torch.mean(attended_melody, dim=0)  # [batch, melody_dim]
            conditioning_vector = self.melody_encoder(melody_summary)  # [batch, hidden_size]
        
        return conditioning_vector
    
    def create_melody_conditioned_hidden(self, conditioning_vector: torch.Tensor, 
                                        batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Create melody-conditioned initial hidden states.
        
        Args:
            conditioning_vector (torch.Tensor): Melody conditioning [batch, hidden_size]
            batch_size (int): Batch size
            device (torch.device): Device
            
        Returns:
            Tuple: Conditioned initial hidden states
        """
        # Project conditioning vector to initial hidden states
        h0_flat = self.h0_projection(conditioning_vector)  # [batch, hidden_size * num_layers]
        h0 = h0_flat.view(batch_size, self.num_layers, self.hidden_size)  # [batch, layers, hidden]
        h0 = h0.transpose(0, 1).contiguous()  # [layers, batch, hidden]
        
        if self.rnn_type == 'LSTM':
            # Create both h_0 and c_0 for LSTM
            c0_flat = self.c0_projection(conditioning_vector)
            c0 = c0_flat.view(batch_size, self.num_layers, self.hidden_size)
            c0 = c0.transpose(0, 1).contiguous()  # [layers, batch, hidden]
            return (h0, c0)
        else:  # GRU
            return h0
    
    def forward(
        self,
        input_sequences: torch.Tensor,
        melody_features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass with melody conditioning.
        
        Args:
            input_sequences (torch.Tensor): Word sequences [batch_size, seq_len]
            melody_features (torch.Tensor): Melody features [batch_size, melody_len, melody_dim]
            hidden (Optional): Hidden states (if None, will use melody conditioning)
            
        Returns:
            Tuple: (output_logits, hidden_states)
        """
        batch_size, seq_len = input_sequences.shape
        
        ####### MELODY CONDITIONING - Process Melody into Hidden States ####
        if hidden is None:
            # Encode melody sequence into conditioning vector
            conditioning_vector = self.encode_melody_sequence(melody_features)  # [batch, hidden_size]
            
            # Create melody-conditioned initial hidden states
            hidden = self.create_melody_conditioned_hidden(
                conditioning_vector, batch_size, input_sequences.device
            )
        
        ####### STANDARD RNN PROCESSING - Use Parent Forward Method ########
        # Use standard word-only processing with melody-conditioned hidden states
        return super().forward(input_sequences, hidden)
    
    def generate_text(
        self,
        seed_sequence: torch.Tensor,
        melody_features: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate text conditioned on melody through initial hidden states.
        """
        self.eval()
        with torch.no_grad():
            # Encode melody for conditioning
            conditioning_vector = self.encode_melody_sequence(melody_features)
            hidden = self.create_melody_conditioned_hidden(
                conditioning_vector, 1, device  # batch_size=1 for generation
            )
            
            # Use parent generation method with melody-conditioned initial states
            current_sequence = seed_sequence.clone()
            generated_sequence = seed_sequence.clone()
            
            for step in range(max_length):
                # Standard forward pass (no melody needed - it's in hidden states)
                output_logits, hidden = super().forward(current_sequence, hidden)
                
                # Sample next word (same as parent)
                next_word_logits = output_logits[0, -1, :]
                
                if temperature != 1.0:
                    next_word_logits = next_word_logits / temperature
                
                if top_k > 0:
                    vocab_size = next_word_logits.size(0)
                    top_k = min(top_k, vocab_size)
                    top_k_logits, top_k_indices = torch.topk(next_word_logits, top_k)
                    next_word_logits = torch.full_like(next_word_logits, -float('inf'))
                    next_word_logits[top_k_indices] = top_k_logits
                
                probabilities = F.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                if next_word.item() == 3:  # EOS
                    break
                
                generated_sequence = torch.cat([generated_sequence, next_word.unsqueeze(0)], dim=1)
                current_sequence = next_word.unsqueeze(0).unsqueeze(0)
            
            return generated_sequence


####################################### MODEL FACTORY - Easy Model Creation ############################
# Utility functions to create specific melody-conditioned models

def create_melody_concatenation_model(vocab_size: int, 
                                    pretrained_embeddings: Optional[torch.Tensor] = None,
                                    **kwargs) -> MelodyConcatenationRNN:
    """
    Create Melody Concatenation model (Approach A).
    
    Returns:
        MelodyConcatenationRNN: Configured model
    """
    return MelodyConcatenationRNN(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        **kwargs
    )


def create_melody_conditioning_model(vocab_size: int,
                                   pretrained_embeddings: Optional[torch.Tensor] = None,
                                   conditioning_method: str = 'projection',
                                   **kwargs) -> MelodyConditioningRNN:
    """
    Create Melody Conditioning model (Approach B).
    
    Returns:
        MelodyConditioningRNN: Configured model
    """
    return MelodyConditioningRNN(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        conditioning_method=conditioning_method,
        **kwargs
    )


####################################### SPECIALIZED TRAINERS - Melody-Aware Training ###################
# Training classes adapted for melody-conditioned models

class MelodyRNNTrainer(LyricsRNNTrainer):
    """
    Trainer for melody-conditioned RNN models.
    Handles melody features in training and validation loops.
    """
    
    def __init__(self, model: nn.Module, learning_rate: float = 0.001, weight_decay: float = 1e-5):
        super().__init__(model, learning_rate, weight_decay)
        self.model_type = type(model).__name__
        print(f"Melody-aware trainer initialized for {self.model_type}")
    
    def train_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor, 
                  melody_batch: torch.Tensor) -> float:
        """
        Single training step with melody features.
        
        Args:
            input_batch (torch.Tensor): Input word sequences
            target_batch (torch.Tensor): Target words
            melody_batch (torch.Tensor): Melody features
            
        Returns:
            float: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with melody
        output_logits, _ = self.model(input_batch, melody_batch)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = output_logits.shape
        output_logits = output_logits.view(-1, vocab_size)
        target_batch = target_batch.view(-1)
        
        # Calculate loss
        loss = self.criterion(output_logits[-target_batch.shape[0]:], target_batch)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor,
                     melody_batch: torch.Tensor) -> float:
        """
        Single validation step with melody features.
        """
        self.model.eval()
        
        with torch.no_grad():
            output_logits, _ = self.model(input_batch, melody_batch)
            
            batch_size, seq_len, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            
            loss = self.criterion(output_logits[-target_batch.shape[0]:], target_batch)
        
        return loss.item()


if __name__ == "__main__":
    # Example usage of both approaches
    vocab_size = 10000
    melody_dim = 84
    
    print("Testing Melody-Conditioned RNN Models...")
    print("=" * 50)
    
    # Test Approach A: Concatenation
    print("\nApproach A: Melody Concatenation")
    model_a = create_melody_concatenation_model(vocab_size)
    
    # Test forward pass
    batch_size, seq_len = 4, 10
    words = torch.randint(1, vocab_size, (batch_size, seq_len))
    melody = torch.randn(batch_size, seq_len, melody_dim)
    
    output_a, hidden_a = model_a(words, melody)
    print(f"  Output shape: {output_a.shape}")
    
    # Test Approach B: Conditioning  
    print("\nApproach B: Melody Conditioning")
    model_b = create_melody_conditioning_model(vocab_size, conditioning_method='projection')
    
    melody_seq = torch.randn(batch_size, 20, melody_dim)  # Longer melody sequence
    output_b, hidden_b = model_b(words, melody_seq)
    print(f"  Output shape: {output_b.shape}")
    
    print("\nBoth approaches successfully tested!")