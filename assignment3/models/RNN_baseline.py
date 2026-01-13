"""
RNN-based lyrics generation model following Deep Learning practical session style.
Implements LSTM/GRU architecture for word-by-word lyrics generation using Word2Vec embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Optional, Tuple, Dict, List


####################################### MODEL DEFINITION - Core RNN Architecture ##########################################
# This section defines the main RNN model class that will learn to generate lyrics word by word
# The model uses either LSTM or GRU cells with Word2Vec embeddings for semantic understanding

class LyricsRNN(nn.Module):
    """
    This is the RNN model for lyrics generation.
    
    During training, receives one word at a time and predicts the next word.
    Uses Word2Vec embeddings (300 dimensions) for text representation.
    """
    
####################################### MODEL INITIALIZATION - Setup Parameters & Config #############################
# Initialize all hyperparameters and model configuration
# Set up vocabulary size, embedding dimensions, RNN architecture choices        

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'LSTM', # TODO: Choose 'LSTM' or 'GRU'
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None # TODO: Load Word2Vec embeddings
    ):
        """
        Initialize RNN model.
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of word embeddings (300 for Word2Vec)
            hidden_size (int): Hidden size of RNN layers
            num_layers (int): Number of RNN layers
            rnn_type (str): Type of RNN ('LSTM' or 'GRU')
            dropout (float): Dropout probability
            pretrained_embeddings (Optional[torch.Tensor]): Pre-trained word embeddings
        """
        super(LyricsRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout

####################################### NEURAL NETWORK ARCHITECTURE - Build Model Layers ##########################################
# Construct the neural network layers: Embeddings → RNN → Dropout → Output Projection
# Each layer serves a specific purpose in the lyrics generation pipeline

        # Embedding layer with Word2Vec initialization (Input: word indexes, Output: word vectors)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) # padding_idx=0 to ignore empty words
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            print(f"Initialized embeddings with pre-trained Word2Vec vectors")
        
        # RNN layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(
                embedding_dim, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(
                embedding_dim, 
                hidden_size, 
                num_layers, 
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Dropout layer to prevent overfitting
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection layer
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self._init_weights()

####################################### WEIGHT INITIALIZATION - Optimize Training Start ##########################################
# Initialize network weights using proven strategies (Xavier, Orthogonal)
# Proper initialization prevents vanishing/exploding gradients and speeds up convergence

    def _init_weights(self):
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

######################################## FORWARD PROPAGATION - Model Inference Pipeline ##########################################
# Core computation: Convert words to embeddings → Process through RNN → Generate predictions
# This is where the actual learning and text generation happens

    def init_hidden(self, batch_size: int, device: torch.device): # Tuple[torch.Tensor, ...]
        """
        Initialize hidden states.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Tuple[torch.Tensor, ...]: Hidden state(s)
        """
        if self.rnn_type == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:  # GRU
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return h0
    
    def forward(
        self, 
        input_sequences: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, ...]] = None): # Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]
        """
        Forward pass.
        
        Args:
            input_sequences (torch.Tensor): Input sequences [batch_size, seq_len]
            hidden (Optional[Tuple[torch.Tensor, ...]]): Hidden states
            
        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]: Output logits and hidden states
        """
        batch_size, seq_len = input_sequences.shape
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, input_sequences.device)
        
        # Word embeddings - converts word indexes to dense vectors
        embedded = self.embedding(input_sequences)  # [batch_size, seq_len, embedding_dim]
        
        # Apply dropout to embeddings
        embedded = self.dropout_layer(embedded)
        
        # RNN forward pass - processes sequence word by word, to learn dependencies and context
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_len, hidden_size]
        
        # Apply dropout to RNN output
        rnn_output = self.dropout_layer(rnn_output)
        
        # Vocabulary to predict next words, give points to each word in it
        output_logits = self.fc_out(rnn_output)  # [batch_size, seq_len, vocab_size]
        
        return output_logits, hidden

######################################## TEXT GENERATION - Creative Lyrics Production ##########################################
# Generate new lyrics by sampling from learned probability distributions
# Uses temperature scaling and top-k sampling for controlled creativity

    def generate_text(
        self,
        seed_sequence: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        device: torch.device = torch.device('cpu')): # torch.Tensor
        """
        Generate lyrics for a provided melody.
        
        Args:
            seed_sequence (torch.Tensor): Seed sequence [1, seq_len]
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling
            device (torch.device): Device to run on
            
        Returns:
            torch.Tensor: Generated sequence
        """
        self.eval()
        with torch.no_grad():
            # Initialize
            current_sequence = seed_sequence.clone()
            generated_sequence = seed_sequence.clone()
            hidden = None
            
            for _ in range(max_length):
                # Forward pass
                output_logits, hidden = self.forward(current_sequence, hidden)
                
                # Get last time word
                next_word_logits = output_logits[0, -1, :]  # [vocab_size]
                
                ####### SAMPLING STRATEGY - Control Text Generation Creativity ############
                # Temperature controls randomness: lower = more deterministic, higher = more creative
                # Top-k sampling focuses on most likely words to maintain coherence
                
                # Apply temperature scaling
                if temperature != 1.0:
                    next_word_logits = next_word_logits / temperature
                
                # Top-k sampling
                if top_k > 0:
                    # Get top-k values and indices
                    vocab_size = next_word_logits.size(0)
                    top_k = min(top_k, vocab_size)  # Ensure top_k doesn't exceed vocab size
                    top_k_logits, top_k_indices = torch.topk(next_word_logits, top_k)
                    # Set other probabilities to very low value
                    next_word_logits = torch.full_like(next_word_logits, -float('inf'))
                    next_word_logits[top_k_indices] = top_k_logits
                
                # Sample next word selection
                probabilities = F.softmax(next_word_logits, dim=-1)
                next_word = torch.multinomial(probabilities, num_samples=1)
                
                # Stop if EOS token (index 3) is generated
                if next_word.item() == 3:  # EOS token
                    break
                
                # Append to sequences
                generated_sequence = torch.cat([generated_sequence, next_word.unsqueeze(0)], dim=1)
                current_sequence = next_word.unsqueeze(0).unsqueeze(0)  # [1, 1] for next input
            
            return generated_sequence


######################################### TRAINING PIPELINE - Model Learning & Optimization ##########################################
# Complete training system: loss calculation, backpropagation, validation, checkpointing
# Handles the entire learning process from data input to model convergence

class LyricsRNNTrainer:
    """
    Trainer class for lyrics RNN.
    """
    
    def __init__(
        self, 
        model: LyricsRNN, 
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model (LyricsRNN): Model to train
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2, 
            verbose=True
        )
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
########################################## TRAINING STEP - Single Learning Iteration ##########################################
    # Forward pass → Calculate loss → Backpropagation → Update weights
    # This is the core learning loop that teaches the model to predict next words
    
    def train_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor): # float
        """
        Single training step.
        
        Args:
            input_batch (torch.Tensor): Input sequences [batch_size, seq_len]
            target_batch (torch.Tensor): Target words [batch_size]
            
        Returns:
            float: Training loss
        """
        self.model.train()
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        output_logits, _ = self.model(input_batch)
        
        # Get predictions for last time step
        # During training, we predict the next word for each position
        batch_size, seq_len, vocab_size = output_logits.shape
        
        # Reshape for loss calculation
        output_logits = output_logits.view(-1, vocab_size)  # [batch_size * seq_len, vocab_size]
        
        # For next word prediction, we need targets shifted by one
        # But since our data is already prepared correctly, we use target_batch directly
        target_batch = target_batch.view(-1)  # [batch_size]
        
        # Calculate loss - predict next word given sequence
        loss = self.criterion(output_logits[-target_batch.shape[0]:], target_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
        
        return loss.item()
    
########################################## VALIDATION STEP - Performance Evaluation Without Learning ##########################################
    # Test model performance on unseen data without updating weights
    # Used to monitor overfitting and guide training decisions
    
    def validate_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor): # float
        """
        Single validation step.
        
        Args:
            input_batch (torch.Tensor): Input sequences
            target_batch (torch.Tensor): Target words
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            # Forward pass
            output_logits, _ = self.model(input_batch)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            
            # Calculate loss
            loss = self.criterion(output_logits[-target_batch.shape[0]:], target_batch)
        
        return loss.item()
    
########################################## MODEL PERSISTENCE - Save/Load Trained Models ##########################################
    # Save and restore model state for inference and continued training
    # Includes optimizer state and training history for complete recovery
    
    def save_model(self, path: str, epoch: int, val_loss: float): # None
        """
        Save model checkpoint.
        
        Args:
            path (str): Save path
            epoch (int): Current epoch
            val_loss (float): Validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_config': {
                'vocab_size': self.model.vocab_size,
                'embedding_dim': self.model.embedding_dim,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'rnn_type': self.model.rnn_type,
                'dropout': self.model.dropout
            }
        }
        
        torch.save(checkpoint, path)
    
    def load_model(self, path: str): # None
        """
        Load model checkpoint.
        
        Args:
            path (str): Checkpoint path
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        print(f"Model loaded from epoch {checkpoint['epoch']} with val_loss {checkpoint['val_loss']:.4f}")


########################################## UTILITY FUNCTIONS - Model Analysis & Metrics ##########################################
# Helper functions for model analysis: parameter counting, perplexity calculation, summaries
# These tools help understand model complexity and performance characteristics

def count_parameters(model: nn.Module): # int
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_perplexity(loss: float): # float
    """Calculate perplexity from loss."""
    return np.exp(loss)


def get_model_summary(model: LyricsRNN): # Dict[str, any]
    """Get model summary information."""
    total_params = count_parameters(model)
    
    summary = {
        'Architecture': model.rnn_type,
        'Vocabulary Size': model.vocab_size,
        'Embedding Dimension': model.embedding_dim,
        'Hidden Size': model.hidden_size,
        'Number of Layers': model.num_layers,
        'Dropout': model.dropout,
        'Total Parameters': f"{total_params:,}",
        'Trainable Parameters': f"{total_params:,}"
    }
    
    return summary
