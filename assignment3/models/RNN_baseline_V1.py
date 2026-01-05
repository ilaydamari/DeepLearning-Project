"""
RNN Baseline V1 - LSTM Configuration
=================================
First model variant for testing - LSTM with conservative settings.
Goal: Test LSTM performance with 2 layers and low dropout for consistency.

Approach:
- LSTM (long memory, more stable)
- 2 layers (moderate depth)
- Low dropout (0.2) for information preservation
- Medium hidden size (256) for speed/performance balance
"""

####################################### IMPORTS - Required Libraries ###########################################
# Import the main model architecture and all dependencies

from RNN_baseline import LyricsRNN, LyricsRNNTrainer, count_parameters, get_model_summary
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


####################################### MODEL CONFIGURATION V1 - LSTM Conservative Setup ####################
# Configuration focused on stability and proven LSTM architecture
# Conservative parameters for reliable training convergence

class LyricsRNN_V1(LyricsRNN):
    """
    Version 1: LSTM-based model with conservative configuration.
    
    Architecture Philosophy:
    - LSTM for robust long-term dependencies
    - Moderate complexity to avoid overfitting
    - Lower dropout for information preservation
    - Proven hyperparameters for lyrics generation
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,  # Standard Word2Vec dimension
        hidden_size: int = 256,   # Moderate size for efficiency
        num_layers: int = 2,      # Classic depth
        dropout: float = 0.2,     # Conservative dropout
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize LSTM-based model (Version 1).
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Word embedding dimension (300 for Word2Vec)
            hidden_size (int): LSTM hidden state size
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout probability
            pretrained_embeddings (Optional[torch.Tensor]): Pre-trained embeddings
        """
        # Force LSTM configuration
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type='LSTM',  # ✓ LSTM Architecture
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        print("Initialized RNN V1 with LSTM configuration:")
        print(f"   - Architecture: LSTM")
        print(f"   - Hidden Size: {hidden_size}")
        print(f"   - Layers: {num_layers}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Parameters: {count_parameters(self):,}")


####################################### TRAINER CONFIGURATION V1 - Conservative Learning ##################
# Training setup optimized for LSTM characteristics and stable convergence

class LyricsRNNTrainer_V1(LyricsRNNTrainer):
    """
    Version 1 Trainer: Conservative training approach for LSTM.
    
    Training Philosophy:
    - Lower learning rate for stable convergence
    - Higher weight decay for regularization
    - Designed for LSTM's training characteristics
    """
    
    def __init__(
        self, 
        model: LyricsRNN_V1, 
        learning_rate: float = 0.0005,  # Lower LR for stability
        weight_decay: float = 1e-4      # Higher regularization
    ):
        """
        Initialize V1 trainer with LSTM-optimized parameters.
        
        Args:
            model (LyricsRNN_V1): LSTM model to train
            learning_rate (float): Learning rate (conservative)
            weight_decay (float): Weight decay (higher regularization)
        """
        super().__init__(model, learning_rate, weight_decay)
        
        print("Trainer V1 configured for LSTM:")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - Weight Decay: {weight_decay}")
        print(f"   - Optimizer: Adam")


####################################### MODEL FACTORY V1 - Easy Model Creation ############################
# Utility function to create and configure V1 model with standard settings

def create_model_v1(vocab_size: int, pretrained_embeddings: Optional[torch.Tensor] = None) -> tuple:
    """
    Factory function to create RNN V1 (LSTM) model and trainer.
    
    Args:
        vocab_size (int): Vocabulary size
        pretrained_embeddings (Optional[torch.Tensor]): Pre-trained embeddings
        
    Returns:
        tuple: (model, trainer) ready for training
    """
    print("Creating RNN Baseline V1 (LSTM Configuration)...")
    
    # Create LSTM model
    model = LyricsRNN_V1(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings
    )
    
    # Create trainer
    trainer = LyricsRNNTrainer_V1(model)
    
    # Print model summary
    summary = get_model_summary(model)
    print("\nModel V1 Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("RNN V1 ready for training!\n")
    
    return model, trainer


####################################### USAGE EXAMPLE V1 #################################################
# Example of how to use this model variant

if __name__ == "__main__":
    print("Testing RNN Baseline V1 (LSTM)...")
    
    # Example parameters
    vocab_size = 10000
    
    # Create model and trainer
    model, trainer = create_model_v1(vocab_size)
    
    # Test forward pass
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Dummy batch for testing
    batch_size, seq_len = 4, 20
    dummy_input = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass test
    output, hidden = model(dummy_input)
    print(f"Forward pass test successful:")
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Hidden states: {len(hidden) if isinstance(hidden, tuple) else 1}")
    
    print("RNN V1 test completed successfully!")
