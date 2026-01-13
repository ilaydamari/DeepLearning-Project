"""
RNN Baseline V2 - GRU Configuration
=================================
Second model variant for testing - GRU with aggressive settings.
Goal: Test GRU performance with 3 layers and higher dropout for robustness.

Approach:
- GRU (simpler than LSTM, faster)
- 3 layers (more depth)
- Higher dropout (0.4) for overfitting prevention
- Larger hidden size (512) for high performance
"""

####################################### IMPORTS - Required Libraries ###########################################
# Import the main model architecture and all dependencies

from RNN_baseline import LyricsRNN, LyricsRNNTrainer, count_parameters, get_model_summary
import torch
import torch.nn as nn
import numpy as np
from typing import Optional


####################################### MODEL CONFIGURATION V2 - GRU Aggressive Setup ######################
# Configuration focused on performance and GRU's efficiency
# Aggressive parameters for maximum model capacity

class LyricsRNN_V2(LyricsRNN):
    """
    Version 2: GRU-based model with aggressive configuration.
    
    Architecture Philosophy:
    - GRU for computational efficiency
    - Higher complexity for better performance
    - Higher dropout for robustness
    - Performance-oriented hyperparameters
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,  # Standard Word2Vec dimension
        hidden_size: int = 512,   # Larger size for performance
        num_layers: int = 3,      # Deeper architecture
        dropout: float = 0.4,     # Aggressive dropout
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize GRU-based model (Version 2).
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Word embedding dimension (300 for Word2Vec)
            hidden_size (int): GRU hidden state size
            num_layers (int): Number of GRU layers
            dropout (float): Dropout probability
            pretrained_embeddings (Optional[torch.Tensor]): Pre-trained embeddings
        """
        # Force GRU configuration
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type='GRU',  # ✓ GRU Architecture
            dropout=dropout,
            pretrained_embeddings=pretrained_embeddings
        )
        
        print("Initialized RNN V2 with GRU configuration:")
        print(f"   - Architecture: GRU")
        print(f"   - Hidden Size: {hidden_size}")
        print(f"   - Layers: {num_layers}")
        print(f"   - Dropout: {dropout}")
        print(f"   - Parameters: {count_parameters(self):,}")


####################################### TRAINER CONFIGURATION V2 - Aggressive Learning ##################
# Training setup optimized for GRU characteristics and fast convergence

class LyricsRNNTrainer_V2(LyricsRNNTrainer):
    """
    Version 2 Trainer: Aggressive training approach for GRU.
    
    Training Philosophy:
    - Higher learning rate for faster convergence
    - Moderate weight decay
    - Designed for GRU's training characteristics
    """
    
    def __init__(
        self, 
        model: LyricsRNN_V2, 
        learning_rate: float = 0.001,   # Higher LR for efficiency
        weight_decay: float = 5e-5      # Moderate regularization
    ):
        """
        Initialize V2 trainer with GRU-optimized parameters.
        
        Args:
            model (LyricsRNN_V2): GRU model to train
            learning_rate (float): Learning rate (aggressive)
            weight_decay (float): Weight decay (moderate regularization)
        """
        super().__init__(model, learning_rate, weight_decay)
        
        print("Trainer V2 configured for GRU:")
        print(f"   - Learning Rate: {learning_rate}")
        print(f"   - Weight Decay: {weight_decay}")
        print(f"   - Optimizer: Adam")


####################################### MODEL FACTORY V2 - Easy Model Creation ############################
# Utility function to create and configure V2 model with performance settings

def create_model_v2(vocab_size: int, pretrained_embeddings: Optional[torch.Tensor] = None) -> tuple:
    """
    Factory function to create RNN V2 (GRU) model and trainer.
    
    Args:
        vocab_size (int): Vocabulary size
        pretrained_embeddings (Optional[torch.Tensor]): Pre-trained embeddings
        
    Returns:
        tuple: (model, trainer) ready for training
    """
    print("Creating RNN Baseline V2 (GRU Configuration)...")
    
    # Create GRU model
    model = LyricsRNN_V2(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings
    )
    
    # Create trainer
    trainer = LyricsRNNTrainer_V2(model)
    
    # Print model summary
    summary = get_model_summary(model)
    print("\nModel V2 Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
    
    print("RNN V2 ready for training!\n")
    
    return model, trainer


####################################### COMPARISON UTILITIES V2 ##########################################
# Tools to compare V2 performance against other models

def compare_architectures():
    """
    Display comparison between V1 (LSTM) and V2 (GRU) configurations.
    """
    print("Architecture Comparison:")
    print("\nRNN V1 (LSTM):")
    print("   + More stable training")
    print("   + Better long-term memory")
    print("   + Proven for text generation")
    print("   - Slower training")
    print("   - More parameters")
    
    print("\nRNN V2 (GRU):")
    print("   + Faster training")
    print("   + Fewer parameters")
    print("   + Good gradient flow")
    print("   - Shorter memory span")
    print("   - Less proven for lyrics")
    
    print("\nExpected Results:")
    print("   - V1: Better lyrics quality, slower training")
    print("   - V2: Faster convergence, efficient memory usage")


####################################### USAGE EXAMPLE V2 #################################################
# Example of how to use this model variant

if __name__ == "__main__":
    print("Testing RNN Baseline V2 (GRU)...")
    
    # Show architecture comparison
    compare_architectures()
    print("\n" + "="*60 + "\n")
    
    # Example parameters
    vocab_size = 10000
    
    # Create model and trainer
    model, trainer = create_model_v2(vocab_size)
    
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
    print(f"   Hidden state shape: {hidden.shape}")
    
    print("RNN V2 test completed successfully!")
