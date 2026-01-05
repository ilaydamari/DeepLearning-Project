####################################### IMPORTS & DEPENDENCIES - Core Training Infrastructure ################
# Complete training pipeline: data loading → model building → training → evaluation → visualization
# Following Deep Learning practical session style with comprehensive lyrics generation workflow

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List

# Add the project root to the path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_utils import TextPreprocessor, parse_lyrics_csv
from models.RNN_baseline import LyricsRNN, LyricsRNNTrainer

# Configure visualization styling for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

####################################### DATA PIPELINE - Lyrics Dataset Processing ############################
# Complete data processing workflow: CSV loading → text preprocessing → sequence generation → DataLoaders
# Integrates with Word2Vec embeddings and handles train/validation/test splitting for model training

class LyricsDataset:
    """Dataset class for lyrics data with Word2Vec embeddings."""
    
    def __init__(self, train_path: str, test_path: str, max_sequence_length: int = 50):
        """
        Initialize dataset.
        
        Args:
            train_path (str): Path to training CSV file
            test_path (str): Path to test CSV file
            max_sequence_length (int): Maximum sequence length
        """
        self.train_path = train_path
        self.test_path = test_path
        self.max_sequence_length = max_sequence_length
        self.preprocessor = TextPreprocessor(min_word_freq=2)
        
        # Load and preprocess data
        print("Loading and preprocessing data...")
        self.train_lyrics = parse_lyrics_csv(train_path)
        self.test_lyrics = parse_lyrics_csv(test_path)
        
        # Build vocabulary
        all_lyrics = self.train_lyrics + self.test_lyrics
        self.preprocessor.build_vocabulary(all_lyrics)
        
        # Load Word2Vec embeddings
        self.preprocessor.load_word2vec_embeddings()
        
        # Prepare sequences
        self.X_train, self.y_train = self.preprocessor.prepare_sequences(
            self.train_lyrics, self.max_sequence_length
        )
        self.X_test, self.y_test = self.preprocessor.prepare_sequences(
            self.test_lyrics, self.max_sequence_length
        )
        
        print(f"Training sequences: {self.X_train.shape}")
        print(f"Test sequences: {self.X_test.shape}")
    
    def get_dataloaders(self, batch_size: int = 32, validation_split: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Args:
            batch_size (int): Batch size
            validation_split (float): Fraction of training data for validation
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders
        """
        # Convert to PyTorch tensors
        X_train_tensor = torch.LongTensor(self.X_train)
        y_train_tensor = torch.LongTensor(self.y_train)
        X_test_tensor = torch.LongTensor(self.X_test)
        y_test_tensor = torch.LongTensor(self.y_test)
        
        # Create training dataset and split into train/validation
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        
        # Calculate split sizes
        total_size = len(train_dataset)
        val_size = int(total_size * validation_split)
        train_size = total_size - val_size
        
        train_subset, val_subset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create test dataset
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader

####################################### TRAINING PIPELINE - Model Learning & Optimization ###################
# Core training loop: forward pass → loss calculation → backpropagation → validation → checkpointing
# Implements early stopping, learning rate scheduling, and comprehensive progress tracking

def train_model(
    model: LyricsRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'models/best_model.pth',
    patience: int = 5
) -> LyricsRNNTrainer:
    """
    Train the lyrics generation model following course training style.
    
    Args:
        model (LyricsRNN): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        device (torch.device): Device to train on
        save_path (str): Path to save the best model
        patience (int): Early stopping patience
        
    Returns:
        LyricsRNNTrainer: Trained model trainer
    """
    # Setup training following course approach
    model = model.to(device)
    trainer = LyricsRNNTrainer(model, learning_rate=0.001)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"  Training setup:")
    print(f"  Device: {device}")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Early stopping patience: {patience}")
    print("-" * 60)
    
    # Training loop following course pattern
    for epoch in range(num_epochs):
        # Training phase - word-by-word learning as required
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f'🏋️  Epoch {epoch+1}/{num_epochs} [Training]')
        for batch_idx, (input_batch, target_batch) in enumerate(train_pbar):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            # Train step - learning to predict next word
            loss = trainer.train_step(input_batch, target_batch)
            train_losses.append(loss)
            
            # Update progress bar with current metrics
            train_pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Avg Loss': f'{np.mean(train_losses):.4f}',
                'Perplexity': f'{np.exp(np.mean(train_losses)):.2f}'
            })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Validation]')
        for input_batch, target_batch in val_pbar:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            loss = trainer.validate_step(input_batch, target_batch)
            val_losses.append(loss)
            
            val_pbar.set_postfix({
                'Val Loss': f'{loss:.4f}',
                'Avg Val Loss': f'{np.mean(val_losses):.4f}'
            })
        
        # Calculate epoch metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        train_perplexity = np.exp(avg_train_loss)
        val_perplexity = np.exp(avg_val_loss)
        
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        trainer.scheduler.step(avg_val_loss)
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Epoch summary following course reporting style
        print(f'\nEpoch {epoch+1}/{num_epochs} Results:')
        print(f'  Training Loss: {avg_train_loss:.4f} | Perplexity: {train_perplexity:.2f}')
        print(f'  Validation Loss: {avg_val_loss:.4f} | Perplexity: {val_perplexity:.2f}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Save best model following course checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            trainer.save_model(save_path, epoch, avg_val_loss)
            print(f' New best model saved! (Val Loss: {avg_val_loss:.4f})')
        else:
            patience_counter += 1
            print(f' No improvement. Patience: {patience_counter}/{patience}')
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\n Early stopping triggered after {epoch+1} epochs')
            print(f'   Best validation loss: {best_val_loss:.4f}')
            break
        
        print('-' * 60)
    
    return trainer

####################################### MODEL EVALUATION - Performance Assessment ##########################
# Test trained model on unseen data to measure generalization capability
# Calculates perplexity metric for language model quality evaluation

def evaluate_model(
    model: LyricsRNN,
    test_loader: DataLoader,
    device: torch.device = torch.device('cpu')
) -> float:
    """
    Evaluate the model on test data.
    
    Args:
        model (LyricsRNN): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to evaluate on
        
    Returns:
        float: Test perplexity
    """
    model.eval()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    test_losses = []
    
    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc='Testing')
        for input_batch, target_batch in test_pbar:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            output_logits, _ = model(input_batch)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            
            loss = criterion(output_logits, target_batch)
            test_losses.append(loss.item())
            
            test_pbar.set_postfix({'Test Loss': f'{loss.item():.4f}'})
    
    avg_test_loss = np.mean(test_losses)
    perplexity = np.exp(avg_test_loss)
    
    print(f'Test Results:')
    print(f'  Test Loss: {avg_test_loss:.4f}')
    print(f'  Perplexity: {perplexity:.2f}')
    
    return perplexity

####################################### TEXT GENERATION - Creative Lyrics Production #######################
# Generate new lyrics from trained model using seed text and sampling strategies
# Controls creativity through temperature and produces coherent lyrical content

def generate_lyrics(
    model: LyricsRNN,
    preprocessor: TextPreprocessor,
    seed_text: str = "love is",
    max_length: int = 100,
    temperature: float = 0.8,
    device: torch.device = torch.device('cpu')
) -> str:
    """
    Generate lyrics using the trained model.
    
    Args:
        model (LyricsRNN): Trained model
        preprocessor (TextPreprocessor): Text preprocessor
        seed_text (str): Seed text to start generation
        max_length (int): Maximum length to generate
        temperature (float): Sampling temperature
        device (torch.device): Device to run on
        
    Returns:
        str: Generated lyrics
    """
    model.eval()
    model = model.to(device)
    
    # Convert seed text to sequence
    seed_sequence = preprocessor.text_to_sequence(seed_text)
    seed_tensor = torch.LongTensor([seed_sequence]).to(device)
    
    # Generate text
    generated_sequence = model.generate_text(
        seed_tensor,
        max_length=max_length,
        temperature=temperature,
        top_k=50,
        device=device
    )
    
    # Convert back to text
    generated_text = preprocessor.sequence_to_text(generated_sequence[0].cpu().tolist())
    
    return generated_text

####################################### TRAINING VISUALIZATION - Learning Progress Analysis #################
# Create comprehensive plots showing training dynamics: loss curves and perplexity trends
# Essential for understanding model convergence and identifying overfitting patterns

def plot_training_curves(trainer: LyricsRNNTrainer, save_path: str = 'training_curves.png'):
    """
    Plot training and validation loss curves.
    
    Args:
        trainer (LyricsRNNTrainer): Trained model trainer
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(trainer.train_losses, label='Training Loss', color='blue')
    plt.plot(trainer.val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    train_perplexity = [np.exp(loss) for loss in trainer.train_losses]
    val_perplexity = [np.exp(loss) for loss in trainer.val_losses]
    
    plt.plot(train_perplexity, label='Training Perplexity', color='blue')
    plt.plot(val_perplexity, label='Validation Perplexity', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to {save_path}")

####################################### MAIN TRAINING ORCHESTRATOR - Complete Workflow Execution ##########
# Master function coordinating entire training pipeline from data loading to final evaluation
# Implements Deep Learning course methodology with comprehensive logging and checkpointing

def main():
    """Main training script following Deep Learning practical session style."""
    
    ####### CONFIGURATION SETUP - Hyperparameter Definition ####################
    # Define all training hyperparameters following Deep Learning best practices
    # Centralized configuration for easy experimentation and reproducibility
    
    # Configuration - following course parameters
    config = {
        'train_path': 'data/sets/lyrics_train_set.csv',
        'test_path': 'data/sets/lyrics_test_set.csv',
        'max_sequence_length': 50,  # Sequence length for RNN input
        'batch_size': 32,
        'embedding_dim': 300,  # Word2Vec dimension as required
        'hidden_size': 512,    # Hidden state size
        'num_layers': 2,       # Number of RNN layers
        'rnn_type': 'LSTM',    # RNN type: 'LSTM' or 'GRU'
        'dropout': 0.3,        # Dropout for regularization
        'num_epochs': 20,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'early_stopping_patience': 5,
        'min_word_freq': 2     # Minimum word frequency for vocabulary
    }
    
    print("=== Lyrics Generation with RNN - Following Course Style ===")
    print(f"Device: {config['device']}")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    
    ####### STEP 1: DATA PREPARATION - Load, Process & Create DataLoaders ######
    # Load CSV files, build vocabulary, integrate Word2Vec embeddings
    # Create train/validation/test splits for proper model evaluation
    
    # 1. Load and prepare data following course data pipeline
    print("\nStep 1: Loading and preparing data...")
    dataset = LyricsDataset(
        config['train_path'],
        config['test_path'],
        config['max_sequence_length']
    )
    
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Get Word2Vec embeddings as required by assignment
    embedding_matrix = dataset.preprocessor.get_embedding_matrix()
    print(f"✓ Data loaded successfully")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Vocabulary size: {dataset.preprocessor.vocab_size}")
    
    ####### STEP 2: MODEL ARCHITECTURE - Initialize RNN with Word2Vec ###########
    # Create model with specified architecture and load pre-trained embeddings
    # Configure LSTM/GRU layers, dropout, and output projection
    
    # 2. Initialize model following course architecture
    print("\nStep 2: Initializing RNN model...")
    model = LyricsRNN(
        vocab_size=dataset.preprocessor.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        rnn_type=config['rnn_type'],
        dropout=config['dropout'],
        pretrained_embeddings=embedding_matrix  # Word2Vec initialization
    )
    
    # Print model summary
    from models.RNN_baseline import get_model_summary
    model_summary = get_model_summary(model)
    print(f"✓ Model created successfully")
    print("Model Summary:")
    for key, value in model_summary.items():
        print(f"  {key}: {value}")
    
    ####### STEP 3: TRAINING EXECUTION - Learn to Predict Next Words ############
    # Execute full training loop with validation, early stopping, and checkpointing
    # Monitor loss curves and learning rate scheduling for optimal convergence
    
    # 3. Train model following course training procedure
    print(f"\nStep 3: Training model for {config['num_epochs']} epochs...")
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_path='models/best_lyrics_model.pth',
        patience=config['early_stopping_patience']
    )
    
    ####### STEP 4: TRAINING ANALYSIS - Visualize Learning Progress ##############
    # Generate loss and perplexity curves to analyze training dynamics
    # Essential for understanding model behavior and debugging training issues
    
    # 4. Plot training curves following course visualization
    print("\nStep 4: Plotting training curves...")
    plot_training_curves(trainer, 'training_curves.png')
    
    ####### STEP 5: MODEL RESTORATION - Load Best Checkpoint #####################
    # Restore model from best validation checkpoint for final evaluation
    # Ensures we evaluate the optimal model state, not the last epoch
    
    # 5. Load best model for evaluation
    print("\nStep 5: Loading best model for evaluation...")
    trainer.load_model('models/best_lyrics_model.pth')
    
    ####### STEP 6: FINAL EVALUATION - Test Set Performance Assessment ###########
    # Evaluate on held-out test set to measure true generalization capability
    # Calculate perplexity as standard language model evaluation metric
    
    # 6. Evaluate model following course evaluation
    print("\nStep 6: Evaluating model...")
    test_perplexity = evaluate_model(model, test_loader, config['device'])
    
    ####### STEP 7: TEXT GENERATION - Demonstrate Model Capabilities ##############
    # Generate sample lyrics with different seeds to showcase learning results
    # Test model's ability to produce coherent, contextually appropriate text
    
    # 7. Generate sample lyrics to demonstrate functionality
    print("\nStep 7: Generating sample lyrics...")
    sample_seeds = [
        "love is",      # Simple emotion seed
        "i want to",    # Action-oriented seed  
        "when the sun", # Descriptive seed
        "in the night", # Time-based seed
        "music makes"   # Music-related seed
    ]
    
    print("Generated lyrics samples:")
    print("=" * 50)
    for seed in sample_seeds:
        print(f"\nSeed: '{seed}'")
        try:
            generated = generate_lyrics(
                model, dataset.preprocessor, seed, 
                max_length=50, temperature=0.8, device=config['device']
            )
            print(f"Generated: {generated}")
        except Exception as e:
            print(f"Generation failed: {e}")
        print("-" * 30)
    
    ####### STEP 8: MODEL PERSISTENCE - Save for Future Use #####################
    # Save preprocessor and final model state for deployment and inference
    # Enables loading trained model in separate generation scripts
    
    # 8. Save preprocessor for future use
    print("\nStep 8: Saving preprocessor...")
    dataset.preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    ####### TRAINING COMPLETION - Final Results Summary ########################
    # Comprehensive summary of training results and saved artifacts
    # Professional reporting following academic standards
    
    # Final summary following course reporting style
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Best model saved: models/best_lyrics_model.pth")
    print(f"Test perplexity: {test_perplexity:.2f}")
    print(f"Preprocessor saved: models/preprocessor.pkl")
    print(f"Training curves: training_curves.png")
    print(f"Architecture: {config['rnn_type']} with {config['embedding_dim']}D Word2Vec embeddings")
    print(f"Vocabulary size: {dataset.preprocessor.vocab_size} words")
    print("=" * 60)

if __name__ == "__main__":
    main()
