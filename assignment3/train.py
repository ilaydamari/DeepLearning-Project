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

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_utils import TextPreprocessor, parse_lyrics_csv
from models.RNN_baseline import LyricsRNN, LyricsRNNTrainer

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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

def train_model(
    model: LyricsRNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 20,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'models/best_model.pth'
) -> LyricsRNNTrainer:
    """
    Train the lyrics generation model.
    
    Args:
        model (LyricsRNN): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        device (torch.device): Device to train on
        save_path (str): Path to save the best model
        
    Returns:
        LyricsRNNTrainer: Trained model trainer
    """
    model = model.to(device)
    trainer = LyricsRNNTrainer(model, learning_rate=0.001)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (input_batch, target_batch) in enumerate(train_pbar):
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            loss = trainer.train_step(input_batch, target_batch)
            train_losses.append(loss)
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss:.4f}',
                'Avg Loss': f'{np.mean(train_losses):.4f}'
            })
        
        # Validation phase
        model.eval()
        val_losses = []
        
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        for input_batch, target_batch in val_pbar:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            loss = trainer.validate_step(input_batch, target_batch)
            val_losses.append(loss)
            
            val_pbar.set_postfix({'Val Loss': f'{loss:.4f}'})
        
        # Calculate average losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        trainer.train_losses.append(avg_train_loss)
        trainer.val_losses.append(avg_val_loss)
        
        # Update learning rate scheduler
        trainer.scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Learning Rate: {trainer.optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            trainer.save_model(save_path, epoch, avg_val_loss)
            print(f'  New best model saved!')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
        
        print('-' * 60)
    
    return trainer

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

def main():
    """Main training script."""
    
    # Configuration - following assignment requirements
    config = {
        'train_path': 'data/sets/lyrics_train_set.csv',
        'test_path': 'data/sets/lyrics_test_set.csv',
        'max_sequence_length': 50,  # Word-by-word sequence length
        'batch_size': 32,
        'embedding_dim': 300,  # Word2Vec dimension (300 entries per term)
        'hidden_size': 512,
        'num_layers': 2,
        'rnn_type': 'LSTM',  # Can be 'LSTM' or 'GRU' as required
        'dropout': 0.3,
        'num_epochs': 20,
        'learning_rate': 0.001,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }
    
    print("=== Lyrics Generation with RNN (Assignment 3) ===")
    print("Implementation: Word-by-word training with Word2Vec embeddings")
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 50)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    dataset = LyricsDataset(
        config['train_path'],
        config['test_path'],
        config['max_sequence_length']
    )
    
    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        batch_size=config['batch_size']
    )
    
    # Get Word2Vec embeddings
    embedding_matrix = dataset.preprocessor.get_embedding_matrix()
    
    # Initialize model
    print("\n2. Initializing model...")
    model = LyricsRNN(
        vocab_size=dataset.preprocessor.vocab_size,
        embedding_dim=config['embedding_dim'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        rnn_type=config['rnn_type'],
        dropout=config['dropout'],
        pretrained_embeddings=embedding_matrix
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
    # Train model
    print("\n3. Training model...")
    trainer = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        device=config['device'],
        save_path='models/best_lyrics_model.pth'
    )
    
    # Plot training curves
    print("\n4. Plotting training curves...")
    plot_training_curves(trainer, 'training_curves.png')
    
    # Load best model for evaluation
    print("\n5. Loading best model for evaluation...")
    trainer.load_model('models/best_lyrics_model.pth')
    
    # Evaluate model
    print("\n6. Evaluating model...")
    test_perplexity = evaluate_model(model, test_loader, config['device'])
    
    # Generate sample lyrics
    print("\n7. Generating sample lyrics...")
    sample_seeds = ["love is", "i want to", "when the sun", "in the night"]
    
    for seed in sample_seeds:
        print(f"\nSeed: '{seed}'")
        generated = generate_lyrics(
            model, dataset.preprocessor, seed, 
            max_length=50, temperature=0.8, device=config['device']
        )
        print(f"Generated: {generated}")
    
    # Save preprocessor
    print("\n8. Saving preprocessor...")
    dataset.preprocessor.save_preprocessor('models/preprocessor.pkl')
    
    print("\nTraining completed successfully!")
    print(f"Best model saved at: models/best_lyrics_model.pth")
    print(f"Test perplexity: {test_perplexity:.2f}")

if __name__ == "__main__":
    main()
