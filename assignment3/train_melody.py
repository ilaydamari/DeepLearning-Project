"""
Melody-Conditioned Training Pipeline
===================================
Enhanced training script supporting melody-conditioned lyrics generation.
Integrates MIDI melody features with text generation models using two approaches:
- Approach A: Melody Concatenation at each timestep  
- Approach B: Melody as initial hidden state conditioning

Comprehensive training with TensorBoard logging and evaluation metrics.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import argparse
import json
from datetime import datetime
import random

# Project imports
from utils.text_utils import TextPreprocessor
from utils.midi_features import MelodyFeatureExtractor
from models.RNN_baseline import LyricsRNN, LyricsRNNTrainer
from models.MelodyRNN import (
    MelodyConcatenationRNN, MelodyConditioningRNN, 
    create_melody_concatenation_model, create_melody_conditioning_model,
    MelodyRNNTrainer
)


####################################### DATASET CLASSES - Data Loading ################################
# Enhanced dataset classes for melody-conditioned training

class MelodyLyricsDataset(Dataset):
    """
    Dataset for melody-conditioned lyrics generation.
    Combines text sequences with aligned melody features.
    """
    
    def __init__(
        self,
        text_sequences: List[List[int]],
        melody_features: List[np.ndarray],
        sequence_length: int = 50,
        melody_feature_dim: int = 84
    ):
        """
        Initialize melody-lyrics dataset.
        
        Args:
            text_sequences (List[List[int]]): Tokenized text sequences
            melody_features (List[np.ndarray]): Melody features per sequence
            sequence_length (int): Fixed sequence length
            melody_feature_dim (int): Melody feature dimension
        """
        self.sequence_length = sequence_length
        self.melody_feature_dim = melody_feature_dim
        self.prepared_data = []
        
        ####### DATA PREPARATION - Sequence Processing ###################
        print("Preparing melody-lyrics dataset...")
        
        for i, (text_seq, melody_feat) in enumerate(zip(text_sequences, melody_features)):
            if len(text_seq) <= 1:  # Skip empty sequences
                continue
                
            # Process text into input-target pairs
            for start_idx in range(0, len(text_seq) - sequence_length, sequence_length // 2):
                end_idx = start_idx + sequence_length
                if end_idx >= len(text_seq):
                    break
                
                # Text sequences
                input_seq = text_seq[start_idx:end_idx]
                target_seq = text_seq[start_idx + 1:end_idx + 1]
                
                # Pad if necessary
                if len(input_seq) < sequence_length:
                    pad_len = sequence_length - len(input_seq)
                    input_seq.extend([0] * pad_len)  # PAD token
                    target_seq.extend([0] * pad_len)
                
                ####### MELODY ALIGNMENT - Temporal Synchronization ##########
                # Align melody features with text sequence
                melody_seq_len = melody_feat.shape[0]
                
                if melody_seq_len == 0:
                    # Default melody if empty
                    aligned_melody = np.zeros((sequence_length, melody_feature_dim))
                elif melody_seq_len >= sequence_length:
                    # Sample or truncate melody to match text
                    step_size = melody_seq_len / sequence_length
                    indices = [int(i * step_size) for i in range(sequence_length)]
                    aligned_melody = melody_feat[indices]
                else:
                    # Repeat melody to match text length
                    repeat_factor = (sequence_length + melody_seq_len - 1) // melody_seq_len
                    repeated_melody = np.tile(melody_feat, (repeat_factor, 1))
                    aligned_melody = repeated_melody[:sequence_length]
                
                self.prepared_data.append({
                    'input_seq': torch.tensor(input_seq, dtype=torch.long),
                    'target_seq': torch.tensor(target_seq, dtype=torch.long),
                    'melody_feat': torch.tensor(aligned_melody, dtype=torch.float32)
                })
        
        print(f"Created {len(self.prepared_data)} melody-lyrics training samples")
    
    def __len__(self) -> int:
        return len(self.prepared_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.prepared_data[idx]


####################################### ENHANCED TRAINING - Melody Integration ########################
# Training functions adapted for melody-conditioned models

def create_melody_datasets(
    text_preprocessor: TextPreprocessor,
    melody_extractor: MelodyFeatureExtractor,
    train_lyrics_path: str,
    val_lyrics_path: str,
    train_midi_dir: str,
    val_midi_dir: str,
    sequence_length: int = 50,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    Create datasets with melody-lyrics alignment.
    
    Args:
        text_preprocessor (TextPreprocessor): Text processing pipeline
        melody_extractor (MelodyFeatureExtractor): MIDI feature extraction
        train_lyrics_path (str): Training lyrics CSV
        val_lyrics_path (str): Validation lyrics CSV  
        train_midi_dir (str): Training MIDI directory
        val_midi_dir (str): Validation MIDI directory
        sequence_length (int): Sequence length
        batch_size (int): Batch size
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and validation loaders
    """
    print("Creating melody-conditioned datasets...")
    print("=" * 50)
    
    ####### TRAINING DATA PROCESSING - Text and Melody ##################
    print("Processing training data...")
    
    # Load and preprocess text data
    train_sequences = text_preprocessor.load_and_preprocess_data(train_lyrics_path)
    
    # Process MIDI files for training data
    train_melody_features = []
    midi_files = [f for f in os.listdir(train_midi_dir) if f.endswith('.mid')]
    
    print(f"Processing {len(midi_files)} training MIDI files...")
    for i, midi_file in enumerate(midi_files):
        midi_path = os.path.join(train_midi_dir, midi_file)
        try:
            features = melody_extractor.extract_melody_features(midi_path)
            if features is not None and features.shape[0] > 0:
                train_melody_features.append(features)
            else:
                # Default features if extraction fails
                train_melody_features.append(np.zeros((10, melody_extractor.get_feature_dimension())))
        except Exception as e:
            print(f"Warning: Failed to process {midi_file}: {e}")
            train_melody_features.append(np.zeros((10, melody_extractor.get_feature_dimension())))
    
    # Align sequences with melody features
    min_length = min(len(train_sequences), len(train_melody_features))
    train_sequences = train_sequences[:min_length]
    train_melody_features = train_melody_features[:min_length]
    
    ####### VALIDATION DATA PROCESSING - Same Pipeline ###################
    print("Processing validation data...")
    
    val_sequences = text_preprocessor.load_and_preprocess_data(val_lyrics_path)
    val_melody_features = []
    
    val_midi_files = [f for f in os.listdir(val_midi_dir) if f.endswith('.mid')]
    print(f"Processing {len(val_midi_files)} validation MIDI files...")
    
    for midi_file in val_midi_files:
        midi_path = os.path.join(val_midi_dir, midi_file)
        try:
            features = melody_extractor.extract_melody_features(midi_path)
            if features is not None and features.shape[0] > 0:
                val_melody_features.append(features)
            else:
                val_melody_features.append(np.zeros((10, melody_extractor.get_feature_dimension())))
        except Exception as e:
            val_melody_features.append(np.zeros((10, melody_extractor.get_feature_dimension())))
    
    min_val_length = min(len(val_sequences), len(val_melody_features))
    val_sequences = val_sequences[:min_val_length]
    val_melody_features = val_melody_features[:min_val_length]
    
    ####### DATASET CREATION - Custom Dataset Classes ###################
    # Create melody-lyrics datasets
    train_dataset = MelodyLyricsDataset(
        train_sequences, train_melody_features, sequence_length, 
        melody_extractor.get_feature_dimension()
    )
    
    val_dataset = MelodyLyricsDataset(
        val_sequences, val_melody_features, sequence_length,
        melody_extractor.get_feature_dimension()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=melody_collate_fn, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=melody_collate_fn, num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


def melody_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for melody-lyrics batches.
    
    Args:
        batch (List[Dict]): Batch of melody-lyrics samples
        
    Returns:
        Dict[str, torch.Tensor]: Batched tensors
    """
    input_seqs = torch.stack([item['input_seq'] for item in batch])
    target_seqs = torch.stack([item['target_seq'] for item in batch]) 
    melody_feats = torch.stack([item['melody_feat'] for item in batch])
    
    return {
        'input_sequences': input_seqs,
        'target_sequences': target_seqs,
        'melody_features': melody_feats
    }


def train_melody_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    device: torch.device = torch.device('cpu'),
    save_path: str = 'models/melody_model.pth',
    log_dir: str = 'runs/melody_training'
) -> Dict[str, List[float]]:
    """
    Train melody-conditioned model with comprehensive monitoring.
    
    Args:
        model (nn.Module): Melody-conditioned model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        num_epochs (int): Number of training epochs
        device (torch.device): Training device
        save_path (str): Model save path
        log_dir (str): TensorBoard log directory
        
    Returns:
        Dict[str, List[float]]: Training history
    """
    print(f"Training melody-conditioned model: {type(model).__name__}")
    print("=" * 60)
    
    ####### TRAINING SETUP - Optimizer and Logging #######################
    model.to(device)
    trainer = MelodyRNNTrainer(model)
    writer = SummaryWriter(log_dir)
    
    history = {'train_loss': [], 'val_loss': [], 'learning_rate': []}
    best_val_loss = float('inf')
    global_step = 0
    
    ####### MAIN TRAINING LOOP - Epoch Processing ########################
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        
        # Training phase
        model.train()
        train_losses = []
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_seqs = batch['input_sequences'].to(device)
            target_seqs = batch['target_sequences'].to(device)
            melody_feats = batch['melody_features'].to(device)
            
            # Training step with melody
            train_loss = trainer.train_step(input_seqs, target_seqs, melody_feats)
            train_losses.append(train_loss)
            
            # Logging
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:4d}/{len(train_loader)}: Loss = {train_loss:.4f}")
                writer.add_scalar('Loss/Train_Step', train_loss, global_step)
            
            global_step += 1
        
        ####### VALIDATION PHASE - Model Evaluation ######################
        model.eval()
        val_losses = []
        
        print("  Validating...")
        for batch in val_loader:
            input_seqs = batch['input_sequences'].to(device)
            target_seqs = batch['target_sequences'].to(device)
            melody_feats = batch['melody_features'].to(device)
            
            val_loss = trainer.validate_step(input_seqs, target_seqs, melody_feats)
            val_losses.append(val_loss)
        
        ####### EPOCH SUMMARY - Statistics and Logging ####################
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        current_lr = trainer.get_learning_rate()
        
        # Update learning rate scheduler
        trainer.scheduler.step(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['learning_rate'].append(current_lr)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'model_config': {
                    'vocab_size': model.vocab_size,
                    'embedding_dim': model.embedding_dim,
                    'hidden_size': model.hidden_size,
                    'num_layers': model.num_layers,
                    'rnn_type': model.rnn_type,
                    'model_type': type(model).__name__
                }
            }, save_path)
            print(f"  New best model saved! Validation loss: {avg_val_loss:.4f}")
        
        # Epoch summary
        epoch_time = datetime.now() - epoch_start_time
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f} | Time: {epoch_time}")
    
    writer.close()
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    
    return history


####################################### MODEL COMPARISON - Approach Evaluation ########################
# Functions to compare different melody conditioning approaches

def compare_melody_approaches(
    vocab_size: int,
    pretrained_embeddings: torch.Tensor,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 5
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compare melody conditioning approaches.
    
    Args:
        vocab_size (int): Vocabulary size
        pretrained_embeddings (torch.Tensor): Pre-trained word embeddings
        train_loader (DataLoader): Training data
        val_loader (DataLoader): Validation data
        device (torch.device): Device
        num_epochs (int): Training epochs per model
        
    Returns:
        Dict[str, Dict]: Comparison results
    """
    print("Comparing Melody Conditioning Approaches")
    print("=" * 50)
    
    results = {}
    
    ####### APPROACH A - MELODY CONCATENATION ############################
    print("\n1. Training Approach A: Melody Concatenation")
    print("-" * 40)
    
    model_a = create_melody_concatenation_model(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        hidden_size=256,
        num_layers=2
    )
    
    history_a = train_melody_model(
        model=model_a,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        save_path='models/melody_concatenation_model.pth',
        log_dir='runs/approach_a_concatenation'
    )
    
    results['approach_a_concatenation'] = history_a
    
    ####### APPROACH B - MELODY CONDITIONING #############################
    print("\n2. Training Approach B: Melody Conditioning (Projection)")
    print("-" * 40)
    
    model_b1 = create_melody_conditioning_model(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        hidden_size=256,
        num_layers=2,
        conditioning_method='projection'
    )
    
    history_b1 = train_melody_model(
        model=model_b1,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        save_path='models/melody_conditioning_projection.pth',
        log_dir='runs/approach_b_projection'
    )
    
    results['approach_b_projection'] = history_b1
    
    ####### APPROACH B - MELODY CONDITIONING (ATTENTION) #################
    print("\n3. Training Approach B: Melody Conditioning (Attention)")
    print("-" * 40)
    
    model_b2 = create_melody_conditioning_model(
        vocab_size=vocab_size,
        pretrained_embeddings=pretrained_embeddings,
        hidden_size=256,
        num_layers=2,
        conditioning_method='attention'
    )
    
    history_b2 = train_melody_model(
        model=model_b2,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
        save_path='models/melody_conditioning_attention.pth',
        log_dir='runs/approach_b_attention'
    )
    
    results['approach_b_attention'] = history_b2
    
    ####### RESULTS SUMMARY - Performance Comparison #####################
    print("\n" + "=" * 60)
    print("MELODY CONDITIONING APPROACH COMPARISON")
    print("=" * 60)
    
    for approach_name, history in results.items():
        best_val_loss = min(history['val_loss'])
        final_val_loss = history['val_loss'][-1]
        print(f"{approach_name}:")
        print(f"  Best Validation Loss: {best_val_loss:.4f}")
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
    
    return results


####################################### MAIN TRAINING PIPELINE ########################################
# Complete training pipeline for melody-conditioned lyrics generation

def main():
    """Main training pipeline for melody-conditioned models."""
    
    ####### ARGUMENT PARSING - Configuration Setup #######################
    parser = argparse.ArgumentParser(description='Train Melody-Conditioned Lyrics Generation Models')
    
    # Data paths
    parser.add_argument('--train_lyrics', default='data/sets/lyrics_train_set.csv', help='Training lyrics CSV')
    parser.add_argument('--val_lyrics', default='data/sets/lyrics_test_set.csv', help='Validation lyrics CSV') 
    parser.add_argument('--train_midi_dir', default='data/midi/train/', help='Training MIDI directory')
    parser.add_argument('--val_midi_dir', default='data/midi/val/', help='Validation MIDI directory')
    
    # Model configuration
    parser.add_argument('--model_type', choices=['concatenation', 'conditioning', 'compare'], 
                       default='compare', help='Type of melody conditioning')
    parser.add_argument('--hidden_size', type=int, default=512, help='RNN hidden size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN layers')
    parser.add_argument('--rnn_type', choices=['LSTM', 'GRU'], default='LSTM', help='RNN type')
    
    # Training configuration
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--sequence_length', type=int, default=50, help='Sequence length')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    # Output paths
    parser.add_argument('--save_dir', default='models/', help='Model save directory')
    parser.add_argument('--log_dir', default='runs/', help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    ####### ENVIRONMENT SETUP - Device and Directories ###################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    ####### DATA PIPELINE INITIALIZATION ##################################
    print("Initializing data processing pipelines...")
    
    # Initialize text preprocessor
    text_preprocessor = TextPreprocessor(
        vocab_size=10000,
        sequence_length=args.sequence_length,
        use_pretrained_embeddings=True
    )
    
    # Initialize melody feature extractor
    melody_extractor = MelodyFeatureExtractor(
        feature_types=['pitch_histogram', 'rhythm_features', 'instrument_features'],
        temporal_resolution=0.25  # 4 features per beat
    )
    
    ####### DATASET CREATION - Melody-Lyrics Alignment ###################
    train_loader, val_loader = create_melody_datasets(
        text_preprocessor=text_preprocessor,
        melody_extractor=melody_extractor,
        train_lyrics_path=args.train_lyrics,
        val_lyrics_path=args.val_lyrics,
        train_midi_dir=args.train_midi_dir,
        val_midi_dir=args.val_midi_dir,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    ####### MODEL TRAINING - Approach Selection ###########################
    vocab_size = text_preprocessor.vocab_size
    pretrained_embeddings = text_preprocessor.get_embedding_matrix()
    
    if args.model_type == 'compare':
        # Compare all approaches
        compare_results = compare_melody_approaches(
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.num_epochs
        )
        
        # Save comparison results
        results_path = os.path.join(args.save_dir, 'melody_comparison_results.json')
        with open(results_path, 'w') as f:
            json.dump(compare_results, f, indent=2)
            
    elif args.model_type == 'concatenation':
        # Train concatenation model only
        model = create_melody_concatenation_model(
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            rnn_type=args.rnn_type
        )
        
        history = train_melody_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=device,
            save_path=os.path.join(args.save_dir, 'melody_concatenation.pth'),
            log_dir=os.path.join(args.log_dir, 'melody_concatenation')
        )
        
    elif args.model_type == 'conditioning':
        # Train conditioning model only
        model = create_melody_conditioning_model(
            vocab_size=vocab_size,
            pretrained_embeddings=pretrained_embeddings,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            rnn_type=args.rnn_type,
            conditioning_method='projection'
        )
        
        history = train_melody_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.num_epochs,
            device=device,
            save_path=os.path.join(args.save_dir, 'melody_conditioning.pth'),
            log_dir=os.path.join(args.log_dir, 'melody_conditioning')
        )
    
    print("\nMelody-conditioned training pipeline completed successfully!")


if __name__ == "__main__":
    main()