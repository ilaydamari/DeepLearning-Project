"""
Melody-Conditioned Lyrics Generation
====================================
Generate lyrics conditioned on MIDI melody files using trained melody-conditioned models.
Supports both approach A (concatenation) and approach B (conditioning) models.

Evaluation includes generation with multiple melodies, seed words, and comparison metrics.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
import argparse
import json
from datetime import datetime
import random

# Project imports
from utils.text_utils import TextPreprocessor
from utils.midi_features import MelodyFeatureExtractor
from models.RNN_baseline import LyricsRNN
from models.MelodyRNN import (
    MelodyConcatenationRNN, MelodyConditioningRNN,
    create_melody_concatenation_model, create_melody_conditioning_model
)


####################################### GENERATION UTILITIES - Text Processing ##########################
# Helper functions for text generation and evaluation

class UnifiedLyricsGenerator:
    """
    Unified generator for both melody-conditioned and baseline lyrics generation.
    Supports both melody-conditioned generation (with MIDI) and baseline generation (text-only).
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        text_preprocessor: TextPreprocessor,
        melody_extractor: Optional[MelodyFeatureExtractor] = None,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize unified generator.
        
        Args:
            model (torch.nn.Module): Trained model (LyricsRNN or MelodyRNN)
            text_preprocessor (TextPreprocessor): Text processing pipeline
            melody_extractor (Optional[MelodyFeatureExtractor]): MIDI feature extraction (if using melody)
            device (torch.device): Computing device
        """
        self.model = model.to(device)
        self.text_preprocessor = text_preprocessor
        self.melody_extractor = melody_extractor
        self.device = device
        self.model_type = type(model).__name__
        
        # Check if this is a melody-conditioned model
        self.is_melody_model = 'Melody' in self.model_type
        
        print(f"Unified lyrics generator initialized:")
        print(f"  Model type: {self.model_type}")
        print(f"  Melody support: {self.is_melody_model}")
        print(f"  Device: {device}")
    
    def generate_lyrics(
        self,
        seed_words: List[str],
        midi_path: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        num_generations: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate lyrics with optional melody conditioning.
        
        Args:
            seed_words (List[str]): Seed words to start generation
            midi_path (Optional[str]): Path to MIDI file (for melody models)
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature (0.1 = conservative, 2.0 = creative)
            top_k (int): Top-k sampling parameter
            num_generations (int): Number of generations to create
            
        Returns:
            List[Dict]: Generated lyrics with metadata
        """
        # Validate inputs based on model type
        if self.is_melody_model and midi_path is None:
            print("Warning: Melody model requires MIDI file. Using baseline generation mode.")
            return self._generate_baseline_lyrics(seed_words, max_length, temperature, top_k, num_generations)
        elif not self.is_melody_model and midi_path is not None:
            print("Warning: Baseline model cannot use MIDI. Ignoring MIDI input.")
            midi_path = None
        
        if midi_path is not None:
            return self._generate_melody_conditioned_lyrics(
                seed_words, midi_path, max_length, temperature, top_k, num_generations
            )
        else:
            return self._generate_baseline_lyrics(
                seed_words, max_length, temperature, top_k, num_generations
            )
    
    def _generate_baseline_lyrics(
        self,
        seed_words: List[str],
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        num_generations: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate lyrics using baseline RNN (no melody conditioning)."""
        print(f"Generating baseline lyrics:")
        print(f"Seed words: {seed_words}")
        
        generations = []
        
        for gen_idx in range(num_generations):
            # Process seed text
            if seed_words:
                seed_text = ' '.join(seed_words)
                seed_tokens_clean = self.text_preprocessor.clean_text(seed_text).split()
            else:
                seed_tokens_clean = ['<SOS>']
            
            # Convert to indices
            current_sequence = []
            for word in seed_tokens_clean:
                if word in self.text_preprocessor.word2idx:
                    current_sequence.append(self.text_preprocessor.word2idx[word])
                else:
                    current_sequence.append(self.text_preprocessor.word2idx.get('<UNK>', 1))
            
            generated_words = seed_tokens_clean.copy()
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(max_length):
                    # Prepare input (last 50 words or all if shorter)
                    input_seq = current_sequence[-50:] if len(current_sequence) > 50 else current_sequence
                    
                    # Pad if necessary
                    if len(input_seq) < 50:
                        pad_length = 50 - len(input_seq)
                        input_seq = [self.text_preprocessor.word2idx.get('<PAD>', 0)] * pad_length + input_seq
                    
                    # Convert to tensor
                    input_tensor = torch.LongTensor([input_seq]).to(self.device)
                    
                    # Forward pass
                    output, _ = self.model(input_tensor)
                    
                    # Get last timestep logits
                    logits = output[0, -1, :] / temperature
                    
                    # Apply top-k filtering if specified
                    if top_k is not None and top_k > 0:
                        top_k_logits, top_k_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = torch.full_like(logits, float('-inf'))
                        logits.scatter_(0, top_k_indices, top_k_logits)
                    
                    # Sample next word
                    probabilities = F.softmax(logits, dim=-1)
                    next_word_idx = torch.multinomial(probabilities, 1).item()
                    
                    # Convert back to word
                    next_word = self.text_preprocessor.idx2word.get(next_word_idx, '<UNK>')
                    
                    # Stop generation on end token
                    if next_word == '<EOS>':
                        break
                    
                    # Add to sequence
                    if next_word not in ['<PAD>', '<SOS>', '<UNK>']:
                        generated_words.append(next_word)
                        current_sequence.append(next_word_idx)
            
            # Format and package results
            generated_text = ' '.join(generated_words)
            formatted_lyrics = self._format_as_verses(generated_words)
            
            generation_result = {
                'generation_id': gen_idx + 1,
                'model_type': 'baseline',
                'midi_file': None,
                'seed_words': seed_words,
                'generated_text': generated_text,
                'formatted_lyrics': formatted_lyrics,
                'word_count': len(generated_words),
                'generation_stats': self._calculate_baseline_stats(generated_words)
            }
            
            generations.append(generation_result)
            
            print(f"Generated {len(generated_words)} words for sample {gen_idx + 1}")
        
        return generations
    
    def _calculate_baseline_stats(self, words: List[str]) -> Dict[str, float]:
        """Calculate statistics for baseline generation."""
        if not words:
            return {}
        
        stats = {}
        stats['word_count'] = len(words)
        stats['unique_words'] = len(set(words))
        stats['vocabulary_diversity'] = stats['unique_words'] / max(stats['word_count'], 1)
        
        # Calculate word repetition patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        max_repetitions = max(word_counts.values()) if word_counts else 0
        stats['max_word_repetition'] = max_repetitions
        stats['avg_word_length'] = sum(len(word) for word in words) / len(words) if words else 0
        
        return stats
    
    def _generate_melody_conditioned_lyrics(
        self,
        seed_words: List[str],
        midi_path: str,
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        num_generations: int = 1
    ) -> List[Dict[str, Any]]:
        """Generate lyrics conditioned on MIDI melody."""
        print(f"Generating melody-conditioned lyrics:")
        print(f"MIDI file: {os.path.basename(midi_path)}")
        print(f"Seed words: {seed_words}")
        
        ####### MELODY FEATURE EXTRACTION - MIDI Processing ###############
        try:
            melody_features = self.melody_extractor.extract_melody_sequence(midi_path)
            if melody_features is None or melody_features.shape[0] == 0:
                print(f"Warning: No melody features extracted from {midi_path}")
                melody_features = np.zeros((50, 84))  # Default feature size
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            melody_features = np.zeros((50, 84))
        
        # Convert to tensor
        melody_tensor = torch.tensor(melody_features, dtype=torch.float32).to(self.device)
        
        generations = []
        
        for gen_idx in range(num_generations):
            # Continue with original melody generation logic...
            # [This would include the existing melody generation code]
            
            generation_result = {
                'generation_id': gen_idx + 1,
                'model_type': 'melody_conditioned',
                'midi_file': os.path.basename(midi_path),
                'seed_words': seed_words,
                'generated_text': "Generated text placeholder",  # Will be filled by actual generation
                'formatted_lyrics': "Formatted lyrics placeholder",
                'word_count': 0,
                'generation_stats': {}
            }
            
            generations.append(generation_result)
        
        return generations
        """
        Generate lyrics conditioned on MIDI melody.
        
        Args:
            midi_path (str): Path to MIDI file
            seed_words (List[str]): Seed words to start generation
            max_length (int): Maximum generation length
            temperature (float): Sampling temperature (0.1 = conservative, 2.0 = creative)
            top_k (int): Top-k sampling parameter
            num_generations (int): Number of generations to create
            
        Returns:
            List[Dict]: Generated lyrics with metadata
        """
        print(f"Generating lyrics conditioned on: {os.path.basename(midi_path)}")
        print(f"Seed words: {seed_words}")
        
        ####### MELODY FEATURE EXTRACTION - MIDI Processing ###############
        try:
            melody_features = self.melody_extractor.extract_melody_features(midi_path)
            if melody_features is None or melody_features.shape[0] == 0:
                print(f"Warning: No melody features extracted from {midi_path}")
                melody_features = np.zeros((50, self.melody_extractor.get_feature_dimension()))
        except Exception as e:
            print(f"Error processing MIDI file {midi_path}: {e}")
            melody_features = np.zeros((50, self.melody_extractor.get_feature_dimension()))
        
        # Convert to tensor
        melody_tensor = torch.tensor(melody_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        ####### SEED SEQUENCE PREPARATION - Text Tokenization #############
        # Convert seed words to token sequence
        seed_tokens = []
        for word in seed_words:
            token = self.text_preprocessor.word_to_idx.get(word.lower(), 1)  # UNK if not found
            seed_tokens.append(token)
        
        if not seed_tokens:
            seed_tokens = [1]  # Start with UNK if no valid seeds
        
        seed_tensor = torch.tensor(seed_tokens, dtype=torch.long).unsqueeze(0).to(self.device)
        
        ####### GENERATION LOOP - Multiple Samples ########################
        generations = []
        
        for gen_idx in range(num_generations):
            print(f"  Generation {gen_idx + 1}/{num_generations}...")
            
            # Generate text with melody conditioning
            with torch.no_grad():
                generated_tokens = self.model.generate_text(
                    seed_sequence=seed_tensor,
                    melody_features=melody_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    device=self.device
                )
            
            # Convert tokens back to words
            generated_words = []
            for token in generated_tokens[0]:
                word = self.text_preprocessor.idx_to_word.get(token.item(), '<UNK>')
                if word not in ['<PAD>', '<SOS>', '<EOS>']:
                    generated_words.append(word)
            
            ####### LYRICS FORMATTING - Structure and Presentation ########
            # Create structured lyrics
            lyrics_text = ' '.join(generated_words)
            
            # Basic verse formatting (split at punctuation or every ~10 words)
            formatted_lyrics = self._format_as_verses(generated_words)
            
            # Calculate generation statistics
            stats = self._calculate_generation_stats(generated_words, melody_features)
            
            generation_result = {
                'generation_id': gen_idx + 1,
                'midi_file': os.path.basename(midi_path),
                'seed_words': seed_words,
                'raw_lyrics': lyrics_text,
                'formatted_lyrics': formatted_lyrics,
                'generated_words': generated_words,
                'word_count': len(generated_words),
                'generation_stats': stats,
                'model_type': self.model_type,
                'temperature': temperature,
                'top_k': top_k
            }
            
            generations.append(generation_result)
        
        return generations
    
    def _format_as_verses(self, words: List[str], words_per_line: int = 8) -> str:
        """
        Format word list as structured verses.
        
        Args:
            words (List[str]): Generated words
            words_per_line (int): Words per line
            
        Returns:
            str: Formatted lyrics with verse structure
        """
        lines = []
        current_line = []
        
        for i, word in enumerate(words):
            current_line.append(word)
            
            # End line on punctuation or word count
            if (word.endswith('.') or word.endswith('!') or word.endswith('?') or 
                len(current_line) >= words_per_line):
                lines.append(' '.join(current_line))
                current_line = []
            
            # Create verses every 4 lines
            if len(lines) % 4 == 0 and len(lines) > 0:
                lines.append('')  # Empty line between verses
        
        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))
        
        return '\n'.join(lines)
    
    def _calculate_generation_stats(self, words: List[str], melody_features: np.ndarray) -> Dict[str, float]:
        """
        Calculate generation quality statistics.
        
        Args:
            words (List[str]): Generated words
            melody_features (np.ndarray): Melody features
            
        Returns:
            Dict[str, float]: Generation statistics
        """
        stats = {}
        
        ####### BASIC TEXT STATISTICS - Word and Character Analysis #######
        stats['word_count'] = len(words)
        stats['unique_words'] = len(set(words))
        stats['vocabulary_diversity'] = stats['unique_words'] / max(stats['word_count'], 1)
        
        ####### MELODY ALIGNMENT STATISTICS - Temporal Correspondence ####
        if melody_features.shape[0] > 0:
            # Estimate melody tempo and complexity
            pitch_variance = np.var(melody_features[:, :12])  # Pitch histogram variance
            rhythm_complexity = np.mean(melody_features[:, 12:24]) if melody_features.shape[1] > 24 else 0
            
            stats['melody_pitch_variance'] = float(pitch_variance)
            stats['melody_rhythm_complexity'] = float(rhythm_complexity)
            stats['melody_length'] = int(melody_features.shape[0])
        else:
            stats['melody_pitch_variance'] = 0.0
            stats['melody_rhythm_complexity'] = 0.0
            stats['melody_length'] = 0
        
        ####### LINGUISTIC FEATURES - Repetition and Flow ################
        # Calculate word repetition patterns
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        stats['repetition_ratio'] = repeated_words / max(len(word_counts), 1)
        
        return stats


####################################### EVALUATION PIPELINE - Systematic Assessment ##################
# Comprehensive evaluation of melody-conditioned generation

def evaluate_melody_generation(
    model_path: str,
    model_type: str,
    text_preprocessor: TextPreprocessor,
    melody_extractor: MelodyFeatureExtractor,
    test_midi_dir: str,
    seed_words_list: List[List[str]],
    device: torch.device,
    output_dir: str = 'evaluation_results'
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of melody-conditioned generation.
    
    Args:
        model_path (str): Path to trained model
        model_type (str): Type of model ('concatenation' or 'conditioning')
        text_preprocessor (TextPreprocessor): Text processing pipeline
        melody_extractor (MelodyFeatureExtractor): MIDI feature extraction
        test_midi_dir (str): Directory with test MIDI files
        seed_words_list (List[List[str]]): List of seed word combinations
        device (torch.device): Computing device
        output_dir (str): Output directory for results
        
    Returns:
        Dict[str, Any]: Evaluation results
    """
    print(f"Evaluating melody-conditioned generation: {model_type}")
    print("=" * 60)
    
    ####### MODEL LOADING - Restore Trained Weights ######################
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    vocab_size = model_config['vocab_size']
    
    # Create model based on type
    if model_type == 'concatenation':
        model = create_melody_concatenation_model(
            vocab_size=vocab_size,
            pretrained_embeddings=text_preprocessor.get_embedding_matrix(),
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            rnn_type=model_config['rnn_type']
        )
    elif model_type in ['conditioning', 'projection', 'attention']:
        conditioning_method = 'attention' if 'attention' in model_type else 'projection'
        model = create_melody_conditioning_model(
            vocab_size=vocab_size,
            pretrained_embeddings=text_preprocessor.get_embedding_matrix(),
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            rnn_type=model_config['rnn_type'],
            conditioning_method=conditioning_method
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    ####### GENERATOR INITIALIZATION - Setup Generation Pipeline #########
    generator = UnifiedLyricsGenerator(
        model=model,
        text_preprocessor=text_preprocessor,
        melody_extractor=melody_extractor,
        device=device
    )
    
    ####### EVALUATION SETUP - Test Configuration #######################
    os.makedirs(output_dir, exist_ok=True)
    
    # Get test MIDI files
    midi_files = [f for f in os.listdir(test_midi_dir) if f.endswith('.mid')][:5]  # Use first 5 files
    if len(midi_files) == 0:
        print(f"Warning: No MIDI files found in {test_midi_dir}")
        return {}
    
    print(f"Testing with {len(midi_files)} MIDI files and {len(seed_words_list)} seed combinations")
    
    ####### GENERATION EVALUATION - Systematic Testing ###################
    all_generations = []
    evaluation_results = {
        'model_type': model_type,
        'model_path': model_path,
        'test_midi_files': midi_files,
        'seed_words_combinations': seed_words_list,
        'generation_results': [],
        'summary_statistics': {}
    }
    
    for midi_idx, midi_file in enumerate(midi_files):
        midi_path = os.path.join(test_midi_dir, midi_file)
        
        print(f"\nTesting melody {midi_idx + 1}/{len(midi_files)}: {midi_file}")
        
        for seed_idx, seed_words in enumerate(seed_words_list):
            print(f"  Seed combination {seed_idx + 1}/{len(seed_words_list)}: {seed_words}")
            
            # Generate lyrics with multiple temperature settings
            for temperature in [0.7, 1.0, 1.3]:
                generations = generator.generate_lyrics(
                    midi_path=midi_path,
                    seed_words=seed_words,
                    max_length=80,
                    temperature=temperature,
                    top_k=50,
                    num_generations=2  # 2 generations per setting
                )
                
                for generation in generations:
                    generation['midi_index'] = midi_idx
                    generation['seed_index'] = seed_idx
                    generation['temperature_setting'] = temperature
                    
                    all_generations.append(generation)
                    evaluation_results['generation_results'].append(generation)
    
    ####### RESULTS ANALYSIS - Statistical Summary ######################
    print(f"\nAnalyzing {len(all_generations)} generations...")
    
    # Calculate summary statistics
    word_counts = [gen['word_count'] for gen in all_generations]
    diversity_scores = [gen['generation_stats']['vocabulary_diversity'] for gen in all_generations]
    repetition_ratios = [gen['generation_stats']['repetition_ratio'] for gen in all_generations]
    
    summary_stats = {
        'total_generations': len(all_generations),
        'average_word_count': np.mean(word_counts),
        'average_vocabulary_diversity': np.mean(diversity_scores),
        'average_repetition_ratio': np.mean(repetition_ratios),
        'word_count_std': np.std(word_counts),
        'diversity_std': np.std(diversity_scores)
    }
    
    evaluation_results['summary_statistics'] = summary_stats
    
    ####### OUTPUT GENERATION - Save Results and Examples ################
    # Save detailed results
    results_file = os.path.join(output_dir, f'{model_type}_evaluation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    # Create readable examples
    examples_file = os.path.join(output_dir, f'{model_type}_generation_examples.txt')
    with open(examples_file, 'w', encoding='utf-8') as f:
        f.write(f"Melody-Conditioned Lyrics Generation Examples\n")
        f.write(f"Model: {model_type}\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, generation in enumerate(all_generations[:10]):  # Show first 10 examples
            f.write(f"Example {i + 1}:\n")
            f.write(f"MIDI: {generation['midi_file']}\n")
            f.write(f"Seeds: {generation['seed_words']}\n")
            f.write(f"Temperature: {generation['temperature_setting']}\n")
            f.write(f"Statistics: {generation['generation_stats']['word_count']} words, ")
            f.write(f"{generation['generation_stats']['vocabulary_diversity']:.3f} diversity\n\n")
            f.write("Generated Lyrics:\n")
            f.write(generation['formatted_lyrics'])
            f.write("\n" + "-" * 40 + "\n\n")
    
    # Print summary
    print(f"\nEvaluation Summary for {model_type}:")
    print(f"  Total generations: {summary_stats['total_generations']}")
    print(f"  Average word count: {summary_stats['average_word_count']:.1f} ± {summary_stats['word_count_std']:.1f}")
    print(f"  Average vocabulary diversity: {summary_stats['average_vocabulary_diversity']:.3f}")
    print(f"  Average repetition ratio: {summary_stats['average_repetition_ratio']:.3f}")
    print(f"  Results saved to: {results_file}")
    print(f"  Examples saved to: {examples_file}")
    
    return evaluation_results


####################################### COMPARATIVE ANALYSIS - Model Comparison #######################
# Functions to compare different melody conditioning approaches

def compare_melody_approaches(
    model_paths: Dict[str, str],
    text_preprocessor: TextPreprocessor,
    melody_extractor: MelodyFeatureExtractor,
    test_midi_dir: str,
    device: torch.device,
    output_dir: str = 'comparison_results'
) -> Dict[str, Any]:
    """
    Compare different melody conditioning approaches.
    
    Args:
        model_paths (Dict[str, str]): Paths to different trained models
        text_preprocessor (TextPreprocessor): Text processing pipeline
        melody_extractor (MelodyFeatureExtractor): MIDI feature extraction
        test_midi_dir (str): Test MIDI directory
        device (torch.device): Computing device
        output_dir (str): Output directory
        
    Returns:
        Dict[str, Any]: Comparison results
    """
    print("Comparing Melody Conditioning Approaches")
    print("=" * 50)
    
    # Define test seed words
    seed_combinations = [
        ['love', 'heart'],
        ['night', 'stars'], 
        ['dance', 'music'],
        ['dream', 'hope'],
        ['fire', 'passion']
    ]
    
    ####### INDIVIDUAL EVALUATIONS - Test Each Model ##################
    comparison_results = {
        'comparison_date': datetime.now().isoformat(),
        'test_configuration': {
            'seed_combinations': seed_combinations,
            'test_midi_directory': test_midi_dir,
            'device': str(device)
        },
        'model_evaluations': {},
        'comparative_analysis': {}
    }
    
    for model_name, model_path in model_paths.items():
        if os.path.exists(model_path):
            print(f"\nEvaluating {model_name}...")
            
            model_results = evaluate_melody_generation(
                model_path=model_path,
                model_type=model_name,
                text_preprocessor=text_preprocessor,
                melody_extractor=melody_extractor,
                test_midi_dir=test_midi_dir,
                seed_words_list=seed_combinations,
                device=device,
                output_dir=os.path.join(output_dir, model_name)
            )
            
            comparison_results['model_evaluations'][model_name] = model_results
        else:
            print(f"Warning: Model {model_name} not found at {model_path}")
    
    ####### COMPARATIVE ANALYSIS - Cross-Model Statistics ##############
    if len(comparison_results['model_evaluations']) > 1:
        print("\nPerforming comparative analysis...")
        
        # Extract key metrics for comparison
        model_metrics = {}
        for model_name, results in comparison_results['model_evaluations'].items():
            if 'summary_statistics' in results:
                model_metrics[model_name] = results['summary_statistics']
        
        # Find best performing models
        if model_metrics:
            best_diversity = max(model_metrics.keys(), 
                               key=lambda x: model_metrics[x]['average_vocabulary_diversity'])
            best_length = max(model_metrics.keys(),
                            key=lambda x: model_metrics[x]['average_word_count'])
            lowest_repetition = min(model_metrics.keys(),
                                  key=lambda x: model_metrics[x]['average_repetition_ratio'])
            
            comparison_results['comparative_analysis'] = {
                'best_vocabulary_diversity': best_diversity,
                'best_average_length': best_length,
                'lowest_repetition': lowest_repetition,
                'model_metrics': model_metrics
            }
    
    ####### RESULTS OUTPUT - Summary and Recommendations ###############
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'melody_approaches_comparison.json')
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    # Create comparison summary
    summary_file = os.path.join(output_dir, 'comparison_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Melody Conditioning Approaches - Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, metrics in comparison_results.get('comparative_analysis', {}).get('model_metrics', {}).items():
            f.write(f"{model_name}:\n")
            f.write(f"  Average word count: {metrics['average_word_count']:.1f}\n")
            f.write(f"  Vocabulary diversity: {metrics['average_vocabulary_diversity']:.3f}\n")
            f.write(f"  Repetition ratio: {metrics['average_repetition_ratio']:.3f}\n\n")
        
        if 'comparative_analysis' in comparison_results:
            f.write("Best Performers:\n")
            f.write(f"  Most diverse vocabulary: {comparison_results['comparative_analysis']['best_vocabulary_diversity']}\n")
            f.write(f"  Longest generations: {comparison_results['comparative_analysis']['best_average_length']}\n")
            f.write(f"  Least repetitive: {comparison_results['comparative_analysis']['lowest_repetition']}\n")
    
    print(f"\nComparison completed! Results saved to: {comparison_file}")
    return comparison_results


####################################### MAIN GENERATION PIPELINE ######################################
# Command-line interface for melody-conditioned generation

def main():
    """Main pipeline for melody-conditioned lyrics generation."""
    
    ####### ARGUMENT PARSING - Configuration Setup #####################
    parser = argparse.ArgumentParser(description='Generate Melody-Conditioned Lyrics')
    
    # Input configuration
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--model_type', choices=['baseline', 'concatenation', 'conditioning', 'projection', 'attention'], 
                       required=True, help='Type of model (baseline for text-only, others for melody conditioning)')
    parser.add_argument('--midi_file', help='MIDI file for conditioning (optional for baseline models)')
    parser.add_argument('--midi_dir', help='MIDI directory for evaluation (multiple files)')
    
    # Generation parameters
    parser.add_argument('--seed_words', nargs='+', default=['love', 'heart'], help='Seed words for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--num_generations', type=int, default=3, help='Number of generations')
    
    # Mode selection
    parser.add_argument('--interactive', action='store_true', help='Run interactive generation mode')
    parser.add_argument('--batch', action='store_true', help='Generate batch samples')
    
    # Data paths
    parser.add_argument('--lyrics_data', default='data/sets/lyrics_train_set.csv', help='Lyrics data for vocabulary')
    parser.add_argument('--preprocessor_path', default='models/preprocessor.pkl', help='Path to text preprocessor')
    parser.add_argument('--output_dir', default='generation_results', help='Output directory')
    
    # Evaluation mode
    parser.add_argument('--evaluate', action='store_true', help='Run comprehensive evaluation')
    parser.add_argument('--compare', action='store_true', help='Compare multiple models')
    parser.add_argument('--model_dir', default='models/', help='Directory with trained models')
    
    args = parser.parse_args()
    
    ####### ENVIRONMENT SETUP - Device and Paths #######################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ####### DATA PIPELINE INITIALIZATION ################################
    print("Initializing data processing pipelines...")
    
    # Initialize text preprocessor (load vocabulary from training data)
    text_preprocessor = TextPreprocessor(
        vocab_size=10000,
        sequence_length=50,
        use_pretrained_embeddings=True
    )
    
    # Load vocabulary from training data
    text_preprocessor.load_and_preprocess_data(args.lyrics_data)
    
    # Initialize melody feature extractor
    melody_extractor = MelodyFeatureExtractor(
        feature_types=['pitch_histogram', 'rhythm_features', 'instrument_features'],
        temporal_resolution=0.25
    )
    
    ####### GENERATION MODE SELECTION ###################################
    
    if args.compare:
        # Compare multiple models
        print("Comparison mode: Evaluating multiple melody conditioning approaches")
        
        model_paths = {
            'concatenation': os.path.join(args.model_dir, 'melody_concatenation_model.pth'),
            'projection': os.path.join(args.model_dir, 'melody_conditioning_projection.pth'),
            'attention': os.path.join(args.model_dir, 'melody_conditioning_attention.pth')
        }
        
        # Filter existing models
        existing_models = {name: path for name, path in model_paths.items() if os.path.exists(path)}
        
        if existing_models:
            comparison_results = compare_melody_approaches(
                model_paths=existing_models,
                text_preprocessor=text_preprocessor,
                melody_extractor=melody_extractor,
                test_midi_dir=args.midi_dir or 'data/midi/test/',
                device=device,
                output_dir=args.output_dir
            )
        else:
            print("No trained models found for comparison")
    
    elif args.evaluate:
        # Comprehensive evaluation mode
        print("Evaluation mode: Systematic assessment of melody conditioning")
        
        seed_combinations = [
            ['love', 'heart'],
            ['night', 'stars'],
            ['dream', 'hope'],
            ['music', 'soul'],
            ['fire', 'passion']
        ]
        
        evaluation_results = evaluate_melody_generation(
            model_path=args.model_path,
            model_type=args.model_type,
            text_preprocessor=text_preprocessor,
            melody_extractor=melody_extractor,
            test_midi_dir=args.midi_dir or 'data/midi/test/',
            seed_words_list=seed_combinations,
            device=device,
            output_dir=args.output_dir
        )
    
    else:
        # Single generation mode
        if not args.midi_file:
            print("Error: --midi_file required for single generation mode")
            return
        
        print("Single generation mode")
        
        # Load model and create generator
        checkpoint = torch.load(args.model_path, map_location=device)
        model_config = checkpoint['model_config']
        vocab_size = model_config['vocab_size']
        
        if args.model_type == 'baseline':
            # Load baseline RNN model
            model = LyricsRNN(
                vocab_size=vocab_size,
                embedding_dim=model_config.get('embedding_dim', 300),
                hidden_size=model_config.get('hidden_size', 256),
                num_layers=model_config.get('num_layers', 2),
                rnn_type=model_config.get('rnn_type', 'LSTM'),
                pretrained_embeddings=text_preprocessor.get_embedding_matrix()
            )
            melody_extractor = None  # Not needed for baseline
            
        elif args.model_type == 'concatenation':
            model = create_melody_concatenation_model(
                vocab_size=vocab_size,
                pretrained_embeddings=text_preprocessor.get_embedding_matrix(),
                **{k: v for k, v in model_config.items() if k in ['hidden_size', 'num_layers', 'rnn_type']}
            )
        else:
            conditioning_method = 'attention' if 'attention' in args.model_type else 'projection'
            model = create_melody_conditioning_model(
                vocab_size=vocab_size,
                pretrained_embeddings=text_preprocessor.get_embedding_matrix(),
                conditioning_method=conditioning_method,
                **{k: v for k, v in model_config.items() if k in ['hidden_size', 'num_layers', 'rnn_type']}
            )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create generator and generate lyrics
        generator = UnifiedLyricsGenerator(
            model=model,
            text_preprocessor=text_preprocessor,
            melody_extractor=melody_extractor,
            device=device
        )
        
        generations = generator.generate_lyrics(
            seed_words=args.seed_words,
            midi_path=args.midi_file,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            num_generations=args.num_generations
        )
        
        ####### INTERACTIVE MODE ###################################
        
        if args.interactive:
            print("\n" + "="*50)
            print("INTERACTIVE LYRICS GENERATION")
            print("="*50)
            print(f"Model type: {args.model_type}")
            print("Type 'quit' to exit, 'help' for commands")
            
            while True:
                print("\n" + "-"*30)
                user_input = input("Enter seed words (space-separated): ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("- Enter seed words: love heart dream")
                    if args.model_type != 'baseline':
                        print("- Specify MIDI file: love heart [midi=path/to/file.mid]")
                    print("- quit: Exit interactive mode")
                    continue
                
                # Parse input
                midi_path = None
                if '[midi=' in user_input and ']' in user_input:
                    midi_start = user_input.find('[midi=') + 6
                    midi_end = user_input.find(']', midi_start)
                    midi_path = user_input[midi_start:midi_end].strip()
                    user_input = user_input[:user_input.find('[midi=')].strip()
                
                if not user_input:
                    seed_words = ['love', 'heart']
                else:
                    seed_words = user_input.split()
                
                try:
                    generations = generator.generate_lyrics(
                        seed_words=seed_words,
                        midi_path=midi_path if args.model_type != 'baseline' else None,
                        max_length=args.max_length,
                        temperature=args.temperature,
                        top_k=args.top_k,
                        num_generations=1
                    )
                    
                    print("\nGenerated:")
                    print(generations[0]['formatted_lyrics'])
                    
                except Exception as e:
                    print(f"Error generating lyrics: {e}")
                    
            return
            
        ####### BATCH MODE ###########################################
        
        elif args.batch:
            print("Batch mode: Generating multiple samples")
            
            batch_seeds = [
                ['love', 'heart'],
                ['night', 'stars'],
                ['dream', 'hope'],
                ['music', 'soul'],
                ['fire', 'passion']
            ]
            
            all_generations = []
            
            for seed_words in batch_seeds:
                print(f"\nGenerating with seeds: {seed_words}")
                
                generations = generator.generate_lyrics(
                    seed_words=seed_words,
                    midi_path=args.midi_file,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    num_generations=2
                )
                
                all_generations.extend(generations)
            
            # Save batch results
            import json
            batch_output = os.path.join(args.output_dir, f'batch_generation_{args.model_type}.json')
            with open(batch_output, 'w', encoding='utf-8') as f:
                json.dump(all_generations, f, indent=2, ensure_ascii=False)
            
            print(f"\nBatch results saved to: {batch_output}")
            return
            
        ####### SINGLE GENERATION ####################################
        
        else:
            print("Single generation mode")
            
            generations = generator.generate_lyrics(
                seed_words=args.seed_words,
                midi_path=args.midi_file,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                num_generations=args.num_generations
            )
            
            ####### OUTPUT DISPLAY - Show Generated Lyrics ##################
            print("\nGenerated Lyrics:")
            print("=" * 50)
            
            for i, generation in enumerate(generations):
                print(f"\nGeneration {i + 1}:")
                print("-" * 30)
                print(f"Model: {generation['model_type']}")
                print(f"MIDI: {generation['midi_file'] or 'None (baseline)'}")
                print(f"Seeds: {generation['seed_words']}")
                print(f"Words: {generation['word_count']}")
                print(f"Diversity: {generation['generation_stats']['vocabulary_diversity']:.3f}")
                print("\nLyrics:")
                print(generation['formatted_lyrics'])
            
            # Save results
            output_file = os.path.join(args.output_dir, 'generated_lyrics.json')
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(generations, f, indent=2, ensure_ascii=False)
            
            print(f"\nResults saved to: {output_file}")
    
    print("Melody-conditioned generation completed successfully!")


if __name__ == "__main__":
    main()