"""
Comprehensive Evaluation Pipeline for Melody-Conditioned Lyrics Generation
=========================================================================
Implements systematic evaluation for melody-conditioned models:
- Test on 5 melody files with 3 different seed word combinations each
- Compare concatenation vs conditioning approaches
- Generate detailed analysis reports

Following course evaluation methodology with statistical analysis.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter

# Project imports
from utils.text_utils import TextPreprocessor
from utils.midi_features import MelodyFeatureExtractor
from models.MelodyRNN import create_melody_concatenation_model, create_melody_conditioning_model
from models.RNN_baseline import LyricsRNN
from generate_melody import UnifiedLyricsGenerator


####################################### EVALUATION CONFIGURATION ######################################
# Define standard evaluation parameters following assignment specifications

EVALUATION_CONFIG = {
    'seed_word_combinations': [
        ['love', 'heart'],      # Romantic theme
        ['music', 'soul'],      # Musical theme  
        ['dream', 'night']      # Dreamy theme
    ],
    
    'generation_parameters': {
        'max_length': 100,
        'temperature_variants': [0.6, 0.8, 1.0],  # Conservative to creative
        'top_k': 50,
        'num_generations_per_combination': 2
    },
    
    'evaluation_metrics': [
        'vocabulary_diversity',
        'melody_alignment_score',
        'lyrical_coherence',
        'creativity_score',
        'musical_structure_fit'
    ],
    
    'output_formats': ['json', 'csv', 'html_report']
}


####################################### ADVANCED METRICS COMPUTATION ##################################
# Calculate sophisticated evaluation metrics beyond basic word count and diversity

def calculate_melody_alignment_score(lyrics_words: List[str], melody_features: np.ndarray) -> float:
    """
    Calculate how well lyrics align with melody rhythm and structure.
    
    Args:
        lyrics_words (List[str]): Generated lyrics as word list
        melody_features (np.ndarray): Melody features [time_steps, feature_dim]
        
    Returns:
        float: Alignment score (0.0 to 1.0)
    """
    if melody_features is None or len(melody_features) == 0:
        return 0.0
    
    # Estimate lyric timing (assuming ~2 words per second)
    words_per_second = 2.0
    lyric_duration = len(lyrics_words) / words_per_second
    melody_duration = len(melody_features) / 16  # 16 fps
    
    # Duration alignment score
    duration_ratio = min(lyric_duration, melody_duration) / max(lyric_duration, melody_duration)
    
    # Rhythm complexity alignment
    rhythm_features = melody_features[:, 17] if melody_features.shape[1] > 17 else np.zeros(len(melody_features))
    rhythm_complexity = np.std(rhythm_features)
    
    # Vocabulary complexity
    unique_words = len(set(lyrics_words))
    vocab_complexity = unique_words / max(len(lyrics_words), 1)
    
    # Combined alignment score
    alignment_score = (duration_ratio * 0.4 + 
                      min(rhythm_complexity / 0.5, 1.0) * 0.3 + 
                      vocab_complexity * 0.3)
    
    return float(np.clip(alignment_score, 0.0, 1.0))


def calculate_lyrical_coherence(lyrics_words: List[str]) -> float:
    """
    Measure semantic coherence and flow of generated lyrics.
    
    Args:
        lyrics_words (List[str]): Generated lyrics as word list
        
    Returns:
        float: Coherence score (0.0 to 1.0)
    """
    if len(lyrics_words) < 3:
        return 0.0
    
    # Repetition analysis (some repetition is good in lyrics)
    word_counts = Counter(lyrics_words)
    total_words = len(lyrics_words)
    unique_words = len(word_counts)
    
    # Calculate healthy repetition ratio (optimal around 0.6-0.8)
    repetition_ratio = unique_words / total_words
    repetition_score = 1.0 - abs(repetition_ratio - 0.7) / 0.3
    repetition_score = max(0.0, repetition_score)
    
    # Word length variety (good lyrics have variety)
    word_lengths = [len(word) for word in lyrics_words]
    length_variance = np.var(word_lengths) / max(np.mean(word_lengths), 1)
    length_score = min(length_variance / 2.0, 1.0)
    
    # Simple sentence structure (look for natural breaks)
    text = ' '.join(lyrics_words)
    sentence_markers = text.count('.') + text.count('!') + text.count('?')
    structure_score = min(sentence_markers / max(len(lyrics_words) / 10, 1), 1.0)
    
    coherence_score = (repetition_score * 0.4 + length_score * 0.3 + structure_score * 0.3)
    return float(np.clip(coherence_score, 0.0, 1.0))


def calculate_creativity_score(lyrics_words: List[str], reference_vocab: set = None) -> float:
    """
    Measure creativity and originality of generated lyrics.
    
    Args:
        lyrics_words (List[str]): Generated lyrics
        reference_vocab (set): Training vocabulary for comparison
        
    Returns:
        float: Creativity score (0.0 to 1.0)
    """
    if len(lyrics_words) == 0:
        return 0.0
    
    # Vocabulary uniqueness
    unique_ratio = len(set(lyrics_words)) / len(lyrics_words)
    
    # Word combination novelty (simple bigram analysis)
    bigrams = [(lyrics_words[i], lyrics_words[i+1]) for i in range(len(lyrics_words)-1)]
    unique_bigrams = len(set(bigrams))
    bigram_creativity = unique_bigrams / max(len(bigrams), 1)
    
    # Length creativity (neither too short nor too long)
    length_score = 1.0 - abs(len(lyrics_words) - 50) / 50  # Optimal around 50 words
    length_score = max(0.0, length_score)
    
    creativity_score = (unique_ratio * 0.4 + bigram_creativity * 0.4 + length_score * 0.2)
    return float(np.clip(creativity_score, 0.0, 1.0))


def calculate_musical_structure_fit(lyrics_words: List[str], target_structure: str = 'verse') -> float:
    """
    Evaluate how well lyrics fit expected musical structure.
    
    Args:
        lyrics_words (List[str]): Generated lyrics
        target_structure (str): Expected structure ('verse', 'chorus', 'bridge')
        
    Returns:
        float: Structure fit score (0.0 to 1.0)
    """
    if len(lyrics_words) == 0:
        return 0.0
    
    # Line length analysis (verses typically 4-8 words per line)
    # Estimate lines by natural breaks
    text = ' '.join(lyrics_words)
    estimated_lines = max(1, len(lyrics_words) // 7)  # ~7 words per line
    
    if target_structure == 'verse':
        # Verses should be 4-8 lines typically
        optimal_lines = 6
        line_score = 1.0 - abs(estimated_lines - optimal_lines) / optimal_lines
        line_score = max(0.0, line_score)
        
        # Verses often tell a story (more concrete nouns)
        concrete_words = ['heart', 'love', 'night', 'day', 'eyes', 'hand', 'soul', 'dream']
        concrete_count = sum(1 for word in lyrics_words if word.lower() in concrete_words)
        concrete_score = min(concrete_count / 5, 1.0)  # Expect ~5 concrete words
        
        structure_score = (line_score * 0.6 + concrete_score * 0.4)
        
    else:
        # Generic structure scoring
        structure_score = 0.7  # Default reasonable score
    
    return float(np.clip(structure_score, 0.0, 1.0))


####################################### SYSTEMATIC EVALUATION PIPELINE #############################
# Main evaluation orchestrator following assignment methodology

class MelodyLyricsEvaluator:
    """
    Comprehensive evaluator for melody-conditioned lyrics generation.
    Implements systematic testing following Deep Learning assignment requirements.
    """
    
    def __init__(self, models_dir: str, data_dir: str, output_dir: str = 'evaluation_results'):
        """
        Initialize evaluator with model and data paths.
        
        Args:
            models_dir (str): Directory containing trained models
            data_dir (str): Directory containing test data
            output_dir (str): Output directory for results
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_preprocessor = None
        self.melody_extractor = None
        self.models = {}
        self.generators = {}
        
        print(f"Melody Lyrics Evaluator initialized")
        print(f"Models directory: {self.models_dir}")
        print(f"Data directory: {self.data_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def load_evaluation_components(self, lyrics_data_path: str):
        """Load text preprocessor and melody extractor."""
        print("\nLoading evaluation components...")
        
        # Initialize text preprocessor
        self.text_preprocessor = TextPreprocessor(vocab_size=10000, sequence_length=50, use_pretrained_embeddings=True)
        self.text_preprocessor.load_and_preprocess_data(lyrics_data_path)
        
        # Initialize melody extractor  
        self.melody_extractor = MelodyFeatureExtractor(
            feature_types=['pitch_histogram', 'rhythm_features', 'instrument_features'],
            temporal_resolution=0.25
        )
        
        print(f"✓ Text preprocessor loaded (vocab: {self.text_preprocessor.vocab_size})")
        print(f"✓ Melody extractor initialized")
    
    def load_trained_models(self):
        """Load all available trained models for comparison."""
        print("\nLoading trained models...")
        
        model_files = {
            'baseline': 'best_lyrics_model.pth',
            'concatenation': 'melody_concatenation_model.pth', 
            'conditioning_projection': 'melody_conditioning_projection.pth',
            'conditioning_attention': 'melody_conditioning_attention.pth'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            
            if model_path.exists():
                try:
                    checkpoint = torch.load(model_path, map_location=self.device)
                    model_config = checkpoint['model_config']
                    vocab_size = model_config['vocab_size']
                    
                    # Create appropriate model
                    if model_name == 'baseline':
                        model = LyricsRNN(
                            vocab_size=vocab_size,
                            embedding_dim=model_config.get('embedding_dim', 300),
                            hidden_size=model_config.get('hidden_size', 256),
                            num_layers=model_config.get('num_layers', 2),
                            rnn_type=model_config.get('rnn_type', 'LSTM'),
                            pretrained_embeddings=self.text_preprocessor.get_embedding_matrix()
                        )
                        melody_extractor = None
                        
                    elif model_name == 'concatenation':
                        model = create_melody_concatenation_model(
                            vocab_size=vocab_size,
                            pretrained_embeddings=self.text_preprocessor.get_embedding_matrix(),
                            **{k: v for k, v in model_config.items() if k in ['hidden_size', 'num_layers', 'rnn_type']}
                        )
                        melody_extractor = self.melody_extractor
                        
                    else:  # conditioning models
                        conditioning_method = 'attention' if 'attention' in model_name else 'projection'
                        model = create_melody_conditioning_model(
                            vocab_size=vocab_size,
                            pretrained_embeddings=self.text_preprocessor.get_embedding_matrix(),
                            conditioning_method=conditioning_method,
                            **{k: v for k, v in model_config.items() if k in ['hidden_size', 'num_layers', 'rnn_type']}
                        )
                        melody_extractor = self.melody_extractor
                    
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(self.device)
                    
                    # Create generator
                    generator = UnifiedLyricsGenerator(
                        model=model,
                        text_preprocessor=self.text_preprocessor,
                        melody_extractor=melody_extractor,
                        device=self.device
                    )
                    
                    self.models[model_name] = model
                    self.generators[model_name] = generator
                    
                    print(f"✓ Loaded {model_name} model")
                    
                except Exception as e:
                    print(f"✗ Failed to load {model_name}: {e}")
            else:
                print(f"✗ Model file not found: {filename}")
        
        if not self.models:
            raise RuntimeError("No models loaded successfully")
        
        print(f"\nSuccessfully loaded {len(self.models)} models: {list(self.models.keys())}")
    
    def run_systematic_evaluation(self) -> Dict[str, Any]:
        """
        Run systematic evaluation following assignment requirements.
        Tests each model on all 5 test melodies with 3 seed combinations each.
        """
        print("\n" + "="*80)
        print("SYSTEMATIC EVALUATION - Following Assignment Requirements")
        print("="*80)
        
        # Find test MIDI files
        test_midi_dir = self.data_dir / 'midi' / 'test'
        midi_files = list(test_midi_dir.glob('*.mid'))[:5]  # Take first 5 as specified
        
        if len(midi_files) < 5:
            print(f"Warning: Only found {len(midi_files)} test MIDI files, need 5")
        
        print(f"Testing on {len(midi_files)} MIDI files:")
        for i, midi_file in enumerate(midi_files):
            print(f"  {i+1}. {midi_file.name}")
        
        # Initialize results storage
        evaluation_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'models_tested': list(self.models.keys()),
                'test_files': [f.name for f in midi_files],
                'config': EVALUATION_CONFIG
            },
            'detailed_results': [],
            'summary_statistics': {},
            'model_comparisons': {}
        }
        
        total_tests = len(self.models) * len(midi_files) * len(EVALUATION_CONFIG['seed_word_combinations'])
        print(f"\nRunning {total_tests} total test combinations...")
        
        test_count = 0
        
        # Main evaluation loop
        for model_name, generator in self.generators.items():
            print(f"\n--- Testing {model_name.upper()} Model ---")
            model_results = []
            
            for midi_idx, midi_file in enumerate(midi_files):
                print(f"\nMIDI {midi_idx + 1}/5: {midi_file.name}")
                
                for seed_idx, seed_words in enumerate(EVALUATION_CONFIG['seed_word_combinations']):
                    test_count += 1
                    print(f"  Test {test_count}/{total_tests}: Seeds {seed_words}")
                    
                    # Test across different temperatures
                    for temp in EVALUATION_CONFIG['generation_parameters']['temperature_variants']:
                        
                        try:
                            # Generate lyrics
                            if model_name == 'baseline':
                                midi_path = None
                            else:
                                midi_path = str(midi_file)
                            
                            generations = generator.generate_lyrics(
                                seed_words=seed_words,
                                midi_path=midi_path,
                                max_length=EVALUATION_CONFIG['generation_parameters']['max_length'],
                                temperature=temp,
                                top_k=EVALUATION_CONFIG['generation_parameters']['top_k'],
                                num_generations=EVALUATION_CONFIG['generation_parameters']['num_generations_per_combination']
                            )
                            
                            # Evaluate each generation
                            for gen_idx, generation in enumerate(generations):
                                lyrics_words = generation['generated_text'].split()
                                
                                # Load melody features if available
                                if midi_path:
                                    melody_features = self.melody_extractor.extract_melody_sequence(midi_path)
                                else:
                                    melody_features = np.array([])
                                
                                # Calculate advanced metrics
                                advanced_metrics = {
                                    'melody_alignment_score': calculate_melody_alignment_score(lyrics_words, melody_features),
                                    'lyrical_coherence': calculate_lyrical_coherence(lyrics_words), 
                                    'creativity_score': calculate_creativity_score(lyrics_words),
                                    'musical_structure_fit': calculate_musical_structure_fit(lyrics_words)
                                }
                                
                                # Compile complete result
                                result = {
                                    'model_name': model_name,
                                    'midi_file': midi_file.name,
                                    'midi_index': midi_idx,
                                    'seed_words': seed_words,
                                    'seed_index': seed_idx,
                                    'temperature': temp,
                                    'generation_index': gen_idx,
                                    'generated_text': generation['generated_text'],
                                    'formatted_lyrics': generation['formatted_lyrics'],
                                    'word_count': generation['word_count'],
                                    'basic_stats': generation['generation_stats'],
                                    'advanced_metrics': advanced_metrics
                                }
                                
                                model_results.append(result)
                                evaluation_results['detailed_results'].append(result)
                                
                                print(f"    Gen {gen_idx+1}: {generation['word_count']} words, "
                                      f"coherence={advanced_metrics['lyrical_coherence']:.3f}")
                        
                        except Exception as e:
                            print(f"    Error generating with {seed_words}: {e}")
                            continue
            
            print(f"✓ {model_name} evaluation complete: {len(model_results)} generations")
        
        # Calculate summary statistics
        evaluation_results['summary_statistics'] = self._calculate_summary_statistics(evaluation_results['detailed_results'])
        evaluation_results['model_comparisons'] = self._generate_model_comparisons(evaluation_results['detailed_results'])
        
        print(f"\n✓ Systematic evaluation complete!")
        print(f"Total successful generations: {len(evaluation_results['detailed_results'])}")
        
        return evaluation_results
    
    def _calculate_summary_statistics(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics across all evaluations."""
        print("\nCalculating summary statistics...")
        
        stats = {}
        
        # Group by model
        by_model = defaultdict(list)
        for result in detailed_results:
            by_model[result['model_name']].append(result)
        
        for model_name, results in by_model.items():
            model_stats = {
                'total_generations': len(results),
                'avg_word_count': np.mean([r['word_count'] for r in results]),
                'avg_vocabulary_diversity': np.mean([r['basic_stats']['vocabulary_diversity'] for r in results]),
                'avg_melody_alignment': np.mean([r['advanced_metrics']['melody_alignment_score'] for r in results]),
                'avg_lyrical_coherence': np.mean([r['advanced_metrics']['lyrical_coherence'] for r in results]),
                'avg_creativity_score': np.mean([r['advanced_metrics']['creativity_score'] for r in results]),
                'avg_structure_fit': np.mean([r['advanced_metrics']['musical_structure_fit'] for r in results])
            }
            
            # Calculate standard deviations
            model_stats.update({
                'std_word_count': np.std([r['word_count'] for r in results]),
                'std_vocabulary_diversity': np.std([r['basic_stats']['vocabulary_diversity'] for r in results]),
                'std_melody_alignment': np.std([r['advanced_metrics']['melody_alignment_score'] for r in results]),
                'std_lyrical_coherence': np.std([r['advanced_metrics']['lyrical_coherence'] for r in results]),
                'std_creativity_score': np.std([r['advanced_metrics']['creativity_score'] for r in results]),
                'std_structure_fit': np.std([r['advanced_metrics']['musical_structure_fit'] for r in results])
            })
            
            stats[model_name] = model_stats
        
        return stats
    
    def _generate_model_comparisons(self, detailed_results: List[Dict]) -> Dict[str, Any]:
        """Generate detailed comparisons between models."""
        print("Generating model comparisons...")
        
        comparisons = {
            'metric_rankings': {},
            'statistical_tests': {},
            'best_performers': {}
        }
        
        # Group by model
        by_model = defaultdict(list)
        for result in detailed_results:
            by_model[result['model_name']].append(result)
        
        metrics = ['melody_alignment_score', 'lyrical_coherence', 'creativity_score', 'musical_structure_fit']
        
        for metric in metrics:
            # Calculate average score per model for this metric
            metric_scores = {}
            for model_name, results in by_model.items():
                if metric in ['melody_alignment_score'] and model_name == 'baseline':
                    continue  # Skip melody metrics for baseline
                scores = [r['advanced_metrics'][metric] for r in results]
                metric_scores[model_name] = np.mean(scores)
            
            # Rank models by this metric
            ranked = sorted(metric_scores.items(), key=lambda x: x[1], reverse=True)
            comparisons['metric_rankings'][metric] = ranked
            comparisons['best_performers'][metric] = ranked[0][0] if ranked else None
        
        return comparisons
    
    def save_evaluation_results(self, results: Dict[str, Any], filename_prefix: str = 'evaluation'):
        """Save evaluation results in multiple formats."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON (complete results)
        json_file = self.output_dir / f'{filename_prefix}_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"✓ Complete results saved: {json_file}")
        
        # Save CSV (detailed results table)
        csv_file = self.output_dir / f'{filename_prefix}_detailed_{timestamp}.csv'
        detailed_df = pd.DataFrame(results['detailed_results'])
        detailed_df.to_csv(csv_file, index=False)
        print(f"✓ Detailed CSV saved: {csv_file}")
        
        # Save summary CSV
        summary_csv = self.output_dir / f'{filename_prefix}_summary_{timestamp}.csv'
        summary_data = []
        for model_name, stats in results['summary_statistics'].items():
            row = {'model': model_name}
            row.update(stats)
            summary_data.append(row)
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_csv, index=False)
        print(f"✓ Summary CSV saved: {summary_csv}")
        
        # Generate HTML report
        self._generate_html_report(results, timestamp)
    
    def _generate_html_report(self, results: Dict[str, Any], timestamp: str):
        """Generate comprehensive HTML evaluation report."""
        html_file = self.output_dir / f'evaluation_report_{timestamp}.html'
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Melody-Conditioned Lyrics Generation - Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 5px; }}
        .metric-table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metric-table th {{ background-color: #f2f2f2; }}
        .best-score {{ background-color: #d4edda; font-weight: bold; }}
        .example {{ background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
        .model-section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Melody-Conditioned Lyrics Generation - Evaluation Report</h1>
        <p><strong>Evaluation Date:</strong> {results['metadata']['evaluation_date']}</p>
        <p><strong>Models Tested:</strong> {', '.join(results['metadata']['models_tested'])}</p>
        <p><strong>Test Files:</strong> {len(results['metadata']['test_files'])} MIDI files</p>
        <p><strong>Total Generations:</strong> {len(results['detailed_results'])}</p>
    </div>

    <h2>Summary Statistics</h2>
    <table class="metric-table">
        <tr>
            <th>Model</th>
            <th>Generations</th>
            <th>Avg Words</th>
            <th>Vocabulary Diversity</th>
            <th>Melody Alignment</th>
            <th>Lyrical Coherence</th>
            <th>Creativity Score</th>
            <th>Structure Fit</th>
        </tr>
"""
        
        for model_name, stats in results['summary_statistics'].items():
            html_content += f"""
        <tr>
            <td><strong>{model_name}</strong></td>
            <td>{stats['total_generations']}</td>
            <td>{stats['avg_word_count']:.1f}</td>
            <td>{stats['avg_vocabulary_diversity']:.3f}</td>
            <td>{stats.get('avg_melody_alignment', 'N/A') if isinstance(stats.get('avg_melody_alignment'), (int, float)) else 'N/A'}</td>
            <td>{stats['avg_lyrical_coherence']:.3f}</td>
            <td>{stats['avg_creativity_score']:.3f}</td>
            <td>{stats['avg_structure_fit']:.3f}</td>
        </tr>
"""
        
        html_content += """
    </table>

    <h2>🏆 Best Performers by Metric</h2>
    <ul>
"""
        
        for metric, best_model in results['model_comparisons']['best_performers'].items():
            html_content += f"        <li><strong>{metric.replace('_', ' ').title()}:</strong> {best_model}</li>\n"
        
        html_content += """
    </ul>

    <h2>Example Generations</h2>
"""
        
        # Add some example generations
        for model_name in results['metadata']['models_tested'][:2]:  # Show examples from first 2 models
            model_results = [r for r in results['detailed_results'] if r['model_name'] == model_name]
            if model_results:
                example = model_results[0]
                html_content += f"""
    <div class="model-section">
        <h3>{model_name.title()} Model Example</h3>
        <p><strong>MIDI:</strong> {example['midi_file']}</p>
        <p><strong>Seeds:</strong> {', '.join(example['seed_words'])}</p>
        <div class="example">
            {example['formatted_lyrics'].replace(chr(10), '<br>')}
        </div>
        <p><strong>Metrics:</strong> 
           Coherence: {example['advanced_metrics']['lyrical_coherence']:.3f}, 
           Creativity: {example['advanced_metrics']['creativity_score']:.3f}</p>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"✓ HTML report saved: {html_file}")


####################################### MAIN EVALUATION ORCHESTRATOR ##################################

def main():
    """Main evaluation pipeline following assignment requirements."""
    
    print("COMPREHENSIVE MELODY-CONDITIONED LYRICS EVALUATION")
    print("=" * 80)
    print("Following Deep Learning Assignment 3 evaluation requirements:")
    print("- Test on 5 melody files with 3 seed combinations each")
    print("- Compare concatenation vs conditioning approaches") 
    print("- Generate detailed analysis reports")
    print("=" * 80)
    
    # Configuration
    models_dir = 'models'
    data_dir = 'data'
    lyrics_data = 'data/sets/lyrics_train_set.csv'
    
    # Initialize evaluator
    evaluator = MelodyLyricsEvaluator(
        models_dir=models_dir,
        data_dir=data_dir,
        output_dir='evaluation_results'
    )
    
    try:
        # Load components
        evaluator.load_evaluation_components(lyrics_data)
        evaluator.load_trained_models()
        
        # Run systematic evaluation
        results = evaluator.run_systematic_evaluation()
        
        # Save results
        evaluator.save_evaluation_results(results)
        
        # Print final summary
        print("\n" + "="*80)
        print("EVALUATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Total generations analyzed: {len(results['detailed_results'])}")
        print(f"Results saved to: evaluation_results/")
        print(f"🏆 Best overall model by coherence: {results['model_comparisons']['best_performers'].get('lyrical_coherence', 'N/A')}")
        print(f"Best creativity: {results['model_comparisons']['best_performers'].get('creativity_score', 'N/A')}")
        print("="*80)
        
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()