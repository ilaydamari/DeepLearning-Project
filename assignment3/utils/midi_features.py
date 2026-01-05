"""
MIDI feature extraction utilities for melody-conditioned lyrics generation.
Extracts musical features from MIDI files to create melody conditioning vectors.
Following the assignment requirements for two-approach melody integration.
"""

import pretty_midi
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
import os
from pathlib import Path


####################################### MIDI LOADING - File Processing ####################################
# Load and parse MIDI files using PrettyMIDI library
# Extract raw musical information for feature computation

def load_midi_file(midi_path: str) -> pretty_midi.PrettyMIDI:
    """
    Load MIDI file using PrettyMIDI.
    
    Args:
        midi_path (str): Path to MIDI file
        
    Returns:
        pretty_midi.PrettyMIDI: Loaded MIDI data
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        return midi_data
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None


####################################### MUSICAL FEATURE EXTRACTION - Core Components ###################
# Extract key musical features: pitches, rhythm, timing, instruments
# Create numerical representations suitable for neural network conditioning

class MelodyFeatureExtractor:
    """
    Extracts musical features from MIDI files for lyrics conditioning.
    Implements comprehensive feature extraction following assignment specifications.
    """
    
    def __init__(self, sample_rate: int = 16, max_duration: float = 30.0):
        """
        Initialize melody feature extractor.
        
        Args:
            sample_rate (int): Features per second for temporal alignment
            max_duration (float): Maximum song duration to process (seconds)
        """
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.feature_dim = 84  # Total feature dimension per timestep
        
        # Feature dimensions breakdown:
        # - Pitch histogram (12 notes): 12D
        # - Rhythm intensity: 1D  
        # - Tempo: 1D
        # - Note density: 1D
        # - Average pitch: 1D
        # - Pitch range: 1D
        # - Instrument encoding (up to 16 instruments): 16D
        # - Dynamics (velocity): 1D
        # - Reserved for extensions: 40D
    
    ####### PITCH ANALYSIS - Note and Harmony Extraction ####################
    # Extract pitch-related features: histograms, averages, ranges
    
    def extract_pitch_features(self, midi_data: pretty_midi.PrettyMIDI, 
                              start_time: float, end_time: float) -> Dict[str, float]:
        """
        Extract pitch-related features for a time window.
        
        Args:
            midi_data: MIDI data object
            start_time: Window start time
            end_time: Window end time
            
        Returns:
            Dict: Pitch feature dictionary
        """
        pitches = []
        velocities = []
        
        # Collect all notes in time window
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                if note.start >= start_time and note.start < end_time:
                    pitches.append(note.pitch)
                    velocities.append(note.velocity)
        
        if not pitches:
            # No notes in window
            return {
                'pitch_histogram': [0.0] * 12,
                'avg_pitch': 60.0,  # Middle C default
                'pitch_range': 0.0,
                'dynamics': 64.0   # Default velocity
            }
        
        # Pitch histogram (chromatic, normalized)
        pitch_histogram = [0.0] * 12
        for pitch in pitches:
            pitch_histogram[pitch % 12] += 1
        
        # Normalize histogram
        total_notes = len(pitches)
        pitch_histogram = [count / total_notes for count in pitch_histogram]
        
        # Statistical features
        avg_pitch = np.mean(pitches)
        pitch_range = np.max(pitches) - np.min(pitches) if len(pitches) > 1 else 0
        avg_velocity = np.mean(velocities)
        
        return {
            'pitch_histogram': pitch_histogram,
            'avg_pitch': avg_pitch,
            'pitch_range': pitch_range,
            'dynamics': avg_velocity
        }
    
    ####### RHYTHM ANALYSIS - Timing and Temporal Features ##################
    # Extract rhythm-related features: note density, tempo, timing patterns
    
    def extract_rhythm_features(self, midi_data: pretty_midi.PrettyMIDI,
                               start_time: float, end_time: float) -> Dict[str, float]:
        """
        Extract rhythm and timing features for a time window.
        
        Args:
            midi_data: MIDI data object
            start_time: Window start time  
            end_time: Window end time
            
        Returns:
            Dict: Rhythm feature dictionary
        """
        note_starts = []
        note_durations = []
        
        # Collect note timing information
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            for note in instrument.notes:
                if note.start >= start_time and note.start < end_time:
                    note_starts.append(note.start)
                    note_durations.append(note.end - note.start)
        
        window_duration = end_time - start_time
        
        if not note_starts:
            return {
                'note_density': 0.0,
                'avg_note_duration': 0.5,  # Default half-beat
                'rhythm_intensity': 0.0
            }
        
        # Note density (notes per second)
        note_density = len(note_starts) / window_duration
        
        # Average note duration
        avg_duration = np.mean(note_durations)
        
        # Rhythm intensity (based on note density and duration variation)
        duration_std = np.std(note_durations) if len(note_durations) > 1 else 0
        rhythm_intensity = note_density * (1 + duration_std)
        
        return {
            'note_density': note_density,
            'avg_note_duration': avg_duration,
            'rhythm_intensity': rhythm_intensity
        }
    
    ####### INSTRUMENT ANALYSIS - Orchestration Features ####################
    # Extract instrument-related features and orchestration patterns
    
    def extract_instrument_features(self, midi_data: pretty_midi.PrettyMIDI) -> List[float]:
        """
        Extract instrument presence encoding.
        
        Args:
            midi_data: MIDI data object
            
        Returns:
            List[float]: 16D instrument presence vector
        """
        # Group instruments into categories (16 categories max)
        instrument_categories = [
            [0, 7],      # Piano family (0-7)
            [8, 15],     # Chromatic Percussion (8-15)  
            [16, 23],    # Organ family (16-23)
            [24, 31],    # Guitar family (24-31)
            [32, 39],    # Bass family (32-39)
            [40, 47],    # Strings (orchestral) (40-47)
            [48, 55],    # Ensemble (48-55)
            [56, 63],    # Brass family (56-63)
            [64, 71],    # Reed family (64-71)
            [72, 79],    # Pipe family (72-79)
            [80, 87],    # Synth Lead (80-87)
            [88, 95],    # Synth Pad (88-95)
            [96, 103],   # Synth Effects (96-103)
            [104, 111],  # Ethnic instruments (104-111)
            [112, 119],  # Percussive (112-119)
            [120, 127]   # Sound effects (120-127)
        ]
        
        presence_vector = [0.0] * 16
        
        for instrument in midi_data.instruments:
            if instrument.is_drum:
                continue
                
            program = instrument.program
            
            # Find category for this instrument
            for i, (start, end) in enumerate(instrument_categories):
                if start <= program <= end:
                    presence_vector[i] = 1.0
                    break
        
        return presence_vector
    
    ####### TEMPORAL FEATURE SEQUENCE - Complete Melody Vector ###############
    # Create complete temporal feature sequence aligned with lyrics timeline
    
    def extract_melody_sequence(self, midi_path: str) -> Optional[np.ndarray]:
        """
        Extract complete melody feature sequence from MIDI file.
        Creates temporal sequence suitable for RNN conditioning.
        
        Args:
            midi_path (str): Path to MIDI file
            
        Returns:
            Optional[np.ndarray]: Melody features [time_steps, feature_dim]
        """
        midi_data = load_midi_file(midi_path)
        if midi_data is None:
            return None
        
        # Get actual song duration (clamp to max_duration)
        song_duration = min(midi_data.get_end_time(), self.max_duration)
        time_steps = int(song_duration * self.sample_rate)
        
        # Initialize feature matrix
        melody_features = np.zeros((time_steps, self.feature_dim))
        
        # Extract global instrument features (same for all timesteps)
        instrument_features = self.extract_instrument_features(midi_data)
        
        # Extract time-varying features per timestep
        for t in range(time_steps):
            start_time = t / self.sample_rate
            end_time = (t + 1) / self.sample_rate
            
            # Extract pitch features
            pitch_feats = self.extract_pitch_features(midi_data, start_time, end_time)
            
            # Extract rhythm features  
            rhythm_feats = self.extract_rhythm_features(midi_data, start_time, end_time)
            
            # Assemble complete feature vector
            feature_vector = []
            
            # Pitch histogram (12D)
            feature_vector.extend(pitch_feats['pitch_histogram'])
            
            # Scalar pitch features (3D)
            feature_vector.append(pitch_feats['avg_pitch'] / 127.0)  # Normalize to [0,1]
            feature_vector.append(pitch_feats['pitch_range'] / 127.0)
            feature_vector.append(pitch_feats['dynamics'] / 127.0)
            
            # Rhythm features (3D)  
            feature_vector.append(min(rhythm_feats['note_density'] / 10.0, 1.0))  # Clamp
            feature_vector.append(min(rhythm_feats['avg_note_duration'] / 4.0, 1.0))  # Normalize
            feature_vector.append(min(rhythm_feats['rhythm_intensity'] / 20.0, 1.0))  # Clamp
            
            # Instrument features (16D)
            feature_vector.extend(instrument_features)
            
            # Tempo and global features (2D)
            tempo = 120.0  # Default tempo if not extractable
            try:
                tempo_changes = midi_data.get_tempo_changes()
                if len(tempo_changes[1]) > 0:
                    tempo = tempo_changes[1][0]
            except:
                pass
            
            feature_vector.append(tempo / 200.0)  # Normalize typical tempo range
            feature_vector.append(song_duration / self.max_duration)  # Song length ratio
            
            # Pad to fixed dimension if needed
            while len(feature_vector) < self.feature_dim:
                feature_vector.append(0.0)
            
            melody_features[t] = feature_vector[:self.feature_dim]
        
        return melody_features
    
    ####### BATCH PROCESSING - Multiple MIDI Files #########################
    # Process multiple MIDI files efficiently for dataset creation
    
    def process_midi_directory(self, midi_dir: str, output_file: str = None) -> Dict[str, np.ndarray]:
        """
        Process all MIDI files in directory and extract features.
        
        Args:
            midi_dir (str): Directory containing MIDI files
            output_file (str, optional): Path to save processed features
            
        Returns:
            Dict[str, np.ndarray]: Mapping from filename to melody features
        """
        midi_features = {}
        midi_dir = Path(midi_dir)
        
        print(f"Processing MIDI files from {midi_dir}...")
        
        # Find all MIDI files
        midi_files = list(midi_dir.glob("*.mid")) + list(midi_dir.glob("*.midi"))
        
        if not midi_files:
            print("No MIDI files found in directory")
            return {}
        
        for midi_file in midi_files:
            print(f"Processing {midi_file.name}...")
            
            try:
                features = self.extract_melody_sequence(str(midi_file))
                if features is not None:
                    midi_features[midi_file.stem] = features
                    print(f"  Extracted features shape: {features.shape}")
                else:
                    print(f"  Failed to extract features")
            except Exception as e:
                print(f"  Error processing {midi_file.name}: {e}")
        
        print(f"Successfully processed {len(midi_features)}/{len(midi_files)} MIDI files")
        
        # Save processed features if requested
        if output_file:
            np.savez_compressed(output_file, **midi_features)
            print(f"Saved processed features to {output_file}")
        
        return midi_features


####################################### MELODY-LYRICS ALIGNMENT - Temporal Synchronization ############
# Align melody features with lyrics sequences for proper conditioning
# Handle different sequence lengths and create synchronized training pairs

def align_melody_with_lyrics(melody_features: np.ndarray, 
                           lyrics_length: int,
                           alignment_method: str = 'interpolate') -> np.ndarray:
    """
    Align melody features with lyrics sequence length.
    
    Args:
        melody_features (np.ndarray): Melody features [melody_steps, feature_dim]
        lyrics_length (int): Target lyrics sequence length
        alignment_method (str): 'interpolate', 'repeat', or 'truncate'
        
    Returns:
        np.ndarray: Aligned features [lyrics_length, feature_dim]
    """
    if melody_features is None or len(melody_features) == 0:
        # Return zero features if no melody
        return np.zeros((lyrics_length, melody_features.shape[1] if melody_features.ndim > 1 else 84))
    
    melody_steps, feature_dim = melody_features.shape
    
    if alignment_method == 'interpolate':
        # Linear interpolation to match lyrics length
        if melody_steps == lyrics_length:
            return melody_features
        
        # Create interpolation indices
        old_indices = np.linspace(0, melody_steps - 1, melody_steps)
        new_indices = np.linspace(0, melody_steps - 1, lyrics_length)
        
        # Interpolate each feature dimension
        aligned_features = np.zeros((lyrics_length, feature_dim))
        for i in range(feature_dim):
            aligned_features[:, i] = np.interp(new_indices, old_indices, melody_features[:, i])
        
        return aligned_features
    
    elif alignment_method == 'repeat':
        # Repeat melody features to match lyrics length
        repeat_factor = lyrics_length // melody_steps + 1
        repeated = np.tile(melody_features, (repeat_factor, 1))
        return repeated[:lyrics_length]
    
    elif alignment_method == 'truncate':
        # Truncate or pad to match length
        if melody_steps >= lyrics_length:
            return melody_features[:lyrics_length]
        else:
            # Pad with last frame
            padding = np.tile(melody_features[-1:], (lyrics_length - melody_steps, 1))
            return np.vstack([melody_features, padding])
    
    else:
        raise ValueError(f"Unknown alignment method: {alignment_method}")


####################################### UTILITY FUNCTIONS - Helper Tools ################################
# Additional utility functions for MIDI processing and feature manipulation

def get_melody_summary(melody_features: np.ndarray) -> Dict[str, float]:
    """
    Compute summary statistics of melody features for analysis.
    
    Args:
        melody_features (np.ndarray): Melody feature sequence
        
    Returns:
        Dict[str, float]: Summary statistics
    """
    if melody_features is None or len(melody_features) == 0:
        return {'error': 'No melody features available'}
    
    return {
        'duration_seconds': len(melody_features) / 16,  # Assuming 16 fps
        'avg_pitch': np.mean(melody_features[:, 12]),  # Avg pitch feature
        'avg_rhythm_intensity': np.mean(melody_features[:, 17]),  # Rhythm intensity
        'active_instruments': np.sum(np.max(melody_features[:, 18:34], axis=0)),  # Active instruments
        'avg_tempo': np.mean(melody_features[:, 34]) * 200,  # Denormalize tempo
        'feature_variance': np.mean(np.var(melody_features, axis=0))
    }


def create_melody_conditioning_vector(melody_features: np.ndarray, method: str = 'mean') -> np.ndarray:
    """
    Create single conditioning vector from melody sequence for global conditioning.
    
    Args:
        melody_features (np.ndarray): Full melody sequence [time_steps, feature_dim]
        method (str): Aggregation method ('mean', 'max', 'weighted_mean')
        
    Returns:
        np.ndarray: Single conditioning vector [feature_dim]
    """
    if melody_features is None or len(melody_features) == 0:
        return np.zeros(84)  # Default feature dimension
    
    if method == 'mean':
        return np.mean(melody_features, axis=0)
    elif method == 'max':
        return np.max(melody_features, axis=0)
    elif method == 'weighted_mean':
        # Weight by rhythm intensity (assuming it's at index 17)
        weights = melody_features[:, 17] + 1e-8  # Add small epsilon
        weights = weights / np.sum(weights)
        return np.average(melody_features, axis=0, weights=weights)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


if __name__ == "__main__":
    # Example usage and testing
    extractor = MelodyFeatureExtractor(sample_rate=16, max_duration=30.0)
    
    # Test with a sample MIDI file (if available)
    test_midi = "sample.mid"
    if os.path.exists(test_midi):
        features = extractor.extract_melody_sequence(test_midi)
        if features is not None:
            print(f"Extracted melody features: {features.shape}")
            summary = get_melody_summary(features)
            print("Melody summary:", summary)
            
            # Test conditioning vector creation
            conditioning = create_melody_conditioning_vector(features, method='weighted_mean')
            print(f"Conditioning vector shape: {conditioning.shape}")
        else:
            print("Failed to extract features from sample MIDI")
    else:
        print("No sample MIDI file found for testing")
