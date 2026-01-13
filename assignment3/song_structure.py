"""
Advanced Song Structure Utilities
=================================
Enhanced utilities for creating professional song structures with:
- Verse/Chorus/Bridge detection
- Rhyme scheme implementation  
- Line length optimization
- Emotional flow analysis

Following music theory principles for lyrics generation.
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import Counter
import string


class SongStructureAnalyzer:
    """
    Analyze and enhance song structure in generated lyrics.
    Implements music theory principles for professional lyric structure.
    """
    
    def __init__(self):
        """Initialize with phonetic and structural patterns."""
        
        # Common rhyme endings for English lyrics
        self.rhyme_patterns = {
            'night_family': ['night', 'light', 'sight', 'bright', 'fight', 'right', 'might', 'flight'],
            'heart_family': ['heart', 'part', 'start', 'art', 'smart', 'apart', 'chart'],
            'love_family': ['love', 'above', 'dove', 'glove', 'shove'],
            'way_family': ['way', 'day', 'say', 'play', 'stay', 'away', 'today'],
            'time_family': ['time', 'rhyme', 'climb', 'prime', 'mime', 'chime'],
            'soul_family': ['soul', 'whole', 'goal', 'roll', 'control', 'hold', 'cold'],
            'dream_family': ['dream', 'stream', 'beam', 'team', 'seem', 'cream'],
            'fire_family': ['fire', 'desire', 'higher', 'wire', 'tire', 'inspire']
        }
        
        # Emotional intensity words
        self.emotion_weights = {
            'high_intensity': ['fire', 'passion', 'burning', 'screaming', 'flying', 'breaking'],
            'medium_intensity': ['feeling', 'wanting', 'hoping', 'trying', 'calling'],
            'low_intensity': ['thinking', 'remembering', 'walking', 'watching', 'sitting']
        }
        
        # Song section characteristics
        self.section_patterns = {
            'verse': {
                'typical_lines': 4,
                'word_density': 'medium',
                'emotional_progression': 'building',
                'rhyme_scheme': 'ABAB'
            },
            'chorus': {
                'typical_lines': 4,
                'word_density': 'high',
                'emotional_progression': 'peak',
                'rhyme_scheme': 'AABB'
            },
            'bridge': {
                'typical_lines': 2,
                'word_density': 'low', 
                'emotional_progression': 'contrasting',
                'rhyme_scheme': 'AA'
            }
        }
    
    def enhance_song_structure(self, words: List[str], target_structure: str = 'verse') -> Dict[str, any]:
        """
        Enhance a word sequence with professional song structure.
        
        Args:
            words (List[str]): Generated words
            target_structure (str): Target section type ('verse', 'chorus', 'bridge')
            
        Returns:
            Dict: Enhanced structure information
        """
        # Create optimized lines
        enhanced_lines = self.create_optimized_lines(words, target_structure)
        
        # Apply rhyme scheme
        rhymed_lines = self.apply_intelligent_rhyme_scheme(
            enhanced_lines, 
            self.section_patterns[target_structure]['rhyme_scheme']
        )
        
        # Format as professional lyrics
        formatted_lyrics = self.format_professional_lyrics(rhymed_lines, target_structure)
        
        # Calculate structure quality metrics
        quality_metrics = self.calculate_structure_quality(rhymed_lines, target_structure)
        
        return {
            'original_words': words,
            'enhanced_lines': rhymed_lines,
            'formatted_lyrics': formatted_lyrics,
            'structure_type': target_structure,
            'quality_metrics': quality_metrics,
            'rhyme_analysis': self.analyze_rhyme_quality(rhymed_lines)
        }
    
    def create_optimized_lines(self, words: List[str], section_type: str) -> List[str]:
        """Create lines optimized for the specific song section type."""
        
        section_config = self.section_patterns[section_type]
        target_lines = section_config['typical_lines']
        
        # Determine optimal words per line based on section type
        if section_type == 'verse':
            words_per_line = len(words) // target_lines if len(words) >= target_lines * 4 else 6
        elif section_type == 'chorus':
            words_per_line = max(4, len(words) // target_lines)  # Choruses can be punchier
        else:  # bridge
            words_per_line = max(6, len(words) // 2)  # Bridges are often longer lines
        
        lines = []
        current_line = []
        
        for i, word in enumerate(words):
            current_line.append(word)
            
            # Smart line break detection
            should_break = self._should_break_line(
                current_line, word, words_per_line, section_type, i, len(words)
            )
            
            if should_break:
                lines.append(' '.join(current_line))
                current_line = []
        
        # Add remaining words
        if current_line:
            lines.append(' '.join(current_line))
        
        # Ensure we have the right number of lines for the section
        lines = self._adjust_line_count(lines, target_lines, section_type)
        
        return lines
    
    def _should_break_line(self, current_line: List[str], word: str, target_length: int, 
                          section_type: str, word_index: int, total_words: int) -> bool:
        """Determine if we should break the line at this point."""
        
        # Basic length check
        if len(current_line) < 3:  # Minimum line length
            return False
        
        if len(current_line) >= target_length + 2:  # Maximum line length
            return True
        
        # Natural break points
        natural_breaks = [
            word.endswith(('.', '!', '?', ',')),
            word.endswith(('ing', 'ed', 'ly')),
            len(current_line) >= target_length and word in ['and', 'but', 'or', 'so'],
            len(current_line) >= target_length - 1 and len(word) >= 6  # Long word at end
        ]
        
        # Emotional intensity breaks (for dramatic effect)
        if section_type == 'chorus' and word.lower() in self.emotion_weights['high_intensity']:
            return len(current_line) >= target_length - 1
        
        return any(natural_breaks) and len(current_line) >= target_length - 2
    
    def _adjust_line_count(self, lines: List[str], target_count: int, section_type: str) -> List[str]:
        """Adjust line count to match section requirements."""
        
        if len(lines) == target_count:
            return lines
        
        elif len(lines) < target_count:
            # Split longer lines to reach target count
            adjusted_lines = []
            lines_to_add = target_count - len(lines)
            
            for i, line in enumerate(lines):
                adjusted_lines.append(line)
                
                # Split this line if we need more lines
                if lines_to_add > 0 and len(line.split()) >= 8:
                    words = line.split()
                    mid_point = len(words) // 2
                    
                    # Replace the line with two shorter lines
                    adjusted_lines[-1] = ' '.join(words[:mid_point])
                    adjusted_lines.append(' '.join(words[mid_point:]))
                    lines_to_add -= 1
            
            return adjusted_lines[:target_count]
        
        else:
            # Combine shorter lines to reach target count
            if target_count == 2 and len(lines) > 2:  # Bridge case
                # Combine into 2 longer lines
                mid_point = len(lines) // 2
                line1 = ' '.join(lines[:mid_point])
                line2 = ' '.join(lines[mid_point:])
                return [line1, line2]
            
            return lines[:target_count]
    
    def apply_intelligent_rhyme_scheme(self, lines: List[str], scheme: str) -> List[str]:
        """Apply intelligent rhyme scheme using phonetic similarity."""
        
        if len(lines) < 2 or not scheme:
            return lines
        
        rhymed_lines = lines.copy()
        
        try:
            if scheme == 'ABAB' and len(lines) >= 4:
                rhymed_lines = self._apply_abab_scheme(lines[:4]) + lines[4:]
            elif scheme == 'AABB' and len(lines) >= 4:
                rhymed_lines = self._apply_aabb_scheme(lines[:4]) + lines[4:]
            elif scheme == 'AA' and len(lines) >= 2:
                rhymed_lines = self._apply_aa_scheme(lines[:2]) + lines[2:]
        except Exception:
            # If rhyming fails, return original lines
            return lines
        
        return rhymed_lines
    
    def _apply_abab_scheme(self, lines: List[str]) -> List[str]:
        """Apply ABAB rhyme scheme to first 4 lines."""
        
        rhymed_lines = lines.copy()
        
        # Try to find rhymes for lines 0&2 and lines 1&3
        line0_words = lines[0].split()
        line1_words = lines[1].split()
        line2_words = lines[2].split()
        line3_words = lines[3].split()
        
        if line0_words and line2_words:
            # Find rhyme family for line 0's last word
            last_word_0 = line0_words[-1].lower().strip(string.punctuation)
            rhyme_family = self._find_rhyme_family(last_word_0)
            
            if rhyme_family and len(rhyme_family) > 1:
                # Try to replace last word of line 2 with a rhyme
                alternative_rhyme = self._find_suitable_rhyme(last_word_0, rhyme_family, line2_words[-1])
                if alternative_rhyme and alternative_rhyme != last_word_0:
                    line2_words[-1] = alternative_rhyme
                    rhymed_lines[2] = ' '.join(line2_words)
        
        if line1_words and line3_words:
            # Same for lines 1&3
            last_word_1 = line1_words[-1].lower().strip(string.punctuation)
            rhyme_family = self._find_rhyme_family(last_word_1)
            
            if rhyme_family and len(rhyme_family) > 1:
                alternative_rhyme = self._find_suitable_rhyme(last_word_1, rhyme_family, line3_words[-1])
                if alternative_rhyme and alternative_rhyme != last_word_1:
                    line3_words[-1] = alternative_rhyme
                    rhymed_lines[3] = ' '.join(line3_words)
        
        return rhymed_lines
    
    def _apply_aabb_scheme(self, lines: List[str]) -> List[str]:
        """Apply AABB rhyme scheme (couplets)."""
        
        rhymed_lines = lines.copy()
        
        # Lines 0&1 rhyme, lines 2&3 rhyme
        for pair_start in [0, 2]:
            if pair_start + 1 < len(lines):
                line_a_words = lines[pair_start].split()
                line_b_words = lines[pair_start + 1].split()
                
                if line_a_words and line_b_words:
                    last_word_a = line_a_words[-1].lower().strip(string.punctuation)
                    rhyme_family = self._find_rhyme_family(last_word_a)
                    
                    if rhyme_family and len(rhyme_family) > 1:
                        alternative_rhyme = self._find_suitable_rhyme(last_word_a, rhyme_family, line_b_words[-1])
                        if alternative_rhyme and alternative_rhyme != last_word_a:
                            line_b_words[-1] = alternative_rhyme
                            rhymed_lines[pair_start + 1] = ' '.join(line_b_words)
        
        return rhymed_lines
    
    def _apply_aa_scheme(self, lines: List[str]) -> List[str]:
        """Apply AA rhyme scheme (two lines rhyme)."""
        
        rhymed_lines = lines.copy()
        
        line0_words = lines[0].split()
        line1_words = lines[1].split()
        
        if line0_words and line1_words:
            last_word_0 = line0_words[-1].lower().strip(string.punctuation)
            rhyme_family = self._find_rhyme_family(last_word_0)
            
            if rhyme_family and len(rhyme_family) > 1:
                alternative_rhyme = self._find_suitable_rhyme(last_word_0, rhyme_family, line1_words[-1])
                if alternative_rhyme and alternative_rhyme != last_word_0:
                    line1_words[-1] = alternative_rhyme
                    rhymed_lines[1] = ' '.join(line1_words)
        
        return rhymed_lines
    
    def _find_rhyme_family(self, word: str) -> Optional[List[str]]:
        """Find the rhyme family for a word."""
        
        word_clean = word.lower().strip(string.punctuation)
        
        for family_name, family_words in self.rhyme_patterns.items():
            if word_clean in family_words:
                return family_words
        
        return None
    
    def _find_suitable_rhyme(self, original_word: str, rhyme_family: List[str], current_word: str) -> Optional[str]:
        """Find a suitable rhyme that's different from the original."""
        
        # Prefer rhymes that are not the same word
        alternatives = [w for w in rhyme_family if w != original_word.lower()]
        
        if alternatives:
            # Simple heuristic: prefer longer words for more sophisticated rhymes
            alternatives.sort(key=len, reverse=True)
            return alternatives[0]
        
        return None
    
    def format_professional_lyrics(self, lines: List[str], section_type: str) -> str:
        """Format lines as professional lyrics with section indicators."""
        
        if not lines:
            return ""
        
        # Add section header
        if section_type == 'verse':
            header = "[Verse]"
        elif section_type == 'chorus':
            header = "[Chorus]"
        elif section_type == 'bridge':
            header = "[Bridge]"
        else:
            header = f"[{section_type.title()}]"
        
        # Format with proper spacing
        formatted_lines = [header, ""] + lines + [""]
        
        return '\n'.join(formatted_lines)
    
    def calculate_structure_quality(self, lines: List[str], section_type: str) -> Dict[str, float]:
        """Calculate various quality metrics for the song structure."""
        
        if not lines:
            return {'overall_quality': 0.0}
        
        metrics = {}
        
        # Line count appropriateness
        target_lines = self.section_patterns[section_type]['typical_lines']
        line_count_score = 1.0 - abs(len(lines) - target_lines) / max(target_lines, 1)
        metrics['line_count_score'] = max(0.0, line_count_score)
        
        # Line length consistency
        line_lengths = [len(line.split()) for line in lines]
        avg_length = np.mean(line_lengths)
        length_variance = np.var(line_lengths)
        length_consistency = 1.0 - min(length_variance / max(avg_length, 1), 1.0)
        metrics['length_consistency'] = length_consistency
        
        # Emotional progression appropriateness
        emotion_progression = self._analyze_emotional_progression(lines, section_type)
        metrics['emotion_progression'] = emotion_progression
        
        # Overall structural quality
        overall_quality = (
            metrics['line_count_score'] * 0.3 +
            metrics['length_consistency'] * 0.3 +
            metrics['emotion_progression'] * 0.4
        )
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def _analyze_emotional_progression(self, lines: List[str], section_type: str) -> float:
        """Analyze if emotional progression matches section type."""
        
        line_emotions = []
        
        for line in lines:
            words = line.lower().split()
            emotion_score = 0
            
            for word in words:
                if word in self.emotion_weights['high_intensity']:
                    emotion_score += 3
                elif word in self.emotion_weights['medium_intensity']:
                    emotion_score += 2
                elif word in self.emotion_weights['low_intensity']:
                    emotion_score += 1
            
            # Normalize by line length
            line_emotion = emotion_score / max(len(words), 1)
            line_emotions.append(line_emotion)
        
        if not line_emotions:
            return 0.5
        
        # Check progression pattern
        expected_pattern = self.section_patterns[section_type]['emotional_progression']
        
        if expected_pattern == 'building':
            # Expect gradual increase
            if len(line_emotions) > 1:
                progression_score = np.corrcoef(range(len(line_emotions)), line_emotions)[0, 1]
                return max(0.0, progression_score)
        
        elif expected_pattern == 'peak':
            # Expect high consistent emotion
            avg_emotion = np.mean(line_emotions)
            return min(avg_emotion / 2.0, 1.0)  # Normalize
        
        elif expected_pattern == 'contrasting':
            # Expect emotional variety
            emotion_variance = np.var(line_emotions)
            return min(emotion_variance, 1.0)
        
        return 0.5  # Default neutral score
    
    def analyze_rhyme_quality(self, lines: List[str]) -> Dict[str, any]:
        """Analyze the quality of rhymes in the lines."""
        
        if len(lines) < 2:
            return {'rhyme_quality': 0.0, 'rhyme_pattern': 'none'}
        
        # Extract last words (potential rhymes)
        last_words = []
        for line in lines:
            words = line.split()
            if words:
                last_word = words[-1].lower().strip(string.punctuation)
                last_words.append(last_word)
        
        # Analyze rhyme patterns
        rhyme_matches = 0
        total_possible_matches = 0
        
        for i in range(len(last_words)):
            for j in range(i + 1, len(last_words)):
                total_possible_matches += 1
                
                if self._words_rhyme(last_words[i], last_words[j]):
                    rhyme_matches += 1
        
        rhyme_quality = rhyme_matches / max(total_possible_matches, 1)
        
        # Detect rhyme pattern
        pattern = self._detect_rhyme_pattern(last_words)
        
        return {
            'rhyme_quality': rhyme_quality,
            'rhyme_pattern': pattern,
            'rhyme_matches': rhyme_matches,
            'total_possible': total_possible_matches,
            'last_words': last_words
        }
    
    def _words_rhyme(self, word1: str, word2: str) -> bool:
        """Simple rhyme detection based on common endings."""
        
        if word1 == word2:
            return False  # Same word doesn't count as rhyme
        
        # Check if words are in the same rhyme family
        for family_words in self.rhyme_patterns.values():
            if word1 in family_words and word2 in family_words:
                return True
        
        # Simple phonetic check (ending sounds)
        common_endings = ['ight', 'art', 'ove', 'ay', 'ime', 'ool', 'eam', 'ire']
        
        for ending in common_endings:
            if word1.endswith(ending) and word2.endswith(ending):
                return True
        
        return False
    
    def _detect_rhyme_pattern(self, last_words: List[str]) -> str:
        """Detect the rhyme pattern (ABAB, AABB, etc.)."""
        
        if len(last_words) < 4:
            if len(last_words) == 2 and self._words_rhyme(last_words[0], last_words[1]):
                return 'AA'
            return 'free'
        
        # Check for ABAB pattern
        if (self._words_rhyme(last_words[0], last_words[2]) and 
            self._words_rhyme(last_words[1], last_words[3])):
            return 'ABAB'
        
        # Check for AABB pattern
        if (self._words_rhyme(last_words[0], last_words[1]) and 
            self._words_rhyme(last_words[2], last_words[3])):
            return 'AABB'
        
        return 'irregular'


def enhance_lyrics_structure(words: List[str], target_structure: str = 'verse') -> str:
    """
    Quick function to enhance lyrics structure.
    
    Args:
        words (List[str]): Generated words
        target_structure (str): Target section type
        
    Returns:
        str: Enhanced lyrics with professional structure
    """
    
    analyzer = SongStructureAnalyzer()
    enhanced = analyzer.enhance_song_structure(words, target_structure)
    
    return enhanced['formatted_lyrics']


# Example usage and testing
if __name__ == "__main__":
    
    # Test with sample lyrics
    test_words = [
        'walking', 'down', 'the', 'street', 'at', 'night', 
        'feeling', 'like', 'everything', 'is', 'right',
        'the', 'stars', 'are', 'shining', 'bright', 'above',
        'and', 'in', 'my', 'heart', 'i', 'feel', 'your', 'love'
    ]
    
    analyzer = SongStructureAnalyzer()
    
    # Test verse structure
    verse_result = analyzer.enhance_song_structure(test_words, 'verse')
    print("Enhanced Verse:")
    print(verse_result['formatted_lyrics'])
    print("\nQuality Metrics:")
    for metric, score in verse_result['quality_metrics'].items():
        print(f"  {metric}: {score:.3f}")
    print("\nRhyme Analysis:")
    for key, value in verse_result['rhyme_analysis'].items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    
    # Test chorus structure
    chorus_result = analyzer.enhance_song_structure(test_words, 'chorus')
    print("Enhanced Chorus:")
    print(chorus_result['formatted_lyrics'])