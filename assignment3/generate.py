import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import List, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.text_utils import TextPreprocessor
from models.RNN_baseline import LyricsRNN

class LyricsGenerator:
    """
    Text generator for lyrics using trained RNN model.
    """
    
    def __init__(self, model_path: str, preprocessor_path: str, device: str = 'auto'):
        """
        Initialize the lyrics generator.
        
        Args:
            model_path (str): Path to the trained model
            preprocessor_path (str): Path to the preprocessor
            device (str): Device to run on ('auto', 'cpu', or 'cuda')
        """
        self.device = self._get_device(device)
        
        # Load preprocessor
        print("Loading preprocessor...")
        self.preprocessor = TextPreprocessor()
        self.preprocessor.load_preprocessor(preprocessor_path)
        
        # Load model
        print("Loading model...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model with correct parameters
        self.model = LyricsRNN(
            vocab_size=self.preprocessor.vocab_size,
            embedding_dim=300,  # Word2Vec dimension
            hidden_size=512,
            num_layers=2,
            rnn_type='LSTM',
            dropout=0.3
        )
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {self.preprocessor.vocab_size}")
    
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device."""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            return torch.device(device)
    
    def generate(
        self,
        seed_text: str = "",
        max_length: int = 100,
        temperature: float = 0.8,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate lyrics text.
        
        Args:
            seed_text (str): Seed text to start generation
            max_length (int): Maximum length to generate
            temperature (float): Sampling temperature (higher = more random)
            top_k (Optional[int]): Top-k sampling (None to disable)
            top_p (Optional[float]): Top-p (nucleus) sampling (None to disable)
            num_samples (int): Number of samples to generate
            
        Returns:
            List[str]: Generated lyrics
        """
        generated_lyrics = []
        
        for i in range(num_samples):
            # Prepare seed sequence
            if seed_text.strip():
                seed_sequence = self.preprocessor.text_to_sequence(seed_text)
            else:
                # Start with just the START token
                seed_sequence = [self.preprocessor.special_tokens['<START>']]
            
            seed_tensor = torch.LongTensor([seed_sequence]).to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_sequence = self.model.generate_text(
                    start_sequence=seed_tensor,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    device=self.device
                )
            
            # Convert back to text
            generated_text = self.preprocessor.sequence_to_text(
                generated_sequence[0].cpu().tolist()
            )
            
            # Clean up the output
            generated_text = self._clean_generated_text(generated_text)
            generated_lyrics.append(generated_text)
        
        return generated_lyrics
    
    def _clean_generated_text(self, text: str) -> str:
        """Clean up generated text."""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        return text
    
    def interactive_generation(self):
        """Interactive text generation session."""
        print("\n=== Interactive Lyrics Generation ===")
        print("Enter a seed phrase to start lyrics generation.")
        print("Leave empty for random generation.")
        print("Type 'quit' to exit.\n")
        
        while True:
            try:
                seed = input("Seed text: ").strip()
                
                if seed.lower() == 'quit':
                    print("Goodbye!")
                    break
                
                # Get generation parameters
                try:
                    max_len = int(input("Max length (default 100): ") or "100")
                    temp = float(input("Temperature (default 0.8): ") or "0.8")
                    num_samples = int(input("Number of samples (default 1): ") or "1")
                except ValueError:
                    print("Invalid input, using defaults.")
                    max_len, temp, num_samples = 100, 0.8, 1
                
                print(f"\nGenerating lyrics...")
                print("=" * 60)
                
                # Generate lyrics
                generated = self.generate(
                    seed_text=seed,
                    max_length=max_len,
                    temperature=temp,
                    num_samples=num_samples
                )
                
                # Display results
                for i, lyrics in enumerate(generated, 1):
                    print(f"\nSample {i}:")
                    print(f"'{lyrics}'")
                
                print("=" * 60)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def generate_with_themes(self, themes: List[str], num_per_theme: int = 3) -> dict:
        """
        Generate lyrics with different themes.
        
        Args:
            themes (List[str]): List of theme words/phrases
            num_per_theme (int): Number of samples per theme
            
        Returns:
            dict: Generated lyrics organized by theme
        """
        results = {}
        
        for theme in themes:
            print(f"Generating lyrics for theme: '{theme}'")
            lyrics = self.generate(
                seed_text=theme,
                max_length=80,
                temperature=0.8,
                num_samples=num_per_theme
            )
            results[theme] = lyrics
        
        return results

def main():
    """Main generation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate lyrics using trained RNN')
    parser.add_argument('--model', type=str, default='models/best_lyrics_model.pth',
                       help='Path to trained model')
    parser.add_argument('--preprocessor', type=str, default='models/preprocessor.pkl',
                       help='Path to preprocessor')
    parser.add_argument('--seed', type=str, default='',
                       help='Seed text for generation')
    parser.add_argument('--length', type=int, default=100,
                       help='Maximum generation length')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--samples', type=int, default=1,
                       help='Number of samples to generate')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--themes', nargs='+', default=[],
                       help='Generate lyrics for specific themes')
    
    args = parser.parse_args()
    
    try:
        # Initialize generator
        generator = LyricsGenerator(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            device='auto'
        )
        
        if args.interactive:
            # Interactive mode
            generator.interactive_generation()
        
        elif args.themes:
            # Theme-based generation
            print(f"Generating lyrics for themes: {args.themes}")
            results = generator.generate_with_themes(args.themes, num_per_theme=2)
            
            for theme, lyrics_list in results.items():
                print(f"\n=== Theme: {theme} ===")
                for i, lyrics in enumerate(lyrics_list, 1):
                    print(f"\nSample {i}:")
                    print(f"'{lyrics}'")
        
        else:
            # Single generation
            print(f"Generating lyrics with seed: '{args.seed}'")
            generated = generator.generate(
                seed_text=args.seed,
                max_length=args.length,
                temperature=args.temperature,
                num_samples=args.samples
            )
            
            for i, lyrics in enumerate(generated, 1):
                print(f"\nSample {i}:")
                print(f"'{lyrics}'")
    
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Make sure you have trained the model first by running train.py")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
