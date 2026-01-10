#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple test script for MelodyRNN models
Tests both Approach A and Approach B
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.MelodyRNN import MelodyConcatenationRNN, MelodyConditioningRNN
    print("✅ Successfully imported MelodyRNN models!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_melody_models():
    """Test both melody-conditioned models"""
    print("\n🎵 Testing Melody-Conditioned RNN Models...")
    print("=" * 60)
    
    # Parameters
    vocab_size = 1000
    batch_size = 2
    seq_len = 8
    melody_len = 10
    melody_dim = 84
    
    # Create test data
    words = torch.randint(1, vocab_size, (batch_size, seq_len))
    melody = torch.randn(batch_size, melody_len, melody_dim)
    
    print(f"Test data shapes:")
    print(f"  Words: {words.shape}")
    print(f"  Melody: {melody.shape}")
    
    # Test Approach A: Concatenation
    print("\n🎼 Testing Approach A: Melody Concatenation")
    try:
        model_a = MelodyConcatenationRNN(vocab_size)
        
        # For concatenation, melody needs same seq_len as words
        melody_aligned = melody[:, :seq_len, :]  # [batch, seq_len, melody_dim]
        
        output_a, hidden_a = model_a(words, melody_aligned)
        print(f"  ✅ Forward pass successful!")
        print(f"  Output shape: {output_a.shape}")
        print(f"  Expected: [{batch_size}, {seq_len}, {vocab_size}]")
        
        # Test generation
        seed = torch.randint(1, vocab_size, (1, 3))
        generated = model_a.generate_text(seed, melody[:1], max_length=5)
        print(f"  ✅ Generation successful! Length: {generated.shape[1]}")
        
    except Exception as e:
        print(f"  ❌ Error in Approach A: {e}")
    
    # Test Approach B: Conditioning + Attention
    print("\n🎵 Testing Approach B: Melody Conditioning + Attention")
    try:
        model_b = MelodyConditioningRNN(vocab_size)
        
        output_b, hidden_b = model_b(words, melody)  # Can use full melody length
        print(f"  ✅ Forward pass successful!")
        print(f"  Output shape: {output_b.shape}")
        print(f"  Expected: [{batch_size}, {seq_len}, {vocab_size}]")
        
        # Test generation
        seed = torch.randint(1, vocab_size, (1, 3))
        generated = model_b.generate_text(seed, melody[:1], max_length=5)
        print(f"  ✅ Generation successful! Length: {generated.shape[1]}")
        
    except Exception as e:
        print(f"  ❌ Error in Approach B: {e}")
    
    print("\n🎯 Summary:")
    print("Both approaches test the FUNDAMENTAL differences:")
    print("  A: Direct concatenation at input level")
    print("  B: Initial conditioning + continuous attention + gating")
    print("Following professor's feedback for significant architectural changes!")

if __name__ == "__main__":
    test_melody_models()