# Architecture and Methodology - Assignment 3

## Model Architecture Overview

This project implements melody-conditioned lyrics generation using two distinct neural architectures that address the assignment's core challenge of integrating MIDI musical features with text generation.

### Baseline Models

**RNN Baselines**: Three variants for comparative analysis:
- **Standard LSTM**: 2 layers, 512 hidden units, dropout=0.3, lr=0.001
- **Conservative LSTM (V1)**: 2 layers, 256 hidden units, dropout=0.2, lr=0.0005  
- **Aggressive GRU (V2)**: 3 layers, 512 hidden units, dropout=0.4, lr=0.001

All baselines use 300D Word2Vec embeddings and cross-entropy loss with padding mask.

### Melody-Conditioned Architectures

Following the assignment specification for two fundamentally different approaches:

#### Approach A: Direct Concatenation (MelodyConcatenationRNN)

**Mathematical Formulation**:
```
h_t = RNN(concat(E(w_t), M_t), h_{t-1})
y_t = Softmax(W_o * h_t + b_o)
```

Where:
- E(w_t) ∈ ℝ^300: Word embedding at timestep t
- M_t ∈ ℝ^84: Melody features at timestep t  
- concat(E(w_t), M_t) ∈ ℝ^384: Combined input vector
- h_t ∈ ℝ^512: Hidden state

**Key Components**:
- Input dimension: 384D (300D words + 84D melody)
- Melody projection layer with tanh activation
- Frame-by-frame temporal alignment between lyrics and music

#### Approach B: Initial Conditioning + Continuous Attention (MelodyConditioningRNN)

**Mathematical Formulation**:
```
c = MelodyEncoder(M_{1:T})                    // Global conditioning
h_0 = σ(W_h * c + b_h)                        // Initial state conditioning
h_t = RNN(E(w_t), h_{t-1})                    // Standard RNN processing  
a_t = MultiHeadAttention(h_t, M_{proj}, M_{proj}) // Continuous attention
g_t = σ(W_g * concat(h_t, a_t) + b_g)        // Gating mechanism
o_t = g_t ⊙ a_t + (1 - g_t) ⊙ h_t           // Gated fusion
y_t = Softmax(W_o * o_t + b_o)               // Output prediction
```

**Key Components**:
- **Melody Encoder**: Global pooling or self-attention over melody sequence
- **Initial Conditioning**: Learned projection from melody summary to initial hidden states
- **Continuous Attention**: 8-head attention between RNN output and projected melody features
- **Gated Fusion**: Learned combination of attention and RNN outputs

### Musical Feature Representation

**MIDI Feature Extraction (84D per timestep)**:
- **Pitch Histogram (12D)**: Chromatic pitch distribution
- **Rhythm Features (12D)**: Note density, tempo variations, syncopation
- **Instrument Features (16D)**: Instrument family encoding
- **Temporal Features (44D)**: Beat-aligned dynamics and harmonic content

**Temporal Resolution**: 0.25 seconds per frame, aligned with lyrical content.

### Training Configuration

| Parameter | Baseline | Approach A | Approach B |
|-----------|----------|------------|------------|
| Input Dimension | 300D | 384D | 300D |
| Hidden Size | 512 | 512 | 512 |
| Batch Size | 32 | 16 | 16 |
| Learning Rate | 0.001 | 0.001 | 0.001 |
| Sequence Length | 50 tokens | 50 tokens | 50 tokens |

**Loss Function**: Cross-entropy with ignore_index=0 for padding
**Optimization**: Adam with ReduceLROnPlateau scheduling
**Regularization**: Gradient clipping (max_norm=1.0) and dropout

### Generation Strategy

**Non-Deterministic Sampling** (per assignment requirements):
```python
logits = model_output / temperature
top_k_logits, indices = torch.topk(logits, k)
probabilities = F.softmax(top_k_logits, dim=-1)
next_word = torch.multinomial(probabilities, 1)
```

**Parameters**: temperature ∈ [0.5, 1.2], top_k ∈ [20, 50] for creativity-quality balance.

### Evaluation Framework

**Quantitative Metrics**:
- **Perplexity**: exp(cross_entropy_loss) on test set
- **Diversity**: Unique n-gram ratios in generated text
- **Melody Alignment**: Correlation between musical features and generated content

**Qualitative Analysis**:
- Song structure assessment (verse/chorus patterns)
- Lyrical coherence and creativity scoring
- Musical-textual thematic consistency

This architecture enables systematic comparison between concatenation-based and attention-based melody conditioning while maintaining rigorous experimental controls and non-deterministic generation as specified in the assignment requirements.