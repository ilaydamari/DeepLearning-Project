import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class LyricsRNN(nn.Module):
    """
    Recurrent Neural Network for lyrics generation using LSTM or GRU.
    Implements word-level language modeling with Word2Vec embeddings.
    
    During each step of the training phase, the architecture receives as input one word
    of the lyrics represented using Word2Vec (300 dimensions) as specified in the assignment.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_size: int = 512,
        num_layers: int = 2,
        rnn_type: str = 'LSTM',
        dropout: float = 0.3,
        pretrained_embeddings: Optional[np.ndarray] = None
    ):
        """
        Initialize the RNN model.
        
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of word embeddings (300 for Word2Vec)
            hidden_size (int): Hidden size of RNN layers
            num_layers (int): Number of RNN layers
            rnn_type (str): Type of RNN ('LSTM' or 'GRU')
            dropout (float): Dropout probability
            pretrained_embeddings (np.ndarray): Pre-trained embedding matrix
        """
        super(LyricsRNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        
        # Word embeddings layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize with pre-trained Word2Vec embeddings if provided
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings))
            # Optionally freeze embeddings
            # self.embedding.weight.requires_grad = False
        
        # RNN layer (LSTM or GRU)
        if rnn_type.upper() == 'LSTM':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        elif rnn_type.upper() == 'GRU':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}. Choose 'LSTM' or 'GRU'.")
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_size, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights."""
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for LSTM
                if self.rnn_type.upper() == 'LSTM' and 'bias_ih' in name:
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
    
    def forward(self, input_sequence: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        Forward pass of the model.
        
        Args:
            input_sequence (torch.Tensor): Input token sequences [batch_size, seq_len]
            hidden (Optional[Tuple]): Initial hidden state
            
        Returns:
            Tuple: (output_logits, hidden_state)
                - output_logits: [batch_size, seq_len, vocab_size]
                - hidden_state: Final hidden state
        """
        batch_size, seq_len = input_sequence.size()
        
        # Word embeddings
        embedded = self.embedding(input_sequence)  # [batch_size, seq_len, embedding_dim]
        embedded = self.dropout_layer(embedded)
        
        # RNN forward pass
        rnn_output, hidden = self.rnn(embedded, hidden)  # [batch_size, seq_len, hidden_size]
        rnn_output = self.dropout_layer(rnn_output)
        
        # Project to vocabulary size
        output_logits = self.output_projection(rnn_output)  # [batch_size, seq_len, vocab_size]
        
        return output_logits, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device):
        """
        Initialize hidden state.
        
        Args:
            batch_size (int): Batch size
            device (torch.device): Device to create tensors on
            
        Returns:
            Tuple or Tensor: Initial hidden state
        """
        if self.rnn_type.upper() == 'LSTM':
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return (h0, c0)
        else:  # GRU
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
            return h0
    
    def generate_text(
        self,
        start_sequence: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: torch.device = torch.device('cpu')
    ) -> torch.Tensor:
        """
        Generate text using the trained model.
        
        Args:
            start_sequence (torch.Tensor): Starting sequence [1, start_len]
            max_length (int): Maximum length to generate
            temperature (float): Sampling temperature
            top_k (Optional[int]): Top-k sampling
            top_p (Optional[float]): Top-p (nucleus) sampling
            device (torch.device): Device to run on
            
        Returns:
            torch.Tensor: Generated sequence
        """
        self.eval()
        
        with torch.no_grad():
            generated = start_sequence.clone()
            hidden = self.init_hidden(1, device)
            
            for _ in range(max_length):
                # Get logits for the last token
                output_logits, hidden = self.forward(generated, hidden)
                last_logits = output_logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = last_logits < torch.topk(last_logits, top_k)[0][..., -1, None]
                    last_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    last_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probabilities = F.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
                
                # Check for end token (assuming index 3 is <END>)
                if next_token.item() == 3:  # <END> token
                    break
            
            return generated

class LyricsRNNTrainer:
    """
    Training utility for the Lyrics RNN model.
    """
    
    def __init__(
        self,
        model: LyricsRNN,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5
    ):
        """
        Initialize trainer.
        
        Args:
            model (LyricsRNN): Model to train
            learning_rate (float): Learning rate
            weight_decay (float): Weight decay for regularization
        """
        self.model = model
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            input_batch (torch.Tensor): Input sequences [batch_size, seq_len]
            target_batch (torch.Tensor): Target sequences [batch_size, seq_len]
            
        Returns:
            float: Training loss
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        output_logits, _ = self.model(input_batch)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = output_logits.shape
        output_logits = output_logits.view(-1, vocab_size)
        target_batch = target_batch.view(-1)
        
        # Calculate loss
        loss = self.criterion(output_logits, target_batch)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update weights
        self.optimizer.step()
        
        return loss.item()
    
    def validate_step(self, input_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
        """
        Perform one validation step.
        
        Args:
            input_batch (torch.Tensor): Input sequences
            target_batch (torch.Tensor): Target sequences
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        
        with torch.no_grad():
            output_logits, _ = self.model(input_batch)
            
            # Reshape for loss calculation
            batch_size, seq_len, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            target_batch = target_batch.view(-1)
            
            loss = self.criterion(output_logits, target_batch)
            
        return loss.item()
    
    def save_model(self, filepath: str, epoch: int, loss: float):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        print(f"Model loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['loss']
