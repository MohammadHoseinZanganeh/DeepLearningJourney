import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Encoder with Embedding + RNN + Linear
    Supports: RNN, LSTM, GRU (bidirectional or not)
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, num_labels, 
                 num_layers=1, dropout=0.0, rnn_type='bilstm'):  #  dropout=0.0 default
        super(Encoder, self).__init__()
        
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.rnn_type = rnn_type
        self.num_layers = num_layers
        
        # 1. Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        
        # 2. RNN layer
        self.bidirectional = 'bi' in rnn_type
        self.num_directions = 2 if self.bidirectional else 1
        
        rnn_input_size = embedding_size
        
        if rnn_type in ['lstm', 'bilstm']:
            self.rnn = nn.LSTM(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type in ['gru', 'bigru']:
            self.rnn = nn.GRU(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif rnn_type in ['birnn']:
            self.rnn = nn.RNN(
                input_size=rnn_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                bidirectional=self.bidirectional,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        else:
            raise ValueError(f"Unknown rnn_type: {rnn_type}")
        
        # 3. Linear layer
        self.fc = nn.Linear(hidden_size * self.num_directions, num_labels)
        
        # Dropout (if dropout > 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        
        Returns:
            logits: [batch_size, seq_len, num_labels]
        """
        # Embedding
        embeddings = self.embedding(input_ids)  # [batch, seq_len, embed]
        
        # RNN
        rnn_output, _ = self.rnn(embeddings)  # [batch, seq_len, hidden*dir]
        
        # Dropout (if exists)
        if self.dropout is not None:
            rnn_output = self.dropout(rnn_output)
        
        # Linear
        logits = self.fc(rnn_output)  # [batch, seq_len, num_labels]
        
        return logits