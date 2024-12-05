import torch 
from torch import nn
import numpy as np 
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_length, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class AttentionPooling(nn.Module):
    def __init__(self, model_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x: (batch_size, seq_length, model_dim)
        scores = self.attention(x).squeeze(-1)  # Shape: (batch_size, seq_length)
        weights = F.softmax(scores, dim=1).unsqueeze(-1)  # Shape: (batch_size, seq_length, 1)
        pooled = torch.sum(x * weights, dim=1)  # Shape: (batch_size, model_dim)
        return pooled

# Transformer Classifier with different transformer configurations for src and notes
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=100, note_dim=1024, src_model_dim=48, note_model_dim=96, 
                src_num_heads=3, note_num_heads=3, src_num_layers=2, note_num_layers=3,
                dropout=0.6, max_len=5000):
        super(TransformerClassifier, self).__init__()

        self.embedding = nn.Linear(input_dim, src_model_dim)
        self.note_embedding = nn.Linear(note_dim, note_model_dim)
        self.batch_norm = nn.BatchNorm1d(src_model_dim)
        self.note_batch_norm = nn.BatchNorm1d(note_model_dim)
        self.pos_encoder = PositionalEncoding(src_model_dim, max_len)
        self.note_pos_encoder = PositionalEncoding(note_model_dim, max_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder for src
        src_encoder_layer = nn.TransformerEncoderLayer(
            d_model=src_model_dim,
            nhead=src_num_heads,
            dim_feedforward=src_model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.src_transformer_encoder = nn.TransformerEncoder(src_encoder_layer, num_layers=src_num_layers)

        # Transformer Encoder for notes
        note_encoder_layer = nn.TransformerEncoderLayer(
            d_model=note_model_dim,
            nhead=note_num_heads,
            dim_feedforward=note_model_dim * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.note_transformer_encoder = nn.TransformerEncoder(note_encoder_layer, num_layers=note_num_layers)

        self.attention_pool = AttentionPooling(src_model_dim)
        self.note_attention_pool = AttentionPooling(note_model_dim)  # Added AttentionPooling for notes
        self.fc = nn.Linear(src_model_dim + note_model_dim, 1)
        # self.sigmoid = nn.Sigmoid() not used, by testing the model, the sigmoid is not needed

    def forward(self, src, notes, src_key_padding_mask, notes_key_padding_mask):
        # src: (batch_size, seq_length, input_dim)
        src = self.embedding(src)  # Shape: (batch_size, seq_length, src_model_dim)
        src = self.batch_norm(src.transpose(1, 2)).transpose(1, 2)
        src = self.pos_encoder(src)
        src = self.dropout(src)
        src_output = self.src_transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        src_output = self.dropout(src_output)
        src_output = self.attention_pool(src_output)

        # notes: (batch_size, seq_length, note_dim)
        notes = self.note_embedding(notes)
        notes = self.note_batch_norm(notes.transpose(1, 2)).transpose(1, 2)
        notes = self.note_pos_encoder(notes)
        notes = self.dropout(notes)
        notes_output = self.note_transformer_encoder(notes, src_key_padding_mask=notes_key_padding_mask)
        notes_output = self.dropout(notes_output)
        notes_output = self.note_attention_pool(notes_output)  # Use the correct AttentionPooling for notes

        logit = self.fc(torch.cat([src_output, notes_output], dim=1))
        return logit.squeeze(-1)

# 6. 定义自定义损失函数
# ----------------------

class LabelSmoothingFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2, smoothing=0.1, reduction='mean'):
        """
        integrating Label Smoothing and Focal Loss
        Args:
            alpha (float): weight for positive class
            gamma (float): focusing parameter for modulating factor (1 - pt)
            smoothing (float): label smoothing parameter
            reduction (str): method to reduce

        """
        super(LabelSmoothingFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):

        inputs = inputs.view(-1)
        targets = targets.view(-1)


        with torch.no_grad():
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
        
