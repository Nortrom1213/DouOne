"""
This file includes the torch models. We wrap the three
models into one class for convenience.
"""



import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention


class LandlordLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)

class FarmerLstmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(162, 128, batch_first=True)
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        lstm_out, (h_n, _) = self.lstm(z)
        lstm_out = lstm_out[:,-1,:]
        x = torch.cat([lstm_out,x], dim=-1)
        x = self.dense1(x)
        x = torch.relu(x)
        x = self.dense2(x)
        x = torch.relu(x)
        x = self.dense3(x)
        x = torch.relu(x)
        x = self.dense4(x)
        x = torch.relu(x)
        x = self.dense5(x)
        x = torch.relu(x)
        x = self.dense6(x)
        if return_value:
            return dict(values=x)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x.shape[0], (1,))[0]
            else:
                action = torch.argmax(x,dim=0)[0]
            return dict(action=action)


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Positional encoding module.
        d_model: feature dimension (here 162)
        dropout: dropout rate
        max_len: maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # even indices
        pe[:, 1::2] = torch.cos(position * div_term)   # odd indices
        pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class LandlordTransformerModel(nn.Module):
    def __init__(self):
        super(LandlordTransformerModel, self).__init__()
        # Positional encoding for historical sequence (z)
        self.pos_encoder = PositionalEncoding(d_model=162, dropout=0.1, max_len=5000)
        # Transformer encoder to process historical information
        self.transformer_layer = TransformerEncoderLayer(d_model=162, nhead=6, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)
        # Attention pooling: compute attention weights for each time step in the sequence
        self.attn_weights = nn.Sequential(
            nn.Linear(162, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Projection layer: project the pooled transformer output from 162 to 128 dimensions (to mimic LSTM hidden state)
        self.transformer_proj = nn.Linear(162, 128)
        # Dense1: input dimension = 373 (state feature) + 128 (projected pooled transformer output) = 501
        self.dense1 = nn.Linear(373 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        """
        z: historical information tensor of shape (batch, seq_len, 162)
        x: current state features for landlord of shape (batch, 373)
        """
        # Add positional encoding to history
        z = self.pos_encoder(z)                      # (batch, seq_len, 162)
        z_encoded = self.transformer_encoder(z)      # (batch, seq_len, 162)

        # Attention pooling: compute weight for each time step
        # Compute raw attention scores: (batch, seq_len, 1)
        attn_scores = self.attn_weights(z_encoded)
        # Apply softmax over the seq_len dimension to get attention weights: (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        # Weighted sum of transformer outputs: (batch, 162)
        pooled = torch.sum(z_encoded * attn_weights, dim=1)

        # Project pooled features to 128 dimensions
        trans_feat = self.transformer_proj(pooled)   # (batch, 128)

        # Concatenate with current state features x (dimension 373)
        combined = torch.cat([trans_feat, x], dim=-1)   # (batch, 128 + 373 = 501)
        x_out = self.dense1(combined)
        x_out = F.relu(x_out)
        x_out = self.dense2(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense3(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense4(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense5(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense6(x_out)
        if return_value:
            return dict(values=x_out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x_out.shape[0], (1,))[0]
            else:
                action = torch.argmax(x_out, dim=0)[0]
            return dict(action=action)

class FarmerTransformerModel(nn.Module):
    def __init__(self):
        super(FarmerTransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=162, dropout=0.1, max_len=5000)
        self.transformer_layer = TransformerEncoderLayer(d_model=162, nhead=6, dropout=0.1, batch_first=True)
        self.transformer_encoder = TransformerEncoder(self.transformer_layer, num_layers=2)
        self.attn_weights = nn.Sequential(
            nn.Linear(162, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # Projection: project pooled transformer output from 162 to 128
        self.transformer_proj = nn.Linear(162, 128)
        # For farmer, state feature x dimension is 484, so dense1 input = 484 + 128 = 612
        self.dense1 = nn.Linear(484 + 128, 512)
        self.dense2 = nn.Linear(512, 512)
        self.dense3 = nn.Linear(512, 512)
        self.dense4 = nn.Linear(512, 512)
        self.dense5 = nn.Linear(512, 512)
        self.dense6 = nn.Linear(512, 1)

    def forward(self, z, x, return_value=False, flags=None):
        z = self.pos_encoder(z)                      # (batch, seq_len, 162)
        z_encoded = self.transformer_encoder(z)      # (batch, seq_len, 162)
        # Attention pooling: compute attention weights for each time step
        attn_scores = self.attn_weights(z_encoded)     # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_scores, dim=1)     # (batch, seq_len, 1)
        pooled = torch.sum(z_encoded * attn_weights, dim=1)  # (batch, 162)
        trans_feat = self.transformer_proj(pooled)       # (batch, 128)
        # Concatenate with current state features x (for farmer, dimension 484)
        combined = torch.cat([trans_feat, x], dim=-1)      # (batch, 128 + 484 = 612)
        x_out = self.dense1(combined)
        x_out = F.relu(x_out)
        x_out = self.dense2(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense3(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense4(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense5(x_out)
        x_out = F.relu(x_out)
        x_out = self.dense6(x_out)
        if return_value:
            return dict(values=x_out)
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(x_out.shape[0], (1,))[0]
            else:
                action = torch.argmax(x_out, dim=0)[0]
            return dict(action=action)


class Model:
    """
    A wrapper for the three models (landlord, landlord_up, landlord_down).
    The model_type parameter can be:
      - 'lstm': original LSTM-based model,
      - 'transformer': full transformer-based multi-modal fusion model,
      - 'lstm_transformer_encoder': baseline structure but with transformer encoder (with positional encoding) replacing LSTM.
    """
    def __init__(self, device=0, model_type="lstm"):
        self.models = {}
        if device != "cpu":
            device = 'cuda:' + str(device)
        if model_type == "transformer":
            self.models['landlord'] = LandlordTransformerModel().to(torch.device(device))
            self.models['landlord_up'] = FarmerTransformerModel().to(torch.device(device))
            self.models['landlord_down'] = FarmerTransformerModel().to(torch.device(device))
        else:
            # default: original LSTM-based models
            self.models['landlord'] = LandlordLstmModel().to(torch.device(device))
            self.models['landlord_up'] = FarmerLstmModel().to(torch.device(device))
            self.models['landlord_down'] = FarmerLstmModel().to(torch.device(device))

    def forward(self, position, z, x, training=False, flags=None):
        model = self.models[position]
        return model.forward(z, x, training, flags)

    def share_memory(self):
        self.models['landlord'].share_memory()
        self.models['landlord_up'].share_memory()
        self.models['landlord_down'].share_memory()

    def eval(self):
        self.models['landlord'].eval()
        self.models['landlord_up'].eval()
        self.models['landlord_down'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models
