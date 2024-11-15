import torch
from torch import nn
import math
from base_model import CryptoSequenceModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, L, C = x.shape
        
        # Multi-head attention
        x = self.norm1(x)
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.out_proj(x)
        
        # FFN
        out = self.norm2(x)
        out = out + self.ffn(out)
        
        return out

class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = math.sqrt(self.head_dim)

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, L, C = x.shape
        
        # Multi-head attention
        x = self.norm1(x)
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Causal mask
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        mask = mask.to(x.device)

        # Attention scores avec masque causal
        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Attention output
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.out_proj(x)
        
        # FFN
        out = self.norm2(x)
        out = out + self.ffn(out)
        
        return out

class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=2.0):
        super().__init__()
        self.beta = beta
        self.d_output = d_output
        
        self.net = nn.Sequential(
            nn.Linear(d_input, d_input * 2),
            nn.GELU(),
            nn.Linear(d_input * 2, d_output)
        )
        
    def forward(self, x):
        # x shape: (batch_size, d_input)
        gate = self.net(x)
        
        # Température adaptative
        volatility = torch.std(x, dim=-1, keepdim=True)
        temp = self.beta * (1 + torch.sigmoid(volatility))
        
        # Normalisation
        gate = self.d_output * torch.softmax(gate / temp, dim=-1)
        return gate

class MASTER(nn.Module):
    def __init__(
        self,
        d_feat=40,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        T_dropout_rate=0.3,
        S_dropout_rate=0.3,
        gate_input_start_index=40,
        gate_input_end_index=60,
        beta=2.0
    ):
        super().__init__()
        
        self.d_feat = d_feat
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        
        # Gate mechanism
        d_gate = gate_input_end_index - gate_input_start_index
        self.feature_gate = Gate(d_gate, d_feat, beta=beta)
        
        # Main transformer pipeline
        self.feature_proj = nn.Linear(d_feat, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.temporal_attn = TAttention(d_model, t_nhead, T_dropout_rate)
        self.spatial_attn = SAttention(d_model, s_nhead, S_dropout_rate)
        
        # Final prediction layers
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length, total_features)
        B, L, _ = x.shape
        
        # Split features and market data
        features = x[:, :, :self.gate_input_start_index]  # (B, L, d_feat)
        market = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]  # (B, d_gate)
        
        # Apply feature gating
        gate_weights = self.feature_gate(market)  # (B, d_feat)
        gate_weights = gate_weights.unsqueeze(1).expand(-1, L, -1)  # (B, L, d_feat)
        features = features * gate_weights
        
        # Transform sequence
        x = self.feature_proj(features)  # (B, L, d_model)
        x = self.pos_enc(x)
        x = self.temporal_attn(x)
        x = self.spatial_attn(x)
        
        # Use last timestep for prediction
        x = x[:, -1]  # (B, d_model)
        
        # Final prediction
        out = self.head(x).squeeze(-1)  # (B,)
        return out

class MASTERModel(CryptoSequenceModel):
    def __init__(
        self,
        d_feat=40,
        d_model=256,
        t_nhead=4,
        s_nhead=2,
        gate_input_start_index=40,
        gate_input_end_index=60,
        T_dropout_rate=0.3,
        S_dropout_rate=0.3,
        beta=2.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.d_feat = d_feat
        self.d_model = d_model
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta
        
        self.init_model()

    def init_model(self):
        self.model = MASTER(
            d_feat=self.d_feat,
            d_model=self.d_model,
            t_nhead=self.t_nhead,
            s_nhead=self.s_nhead,
            T_dropout_rate=self.T_dropout_rate,
            S_dropout_rate=self.S_dropout_rate,
            gate_input_start_index=self.gate_input_start_index,
            gate_input_end_index=self.gate_input_end_index,
            beta=self.beta
        )
        super().init_model()