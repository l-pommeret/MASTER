import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
import math

from base_model import CryptoSequenceModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=120):  # 120 minutes au lieu de 100
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class SAttention(nn.Module):
    """Attention spatiale adaptée pour les cryptos"""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        # Projections Q/K/V
        self.qtrans = nn.Linear(d_model, d_model, bias=True)  # Ajout du bias
        self.ktrans = nn.Linear(d_model, d_model, bias=True)
        self.vtrans = nn.Linear(d_model, d_model, bias=True)

        # Dropout par tête avec régularisation
        self.attn_dropout = nn.ModuleList([
            Dropout(p=dropout) for _ in range(nhead)
        ])

        # Normalisation et FFN
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model * 2),
            nn.GELU(),  # GELU au lieu de ReLU
            Dropout(p=dropout),
            Linear(d_model * 2, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x).transpose(0,1)
        k = self.ktrans(x).transpose(0,1)
        v = self.vtrans(x).transpose(0,1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        
        for i in range(self.nhead):
            if i == self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # Attention avec scaling dynamique
            scores = torch.matmul(qh, kh.transpose(1, 2))
            scaling = self.temperature * torch.sqrt(torch.tensor(1.0 + torch.std(scores)))
            scores = scores / scaling
            
            atten_weights = torch.softmax(scores, dim=-1)
            atten_weights = self.attn_dropout[i](atten_weights)
            
            att_output.append(torch.matmul(atten_weights, vh).transpose(0, 1))
            
        att_output = torch.concat(att_output, dim=-1)

        # FFN avec connexion résiduelle
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output

class TAttention(nn.Module):
    """Attention temporelle adaptée pour la haute fréquence"""
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Projections avec bias
        self.qtrans = nn.Linear(d_model, d_model, bias=True)
        self.ktrans = nn.Linear(d_model, d_model, bias=True)
        self.vtrans = nn.Linear(d_model, d_model, bias=True)

        # Dropout adaptatif
        self.attn_dropout = nn.ModuleList([
            Dropout(p=dropout) for _ in range(nhead)
        ])

        # Normalisation et FFN
        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model * 2),
            nn.GELU(),
            Dropout(p=dropout),
            Linear(d_model * 2, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        
        for i in range(self.nhead):
            if i == self.nhead-1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            # Attention avec masque causal
            scores = torch.matmul(qh, kh.transpose(1, 2))
            
            # Masque pour donner plus de poids aux données récentes
            seq_len = scores.size(-1)
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(x.device)
            scores = scores.masked_fill(mask == 1, float('-inf'))
            
            atten_weights = torch.softmax(scores, dim=-1)
            if self.attn_dropout:
                atten_weights = self.attn_dropout[i](atten_weights)
                
            att_output.append(torch.matmul(atten_weights, vh))
            
        att_output = torch.concat(att_output, dim=-1)

        # FFN avec connexion résiduelle
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output

class Gate(nn.Module):
    """Gate adapté pour les cryptos"""
    def __init__(self, d_input, d_output, beta=2.0):  # beta réduit pour plus de réactivité
        super().__init__()
        self.trans = nn.Sequential(
            nn.Linear(d_input, d_input * 2),
            nn.GELU(),
            nn.Linear(d_input * 2, d_output)
        )
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        # Softmax avec température adaptative
        volatility = torch.std(gate_input, dim=-1, keepdim=True)
        temp = self.t * (1 + torch.sigmoid(volatility))
        return self.d_output * torch.softmax(output/temp, dim=-1)

class TemporalAttention(nn.Module):
    """Attention temporelle finale avec biais de récence"""
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=True)

    def forward(self, z):
        h = self.trans(z)
        query = h[:, -1, :].unsqueeze(-1)
        
        # Scores d'attention avec biais de récence
        base_scores = torch.matmul(h, query).squeeze(-1)
        positions = torch.arange(base_scores.size(1), device=z.device).float()
        recency_bias = torch.exp(-0.1 * (positions[-1] - positions))
        
        scores = base_scores * recency_bias.unsqueeze(0)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        
        return torch.matmul(weights, z).squeeze(1)

class MASTER(nn.Module):
    """MASTER adapté pour le trading crypto"""
    def __init__(self, 
                d_feat=40,  # Réduit pour crypto
                d_model=256,
                t_nhead=4,
                s_nhead=2,
                T_dropout_rate=0.3,  # Réduit
                S_dropout_rate=0.3,
                gate_input_start_index=40,
                gate_input_end_index=60,  # 20 features de marché
                beta=2.0):  # Plus réactif
        super(MASTER, self).__init__()
        
        # Configuration du gate
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        # Pipeline principal
        self.layers = nn.Sequential(
            nn.Linear(d_feat, d_model),
            PositionalEncoding(d_model),
            TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate),
            SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate),
            TemporalAttention(d_model=d_model),
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(p=0.1),
                nn.Linear(d_model // 2, 1)
            )
        )

    def forward(self, x):
        # Séparation des features et du marché
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        
        # Application du gate
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)
        
        # Forward pass
        return self.layers(src).squeeze(-1)

class MASTERModel(CryptoSequenceModel):
    def __init__(
            self, 
            d_feat: int = 40,
            d_model: int = 256,
            t_nhead: int = 4,
            s_nhead: int = 2,
            gate_input_start_index=40,
            gate_input_end_index=60,
            T_dropout_rate=0.3,
            S_dropout_rate=0.3,
            beta=2.0,
            **kwargs):
        super(MASTERModel, self).__init__(**kwargs)
        
        self.d_model = d_model
        self.d_feat = d_feat
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
        super(MASTERModel, self).init_model()