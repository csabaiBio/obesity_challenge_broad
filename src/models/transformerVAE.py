import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np  
import torchmetrics
from torchmetrics.wrappers import ClasswiseWrapper
from torchmetrics.classification import MulticlassConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
from lightning.pytorch.loggers import MLFlowLogger, WandbLogger
import os

class TransformerVAEEncoder(nn.Module):
    def __init__(self, input_dim=675, d_model=512, n_layers=6, n_heads=8, z_dim=32):
        super().__init__()

        self.token_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.mu = nn.Linear(d_model, z_dim)
        self.logvar = nn.Linear(d_model, z_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
    def forward(self, x):
        # x: (B, 32, 675)
        h = self.token_proj(x)
        h = self.encoder(h)
        # Optionally add layernorm here
        h_pooled = h.mean(dim=1)
        return self.mu(h_pooled), self.logvar(h_pooled)
    




class TransformerVAEDecoder(nn.Module):
    def __init__(self,output_dim=675,num_tokens=32,d_model=512,n_layers=6,n_heads=8,z_dim=32):
        super().__init__()

        self.num_tokens = num_tokens

        # Project latent to transformer dimension
        self.latent_proj = nn.Linear(z_dim, d_model)

        # Learned token queries (permutation invariant)
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, n_layers)

        # Project back to original feature space
        self.film_gamma = nn.Linear(z_dim, d_model)
        self.film_beta  = nn.Linear(z_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.outact = nn.ReLU()

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)

        latent = self.latent_proj(z)              # (B, d_model)
        latent = latent.unsqueeze(1)              # (B, 1, d_model)
        queries = self.token_queries.expand(B, -1, -1)  # (B, 32, d_model)
        # Inject latent into every token
        h = queries + latent
        gamma = self.film_gamma(z).unsqueeze(1)
        beta  = self.film_beta(z).unsqueeze(1)
        
        h = self.decoder(h)
        h = gamma * h + beta                      # FiLM modulation
        x_hat = self.output_proj(h)                # (B, 32, 675)
        #x_hat = self.outact(x_hat)
        return x_hat
    

class TransformerVAEDecoder_conditioned(nn.Module):
    def __init__(self,output_dim=675,num_tokens=32,d_model=512,n_layers=6,n_heads=8,z_dim=32,cond_dim=128):
        super().__init__()

        self.num_tokens = num_tokens

        # Project latent to transformer dimension
        self.latent_proj = nn.Linear(z_dim, d_model)

        # Learned token queries (permutation invariant)
        self.token_queries = nn.Parameter(torch.randn(1, num_tokens, d_model))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, n_layers)

        # Project back to original feature space
        self.film_gamma = nn.Linear(z_dim, d_model)
        self.film_beta  = nn.Linear(z_dim, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.outact = nn.ReLU()
        self.cond_proj_gamma = nn.Linear(cond_dim, d_model)
        self.cond_proj_beta  = nn.Linear(cond_dim, d_model)
    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)

        latent = self.latent_proj(z)              # (B, d_model)
        latent = latent.unsqueeze(1)              # (B, 1, d_model)
        queries = self.token_queries.expand(B, -1, -1)  # (B, 32, d_model)
        # Inject latent into every token
        h = queries + latent
        gamma = self.film_gamma(z).unsqueeze(1)
        beta  = self.film_beta(z).unsqueeze(1)
        
        h = self.decoder(h)
        h = gamma * h + beta                      # FiLM modulation
        x_hat = self.output_proj(h)                # (B, 32, 675)
        #x_hat = self.outact(x_hat)
        return x_hat
    



class TransformerVAEEncoder_conditioned(nn.Module):
    def __init__(self, input_dim=675, d_model=512, n_layers=6, n_heads=8, z_dim=32):
        super().__init__()

        self.token_proj = nn.Linear(input_dim, d_model)
        self.condition_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.mu = nn.Linear(d_model, z_dim)
        self.logvar = nn.Linear(d_model, z_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)
    def forward(self, x, condition):
        # x: (B, 32, 675)
        h = self.token_proj(x)
        cond = self.condition_proj(condition)
        h = h + cond
        h = self.encoder(h)
        # Optionally add layernorm here
        h_pooled = h.mean(dim=1)
        return self.mu(h_pooled), self.logvar(h_pooled)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=675, d_model=512, n_layers=4, n_heads=8, num_classes=5):
        super().__init__()

        self.token_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, 32, 675)
        h = self.token_proj(x)
        h = self.encoder(h)
        # Optionally add layernorm here
        h_pooled = h.mean(dim=1)
        class_logits = self.classifier(h_pooled)
        return class_logits

class Transfomer_latent_Classifier(nn.Module):
    def __init__(self, z_dim=32, d_model=128, n_layers=2, n_heads=8, num_classes=4):
        super().__init__()

        self.latent_proj = nn.Linear(z_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, z):
        # z: (B, z_dim)
        B = z.size(0)
        h = self.latent_proj(z)
        h = h.unsqueeze(1)  # (B, 1, d_model)
        h = self.encoder(h)
        h_pooled = h.mean(dim=1)
        class_logits = self.classifier(h_pooled)
        return class_logits
    

class ConditionEncoder(nn.Module):
    def __init__(self, input_dim=675, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.out   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, c):
        # c: [B, T, Input_dim]
        h = self.embed(c)             # [B, T, H]
        h = h.transpose(1, 2)         # [B, H, T]
        h = self.pool(h).squeeze(-1)  # [B, H]
        return self.out(h)  


class AttentionConditionEncoder(nn.Module):
    def __init__(self, input_dim=675, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden_dim)
        
        # Attention mechanism
        # "What is important in this condition?"
        self.attention_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1) # Scalar score per token
        )
        
        self.out = nn.Linear(hidden_dim, latent_dim)

    def forward(self, c):
        # c: [B, T, Input_dim]
        h = self.embed(c)           # [B, T, H]
        
        # Calculate Attention Weights
        attn_scores = self.attention_net(h)      # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Weighted Sum (Attention Pooling)
        # Instead of mean(), we take the weighted average
        weighted_h = torch.sum(h * attn_weights, dim=1) # [B, H]
        
        return self.out(weighted_h)

class SimpleLatentTransition(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim=256):
        super().__init__()
        # Input is z + condition
        self.input_proj = nn.Linear(latent_dim + cond_dim, hidden_dim) 
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Zero-init ensures the *change* is zero at start
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, cond):
        # Concatenate z and condition
        combined = torch.cat([z, cond], dim=-1)
        
        # Calculate the "flow" (delta)
        delta = self.mlp(self.input_proj(combined)) # Assuming intermediate processing
        
        # Apply Residual: z_new = z + delta
        return delta

class FiLMLatentTransition(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim=512):
        super().__init__()
        
        # 1. Process z (The State)
        self.z_mlp = nn.Linear(latent_dim, hidden_dim)
        
        # 2. Process Condition into FiLM params (Gamma, Beta)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2) # Output both Gamma and Beta
        )
        
        # 3. Output Processing
        self.out_mlp = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Zero-init the last layer for Identity start
        nn.init.zeros_(self.out_mlp[-1].weight)
        nn.init.zeros_(self.out_mlp[-1].bias)

    def forward(self, z, cond):
        # Embed State
        h_z = self.z_mlp(z) # [B, Hidden]
        
        # Get FiLM params
        film_params = self.cond_mlp(cond) # [B, 2 * Hidden]
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        
        # Apply Modulation: Scale * State + Shift
        # This is the "Physical Interaction"
        h_modulated = gamma * h_z + beta
        
        # Calculate Delta
        delta = self.out_mlp(h_modulated)
        
        return delta


class ConditionalLatentTransition(nn.Module):
    """
    z_cond = T(z, cond)

    z:     [B, N, D] latent tokens
    cond:  [B, M, C] condition tokens (or [B, C])
    """

    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # --- Self-attention on latent ---
        self.self_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        # --- Cross-attention: latent queries, condition keys/values ---
        self.cross_attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.cond_proj = nn.Linear(cond_dim, latent_dim)

        # --- Feed-forward ---
        hidden_dim = int(latent_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # --- Normalization ---
        #self.norm1 = nn.LayerNorm(latent_dim)
        #self.norm2 = nn.LayerNorm(latent_dim)
        #self.norm3 = nn.LayerNorm(latent_dim)

        # --- Zero-init last layer for stable identity start ---
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.self_attn.out_proj.weight.data.zero_()
        self.self_attn.out_proj.bias.data.zero_()
        
        self.cross_attn.out_proj.weight.data.zero_()
        self.cross_attn.out_proj.bias.data.zero_()

    def forward(self, z, cond):
        """
        z:    [B, N, D]
        cond: [B, M, C] or [B, C]
        """

        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # [B, 1, C]

        cond = self.cond_proj(cond)  # [B, M, D]
        z = z.unsqueeze(1) 
        # --- Self-attention (latent coherence) ---
        z_res = z
        #z = self.norm1(z)
        z_attn, _ = self.self_attn(z, z, z)
        z = z_res + z_attn

        # --- Cross-attention (conditioning) ---
        z_res = z
        #z = self.norm2(z)
        z_attn, _ = self.cross_attn(
            query=z,
            key=cond,
            value=cond,
        )
        z = z_res + z_attn

        # --- Feed-forward ---
        z_res = z
        #z = self.norm3(z)
        z = z_res + self.mlp(z)
        return z.squeeze(1)  # [B, D]