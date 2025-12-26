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