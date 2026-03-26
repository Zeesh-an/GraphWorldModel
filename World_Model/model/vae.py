# -*- coding: utf-8 -*-
"""
VAE — Encoder / Decoder / VAEModel
====================================
Identical role to the baseline's Encoder/Decoder/VAEModel in DeepIM.
Encodes/decodes N-dimensional binary seed vectors through a latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    3-layer MLP encoder: seed_vec (N,) → latent (latent_dim,)
    Mirrors baseline Encoder exactly.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """
    4-layer MLP decoder: latent (latent_dim,) → seed_vec (N,) ∈ [0,1]
    Mirrors baseline Decoder exactly.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(z))


class VAEModel(nn.Module):
    """
    VAE wrapper: encode → reparameterise → decode.
    For inference phase (latent optimisation), the encoder is bypassed —
    z is optimised directly as a free variable.
    """
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z    = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
