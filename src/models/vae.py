"""
Variational Autoencoder (VAE) for battery data anomaly detection.
Filters noisy/defective battery cycling data before ML training.

References:
- Chan et al. (2024), J. Energy Storage — DVAA-SVDD for battery anomaly detection
- Lee et al. (2025), MDPI Batteries — DNN with anomaly detection for battery lifetime
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import VAE_CONFIG, DEVICE


class VAE(nn.Module):
    """
    Variational Autoencoder for anomaly detection.

    Architecture:
        Input → Encoder (FC layers) → mu, log_var → Reparameterize → Decoder → Reconstruction
        Anomaly = high reconstruction error
    """

    def __init__(self, input_dim: int, latent_dim: int = None, hidden_dims: list = None):
        super().__init__()

        self.latent_dim = latent_dim or VAE_CONFIG["latent_dim"]
        hidden_dims = hidden_dims or VAE_CONFIG["hidden_dims"]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], self.latent_dim)

        # Decoder
        decoder_layers = []
        reversed_dims = list(reversed(hidden_dims))
        prev_dim = self.latent_dim
        for h_dim in reversed_dims:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(reversed_dims[-1], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

    @staticmethod
    def loss_function(reconstruction, x, mu, log_var, beta=1.0):
        """
        VAE loss = Reconstruction loss + beta * KL divergence

        Args:
            reconstruction: Decoder output
            x: Original input
            mu: Latent mean
            log_var: Latent log-variance
            beta: KL weight (beta-VAE)
        """
        recon_loss = F.mse_loss(reconstruction, x, reduction="mean")
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_loss + beta * kl_loss, recon_loss, kl_loss


def compute_reconstruction_error(model: VAE, data: torch.Tensor) -> np.ndarray:
    """Compute per-sample reconstruction error for anomaly detection."""
    model.eval()
    with torch.no_grad():
        data = data.to(DEVICE)
        reconstruction, _, _ = model(data)
        error = F.mse_loss(reconstruction, data, reduction="none")
        error = error.mean(dim=1).cpu().numpy()
    return error


def detect_anomalies(model: VAE, data: torch.Tensor, percentile: float = None) -> np.ndarray:
    """
    Flag anomalous samples based on reconstruction error threshold.

    Returns:
        Boolean array: True = anomalous, False = normal
    """
    percentile = percentile or VAE_CONFIG["anomaly_percentile"]
    errors = compute_reconstruction_error(model, data)
    threshold = np.percentile(errors, percentile)
    return errors > threshold, errors, threshold
