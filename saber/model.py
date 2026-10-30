from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class SABER(nn.Module):
    """Prior-informed bilinear model for single-cell proteomics.

    This core model exposes a cell encoder, a protein prior encoder, and a
    bilinear decoder.
    """

    def __init__(
        self,
        n_proteins: int,
        protein_features: torch.Tensor,
        latent_dim: int = 16,
        ae_h1: int = 128,
        ae_h2: int = 64,
        protein_proj_dim: int = 256,
        protein_proj_hidden_dim: int = 256,
        protein_proj_dropout: float = 0.1,
        bilinear_with_w: bool = True,
        trainable_protein_features: bool = False,
    ) -> None:
        super().__init__()
        if protein_features.ndim != 2:
            raise ValueError("protein_features must be a 2D tensor")
        if int(n_proteins) != int(protein_features.shape[0]):
            raise ValueError(
                f"n_proteins={n_proteins} but protein_features has "
                f"{protein_features.shape[0]} rows"
            )

        self.n_proteins = int(n_proteins)
        self.latent_dim = int(latent_dim)
        self.bilinear_with_w = bool(bilinear_with_w)
        feature_tensor = protein_features.float()
        if bool(trainable_protein_features):
            self.protein_features = nn.Parameter(feature_tensor)
        else:
            self.register_buffer("protein_features", feature_tensor)

        self.cell_encoder = nn.Sequential(
            nn.Linear(self.n_proteins, int(ae_h1)),
            nn.ReLU(),
            nn.Linear(int(ae_h1), int(ae_h2)),
            nn.ReLU(),
            nn.Linear(int(ae_h2), self.latent_dim),
        )

        feature_dim = int(protein_features.shape[1])
        self.protein_projection = nn.Sequential(
            nn.Linear(feature_dim, int(protein_proj_hidden_dim)),
            nn.LayerNorm(int(protein_proj_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(protein_proj_dropout)),
            nn.Linear(int(protein_proj_hidden_dim), int(protein_proj_dim)),
        )
        self.protein_output = nn.Linear(int(protein_proj_dim), self.latent_dim)
        self.protein_encoder = nn.Sequential(self.protein_projection, self.protein_output)

        if self.bilinear_with_w:
            self.W = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
            nn.init.xavier_uniform_(self.W)
        else:
            self.register_parameter("W", None)

    def encode_cells(self, x: torch.Tensor) -> torch.Tensor:
        return self.cell_encoder(x)

    def encode_proteins(self, protein_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        features = self.protein_features if protein_features is None else protein_features.float()
        return self.protein_encoder(features)

    def decode(self, z: torch.Tensor, protein_embeddings: torch.Tensor) -> torch.Tensor:
        z_use = z @ self.W if self.W is not None else z
        return z_use @ protein_embeddings.t()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode_cells(x)
        e = self.encode_proteins()
        x_hat = self.decode(z, e)
        return x_hat, z, e

    @torch.no_grad()
    def predict_new_proteins(self, x: torch.Tensor, protein_features: torch.Tensor) -> torch.Tensor:
        z = self.encode_cells(x)
        e = self.encode_proteins(protein_features)
        z_use = z @ self.W if self.W is not None else z
        return z_use @ e.t()
