from __future__ import annotations

import torch

from saber.model import SABER


def test_saber_forward_shapes() -> None:
    protein_features = torch.randn(7, 11)
    model = SABER(
        n_proteins=7,
        protein_features=protein_features,
        latent_dim=5,
        ae_h1=13,
        ae_h2=9,
        protein_proj_dim=17,
        protein_proj_hidden_dim=19,
        protein_proj_dropout=0.0,
    )

    pred, z, e = model(torch.randn(4, 7))

    assert pred.shape == (4, 7)
    assert z.shape == (4, 5)
    assert e.shape == (7, 5)
    assert torch.isfinite(pred).all()


def test_predict_new_proteins_shapes() -> None:
    model = SABER(
        n_proteins=5,
        protein_features=torch.randn(5, 3),
        latent_dim=2,
        ae_h1=7,
        ae_h2=5,
        protein_proj_dim=6,
        protein_proj_hidden_dim=8,
        protein_proj_dropout=0.0,
    )

    pred = model.predict_new_proteins(torch.randn(6, 5), torch.randn(4, 3))

    assert pred.shape == (6, 4)


def test_frozen_prior_is_buffer_not_parameter() -> None:
    features = torch.randn(5, 3)
    model = SABER(
        n_proteins=5,
        protein_features=features,
        latent_dim=2,
        ae_h1=7,
        ae_h2=5,
        protein_proj_dim=6,
        protein_proj_hidden_dim=8,
        protein_proj_dropout=0.0,
        trainable_protein_features=False,
    )
    assert "protein_features" not in dict(model.named_parameters())
    assert "protein_features" in dict(model.named_buffers())


def test_no_protein_bias_parameter() -> None:
    model = SABER(
        n_proteins=5,
        protein_features=torch.randn(5, 3),
        latent_dim=2,
        ae_h1=7,
        ae_h2=5,
        protein_proj_dim=6,
        protein_proj_hidden_dim=8,
        protein_proj_dropout=0.0,
    )
    assert "protein_bias" not in dict(model.named_parameters())
