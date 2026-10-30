"""Shared fixtures: synthetic SABER_* roots so no test touches real data."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def saber_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point every SABER_* root at a temporary tree with minimal contents."""
    data_root = tmp_path / "proteomics"
    plm_root = tmp_path / "precomputed_plm"
    external_repos = tmp_path / "external_repos"
    outputs_root = tmp_path / "outputs"
    for d in (data_root / "anndata", data_root / "saber_benchmark",
              plm_root, external_repos, outputs_root):
        d.mkdir(parents=True, exist_ok=True)
    # Upstream-clone stubs for the adapters used by the specs in
    # saber/external_methods.py (the adapters themselves resolve to the
    # repository's external_adapters/; dry runs only need the repo root).
    for m in ("pimms", "omicsgmf", "scPROTEIN"):
        (external_repos / m).mkdir(parents=True, exist_ok=True)
    # Legacy annotation source for saber prepare-benchmark-data (the registry's
    # external_annotations block references it; sources may be absent in tests).
    protae_root = tmp_path / "protae_legacy"
    protae_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SABER_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SABER_PLM_ROOT", str(plm_root))
    monkeypatch.setenv("SABER_EXTERNAL_REPOS", str(external_repos))
    monkeypatch.setenv("SABER_PROTAE_ROOT", str(protae_root))
    monkeypatch.setenv("SABER_OUTPUTS_ROOT", str(outputs_root))
    return {
        "data_root": data_root,
        "plm_root": plm_root,
        "external_repos": external_repos,
        "protae_root": protae_root,
        "outputs_root": outputs_root,
    }
