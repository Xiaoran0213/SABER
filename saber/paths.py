"""Portable data and cache roots for SABER.

Everything is configured through environment variables so the repository is
machine-independent; no author-cluster default is baked into the code. A
`.saber_env` file at the repository root (KEY=VALUE lines, see `env.example`)
is loaded first and can be used instead of exporting variables in every shell.

| Variable              | What it points to                                          | Default        |
|-----------------------|------------------------------------------------------------|----------------|
| ``SABER_DATA_ROOT``   | proteomics data root (``anndata/``, ``easyscp_liver/``, …) | required       |
| ``SABER_PLM_ROOT``    | precomputed protein-language-model caches                  | required       |
| ``SABER_BENCHMARK_ROOT`` | benchmark slice copy                                     | ``DATA_ROOT/saber_benchmark`` |
| ``SABER_EXT_METHODS_ROOT`` | external-baseline adapters                              | repo ``external_adapters/`` |
| ``SABER_EXTERNAL_REPOS`` | upstream clones of the external baselines               | ``PROTAE_ROOT/external`` |
| ``SABER_PROTAE_ROOT`` | annotation source for materializing the benchmark slice (optional, legacy) | unset |
| ``SABER_HF_CACHE``    | HuggingFace weight cache for rebuilding PLM features       | HF_HOME        |
| ``SABER_OUTPUTS_ROOT``| training-run outputs                                       | ``./outputs``  |
| ``SABER_FIGURES_ROOT``| rendered figures and per-panel source tables               | ``./figures``  |

See ``docs/DATA.md`` for the layout each root is expected to contain.
"""

from __future__ import annotations

import os
from pathlib import Path


def _load_saber_env() -> None:
    """Load KEY=VALUE lines from a repo-root ``.saber_env`` (never overrides)."""
    # saber/paths.py -> repo root. Under a regular (non-editable) install this
    # directory simply does not contain the file and the loader is a no-op.
    env_file = Path(__file__).resolve().parents[1] / ".saber_env"
    if not env_file.is_file():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass  # unreadable file: fall through to plain environment resolution


_load_saber_env()


def _resolve(env_name: str, default: str | Path | None, what: str) -> Path:
    override = os.environ.get(env_name, "").strip()
    if override:
        return Path(override).expanduser()
    if default is not None:
        return Path(default).expanduser()
    raise RuntimeError(
        f"{what} is not configured. Export {env_name} (or add it to .saber_env, "
        f"see env.example and docs/DATA.md)."
    )


def data_root() -> Path:
    """Proteomics data root: ``anndata/`` matrices and case-study folders."""
    return _resolve("SABER_DATA_ROOT", None, "Proteomics data root")


def plm_root() -> Path:
    """Precomputed PLM caches (one subdirectory per model, e.g. ``esm2_650m/``)."""
    return _resolve("SABER_PLM_ROOT", None, "Precomputed PLM cache root")


def benchmark_root() -> Path:
    """Benchmark slice root (materialized copies + annotations)."""
    override = os.environ.get("SABER_BENCHMARK_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    return data_root() / "saber_benchmark"


def protae_root() -> Path | None:
    """Annotation source for materializing the benchmark slice (optional, legacy).

    Only needed by ``saber prepare-benchmark-data`` when rebuilding
    ``saber_benchmark/`` from the original evaluation tree; the archived
    data package ships the materialized slice.
    """
    override = os.environ.get("SABER_PROTAE_ROOT", "").strip()
    return Path(override).expanduser() if override else None


def ext_methods_root() -> Path | None:
    """External-baseline adapter directory (ships with the repository).

    Resolution order: ``SABER_EXT_METHODS_ROOT``, then the in-repository
    ``external_adapters/`` (present in a git checkout or editable install),
    then the legacy ``$SABER_PROTAE_ROOT/ext_methods`` layout.
    """
    override = os.environ.get("SABER_EXT_METHODS_ROOT", "").strip()
    if override:
        return Path(override).expanduser()
    in_repo = Path(__file__).resolve().parents[1] / "external_adapters"
    if in_repo.is_dir():
        return in_repo
    root = protae_root()
    if root is not None and (root / "ext_methods").is_dir():
        return root / "ext_methods"
    return None


def external_repos_root() -> Path | None:
    """Upstream clones of the external baselines (passed to adapters as --repo).

    Resolution order: ``SABER_EXTERNAL_REPOS``, then the legacy
    ``$SABER_PROTAE_ROOT/external`` layout. See external_adapters/README.md.
    """
    override = os.environ.get("SABER_EXTERNAL_REPOS", "").strip()
    if override:
        return Path(override).expanduser()
    root = protae_root()
    if root is not None and (root / "external").is_dir():
        return root / "external"
    return None


def hf_cache() -> Path:
    override = os.environ.get("SABER_HF_CACHE", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def outputs_root() -> Path:
    """Training-run outputs. Defaults to ``./outputs`` next to the invocation."""
    override = os.environ.get("SABER_OUTPUTS_ROOT", "").strip()
    return Path(override).expanduser() if override else Path("outputs")


def figures_root() -> Path:
    """Rendered figures + per-panel source tables. Defaults to ``./figures``."""
    override = os.environ.get("SABER_FIGURES_ROOT", "").strip()
    return Path(override).expanduser() if override else Path("figures")


def esm2_cache_dir() -> Path:
    return plm_root() / "esm2_650m"


def campaign_run_dir(run: str | os.PathLike) -> Path:
    """Resolve a campaign run name or explicit path to its directory.

    A name (e.g. ``paper_benchmark_20260801_final_10seed``) resolves under
    ``outputs_root()``; anything that already looks like a path is used as-is,
    so ``--run results`` can point at the shipped source tables.
    """
    candidate = Path(run)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    named = outputs_root() / str(run)
    if named.exists():
        return named
    return named  # keep the name; loaders raise a descriptive error later


def resolve_outdir(outdir: str | os.PathLike) -> Path:
    """Resolve a config ``project.outdir``.

    Relative paths whose first component is ``outputs`` are rebased onto
    ``outputs_root()`` so ``SABER_OUTPUTS_ROOT`` redirects every run at once;
    anything else is used as given (relative to the invocation directory).
    """
    p = Path(outdir)
    if p.is_absolute() or not p.parts or p.parts[0] != "outputs":
        return p
    return outputs_root().joinpath(*p.parts[1:])
