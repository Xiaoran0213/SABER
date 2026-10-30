"""Tests for saber.paths: environment-variable roots."""

from __future__ import annotations

import pytest

import saber.paths as paths


def test_required_roots_raise_without_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("SABER_DATA_ROOT", "SABER_PLM_ROOT"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(RuntimeError, match="SABER_DATA_ROOT"):
        paths.data_root()
    with pytest.raises(RuntimeError, match="SABER_PLM_ROOT"):
        paths.plm_root()


def test_roots_follow_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SABER_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("SABER_PLM_ROOT", str(tmp_path / "plm"))
    assert paths.data_root() == tmp_path / "data"
    assert paths.plm_root() == tmp_path / "plm"
    assert paths.esm2_cache_dir() == tmp_path / "plm" / "esm2_650m"
    assert paths.benchmark_root() == tmp_path / "data" / "saber_benchmark"
    monkeypatch.delenv("SABER_BENCHMARK_ROOT", raising=False)


def test_optional_roots_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SABER_PROTAE_ROOT", raising=False)
    monkeypatch.delenv("SABER_OUTPUTS_ROOT", raising=False)
    monkeypatch.delenv("SABER_FIGURES_ROOT", raising=False)
    assert paths.protae_root() is None
    assert paths.outputs_root() == paths.Path("outputs")
    assert paths.figures_root() == paths.Path("figures")


def test_ext_methods_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    root = paths.Path(paths.__file__).resolve().parents[1] / "external_adapters"
    monkeypatch.delenv("SABER_EXT_METHODS_ROOT", raising=False)
    monkeypatch.delenv("SABER_PROTAE_ROOT", raising=False)
    assert paths.ext_methods_root() == root  # ships with the repository
    monkeypatch.setenv("SABER_EXT_METHODS_ROOT", str(tmp_path / "custom"))
    assert paths.ext_methods_root() == tmp_path / "custom"  # explicit override wins


def test_external_repos_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("SABER_EXTERNAL_REPOS", raising=False)
    monkeypatch.delenv("SABER_PROTAE_ROOT", raising=False)
    assert paths.external_repos_root() is None  # nothing configured
    monkeypatch.setenv("SABER_EXTERNAL_REPOS", str(tmp_path / "repos"))
    assert paths.external_repos_root() == tmp_path / "repos"
    monkeypatch.delenv("SABER_EXTERNAL_REPOS", raising=False)
    protae = tmp_path / "protae"
    (protae / "external").mkdir(parents=True)
    monkeypatch.setenv("SABER_PROTAE_ROOT", str(protae))
    assert paths.external_repos_root() == protae / "external"  # legacy layout


def test_resolve_outdir_rebases_outputs_prefix(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SABER_OUTPUTS_ROOT", str(tmp_path / "runs"))
    assert paths.resolve_outdir("outputs/easyscp/x") == tmp_path / "runs" / "easyscp" / "x"
    assert paths.resolve_outdir("elsewhere/x") == paths.Path("elsewhere/x")
    assert paths.resolve_outdir("/abs/x") == paths.Path("/abs/x")


def test_campaign_run_dir_prefers_explicit_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("SABER_OUTPUTS_ROOT", str(tmp_path))
    (tmp_path / "paper_run").mkdir()
    assert paths.campaign_run_dir("paper_run") == tmp_path / "paper_run"
    assert paths.campaign_run_dir(str(tmp_path / "paper_run")) == tmp_path / "paper_run"


# Cluster-style absolute paths and numeric user IDs must not appear anywhere
# in the shipped tree.
_FORBIDDEN = (r"/lustre/", r"/storage\d*/", r"/home/\w+/", r"\bs\d{7,}\b")


def _path_offenders(files) -> list:
    import re
    offenders = []
    for py in files:
        text = paths.Path(py).read_text(encoding="utf-8", errors="ignore")
        for pat in _FORBIDDEN:
            if re.search(pat, text):
                offenders.append(f"{py}: {pat}")
    return offenders


def test_no_cluster_defaults_in_source() -> None:
    """No author-specific absolute paths in the package."""
    import saber
    pkg = paths.Path(saber.__file__).parent
    offenders = _path_offenders(pkg.rglob("*.py"))
    assert not offenders, f"hardcoded paths remain: {offenders}"


def test_no_cluster_defaults_in_scripts() -> None:
    """No author-specific absolute paths in scripts, configs, and adapters."""
    root = paths.Path(__file__).resolve().parents[1]
    files: list[paths.Path] = []
    for pattern in (
        "scripts/**/*.py",
        "scripts/*.sh",
        "configs/**/*.yaml",
        "external_adapters/**/*.py",
        "external_adapters/**/*.R",
        "external_adapters/**/*.json",
        "external_adapters/**/*.yml",
    ):
        files.extend(root.glob(pattern))
    assert not _path_offenders(files), f"hardcoded paths remain: {_path_offenders(files)}"


def test_external_adapters_are_ascii() -> None:
    """Adapter sources stay ASCII (English comments only)."""
    root = paths.Path(__file__).resolve().parents[1] / "external_adapters"
    for path in root.rglob("*"):
        if path.suffix in {".py", ".R", ".json", ".yml", ".md"}:
            text = path.read_text(encoding="utf-8")
            assert all(ord(ch) < 128 for ch in text), f"non-ASCII content in {path}"
