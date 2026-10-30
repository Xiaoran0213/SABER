from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


DATASET_LABELS = {
    "Specht_2019": "Specht 2019",
    "Derks_2022": "Derks 2022",
    "Derks_2023": "Derks 2023",
    "Huffman_2023_melanoma_HEK": "Huffman 2023",
    "Woo_2021": "Woo 2021",
    "PXD071205_LFQ": "LFQ spike-in",
}

METHOD_LABELS = {
    "saber": "SABER",
    "cell_mean": "Cell Mean",
    "protein_mean": "Protein Mean",
    "pimms_dae": "PIMMS-DAE",
    "pimms_vae": "PIMMS-VAE",
    "omicsgmf": "omicsGMF",
    "scprotein": "scPROTEIN",
    "raw_true": "Raw",
    "raw_plm": "Raw ESM-2",
}

PALETTE = {
    "SABER": "#006D77",
    "SABER no PLM": "#56A6A8",
    "SABER no W": "#3A86FF",
    "Protein Mean": "#8D99AE",
    "Cell Mean": "#D0D3D4",
    "PIMMS-DAE": "#E07A5F",
    "PIMMS-VAE": "#F2CC8F",
    "omicsGMF": "#81B29A",
    "scPROTEIN": "#B07AA1",
    "Raw": "#6C757D",
    "Raw ESM-2": "#6C757D",
}

METHOD_ORDER = [
    "SABER",
    "SABER no PLM",
    "SABER no W",
    "PIMMS-DAE",
    "PIMMS-VAE",
    "omicsGMF",
    "scPROTEIN",
    "Raw ESM-2",
    "Protein Mean",
    "Cell Mean",
    "Raw",
]

DATASET_ORDER = [
    "Specht_2019",
    "Derks_2022",
    "Huffman_2023_melanoma_HEK",
    "Woo_2021",
    "EasySCP_liver",
    "Brunner_2022",
    "PXD071205_LFQ",
]

MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]


def require_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def apply_paper_style() -> None:
    plt = require_matplotlib()
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "axes.axisbelow": True,
            "lines.linewidth": 1.8,
            "patch.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def dataset_label(name: object) -> str:
    text = str(name)
    return DATASET_LABELS.get(text, text.replace("_", " "))


def method_label(method: object, condition: object = "") -> str:
    method_text = str(method)
    condition_text = str(condition).lower()
    if method_text == "saber" and ("noplm" in condition_text or "no_plm" in condition_text):
        return "SABER no PLM"
    if method_text == "saber" and ("now" in condition_text or "no_w" in condition_text):
        return "SABER no W"
    if method_text == "saber":
        return "SABER"
    return METHOD_LABELS.get(method_text, method_text.replace("_", " "))


def color_for(label: object) -> str:
    return PALETTE.get(str(label), "#4B5563")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def save_figure(fig, path: str | Path, *, formats: Sequence[str] = ("png", "pdf")) -> list[str]:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    stem = out.with_suffix("")
    for ext in formats:
        target = stem.with_suffix(f".{ext}")
        fig.savefig(target, bbox_inches="tight", facecolor="white")
        saved.append(str(target.resolve()))
    require_matplotlib().close(fig)
    return saved


def place_legend(ax, *, ncol: int = 1, outside: bool = False) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    seen: set[str] = set()
    uniq_h = []
    uniq_l = []
    for handle, label in zip(handles, labels):
        if label in seen or label.startswith("_"):
            continue
        seen.add(label)
        uniq_h.append(handle)
        uniq_l.append(label)
    if not uniq_h:
        return
    kwargs = {"frameon": False, "ncol": ncol}
    if outside:
        kwargs.update({"loc": "upper center", "bbox_to_anchor": (0.5, -0.22)})
    else:
        kwargs.update({"loc": "best"})
    ax.legend(uniq_h, uniq_l, **kwargs)


def require_columns(columns: Iterable[str], available: Iterable[str], name: str) -> None:
    available_set = set(available)
    missing = [col for col in columns if col not in available_set]
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")
