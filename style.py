"""Shared typography, colors and model labels."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

cache = Path(tempfile.mkdtemp(prefix="ncm-figures-"))
os.environ.setdefault("MPLCONFIGDIR", str(cache))
import matplotlib

if "FONTCONFIG_FILE" not in os.environ:
    config = cache / "fonts.conf"
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    config.write_text(
        '<?xml version="1.0"?><fontconfig>'
        f'<dir>{font_dir}</dir><cachedir>{cache}</cachedir></fontconfig>'
    )
    os.environ["FONTCONFIG_FILE"] = str(config)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "NCM": "#ff1a1a", "MMarkov-4": "#8b0000", "MMarkov-3": "#555555",
    "ORAS5": "#111111", "3D-Geoformer": "#2D9CDB", "DESN": "#D55E00",
    "NASA GMAO (IRI)": "#E69F00", "NCEP CFSv2 (IRI)": "#7B2CBF",
    "SINTEX-F (IRI)": "#CC79A7", "CPC CA (IRI)": "#8C9A2B",
    "CPC MRKOV (IRI)": "#3F3C99", "CSU CLIPR (IRI)": "#A6672A",
    "GFDL-SPEAR (NMME)": "#AA3377", "NASA-GEOSS2S (NMME)": "#44AA99",
    "CCSM4 (NMME)": "#008B9A", "CFSv2": "#E69F00",
    "CMC1": "#56B4E9", "CMC2": "#009E73", "CanCM4i": "#56B4E9",
    "CanESM5": "#56B4E9", "GFDL": "#0072B2", "GFDL_FLOR": "#D55E00",
    "NASA": "#CC79A7", "NCAR_CCSM4": "#6A3D9A", "GEM5_NEMO": "#009E73",
    "GEM_NEMO": "#009E73", "GFDL_SPEAR": "#D55E00",
    "NASA_GEOS5v2": "#0072B2", "NCAR_CESM1": "#CC79A7",
    "NASA GEOS5v2": "#0072B2", "NCAR CCSM4": "#6A3D9A",
    "NCAR CESM1": "#CC79A7", "GEM5.2 NEMO": "#009E73",
}
DISPLAY = {
    "GFDL": "GFDL CM2.1", "GFDL_FLOR": "GFDL FLOR", "NASA": "NASA GMAO",
    "NCAR_CCSM4": "NCAR CCSM4", "GEM5_NEMO": "GEM5 NEMO",
    "GEM_NEMO": "GEM NEMO", "GFDL_SPEAR": "GFDL SPEAR",
    "NASA_GEOS5v2": "NASA GEOS5v2", "NCAR_CESM1": "NCAR CESM1",
    "ORAS5": "ORAS5 verification Niño3.4",
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ACC_ORDER = [
    "NCM", "MMarkov-4", "MMarkov-3", "3D-Geoformer", "DESN",
    "NASA GMAO (IRI)", "NCEP CFSv2 (IRI)", "SINTEX-F (IRI)",
    "CPC CA (IRI)", "CPC MRKOV (IRI)", "CSU CLIPR (IRI)",
    "GFDL-SPEAR (NMME)", "NASA-GEOSS2S (NMME)", "CCSM4 (NMME)",
]
WARM_MODELS = {
    2015: ["CFSv2", "CMC1", "CMC2", "GFDL", "GFDL_FLOR", "NASA", "NCAR_CCSM4"],
    2023: ["CFSv2", "CanCM4i", "GEM5_NEMO", "GFDL_SPEAR",
           "NASA_GEOS5v2", "NCAR_CCSM4"],
}


def set_style(size=9, bold=False):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": size,
        "font.weight": "bold" if bold else "normal",
        "axes.labelweight": "bold" if bold else "normal",
        "axes.titleweight": "bold" if bold else "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "axes.grid": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "xtick.direction": "out", "ytick.direction": "out",
    })


def label_panel(fig, axis, letter, size=13):
    position = axis.get_position()
    fig.text(position.x0-0.027,
             position.y1+8/(72*fig.get_figheight()), f"({letter})",
             fontsize=size, fontweight="bold", ha="left", va="bottom")


def save_figure(fig, folder, number, dpi):
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(folder / f"Figure_{number}.{extension}",
                    dpi=dpi, facecolor="white")
    plt.close(fig)
