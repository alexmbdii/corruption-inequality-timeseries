"""Matplotlib style preset matching the colleagues' Mines/PSL report layout.

The report is A4 portrait, serif body (Garamond-style), with navy headings
and coral-red section titles. Figures should feel like a math-textbook
figure: serif fonts, classical palette, light grid, no chartjunk, sized
to drop full-page-width into the document.

Usage:
    from src.course.style import apply_style, FULL_PAGE, HALF_PAGE
    with apply_style():
        fig, ax = plt.subplots(figsize=FULL_PAGE)
        ...
"""
from __future__ import annotations

import contextlib

import matplotlib as mpl
import matplotlib.pyplot as plt

# Document palette — sampled from the report's headings/body
NAVY = "#1F3A5F"        # body & section subheadings, OECD line
CORAL = "#B22B2B"       # Granger/Conclusion section headings, C→I direction
OCHRE = "#B7791F"       # Émergents — third axis, warm earth tone
TEAL = "#2F6F77"        # Afrique — muted teal, distinct from navy
PURPLE = "#5E3A82"      # Bidirectional Granger
GREY = "#9CA3AF"        # "Aucune" / no-result / neutral
LIGHT_GREY = "#D6D6D6"  # gridlines, faint backdrops
INK = "#222222"         # primary text on plots

# Group palette (3 country groups)
GROUP_COLORS_DOC = {
    "OECD":      NAVY,
    "Émergents": OCHRE,
    "Afrique":   TEAL,
}

# Granger 4-category palette
CAT_COLORS_DOC = {
    "C→I":    CORAL,
    "I→C":    NAVY,
    "Bidir":  PURPLE,
    "Aucune": GREY,
}

# Figure sizes (inches). The report's text block is ~16 cm wide ≈ 6.3".
FULL_PAGE = (6.3, 4.0)            # full text-width landscape figure
FULL_PAGE_TALL = (6.3, 5.2)       # taller variant for two-row layouts
TWO_PANEL = (6.3, 3.2)            # side-by-side, e.g. corruption + Gini curves
SQUARE = (4.5, 4.5)               # pies, scatters
WIDE = (6.3, 2.6)                 # heatmap rows, IRF strips
TALL = (4.6, 7.0)                 # vertical heatmaps with country labels

DPI = 220                          # matches roughly 1500 px wide


@contextlib.contextmanager
def apply_style():
    """Context manager applying the report-aligned matplotlib rcParams."""
    rc = {
        # Fonts — serif, classical look
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Nimbus Roman",
                        "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,

        # Colors and lines
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",

        # Grid: very light, dotted
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": LIGHT_GREY,
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,

        # Spines
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Title placement
        "axes.titlelocation": "left",
        "axes.titlepad": 8.0,

        # Legend
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.4,

        # Saving
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # Cycle uses the doc palette in a sensible default order
        "axes.prop_cycle": mpl.cycler(color=[NAVY, OCHRE, TEAL, CORAL,
                                              PURPLE, GREY, INK]),
    }
    with mpl.rc_context(rc):
        yield


def add_caption(fig, text: str, y: float = -0.04) -> None:
    """Add an italic 'Figure N — caption' below the figure (academic style)."""
    fig.text(0.5, y, text, ha="center", va="top",
             fontsize=9, style="italic", color=INK)
