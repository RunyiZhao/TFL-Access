"""
twc_style.py -- shared Matplotlib styling for IEEE TWC-quality figures.

Import and call `apply_twc_style()` at the top of any plotting script, then use
COL_W / DBL_W for single- and double-column figure widths (inches), and the
PALETTE / MARKERS for consistent colors and markers across all figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE column widths (inches): single column ~3.5", double column ~7.16"
COL_W = 3.5
DBL_W = 7.16

# A restrained, print-safe, color-blind-friendly palette
PALETTE = {
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "green":  "#009E73",
    "purple": "#7B3FA0",
    "gray":   "#555555",
    "orange": "#E69F00",
}
MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_twc_style():
    plt.rcParams.update({
        # fonts: serif to match IEEE body text
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        # lines / markers
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        # grid
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        # ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.7",
        "legend.fancybox": False,
        # layout / output
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # embed TrueType (editable, no Type-3 warnings)
        "ps.fonttype": 42,
    })


def panel_label(ax, text, x=-0.18, y=1.02):
    """Add an (a)/(b) panel label to a subplot, top-left, bold."""
    ax.text(x, y, text, transform=ax.transAxes,
            fontsize=8, fontweight="bold", va="bottom", ha="left")
