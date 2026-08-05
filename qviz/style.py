"""Repo-wide visual system: palette, colormaps, matplotlib defaults.

DRAWING ONLY. Nothing in ``qviz`` may compute quantum mechanics — Track A
notebooks derive their own math inline, or they stop teaching "under the hood".

The palette is a validated categorical set (light surface #fcfcfb): it clears the
lightness band, chroma floor, adjacent-pair colour-vision-deficiency separation
(worst ΔE 9.1) and the normal-vision floor (worst ΔE 19.6). Three slots sit below
3:1 contrast on the surface, so charts always carry a legend or direct labels —
identity is never colour-alone.

The phase colormap is generated, not borrowed. Phase is *cyclic*, so it needs a
colormap whose two ends meet; and phase 0 is by far the most common value in a
statevector, so it must not land on a washed-out colour. ``twilight`` fails that
second test (near-white at 0). Instead we sweep hue at **constant OKLCH
lightness**, which makes the map perceptually uniform, cyclic by construction, and
equally legible at every phase.
"""

from __future__ import annotations

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --------------------------------------------------------------------------
# Surfaces and ink
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"    # chart surface; notebook PNGs bake this in
INK = "#0b0b0b"        # primary text
INK_2 = "#52514e"      # secondary text
MUTED = "#898781"      # axis labels, ticks
GRID = "#e1e0d9"       # hairline gridlines
AXIS = "#c3c2b7"       # baseline / spine

# --------------------------------------------------------------------------
# Categorical palette — fixed order, never cycled.
# Past 3 series in a scatter (all-pairs), fold to "Other" or facet.
# --------------------------------------------------------------------------
CAT = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT

# Status colours — reserved, never reused as "series 4".
GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"

# --------------------------------------------------------------------------
# Sequential + diverging ramps
# --------------------------------------------------------------------------
_BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6",
              "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
_ORANGE_RAMP = ["#fbe0d2", "#f7c2a6", "#f2a179", "#ee8352", "#eb6834",
                "#d4562a", "#b64720", "#963818", "#762a11", "#571d0b"]

SEQ = LinearSegmentedColormap.from_list("quth_seq", _BLUE_RAMP)
SEQ2 = LinearSegmentedColormap.from_list("quth_seq2", _ORANGE_RAMP)
# Diverging: two hues + a NEUTRAL GRAY midpoint (never a hue at the middle).
DIV = LinearSegmentedColormap.from_list(
    "quth_div", ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec",
                 "#f3a3a2", "#e34948", "#8e1f1e"])


# --------------------------------------------------------------------------
# OKLCH -> sRGB, used to build the cyclic phase map
# --------------------------------------------------------------------------
def _oklab_to_linear_srgb(L, a, b):
    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b
    l3, m3, s3 = l_ ** 3, m_ ** 3, s_ ** 3
    r = 4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
    g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
    bb = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
    return np.stack([r, g, bb], axis=-1)


def _gamma(rgb_linear):
    rgb = np.clip(rgb_linear, 0.0, 1.0)
    return np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)


def _oklch_to_srgb(L, C, h_rad):
    return _gamma(_oklab_to_linear_srgb(L, C * np.cos(h_rad), C * np.sin(h_rad)))


def _in_gamut(L, C, h_rad, tol=1e-4):
    lin = _oklab_to_linear_srgb(L, C * np.cos(h_rad), C * np.sin(h_rad))
    return bool(np.all(lin >= -tol) and np.all(lin <= 1 + tol))


def _max_chroma(L, h_rad, ceiling=0.30):
    """Largest in-gamut chroma at this lightness/hue (bisection)."""
    lo, hi = 0.0, ceiling
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _in_gamut(L, mid, h_rad):
            lo = mid
        else:
            hi = mid
    return lo


def _build_phase_cmap(n=512, lightness=0.62, chroma=0.125, hue_offset_deg=25.0):
    """Constant-lightness cyclic hue sweep.

    Chroma is held at ``chroma`` wherever the sRGB gamut allows and reduced only
    where it must be — lightness stays constant everywhere, which is what makes
    every phase equally readable. ``hue_offset_deg`` places phase 0 on a blue that
    matches the repo's primary series colour.
    """
    h = np.linspace(0.0, 2 * np.pi, n, endpoint=False) + np.deg2rad(hue_offset_deg)
    cols = np.empty((n, 3))
    for i, hi in enumerate(h):
        c = min(chroma, _max_chroma(lightness, hi))
        cols[i] = _oklch_to_srgb(lightness, c, hi)
    return ListedColormap(np.clip(cols, 0, 1), name="quth_phase")


PHASE = _build_phase_cmap()
"""Cyclic colormap for complex phase. Domain is [0, 2*pi); use with
``norm=Normalize(0, 2*np.pi)`` or via :func:`phase_colors`."""


def phase_colors(amplitudes):
    """RGBA colours encoding the phase of each complex amplitude."""
    ang = np.angle(np.asarray(amplitudes)) % (2 * np.pi)
    return PHASE(ang / (2 * np.pi))


# --------------------------------------------------------------------------
# matplotlib defaults
# --------------------------------------------------------------------------
def use():
    """Apply the repo's matplotlib defaults. Call once per notebook."""
    import matplotlib as mpl

    mpl.rcParams.update({
        # Size budget: dpi 110, not 200. 30 notebooks x many figures must stay
        # under ~500 KB each, and a 300-dpi PNG is ~10x the bytes for no gain
        # on a screen.
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "figure.figsize": (7.0, 4.2),
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11.5,
        "axes.titleweight": "medium",
        "axes.titlelocation": "left",
        "axes.titlepad": 9,
        "axes.labelsize": 9.5,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "text.color": INK,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "font.size": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "lines.linewidth": 2.0,       # 2px lines
        "lines.markersize": 5.0,      # >= 8px markers
        "lines.solid_capstyle": "round",
        "image.cmap": "quth_seq",
        "figure.constrained_layout.use": True,
    })
    # Register colormaps by name so rcParams["image.cmap"] resolves.
    for cm in (SEQ, SEQ2, DIV, PHASE):
        try:
            mpl.colormaps.register(cm)
        except ValueError:
            pass  # already registered (notebook re-run)


def title(ax, headline, sub=None):
    """Left-aligned headline with an optional muted subtitle underneath."""
    ax.set_title(headline, loc="left", color=INK)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=8.5, color=MUTED)
        ax.title.set_position((0.0, 1.10))
    return ax


def phase_wheel(ax, label="phase  arg(amplitude)"):
    """Circular legend for the cyclic phase colormap.

    A normal colorbar is wrong for cyclic data: it implies 0 and 2*pi are far
    apart when they are the same value. Draw the wheel instead.
    """
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    # pcolormesh, not 360 overlapping bars: overlapping wedges alias into
    # visible moire fringes that read as structure the data does not have.
    n = 720
    theta_edges = np.linspace(0, 2 * np.pi, n + 1)
    r_edges = np.array([0.62, 1.0])
    T, R = np.meshgrid(theta_edges, r_edges)
    vals = ((theta_edges[:-1] + theta_edges[1:]) / 2)[None, :]
    ax.pcolormesh(T, R, vals, cmap=PHASE, vmin=0, vmax=2 * np.pi,
                  shading="flat", linewidth=0, rasterized=True)
    ax.set_yticks([])
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$"],
                       fontsize=8, color=MUTED)
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    if label:
        ax.set_title(label, fontsize=8.5, color=MUTED, loc="center", pad=6)
    return ax
