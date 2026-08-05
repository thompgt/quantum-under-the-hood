"""Bloch sphere / Bloch ball rendering. DRAWING ONLY.

This module knows nothing about quantum mechanics — it draws arrows in a
3-sphere. Notebooks compute their own Bloch vectors (r = (<X>,<Y>,<Z>)) and hand
them here. That keeps the "under the hood" derivation inside the notebook where
a reader can see it.
"""

from __future__ import annotations

import numpy as np

from qviz.style import AQUA, AXIS, BLUE, GRID, INK, INK_2, MUTED, ORANGE, SURFACE

_LABELS = {
    "z": (r"$|0\rangle$", r"$|1\rangle$"),
    "x": (r"$|+\rangle$", r"$|-\rangle$"),
    "y": (r"$|{+}i\rangle$", r"$|{-}i\rangle$"),
}


def axes3d(fig=None, rect=111, figsize=(4.0, 4.0)):
    """Create a 3D axis sized and framed for a Bloch sphere."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(rect, projection="3d")
    return ax


def sphere(ax, *, labels=True, equator=True, alpha=0.055, wire=True,
           label_kets=True, tick_axes=True):
    """Draw the Bloch sphere frame on a 3D axis.

    Returns the axis. Call before :func:`vector` / :func:`path`.
    """
    u = np.linspace(0, 2 * np.pi, 48)
    v = np.linspace(0, np.pi, 26)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=BLUE, alpha=alpha, linewidth=0,
                    antialiased=True, shade=False, zorder=0)

    if wire:
        for ang in np.linspace(0, np.pi, 7)[1:-1]:  # lines of latitude
            t = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.sin(ang) * np.cos(t), np.sin(ang) * np.sin(t),
                    np.cos(ang) * np.ones_like(t),
                    color=GRID, lw=0.5, alpha=0.85, zorder=1)
        for ang in np.linspace(0, np.pi, 7)[:-1]:   # lines of longitude
            t = np.linspace(0, 2 * np.pi, 100)
            ax.plot(np.cos(ang) * np.sin(t), np.sin(ang) * np.sin(t), np.cos(t),
                    color=GRID, lw=0.5, alpha=0.85, zorder=1)

    if equator:
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), np.zeros_like(t),
                color=AXIS, lw=1.0, zorder=2)

    if tick_axes:
        ax.plot([-1, 1], [0, 0], [0, 0], color=AXIS, lw=0.9, zorder=2)
        ax.plot([0, 0], [-1, 1], [0, 0], color=AXIS, lw=0.9, zorder=2)
        ax.plot([0, 0], [0, 0], [-1, 1], color=AXIS, lw=0.9, zorder=2)

    if labels:
        pad = 1.40
        pairs = [((0, 0, pad), (0, 0, -pad), "z"),
                 ((pad, 0, 0), (-pad, 0, 0), "x"),
                 ((0, pad, 0), (0, -pad, 0), "y")]
        for pos, neg, key in pairs:
            hi, lo = _LABELS[key] if label_kets else (f"+{key}", f"-{key}")
            ax.text(*pos, hi, color=INK_2, fontsize=9, ha="center", va="center")
            ax.text(*neg, lo, color=INK_2, fontsize=9, ha="center", va="center")

    _frame(ax)
    return ax


def _frame(ax, lim=1.52):
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=18, azim=32)
    try:
        ax.set_facecolor(SURFACE)
    except Exception:
        pass
    return ax


def vector(ax, r, *, color=ORANGE, lw=2.4, label=None, alpha=1.0,
           head=0.09, tip_marker=True):
    """Draw a Bloch vector ``r = (x, y, z)`` from the origin."""
    r = np.asarray(r, dtype=float)
    ax.plot([0, r[0]], [0, r[1]], [0, r[2]],
            color=color, lw=lw, alpha=alpha, zorder=6,
            solid_capstyle="round")
    if tip_marker:
        n = np.linalg.norm(r)
        ax.scatter([r[0]], [r[1]], [r[2]], color=color, s=26 if n > 1e-6 else 40,
                   depthshade=False, alpha=alpha, zorder=7,
                   edgecolors=SURFACE, linewidths=0.8)
    if label:
        off = 1.10 if np.linalg.norm(r) > 0.2 else 1.0
        ax.text(r[0] * off, r[1] * off, r[2] * off + 0.10, label,
                color=color, fontsize=8.5, ha="center", zorder=8)
    return ax


def path(ax, pts, *, color=BLUE, lw=1.8, fade=True, alpha=0.95):
    """Draw a trajectory across the sphere.

    With ``fade=True`` the tail is drawn progressively more transparent, which
    reads as direction of travel in a static frame — the trick that lets a frame
    grid stand in for an animation.
    """
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return ax
    if not fade:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=color, lw=lw,
                alpha=alpha, zorder=5)
        return ax
    n = len(pts) - 1
    for i in range(n):
        ax.plot(pts[i:i + 2, 0], pts[i:i + 2, 1], pts[i:i + 2, 2],
                color=color, lw=lw, alpha=alpha * (0.10 + 0.90 * (i + 1) / n),
                zorder=5, solid_capstyle="round")
    return ax


def ellipsoid(ax, matrix, offset=(0, 0, 0), *, color=AQUA, alpha=0.30,
              wire=True, n=40):
    """Draw the image of the Bloch ball under an affine map.

    ``matrix`` is the 3x3 linear part and ``offset`` the translation — together
    they are how a CPTP channel acts on the Bloch ball. Used by A12/B26 to make
    "what a noise channel *is*" a picture rather than algebra.
    """
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n // 2)
    pts = np.stack([np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v))], axis=-1)
    M = np.asarray(matrix, dtype=float)
    out = pts @ M.T + np.asarray(offset, dtype=float)
    ax.plot_surface(out[..., 0], out[..., 1], out[..., 2], color=color,
                    alpha=alpha, linewidth=0, shade=False, zorder=3)
    if wire:
        ax.plot_wireframe(out[..., 0], out[..., 1], out[..., 2], color=color,
                          alpha=min(1.0, alpha + 0.25), linewidth=0.35,
                          rstride=4, cstride=4, zorder=4)
    return ax


def ball(ax, *, alpha=0.05, color=BLUE):
    """Translucent solid ball — for showing that mixed states live *inside*."""
    return sphere(ax, alpha=alpha)


def label(ax, text, *, y=-0.06, color=INK, size=9.0):
    """Caption under a Bloch panel (3D axes ignore set_title spacing)."""
    ax.text2D(0.5, y, text, transform=ax.transAxes, ha="center", va="top",
              fontsize=size, color=color)
    return ax
