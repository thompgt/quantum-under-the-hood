"""Plot primitives reused across the repo. DRAWING ONLY.

The two that matter most:

``amp_bars`` - the repo's signature plot. Bar height is |amplitude| and bar
colour is arg(amplitude) on the cyclic phase map. One picture carries the whole
complex vector, which is what makes interference, phase kickback and Grover
legible instead of abstract.

``frames`` - "animation as a static frame grid". A notebook committed with
outputs cannot animate, so evolution is shown as a grid of small panels with a
fading trail. Every time-evolution figure in the repo uses this.
"""

from __future__ import annotations

import numpy as np

from qviz.style import (AXIS, DIV, GRID, INK, INK_2, MUTED, PHASE, SEQ, SURFACE,
                        phase_colors)


def frames(n, ncols=4, *, panel=(1.9, 1.9), projection=None, sharex=False,
           sharey=False):
    """Grid of ``n`` small panels - the animation stand-in.

    Returns ``(fig, axes_flat)`` with unused trailing axes already removed.
    """
    import matplotlib.pyplot as plt

    nrows = int(np.ceil(n / ncols))
    figsize = (panel[0] * ncols, panel[1] * nrows)
    if projection is None:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 sharex=sharex, sharey=sharey)
        axes = np.atleast_1d(axes).ravel()
    else:
        fig = plt.figure(figsize=figsize)
        axes = np.array([fig.add_subplot(nrows, ncols, i + 1,
                                         projection=projection)
                         for i in range(nrows * ncols)])
    for ax in axes[n:]:
        ax.remove()
    return fig, axes[:n]


def _basis_labels(n_states, n_qubits=None):
    if n_qubits is None:
        n_qubits = int(np.log2(n_states))
    return [format(i, f"0{n_qubits}b") for i in range(n_states)]


def ket(s):
    r"""Render a ket as mathtext.

    The literal Unicode bracket U+27E9 is missing from Segoe UI and from most
    Windows UI faces, so it rasterises as a tofu box. Always go through mathtext
    ``\rangle`` in figure text.
    """
    return rf"$|{s}\rangle$"


def phase_hands(ax, positions, heights, phases, *, radius=6.0, color=INK,
                lw=1.1, alpha=0.75):
    """Little clock hands marking phase - the colour-independent channel.

    No cyclic colormap can be colour-vision-deficiency safe: CVD collapses the
    hue circle onto a line, so any map that returns to its start must
    self-intersect under simulation. Ours is fine on the case that matters most
    (phase 0 vs pi - a sign flip - separates at deltaE 11.5) but 0 vs pi/2 is
    confusable for deutan viewers. So phase always ships with this second
    encoding: a hand pointing at the angle, drawn in display space so it stays
    circular whatever the axis scaling.
    """
    from matplotlib.transforms import offset_copy

    fig = ax.figure
    for x, h, ph in zip(positions, heights, phases):
        if h < 1e-12:
            continue
        dx, dy = radius * np.cos(ph), radius * np.sin(ph)
        # Both ends are point-offsets from the same data point, so the hand is a
        # true circle on screen regardless of the axis aspect ratio.
        tail = offset_copy(ax.transData, fig=fig, x=-dx, y=-dy, units="points")
        head = offset_copy(ax.transData, fig=fig, x=dx, y=dy, units="points")
        ax.annotate("", xy=(x, h), xycoords=head, xytext=(x, h), textcoords=tail,
                    annotation_clip=False,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    alpha=alpha, shrinkA=0, shrinkB=0,
                                    mutation_scale=7))
    return ax


def amp_bars(ax, amplitudes, *, labels=None, n_qubits=None, ylim=None,
             show_zero_phase=True, edge=True, label_rotation=None,
             ymax_pad=1.30, bar_width=0.78, hands=True,
             ylabel="|amplitude|", tick_every=1):
    """THE signature plot: height = |amplitude|, fill colour = phase.

    ``amplitudes`` is a complex vector in the repo's little-endian convention
    (index i -> bitstring with qubit 0 as the RIGHTMOST character).

    Zero amplitudes are drawn as a faint baseline tick rather than a coloured bar
    of height 0 - a colour there would claim a phase the state does not have.

    ``hands=True`` adds a clock hand above each bar pointing at the phase angle,
    so phase survives colour-vision deficiency and greyscale printing. It is
    dropped automatically past 32 states, where the bars are too dense.

    ``tick_every=k`` labels every k-th bar; past ~16 states one ket per bar is
    unreadable. ``ylabel=None`` drops the y-label, which frees that slot for a
    row name in a frame grid.
    """
    amps = np.asarray(amplitudes).ravel()
    mag = np.abs(amps)
    cols = phase_colors(amps)
    idx = np.arange(len(amps))

    if labels is None:
        labels = _basis_labels(len(amps), n_qubits)

    near_zero = mag < 1e-12
    bars = ax.bar(idx, mag, width=bar_width, color=cols,
                  edgecolor=SURFACE if edge else "none", linewidth=1.2,
                  zorder=3)
    if show_zero_phase and near_zero.any():
        ax.scatter(idx[near_zero], np.zeros(near_zero.sum()), marker="_",
                   s=48, color=MUTED, zorder=4, linewidths=1.2)
    if hands and len(amps) <= 32:
        phase_hands(ax, idx, mag, np.angle(amps))

    keep = idx[::max(1, int(tick_every))]
    ax.set_xticks(keep)
    if label_rotation is None:
        label_rotation = 0 if len(keep) <= 16 else 90
    ax.set_xticklabels([ket(labels[i]) for i in keep], rotation=label_rotation,
                       fontsize=8 if len(keep) <= 16 else 6.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, ylim if ylim is not None else max(mag.max() * ymax_pad, 1e-9))
    ax.set_xlim(-0.6, len(amps) - 0.4)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    return bars


def prob_bars(ax, probs, *, labels=None, n_qubits=None, color=None,
              analytic=None, label_rotation=None, ymax_pad=1.25,
              ylabel="probability", tick_every=1):
    """Probability bars, optionally overlaid with an analytic reference."""
    from qviz.style import BLUE

    p = np.asarray(probs, dtype=float).ravel()
    idx = np.arange(len(p))
    if labels is None:
        labels = _basis_labels(len(p), n_qubits)
    ax.bar(idx, p, width=0.78, color=color or BLUE, edgecolor=SURFACE,
           linewidth=1.2, zorder=3, label="measured")
    top = p.max()
    if analytic is not None:
        a = np.asarray(analytic, dtype=float).ravel()
        ax.plot(idx, a, ls="none", marker="_", ms=13, mew=2.0, color=INK,
                zorder=5, label="analytic")
        top = max(top, a.max())
    keep = idx[::max(1, int(tick_every))]
    ax.set_xticks(keep)
    if label_rotation is None:
        label_rotation = 0 if len(keep) <= 16 else 90
    ax.set_xticklabels([ket(labels[i]) for i in keep], rotation=label_rotation,
                       fontsize=8 if len(keep) <= 16 else 6.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(top * ymax_pad, 1e-9))
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    return ax


def signed_bars(ax, amps, *, labels=None, n_qubits=None, mean=True,
                ghost=None, highlight=None, ylabel="amplitude",
                tick_every=1, label_rotation=None, ymax_pad=1.35):
    """Real amplitudes as signed bars, with the mean drawn across them.

    The encoding for inversion-about-the-mean: bars carry sign as direction
    rather than folding it into a colour, so "reflect through the mean" is a
    visible geometric operation. Only valid when every amplitude is real -
    Grover and amplitude amplification keep them real, most circuits do not.

    ``ghost`` draws a previous amplitude vector as hollow outlines behind the
    bars, so a step shows where each bar moved from. ``highlight`` is an index
    (the marked item) drawn in the accent colour.
    """
    from qviz.style import BLUE, ORANGE

    a = np.asarray(amps, dtype=float).ravel()
    idx = np.arange(len(a))
    if labels is None:
        labels = _basis_labels(len(a), n_qubits)

    cols = [ORANGE if (highlight is not None and i == highlight) else BLUE
            for i in idx]
    if ghost is not None:
        g = np.asarray(ghost, dtype=float).ravel()
        ax.bar(idx, g, width=0.78, facecolor="none", edgecolor=MUTED,
               linewidth=0.9, linestyle=":", zorder=2)
    ax.bar(idx, a, width=0.78, color=cols, edgecolor=SURFACE, linewidth=1.1,
           zorder=3)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=4)

    if mean:
        m = a.mean()
        ax.axhline(m, color=INK, lw=1.3, ls="--", zorder=5)
        ax.annotate(f"mean {m:+.3f}", xy=(len(a) - 0.4, m),
                    xytext=(-2, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.5, color=INK)

    keep = idx[::max(1, int(tick_every))]
    ax.set_xticks(keep)
    if label_rotation is None:
        label_rotation = 0 if len(keep) <= 16 else 90
    ax.set_xticklabels([ket(labels[i]) for i in keep], rotation=label_rotation,
                       fontsize=8 if len(keep) <= 16 else 6.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    lim = max(np.abs(a).max(), 1e-9) * ymax_pad
    ax.set_ylim(-lim, lim)
    ax.set_xlim(-0.6, len(a) - 0.4)
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    return ax


def matrix(ax, M, *, part="re", labels=None, annot=None, cbar=True,
           vmax=None, title=None, grid_lines=True, fmt="{:.2f}",
           hide_zeros=True, annot_size=6.5):
    """Matrix heatmap.

    ``part`` is one of ``"re"``, ``"im"``, ``"abs"``, ``"phase"``,
    ``"nonzero"``. Real and imaginary parts use the diverging map anchored
    symmetrically at 0 so the sign is readable; magnitude uses the sequential
    map; phase uses the cyclic map, since phase wraps.

    ``"nonzero"`` is the sparsity view: every non-zero entry gets one ink and
    exact zeros go grey. Use it when the question is "where are the entries",
    not "how big are they" - an ``"abs"`` ramp makes a zero read as "small"
    rather than "absent".

    ``annot_size`` is the annotation font size; bump it when a small matrix is
    blown up to a large panel.
    """
    M = np.asarray(M)
    n = M.shape[0]
    if part == "re":
        data, cmap = M.real, DIV
        v = vmax if vmax is not None else max(np.abs(M.real).max(), 1e-12)
        kw = dict(vmin=-v, vmax=v)
    elif part == "im":
        data, cmap = M.imag, DIV
        v = vmax if vmax is not None else max(np.abs(M.imag).max(), 1e-12)
        kw = dict(vmin=-v, vmax=v)
    elif part == "abs":
        data, cmap = np.abs(M), SEQ
        kw = dict(vmin=0,
                  vmax=vmax if vmax is not None else max(np.abs(M).max(), 1e-12))
    elif part == "phase":
        # Phase of a (near) zero entry is meaningless noise - mask it rather
        # than paint a confident hue on numerical dust.
        data = np.ma.masked_where(np.abs(M) < 1e-12, np.angle(M) % (2 * np.pi))
        cmap = PHASE
        kw = dict(vmin=0, vmax=2 * np.pi)
    elif part == "nonzero":
        # One ink for "present", grey for "absent". No magnitude ramp, because
        # magnitude is not the question being asked.
        data = np.ma.masked_where(np.abs(M) < 1e-12, np.ones_like(np.abs(M)))
        cmap = SEQ
        kw = dict(vmin=0, vmax=1)
    else:
        raise ValueError(f"part must be re/im/abs/phase, got {part!r}")

    cmap = cmap.copy()
    cmap.set_bad(GRID)
    im = ax.imshow(data, cmap=cmap, interpolation="nearest", **kw)

    if labels is None and n <= 16:
        labels = _basis_labels(n)
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        size = 8 if n <= 8 else 6
        ax.set_xticklabels(labels, fontsize=size, rotation=90 if n > 8 else 0)
        ax.set_yticklabels(labels, fontsize=size)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if grid_lines and n <= 16:
        ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=1.1)
        ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    if annot is None:
        # "nonzero" carries no magnitude, so annotating it prints 1.00 in every
        # occupied cell - noise, not information.
        annot = n <= 8 and part != "nonzero"
    if annot:
        plain = np.ma.filled(data, 0.0)
        thr = 0.55 * max(np.abs(plain).max(), 1e-12)
        for i in range(n):
            for j in range(n):
                val = plain[i, j]
                if hide_zeros and abs(val) < 1e-12:
                    continue  # a grid of "0.00" is clutter, not information
                ax.text(j, i, fmt.format(val), ha="center", va="center",
                        fontsize=annot_size,
                        color="white" if abs(val) > thr and part != "phase" else INK)
    if cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7, color=MUTED, labelcolor=MUTED)
        if part == "phase":
            cb.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
            cb.set_ticklabels(["0", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return im


def hinton(ax, M, *, max_weight=None, labels=None, title=None):
    """Hinton diagram: square AREA encodes magnitude, colour encodes phase.

    Better than a heatmap for density matrices, because the eye reads "this
    off-diagonal block is shrinking" far faster from area than from colour.
    """
    from matplotlib.patches import Rectangle

    M = np.asarray(M)
    n = M.shape[0]
    if max_weight is None:
        max_weight = max(np.abs(M).max(), 1e-12)

    ax.patch.set_facecolor(SURFACE)
    ax.set_aspect("equal", "box")

    cols = phase_colors(M.ravel()).reshape(n, n, 4)
    for i in range(n):
        for j in range(n):
            w = abs(M[i, j])
            if w < 1e-12:
                continue
            size = np.sqrt(w / max_weight)
            ax.add_patch(Rectangle((j - size / 2, i - size / 2), size, size,
                                   facecolor=cols[i, j], edgecolor=SURFACE,
                                   linewidth=0.6))
    ax.set_xlim(-0.6, n - 0.4)
    ax.set_ylim(n - 0.4, -0.6)
    if labels is None and n <= 16:
        labels = _basis_labels(n)
    if labels is not None:
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        s = 8 if n <= 8 else 6
        ax.set_xticklabels(labels, fontsize=s, rotation=90 if n > 8 else 0)
        ax.set_yticklabels(labels, fontsize=s)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=INK)
    return ax


def show_state(vec, *, n_show=8, precision=4, n_qubits=None):
    """Truncating statevector repr - never print 2**10 amplitudes into a cell."""
    v = np.asarray(vec).ravel()
    if n_qubits is None:
        n_qubits = int(np.log2(len(v)))
    order = np.argsort(-np.abs(v))
    keep = sorted(i for i in order if abs(v[i]) > 1e-10)[:n_show]
    parts = []
    for i in keep:
        a = complex(v[i])
        parts.append(f"  |{format(i, f'0{n_qubits}b')}>  "
                     f"{a.real:+.{precision}f}{a.imag:+.{precision}f}j")
    hidden = int((np.abs(v) > 1e-10).sum()) - len(keep)
    out = "\n".join(parts)
    if hidden > 0:
        out += f"\n  ... and {hidden} more non-zero amplitude(s)"
    return out


def annotate(ax, text, xy, xytext, *, color=None, size=8.5, arrow=True):
    """Muted callout with a thin arrow - used to point at the punchline."""
    color = color or INK_2
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=size, color=color,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.9,
                                shrinkA=2, shrinkB=3) if arrow else None)
    return ax
