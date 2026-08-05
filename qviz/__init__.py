"""qviz — the repo's shared drawing layer.

HARD RULE: this package may only *draw*. It must never compute quantum
mechanics on behalf of a Track A notebook. If A03 imported ``apply_gate`` from
here, the notebook would stop teaching what is under the hood — which is the
entire point of the repo. Statevectors, gates, measurement and channels are
derived inline, in the notebook, every time.

Public API (frozen — notebooks depend on these signatures):

    from qviz import style, bloch, grid, backends

    style.use()                      # matplotlib defaults; call once per notebook
    style.CAT / SEQ / DIV / PHASE    # palette + colormaps
    style.phase_colors(amps)         # RGBA encoding arg(amplitude)
    style.phase_wheel(polar_ax)      # cyclic legend for the above

    bloch.sphere(ax)                 # draw the Bloch sphere frame
    bloch.vector(ax, v)              # a state arrow
    bloch.path(ax, pts)              # a trajectory polyline
    bloch.ellipsoid(ax, M, c)        # channel-deformed ball (A12/B26)

    grid.frames(n, ncols=4)          # "animation as a static frame grid"
    grid.amp_bars(ax, amps)          # THE signature plot: height=|amp|, hue=phase
    grid.matrix(ax, M, part=...)     # matrix heatmap (re/im/abs/phase)
    grid.hinton(ax, M)               # area-encoded matrix

    backends.get_backend()           # single point of Aer fallback
    backends.SEED                    # base seed; notebooks add their own offset
"""

from qviz import backends, bloch, grid, style  # noqa: F401

__all__ = ["style", "bloch", "grid", "backends"]
