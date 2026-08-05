"""Validate the generated phase colormap.

Checks the two properties the signature amplitude-bar plot depends on:
  1. constant perceptual lightness across the full cycle (no phase is privileged)
  2. adequate contrast against the chart surface at every phase

Also prints 8 evenly-spaced phase hues so they can be fed to the dataviz
validator for colour-vision-deficiency separation.
"""

import numpy as np

from qviz import style


def relative_luminance(rgb):
    c = np.asarray(rgb[:3])
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(rgb, surface_hex):
    sr = np.array([int(surface_hex[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    l1, l2 = relative_luminance(rgb), relative_luminance(sr)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def srgb_to_oklab_L(rgb):
    c = np.asarray(rgb[:3])
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                  [0.2119034982, 0.6806995451, 0.1073969566],
                  [0.0883024619, 0.2817188376, 0.6299787005]])
    lms = np.cbrt(m @ lin)
    return float(np.array([0.2104542553, 0.7936177850, -0.0040720468]) @ lms)


def main():
    samples = style.PHASE(np.linspace(0, 1, 64, endpoint=False))
    Ls = np.array([srgb_to_oklab_L(c) for c in samples])
    cs = np.array([contrast(c, style.SURFACE) for c in samples])

    print(f"phase cmap: {len(style.PHASE.colors)} steps")
    print(f"  OKLab lightness  min {Ls.min():.4f}  max {Ls.max():.4f}  "
          f"spread {Ls.max() - Ls.min():.4f}")
    print(f"  contrast vs {style.SURFACE}  min {cs.min():.2f}  max {cs.max():.2f}")

    ok = True
    if Ls.max() - Ls.min() > 0.01:
        print("  FAIL lightness is not constant across the cycle")
        ok = False
    else:
        print("  ok   lightness constant (spread <= 0.01) - no phase privileged")
    if cs.min() < 3.0:
        print(f"  FAIL min contrast {cs.min():.2f} < 3:1")
        ok = False
    else:
        print(f"  ok   every phase clears 3:1 on the surface")

    eight = style.PHASE(np.linspace(0, 1, 8, endpoint=False))
    hexes = ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3]) for c in eight]
    print("\n  8 evenly-spaced phase hues (feed to validate_palette.js):")
    print("  " + ",".join(hexes))

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
