"""Extract a notebook's figures to PNGs so they can be eyeballed.

    python tools/peek.py A01 [outdir]

The build gate checks colour and structure; it cannot tell you that two labels
collide or that a panel overflowed. Look at the pictures.
"""

import base64
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent


USAGE = "usage: python tools/peek.py <ID> [outdir]      e.g. python tools/peek.py A01"


def known_ids():
    return sorted(p.name.split("_")[0] for p in (ROOT / "notebooks").glob("*.ipynb"))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(USAGE)
        print("  ids: " + ", ".join(known_ids()))
        return 0 if len(sys.argv) > 1 else 2

    nb_id = sys.argv[1]
    outdir = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "_peek"

    hits = sorted((ROOT / "notebooks").glob(f"{nb_id}_*.ipynb"))
    if not hits:
        print(f"no notebook matching {nb_id}_*.ipynb")
        print("  ids: " + ", ".join(known_ids()))
        return 2
    path = hits[0]

    outdir.mkdir(parents=True, exist_ok=True)
    nb = nbformat.read(path, as_version=4)

    n = 0
    for ci, cell in enumerate(nb.cells):
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png:
                n += 1
                dest = outdir / f"{nb_id}_fig{n:02d}_cell{ci}.png"
                dest.write_bytes(base64.b64decode(png))
                print(dest)
    print(f"{n} figure(s) from {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
