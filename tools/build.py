"""Regenerate, execute, lint and size-audit notebooks.

    python tools/build.py                # everything
    python tools/build.py A03 A04        # just these
    python tools/build.py --lint-only    # no execution, just the gates
    python tools/build.py --no-exec A05  # regenerate but do not execute

Each generator in tools/gen/gen_<ID>.py writes notebooks/<ID>_*.ipynb. The
generator is the source of truth; the .ipynb is a build artifact that happens to
be committed (with outputs) because rendering on GitHub is the product.
"""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN_DIR = ROOT / "tools" / "gen"
NB_DIR = ROOT / "notebooks"

MAX_BYTES = 2 * 1024 * 1024      # hard fail
WARN_BYTES = 700 * 1024          # nudge

# Kernel registered from the repo venv:
#   .venv/Scripts/python -m ipykernel install --user --name quth
KERNEL = os.environ.get("QUTH_KERNEL", "quth")

# Dead Qiskit APIs. A hit anywhere in a notebook or generator fails the build --
# this is the cheapest possible guard against the project's top failure mode.
DEAD_APIS = [
    (r"\bfrom\s+qiskit\s+import\s+[^\n]*\bexecute\b", "qiskit.execute (removed in 1.0)"),
    (r"(?<!\.)\bexecute\s*\(\s*(qc|circ|circuit)", "execute(...) (removed in 1.0)"),
    (r"\bAer\.get_backend\b", "Aer.get_backend (use AerSimulator)"),
    (r"\bfrom\s+qiskit\s+import\s+[^\n]*\bAer\b", "from qiskit import Aer"),
    (r"\bqiskit\.providers\.aer\b", "qiskit.providers.aer (use qiskit_aer)"),
    (r"\bqiskit\.opflow\b", "qiskit.opflow (removed in 1.0)"),
    (r"\bqiskit\.algorithms\b", "qiskit.algorithms (removed in 1.0)"),
    (r"\bqiskit_algorithms\b", "qiskit-algorithms (unsupported; hand-roll instead)"),
    (r"\bQuantumInstance\b", "QuantumInstance (removed)"),
    (r"\.c_if\s*\(", ".c_if (removed in 2.0; use `with qc.if_test(...)`)"),
    (r"\bfrom\s+qiskit\.primitives\s+import\s+(?![^\n]*V2)[^\n]*\b(Sampler|Estimator)\b(?!V2)",
     "Primitives V1 (removed in 2.0; use StatevectorSampler/StatevectorEstimator)"),
    (r"\bBackendSampler\b(?!V2)", "BackendSampler V1 (use BackendSamplerV2)"),
    (r"\bBackendEstimator\b(?!V2)", "BackendEstimator V1 (use BackendEstimatorV2)"),
    (r"\bqiskit\.pulse\b", "qiskit.pulse (removed in 2.0)"),
    (r"\bqiskit\.assemble\b", "qiskit.assemble (removed in 2.0)"),
    (r"\.qasm\s*\(\s*\)", "QuantumCircuit.qasm() (use qiskit.qasm2.dumps)"),
    (r"\bBackendV1\b", "BackendV1 (removed in 2.0)"),
    (r"\bqiskit\.providers\.models\b", "qiskit.providers.models (removed in 2.0)"),
]

# U+27E9/U+27E8 are missing from Segoe UI and rasterise as tofu in figures.
TOFU = re.compile(r"[⟨⟩]")

# Markdown links to another notebook. Every notebook ends with a "Next:" link,
# and renaming a notebook without updating the ones pointing at it produces a
# 404 that only a sequential reader ever hits -- the README index stays right.
NB_LINK = re.compile(r"\]\(\s*(?!https?:)([^)\s#]+\.ipynb)[^)]*\)")

# Mojibake: UTF-8 bytes that were decoded as cp1252 somewhere upstream. On
# Windows, `Get-Content -Raw` / `Set-Content` in PowerShell 5.1 reads a BOM-less
# UTF-8 file as ANSI and silently rewrites every non-ASCII character this way --
# an em dash becomes "â€”". It renders as garbage in the committed notebook but
# is still perfectly valid Python, so nothing else here would catch it.
MOJIBAKE = re.compile(r"â€|Ã[©¨¢«»]|Â[ §°]|â„¢")


def sh(msg):
    print(msg, flush=True)


def notebook_ids():
    ids = []
    for p in sorted(GEN_DIR.glob("gen_*.py")):
        ids.append(p.stem[len("gen_"):])
    return ids


def notebook_path(nb_id):
    hits = sorted(NB_DIR.glob(f"{nb_id}_*.ipynb"))
    return hits[0] if hits else None


def generate(nb_id):
    gen = GEN_DIR / f"gen_{nb_id}.py"
    if not gen.exists():
        return False, f"no generator {gen.name}"
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONNOUSERSITE="1")
    r = subprocess.run([sys.executable, str(gen)], cwd=str(ROOT), env=env,
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        return False, (r.stdout or "") + (r.stderr or "")
    return True, ""


def execute(nb_id, timeout=900):
    path = notebook_path(nb_id)
    if path is None:
        return False, f"no notebook for {nb_id}"
    # PYTHONNOUSERSITE is not optional: this venv otherwise picks up the user
    # site-packages in AppData\Roaming, which shadows the venv's own nbconvert
    # and hands execution to the system interpreter (where qviz is absent).
    # Do NOT set MPLBACKEND=Agg here. ipykernel's inline backend is what captures
    # figures into the notebook; forcing Agg is headless-safe but silently emits
    # no images, and a notebook with no figures is a failed notebook in this repo.
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               PYTHONNOUSERSITE="1",
               QUTH_MAX_THREADS=os.environ.get("QUTH_MAX_THREADS", "2"),
               OMP_NUM_THREADS=os.environ.get("QUTH_MAX_THREADS", "2"))
    env.pop("MPLBACKEND", None)
    # Execute a COPY and move it into place only on success. `--inplace` on the
    # real path leaves a half-executed notebook behind when a cell raises, and
    # main() skips lint() on an execute failure -- so the damage would ship
    # unlinted, invisibly, since .gitattributes marks .ipynb as -diff.
    # The copy lives beside the original: nbconvert runs the kernel with the
    # notebook's own directory as cwd, and a few notebooks rely on that.
    # Named so that notebook_path()'s "<ID>_*.ipynb" glob cannot pick up a
    # stray copy left behind by a hard kill.
    tmp = path.with_name(f"_executing_{nb_id}.ipynb")
    shutil.copyfile(path, tmp)
    # `-m nbconvert`, not `-m jupyter nbconvert`: the jupyter dispatcher shells
    # out to whichever jupyter-nbconvert.EXE is first on PATH, which on this
    # machine is the system one.
    cmd = [sys.executable, "-m", "nbconvert", "--to", "notebook",
           "--execute", "--inplace",
           f"--ExecutePreprocessor.timeout={timeout}",
           f"--ExecutePreprocessor.kernel_name={KERNEL}",
           str(tmp)]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        return False, "\n".join(tail[-40:])
    os.replace(tmp, path)
    return True, ""


def lint(nb_id):
    """Dead-API, seeding, tofu and output-presence gates."""
    import nbformat

    problems = []
    path = notebook_path(nb_id)
    gen = GEN_DIR / f"gen_{nb_id}.py"

    if path is not None:
        nb = nbformat.read(path, as_version=4)
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    else:
        nb = None
        code = ""

    # Scan the notebook's CODE cells only. Markdown is deliberately allowed to
    # name the dead APIs -- warning readers off them is part of the teaching --
    # and scanning the generator source would flag that prose too, since the
    # generator is where the markdown is authored.
    for pattern, label in DEAD_APIS:
        for m in re.finditer(pattern, code):
            line = code[:m.start()].count("\n") + 1
            problems.append(f"dead API in code line {line}: {label}")

    # Tofu bracket in figure text (code cells only; markdown renders fine).
    for m in TOFU.finditer(code):
        line = code[:m.start()].count("\n") + 1
        problems.append(f"literal U+27E8/9 in code line {line}: use grid.ket() or mathtext")

    # Mojibake, in the generator AND in every cell of the notebook. Unlike the
    # checks above this one covers markdown too, because prose em dashes are
    # exactly what a bad cp1252 round-trip corrupts first.
    scan = []
    if gen.exists():
        scan.append((f"gen_{nb_id}.py", gen.read_text(encoding="utf-8", errors="replace")))
    if nb is not None:
        scan.append((f"{nb_id} notebook",
                     "\n".join(c.source for c in nb.cells)))
    for where, text in scan:
        seen = {m.group(0) for m in MOJIBAKE.finditer(text)}
        if seen:
            problems.append(
                f"mojibake in {where}: {sorted(seen)} -- a UTF-8 file was read as "
                f"cp1252 (do not edit these through PowerShell text pipelines)")

    # Cross-notebook links must resolve on disk. Relative to notebooks/, since
    # that is where the .ipynb lives and how GitHub resolves the href.
    if nb is not None:
        prose = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
        for m in NB_LINK.finditer(prose):
            target = m.group(1)
            if not (NB_DIR / target).exists():
                problems.append(f"dead notebook link: {target} does not exist")

    if nb is not None:
        if "seed" not in code and "default_rng" not in code:
            problems.append("no seeding found - determinism is mandatory")
        if "style.use()" not in code:
            problems.append("missing `style.use()` - notebook will not match the repo look")
        n_out = sum(1 for c in nb.cells
                    if c.cell_type == "code" and c.get("outputs"))
        n_img = sum(1 for c in nb.cells if c.cell_type == "code"
                    for o in c.get("outputs", [])
                    if "image/png" in o.get("data", {}))
        if n_out == 0:
            problems.append("no executed outputs - notebook must ship executed")
        if n_img == 0:
            problems.append("no figures - this repo is about visualization")
        for c in nb.cells:
            for o in c.get("outputs", []):
                if o.get("output_type") == "error":
                    problems.append(f"cell raised {o.get('ename')}: {o.get('evalue')}")

        size = path.stat().st_size
        if size > MAX_BYTES:
            problems.append(f"{size/1024:.0f} KB exceeds the {MAX_BYTES//1024} KB limit")

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="*", help="notebook ids, e.g. A03 B19")
    ap.add_argument("--lint-only", action="store_true")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    ids = args.ids or notebook_ids()
    if not ids:
        sh("no generators found in tools/gen/")
        return 0

    failures = {}
    for nb_id in ids:
        t0 = time.time()
        sh(f"=== {nb_id}")
        if not args.lint_only:
            ok, msg = generate(nb_id)
            if not ok:
                sh(f"  GENERATE FAILED\n{msg}")
                failures[nb_id] = "generate"
                continue
            sh("  generated")
            if not args.no_exec:
                ok, msg = execute(nb_id, args.timeout)
                if not ok:
                    sh(f"  EXECUTE FAILED\n{msg}")
                    failures[nb_id] = "execute"
                    continue
                sh(f"  executed ({time.time()-t0:.0f}s)")

        problems = lint(nb_id)
        path = notebook_path(nb_id)
        if path is not None:
            kb = path.stat().st_size / 1024
            flag = "  <-- large" if kb * 1024 > WARN_BYTES else ""
            sh(f"  {path.name}  {kb:.0f} KB{flag}")
        if problems:
            for p in problems:
                sh(f"  LINT: {p}")
            failures[nb_id] = "lint"
        else:
            sh("  lint clean")

    sh("")
    if failures:
        sh(f"FAILED: {', '.join(f'{k} ({v})' for k, v in failures.items())}")
        return 1
    sh(f"ALL GREEN ({len(ids)} notebook(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
