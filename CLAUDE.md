# Repo conventions — read before writing any notebook

This repo teaches quantum computing twice: once by building it from scratch in
NumPy (Track A), once with the Qiskit SDK (Track B). Every notebook ships
executed, with outputs committed, so it renders on GitHub.

---

## 1. DO NOT WRITE these APIs — they are REMOVED in Qiskit 2.x

Most quantum tutorials on the internet — and therefore most of what a language
model has memorised — target Qiskit 0.x/1.x. **Every item below raises
`ImportError` or `AttributeError` on the pinned Qiskit 2.5.1.** This is the
single highest-risk failure mode in this project.

| Never write | Write instead |
|---|---|
| `from qiskit import execute` / `execute(qc, backend)` | `backend.run(transpile(qc, backend))` or a primitive |
| `from qiskit import Aer`, `Aer.get_backend('qasm_simulator')` | `from qiskit_aer import AerSimulator` |
| `from qiskit.providers.aer import ...` | `from qiskit_aer import ...` |
| `qiskit.opflow` (`PauliSumOp`, `StateFn`, `I/X/Z` operators) | `qiskit.quantum_info.SparsePauliOp`, `Statevector` |
| `qiskit.algorithms` (`VQE`, `QAOA`, `Grover`, `Shor`, `COBYLA`, `SPSA`) | hand-roll on `EstimatorV2` + `scipy.optimize.minimize` |
| `qiskit.utils.QuantumInstance` | primitives |
| `from qiskit.primitives import Sampler, Estimator` (V1) | `StatevectorSampler`, `StatevectorEstimator` (V2) |
| `BackendSampler`, `BackendEstimator` | `BackendSamplerV2`, `BackendEstimatorV2` |
| **`.c_if(creg, val)`**, `Instruction.condition` | `with qc.if_test((creg, val)): ...` |
| `qiskit.pulse`, anything pulse-related | (removed entirely) |
| `qiskit.assemble`, `Qobj` | (removed entirely) |
| `BackendV1`, `qiskit.providers.models`, V1 fake backends | `BackendV2` |
| `QuantumCircuit.qasm()` | `qiskit.qasm2.dumps(qc)` / `qasm3.dumps(qc)` |
| `qc.calibrations`, `qc.add_calibration()`, `Instruction.duration` | (removed entirely) |
| `ASAPSchedule`, `ALAPSchedule`, `DynamicalDecoupling` | (removed entirely) |
| `qiskit.circuit.classicalfunction`, old `PhaseOracle` | `PhaseOracleGate`, `BitFlipOracleGate` |

Prefer the **function forms** from `qiskit.circuit.library` — `efficient_su2()`,
`n_local()`, `real_amplitudes()`, `zz_feature_map()`, `grover_operator()`,
`phase_estimation()`, and `QFTGate`. The old `BlueprintCircuit` classes still
exist in 2.5 but transpile worse and are on their way out.

`transpile()` is **not** deprecated — use it for the simple path, and
`generate_preset_pass_manager` when the pass pipeline is the subject.

**`qiskit-algorithms` is not installed and must not be added.** It was spun out
of the SDK in 1.0 and is no longer supported by IBM. C28/C29 hand-roll VQE and
QAOA, which teaches better anyway.

## 2. Two Qiskit traps that bite silently

**Primitive result access goes through the classical register's NAME.**
`measure_all()` creates a register called `meas`, so it is
`result[0].data.meas.get_counts()`. This repo standardises on an explicit
`ClassicalRegister(n, "c")`, so it is `result[0].data.c.get_counts()`. Using the
wrong attribute is an `AttributeError`, not a wrong number — but only at runtime.

**Aer's two primitives seed differently.** `SamplerV2(seed=...)` works;
`EstimatorV2` takes **no** `seed` argument and must be seeded with
`options={"backend_options": {"seed_simulator": N}}`. Go through
`qviz.backends.sampler()` / `.estimator()` and you get this right for free.

## 3. Endianness — decided once, for the whole repo

**Qiskit is little-endian and Track A matches it.** Statevector index `i` maps to
the bitstring where **qubit 0 is the RIGHTMOST character**. `X` on qubit 0 of a
2-qubit register gives index 1, printed `|01>`.

A from-scratch simulator written the textbook way (`np.kron(q0, q1)`) comes out
big-endian and will silently disagree with every Track B result. So Track A
builds operators little-endian throughout, and A05 spends a section on why.

This is what makes the repo's regression test possible:

```python
assert np.allclose(my_state, Statevector(qc).data)
```

Every Track B notebook closes with an assertion of this shape against the Track A
result it mirrors.

## 4. Determinism is mandatory

Every notebook seeds everything:

```python
from qviz import backends
SEED = backends.seed_for("A03")     # deterministic per notebook
rng = np.random.default_rng(SEED)
```

plus `seed_simulator=`, `seed_transpiler=`, and the primitive seeds. Without this
every rebuild produces a different diff and review becomes impossible.

## 5. `qviz/` may only draw

`qviz` is the shared **drawing** layer and is **frozen** during notebook
authoring — do not edit it. If you need a change, say so in your final message
and the orchestrator will apply it between waves. That is how `signed_bars`,
`matrix(part="nonzero")`, `sphere(zoom=)` and the `ylabel=`/`tick_every=`
pass-throughs on `amp_bars`/`prob_bars` got there — check whether the helper you
want already exists before writing it inline.

The hard rule: **`qviz` must never compute quantum mechanics for a Track A
notebook.** If A03 imported `apply_gate` from a helper, the notebook would stop
teaching what is under the hood, which is the entire point. Track A derives its
statevectors, gates, measurement and channels inline, every time. Track B may of
course use `qiskit.quantum_info` — that IS its subject.

## 6. Notebook mechanics

- **Author a generator, not the `.ipynb`.** Write `tools/gen/gen_A03.py` using
  `nbformat`; `tools/build.py` runs it and executes the result. Hand-written
  notebook JSON breaks on LaTeX escaping and is unreviewable in a diff.
- Generators must be **idempotent** — running twice gives the same notebook.
- **Write cell sources as raw strings** (`r'''...'''`) rather than escaping
  backslashes. `r"$\pi$"` beats `"$\\\\pi$"`; the emitted notebook is identical
  and the generator stays reviewable. A01 predates this and still uses the
  doubled form — follow A09/A10, not A01, on this one point.
- **Never edit a generator through a PowerShell text pipeline.** `Get-Content
  -Raw` / `Set-Content` in PS 5.1 reads a BOM-less UTF-8 file as ANSI and
  silently corrupts every non-ASCII character — em dashes become `â€”`. The
  result is still valid Python, so it executes and ships. Use the Edit tool.
  `tools/build.py` now fails on the resulting mojibake, but do not rely on it.
- First code cell of every notebook: `from qviz import style; style.use()`.
- **No emoji in `print()`.** Windows stdout is cp1252 and will raise
  `UnicodeEncodeError`. Emoji in markdown cells is fine.
- **No literal `⟩` (U+27E9) in figure text** — missing from Segoe UI, renders as
  a tofu box. Use `qviz.grid.ket("01")` or mathtext `$|01\rangle$`. In markdown
  cells the Unicode character is fine.
- Figures: `dpi=110`. The gate nudges at 700 KB per notebook and hard-fails at
  2 MB. Going over the nudge is acceptable for a genuinely 3-D-heavy notebook
  (A02, A12) — say why in the commit message rather than cutting a figure or
  flattening the render to squeeze under it.
  `rasterized=True` on scatter plots over ~5k points. One figure with subplots,
  never 16 separate figures. `plt.close(fig)` inside loops.
- Never print a full statevector past a few qubits — use `qviz.grid.show_state`.

## 7. What a notebook must contain

Follow `notebooks/A01_*.ipynb` (Track A) or `notebooks/B13_*.ipynb` (Track B) as
the template. Structure:

1. **Title + a plain-language on-ramp.** Lead with intuition a beginner can
   follow. Rigor comes after, and is clearly marked as a deep dive.
2. **Derivation from first principles**, in the notebook, with the code visible.
3. **The visualizations** — this repo exists for them. Every notebook carries
   several substantive figures. A bar chart of measurement counts is the *least*
   interesting thing you can draw; use it only where it is genuinely the point.
4. **Honest limits.** Where does this break? What does the picture hide? Say so.
   (e.g. Grover's over-rotation, `plot_bloch_multivector` lying about
   entanglement, QEC below threshold making things worse.)
5. **A Checkpoint cell** that asserts the result — analytic for Track A, plus
   `np.allclose` against the paired Track A result for Track B.

Write prose a curious reader actually reads. Explain *why*, not just *what*.
