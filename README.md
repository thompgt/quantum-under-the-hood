# Quantum Under The Hood

**Learn quantum computing by seeing it.** Thirty Jupyter notebooks that teach every
concept twice — once by building it from scratch in NumPy, once with the real
Qiskit SDK — with the visualization doing the explaining.

Every notebook is committed **already executed**, so you can read the whole thing
on GitHub without installing anything.

```
Track A  —  under the hood   pure NumPy. Qiskit is never imported.
Track B  —  the Qiskit SDK   the same ideas, the tool you'd actually use.
Track C  —  capstones        error correction, VQE, QAOA, transpilation.
```

Why both? From-scratch code alone doesn't transfer to real work. SDK-only
tutorials hide the linear algebra that makes the intuition click. Doing it twice
is the point: Track B notebooks end by asserting `np.allclose` against the Track A
result they mirror, so the two halves keep each other honest.

---

## Skills & topics covered

<p>
<img alt="Qiskit 2.5" src="https://img.shields.io/badge/Qiskit-2.5-6929C4?style=for-the-badge&logo=qiskit&logoColor=white">
<img alt="Qiskit Aer" src="https://img.shields.io/badge/Qiskit_Aer-0.17-6929C4?style=for-the-badge">
<img alt="Python 3.13" src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-2.3-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img alt="SciPy" src="https://img.shields.io/badge/SciPy-1.16-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white">
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-3.10-11557C?style=for-the-badge">
<img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-executed-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
</p>

<table>
<tr><td width="33%" valign="top">

**⚛️ Quantum foundations**

```text
Complex amplitudes
Superposition
Global vs relative phase
Born rule
Bloch sphere
Measurement & collapse
Basis choice
Tensor products
Endianness
Entanglement
No-cloning
```
</td><td width="33%" valign="top">

**🔀 Gates & circuits**

```text
Unitarity
Pauli matrices
Rotations exp(-iθP/2)
Hadamard
CNOT / CZ / SWAP
Toffoli (CCX)
Controlled-U
Multi-controlled gates
Circuit depth & DAGs
Basis translation
Transpilation
```
</td><td width="33%" valign="top">

**🧮 Algorithms**

```text
Interference
Phase kickback
Deutsch-Jozsa
Bernstein-Vazirani
Grover search
Amplitude amplification
Quantum Fourier Transform
Phase estimation
Shor order-finding
Teleportation
Superdense coding
```
</td></tr>
<tr><td valign="top">

**📉 Noise & error**

```text
Density matrices
Mixed states
Partial trace
Purity & von Neumann entropy
Kraus operators
CPTP maps
Depolarizing channel
Amplitude/phase damping
T1 / T2 decoherence
Pauli transfer matrices
Readout error
```
</td><td valign="top">

**🛡️ Error correction & NISQ**

```text
Bit-flip & phase-flip codes
Shor 9-qubit code
Syndrome measurement
Stabilizers
Break-even threshold
Variational circuits (VQE)
QAOA / MaxCut
Barren plateaus
Ansatz design
Approximation ratio
```
</td><td valign="top">

**🛠️ Engineering**

```text
Qiskit 2.x primitives (V2)
SamplerV2 / EstimatorV2
qiskit.quantum_info
AerSimulator & noise models
PassManager & opt levels
Dynamic circuits (if_test)
Statevector simulation
Reproducible seeding
nbformat notebook generation
Headless nbconvert CI
Perceptual colormap design
```
</td></tr>
</table>

**Bell inequalities & CHSH**, **Haar-random sampling**, **Schmidt rank**,
**continued fractions**, and **the exponential simulation wall** get their own
treatment too — see the notebook index below.

---

## The visual language

Four ideas recur across all thirty notebooks.

**Amplitude bars.** A complex vector drawn in one picture: bar height is
`|amplitude|`, bar colour is the phase, and a little clock hand on top repeats
the phase as an angle. Interference stops being algebra and becomes something you
watch happen.

**The phase colormap is generated, not borrowed.** Phase is cyclic, so it needs a
colormap whose ends meet — and phase 0 is the most common value in a statevector,
so it must not land on a washed-out colour. `twilight` fails that second test. We
sweep hue at **constant OKLCH lightness** instead: measured lightness spread
`0.0000`, every phase clearing 3:1 contrast, no phase privileged over another.

**Phase always ships with a second, colour-independent encoding.** No cyclic
colormap can be colourblind-safe — colour-vision deficiency collapses the hue
circle onto a line, so any map that returns to its start must self-intersect. Ours
is strong on the case that matters most (phase 0 vs π, a sign flip, separates at
ΔE 11.5) but 0 vs π/2 is confusable for deutan viewers. Hence the clock hands.

**Animation as a static frame grid.** A notebook with committed outputs can't
animate, so time evolution is a grid of small panels with a fading trail.

---

## The notebooks

### Track A — Under the Hood
*Pure NumPy and matplotlib. Nothing is a black box; every number is one you could
derive with a pen.*

| # | Notebook | What you'll see |
|---|---|---|
| A01 | [Complex Amplitudes and the Qubit State](notebooks/A01_Complex_Amplitudes.ipynb) | Amplitudes as arrows in ℂ; the signature phase-hue bars; why global phase is invisible but relative phase is everything |
| A02 | [The Bloch Sphere, Built From Scratch](notebooks/A02_Bloch_Sphere.ipynb) | A hand-rolled sphere; Haar-random states vs. naive (θ,φ) sampling and the pole-clustering it causes |
| A03 | [Single-Qubit Gates as 2×2 Unitaries](notebooks/A03_Single_Qubit_Gates.ipynb) | Gate matrices as heatmaps; Bloch trajectories as frame grids; a commutator heatmap showing what doesn't commute |
| A04 | [Measurement, Sampling, and Collapse](notebooks/A04_Measurement.ipynb) | Shot-convergence fan chart with the ±1/√N envelope; collapse drawn on three spheres |
| A05 | [Tensor Products and Multi-Qubit States](notebooks/A05_Tensor_Products.ipynb) | The `kron` block structure made visible; the endianness Rosetta table; where your laptop dies |
| A06 | [Two-Qubit Gates and Controlled Operations](notebooks/A06_Two_Qubit_Gates.ipynb) | Block structure of CNOT/CZ/SWAP; the 8×8 Toffoli; gates as permutation graphs |
| A07 | [A Tiny Statevector Simulator](notebooks/A07_Statevector_Simulator.ipynb) | Reshape-and-tensordot vs. naive kron, benchmarked; the 2ⁿ-vs-4ⁿ wall |
| A08 | [Interference and Phase Kickback](notebooks/A08_Interference_Phase_Kickback.ipynb) | Amplitudes added head-to-tail in ℂ; the interference waterfall |
| A09 | [Deutsch–Jozsa and Bernstein–Vazirani](notebooks/A09_Deutsch_Jozsa_Bernstein_Vazirani.ipynb) | Amplitude-evolution filmstrips; 16 secrets recovered as one binary image |
| A10 | [Grover From Scratch](notebooks/A10_Grover_From_Scratch.ipynb) | Inversion about the mean, step by step; the **over-rotation dip** most tutorials hide |
| A11 | [QFT and Phase Estimation From Scratch](notebooks/A11_QFT_Phase_Estimation.ipynb) | The QFT matrix as a phase swirl; exact vs. inexact phase and spectral leakage |
| A12 | [Density Matrices, Mixed States, Noise Channels](notebooks/A12_Density_Matrices_Noise.ipynb) | The Bloch *ball*; a channel drawn as a deformation of that ball |

### Track B — The Qiskit SDK

| # | Notebook | What you'll see |
|---|---|---|
| B13 | [Qiskit Tour: Circuits, Statevector, Operator](notebooks/B13_Qiskit_Tour.ipynb) | One Bell state in five renderings — the repo's visualization index card |
| B14 | [The Visualization Suite](notebooks/B14_Visualization_Suite.ipynb) | Why `plot_bloch_multivector` **lies about entanglement**, shown side by side |
| B15 | [Single-Qubit Gates in Qiskit](notebooks/B15_Single_Qubit_Gates_Qiskit.ipynb) | A 12-gate contact sheet; an `Operator.equiv` heatmap that discovers gate identities |
| B16 | [Measurement with SamplerV2](notebooks/B16_Measurement_SamplerV2.ipynb) | The raw shot record as a bitmap — the actual randomness, which tutorials never show |
| B17 | [Multi-Qubit Circuits and Ordering](notebooks/B17_Multi_Qubit_Ordering.ipynb) | The endianness trap, with the wrong answer struck through |
| B18 | [Entanglement and Bell States](notebooks/B18_Entanglement_Bell_States.ipynb) | An entanglement dial: entropy, concurrence and Bloch-vector length as one knob turns |
| B19 | Bell Inequalities and CHSH | The money shot — S rising above the classical bound of 2 toward 2√2, then dying under noise |
| B20 | Interference and Phase Kickback in Qiskit | A Ramsey interferogram, and its fringe contrast collapsing under dephasing |
| B21 | Deutsch–Jozsa and Bernstein–Vazirani in Qiskit | Oracles as circuits; the one-query punchline |
| B22 | Grover in Qiskit | The Grover heatmap: iteration × basis state, marked column brightening |
| B23 | QFT and Phase Estimation in Qiskit | Qiskit's QFT beside the hand-built one, with a 1e-16 diff panel |
| B24 | Shor's Order Finding (N = 15, 21) | Honest success rates over every base; and how far RSA-2048 really is |
| B25 | Teleportation and Superdense Coding | Fidelity over 20 random states; dynamic circuits with `if_test` |
| B26 | Density Matrices and Noise in Qiskit | Pauli transfer matrices — the cleanest picture of what a channel *is*; T1/T2 decay fits |

### Track C — Capstones

| # | Notebook | What you'll see |
|---|---|---|
| C27 | Error Correction: Bit-Flip, Phase-Flip, Shor-9 | Logical vs. physical error rate, with the break-even crossing and the region where **QEC makes things worse** |
| C28 | Variational Circuits and VQE (H₂) | The H₂ dissociation curve against exact diagonalization; barren plateaus, measured |
| C29 | QAOA on MaxCut | Probability mass migrating toward good cuts as depth grows |
| C30 | Transpilation, Optimization, Simulator Limits | SWAP routing on the coupling graph; why Clifford circuits simulate in polynomial time; *50 qubits = 16 PB* |

---

## Running it yourself

You don't need to — the notebooks are committed with outputs. But if you want to
change things:

```bash
git clone https://github.com/thompgt/quantum-under-the-hood
cd quantum-under-the-hood

python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
pip install -e .                # the qviz drawing helpers

python -m ipykernel install --user --name quth
python tools/smoke_gate.py      # verifies the Qiskit API surface actually works
```

Then open any notebook, or rebuild from the command line:

```bash
python tools/build.py           # regenerate + execute + lint + size-audit everything
python tools/build.py A03 B19   # just these
python tools/peek.py A03        # dump a notebook's figures to PNG to look at them
```

**No IBM Quantum account is needed.** Everything runs on a local simulator.

### How notebooks are built

Each notebook has a generator at `tools/gen/gen_<ID>.py` that emits the `.ipynb`
via `nbformat`; `tools/build.py` runs it, executes it, and gates it. **The
generator is the source of truth** — edit that, not the notebook. Hand-written
notebook JSON breaks on LaTeX escaping and is unreviewable in a diff.

The build gate fails on: any dead Qiskit API, a notebook with no figures, an
unseeded RNG, a cell that raised, or a file over 2 MB.

---

## Two decisions worth knowing about

**This targets Qiskit 2.5, and Qiskit 2.x removed a lot.** `execute()`,
`Aer.get_backend()`, `.c_if()`, `qiskit.algorithms`, `qiskit.opflow` and
Primitives V1 are all gone. Nearly every quantum tutorial you'll find online — and
most of what an LLM has memorised — is written against those dead APIs. The
current idioms are in [`CLAUDE.md`](CLAUDE.md), and `tools/build.py` enforces them
by regex so nothing stale can ship.

**Everything is little-endian, including Track A.** Qiskit indexes statevectors
with qubit 0 as the *rightmost* bit. A from-scratch simulator written the textbook
way (`np.kron(q0, q1)`) comes out big-endian and silently disagrees with every
Qiskit result. So Track A is built to match, and A05 explains the choice. That's
what makes `assert np.allclose(mine, Statevector(qc).data)` a valid cross-track
regression test.

## Layout

```
notebooks/     A01..A12, B13..B26, C27..C30 — executed, outputs committed
qviz/          shared DRAWING layer (style, bloch, grid, backends)
tools/gen/     one generator per notebook — the source of truth
tools/         build.py, smoke_gate.py, check_style.py, peek.py
```

`qviz/` may only *draw*. It never computes quantum mechanics on behalf of a Track
A notebook — if it did, the notebook would stop teaching what's under the hood,
which is the whole point.

## License

MIT.
