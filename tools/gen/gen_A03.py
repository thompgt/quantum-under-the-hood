"""Generator for A03 - Single-Qubit Gates as 2x2 Unitaries.

Track A. Pure NumPy; Qiskit is never imported. Structure follows gen_A01.py:
markdown on-ramp -> derivation in visible code -> substantive figures ->
honest limits -> Checkpoint assertions.

Thesis: every single-qubit gate is a rotation of the Bloch sphere, and the 2x2
matrix is just the algebra for that rotation.
"""

import sys
from pathlib import Path as _Path

import nbformat as nbf

sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
from nbmeta import notebook_metadata  # noqa: E402

NB_ID = "A03"
TITLE = "Single-Qubit Gates as 2x2 Unitaries"
OUT = f"notebooks/{NB_ID}_Single_Qubit_Gates.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# NOTE: plain r-strings everywhere. Markdown is full of LaTeX braces
# (\frac{a}{b}, \tfrac{\theta}{2}) and an f-string would read them as
# interpolation slots. Code cells are r-strings too, so that "\n" inside a
# print() lands in the notebook as the two characters the reader should see.
cells.append(md(r"""# A03 — Single-Qubit Gates as 2x2 Unitaries

**Track A — under the hood.** Pure NumPy. Qiskit is not imported anywhere in this
notebook; every matrix here is one you can write down with a pen.

---

## The one-sentence version

A02 showed that a qubit is a point on a sphere. This notebook shows that **every
single-qubit gate is a rotation of that sphere** — and that the 2x2 complex matrix
you see in every textbook is nothing more exotic than the algebra for the
rotation.

That is the whole notebook. `X` is a half-turn about the $x$-axis. `Z` is a
half-turn about the $z$-axis. `T` is an eighth-turn about the $z$-axis. `H`, which
gets described as a "50/50 splitter", is really a half-turn about a *tilted* axis
— and once you see the axis, the fact that $H^2 = I$ stops being a curious
identity and becomes obvious: turn twice about any axis by 180°, you're back.

## Why rotations, and not something more general?

Because a gate is not allowed to change the *length* of the state vector.

Probabilities must keep summing to 1 — before the gate, after the gate, after ten
thousand gates. A qubit is a unit vector in $\mathbb{C}^2$, and the only linear
maps that send every unit vector to a unit vector are the **unitary** ones:

$$U^\dagger U = I.$$

Unit vectors in $\mathbb{C}^2$ live on the Bloch sphere. Length-preserving linear
maps of a sphere are rotations. So the chain is:

$$\text{probability conserved} \;\Rightarrow\; \text{unitary} \;\Rightarrow\;
\text{rotation of the Bloch sphere}.$$

Everything below is that sentence, made concrete and then verified numerically."""))

cells.append(code(r"""from qviz import backends, bloch, grid, style

import matplotlib.pyplot as plt
import numpy as np

style.use()

SEED = backends.seed_for("A03")
rng = np.random.default_rng(SEED)
print("seed:", SEED)"""))

# ------------------------------------------------------------------ unitarity
cells.append(md(r"""## 1. Unitarity, derived rather than asserted

Suppose a gate is some linear map $M$, so $|\psi\rangle \mapsto M|\psi\rangle$. The
new state's squared norm is

$$\langle \psi | M^\dagger M | \psi \rangle.$$

We need that to equal $\langle\psi|\psi\rangle = 1$ **for every** $|\psi\rangle$, not
just for a lucky few. A bilinear form that agrees with the identity on every
vector *is* the identity, so the requirement collapses to

$$M^\dagger M = I,$$

which is the definition of unitary. Three immediate consequences, all of which we
will lean on:

- $M^{-1} = M^\dagger$ — **every gate is reversible**, and undoing it costs nothing
  more than a conjugate transpose. There is no quantum `AND` gate, because `AND`
  throws information away.
- $|\det M| = 1$ — unitaries preserve volume as well as length.
- Unitaries are closed under multiplication, so **a circuit is a gate**.

Let's build the standard gate set and check all of this numerically."""))

cells.append(code(r'''I2 = np.eye(2, dtype=complex)

# The three Pauli matrices. Everything else in this notebook is built from them.
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

H = (X + Z) / np.sqrt(2)                          # note: H is literally (X+Z)/sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)    # quarter turn about z
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)   # eighth turn


def dag(M):
    """Conjugate transpose."""
    return np.asarray(M).conj().T


def is_unitary(M, tol=1e-12):
    M = np.asarray(M)
    return np.allclose(dag(M) @ M, np.eye(M.shape[0]), atol=tol)


GATES = [("I", I2), ("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T)]

print(f"{'gate':>5} {'unitary?':>9} {'||U*U - I||':>13} {'det U':>22}")
for name, M in GATES:
    err = np.linalg.norm(dag(M) @ M - I2)
    d = np.linalg.det(M)
    print(f"{name:>5} {str(is_unitary(M)):>9} {err:13.2e}   {d.real:+.4f}{d.imag:+.4f}j")'''))

cells.append(md(r"""Note the determinants: $\det X = \det Y = \det Z = -1$, $\det H = -1$, but
$\det S = i$ and $\det T = e^{i\pi/4}$. All have modulus 1, as promised. The
*phase* of the determinant is the global phase the gate carries around, and
Section 7 is about why that phase is simultaneously invisible and important.

### Figure 1 — what unitarity buys you, and what its absence costs

Left: apply a matrix to a state over and over and watch the norm. A unitary holds
it at exactly 1 forever. A matrix that is merely *close* to unitary drifts
geometrically — $0.9H$ has bled away 88% of the state after twenty gates, and
$1.1H$ has inflated it sevenfold. There is no "approximately conserved
probability": the error compounds, and a circuit is thousands of gates long.

Right: the unitarity defect $\|U^\dagger U - I\|_F$ for the standard gates and for
three impostors, on a log axis. The real gates sit at $10^{-16}$ or below —
floating-point dust — and the impostors sit fifteen orders of magnitude higher.
This is not a close call, which is exactly why it makes a good assertion."""))

cells.append(code(r'''# Three impostors: a shrunken and a stretched Hadamard, and a random matrix.
SHRUNK, STRETCH = 0.9 * H, 1.1 * H
RANDM = (rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))) / 2

psi0 = np.array([1, 0], dtype=complex)
ks = np.arange(0, 21)

fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))

for name, M, col in [("H  (unitary)", H, style.BLUE),
                     ("1.1 H", STRETCH, style.MAGENTA),
                     ("0.9 H", SHRUNK, style.ORANGE),
                     ("random 2x2", RANDM, style.AQUA)]:
    v = psi0.copy()
    norms = []
    for _ in ks:
        norms.append(np.linalg.norm(v))
        v = M @ v
    axes[0].plot(ks, norms, color=col, label=name, marker="o", ms=3.2)

axes[0].axhline(1.0, color=style.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
axes[0].set_yscale("log")
axes[0].set_ylim(3e-3, 3e2)
axes[0].set_xlim(-0.6, 24.5)
axes[0].set_xticks([0, 5, 10, 15, 20])
axes[0].set_xlabel("gates applied")
axes[0].set_ylabel(r"$\||\psi\rangle\|$")
axes[0].set_title("Only a unitary conserves probability", loc="left", fontsize=10)
axes[0].legend(loc="lower left")
for lab, y in [("blows up", 1.1 ** 20), ("evaporates", 0.9 ** 20)]:
    axes[0].text(20.6, y, lab, fontsize=8, color=style.MUTED, va="center")

cands = GATES + [("0.9H", SHRUNK), ("1.1H", STRETCH), ("rand", RANDM)]
FLOOR = 3e-17     # exact-zero defects are drawn on the floor so the bar is seen
defects = [max(np.linalg.norm(dag(M) @ M - I2), FLOOR) for _, M in cands]
cols = [style.BLUE] * len(GATES) + [style.RED] * 3
axes[1].bar(range(len(cands)), defects, color=cols, width=0.7, zorder=3)
axes[1].set_yscale("log")
axes[1].set_ylim(1e-18, 1e2)
axes[1].axhline(1e-12, color=style.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=2)
axes[1].text(0.02, 2e-12, "numerical zero", fontsize=8, color=style.MUTED,
             va="bottom")
axes[1].set_xticks(range(len(cands)))
axes[1].set_xticklabels([n for n, _ in cands], fontsize=8.5)
axes[1].set_ylabel(r"$\|U^\dagger U - I\|_F$")
axes[1].set_title("Unitary (blue) vs not (red)", loc="left", fontsize=10)
plt.show()'''))

# --------------------------------------------------------------------- Paulis
cells.append(md(r"""## 2. The Pauli matrices are the whole alphabet

$$X=\begin{pmatrix}0&1\\1&0\end{pmatrix},\quad
Y=\begin{pmatrix}0&-i\\i&0\end{pmatrix},\quad
Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}.$$

They are unusual in being **both Hermitian and unitary** ($P = P^\dagger$ *and*
$P^\dagger P = I$), which forces $P^2 = I$ and hence eigenvalues $\pm 1$. That
double life is why they play two roles at once: as *observables* (A04 measures
them) and as *gates* (here).

Their algebra is small enough to memorise:

$$X^2 = Y^2 = Z^2 = I, \qquad XY = iZ,\ YZ = iX,\ ZX = iY,$$

and, crucially, **distinct Paulis anticommute**: $PQ = -QP$. That single fact is
the source of nearly every non-commutation result later in the repo, and — as
Section 6 will show — it also produces a nice trap.

Together with $I$ they form a basis for all $2\times 2$ complex matrices, so
*every* single-qubit gate can be written $aI + bX + cY + dZ$. That is the reason
this three-element alphabet is enough."""))

cells.append(code(r'''PAULI = [("X", X), ("Y", Y), ("Z", Z)]

print("Hermitian and unitary at the same time:")
for name, P in PAULI:
    print(f"  {name}:  P = P-dagger -> {np.allclose(P, dag(P))}"
          f"   P@P = I -> {np.allclose(P @ P, I2)}"
          f"   eigenvalues {np.round(np.linalg.eigvalsh(P), 6)}")

print("\nProducts (cyclic):")
for (a, A), (b, B), (c, C) in [(PAULI[0], PAULI[1], PAULI[2]),
                               (PAULI[1], PAULI[2], PAULI[0]),
                               (PAULI[2], PAULI[0], PAULI[1])]:
    print(f"  {a}{b} = i{c} -> {np.allclose(A @ B, 1j * C)}")

print("\nAnticommutation, {P,Q} = PQ + QP = 0 for P != Q:")
for i, (a, A) in enumerate(PAULI):
    for b, B in PAULI[i + 1:]:
        print(f"  {a}{b} + {b}{a} = 0 -> {np.allclose(A @ B + B @ A, 0)}")


def pauli_coeffs(M):
    """Decompose a 2x2 matrix as a*I + b*X + c*Y + d*Z.  Coefficient of P is
    Tr(P M)/2, because Tr(P Q) = 2*delta_PQ."""
    return np.array([np.trace(B @ M) / 2 for B in (I2, X, Y, Z)])


print("\nEvery 2x2 gate is a combination of I, X, Y, Z:")
for name, M in [("H", H), ("S", S), ("T", T)]:
    a, b, c, d = np.round(pauli_coeffs(M), 4)
    print(f"  {name} = ({a})I + ({b})X + ({c})Y + ({d})Z"
          f"   reconstructs -> "
          f"{np.allclose(a * I2 + b * X + c * Y + d * Z, M)}")'''))

cells.append(md(r"""$H = \tfrac{1}{\sqrt2}(X + Z)$ falls straight out — equal parts $X$ and $Z$, no
$I$ and no $Y$. Hold onto that; in Section 4 it turns into the tilted rotation
axis, and the "$H$ is a coin flip" story quietly dies.

### Figure 2 — the gate set, as pictures

A $2\times 2$ complex matrix is four complex numbers, which is two $2\times 2$
real grids: the real part and the imaginary part. Every gate in the standard set,
drawn on one shared colour scale so the panels are directly comparable. Blue is
negative, red is positive, pale is zero.

Reading the grid: an entirely blank imaginary row means a **real** gate. $Y$ is
the only Pauli with imaginary entries; $S$ and $T$ push their phase into the
bottom-right corner and nowhere else; $R_x$ is the first gate whose real and
imaginary parts are *both* non-trivial, which is what lets it interpolate rather
than jump."""))

cells.append(code(r'''def rot(P, theta):
    """exp(-i * theta * P / 2) for a Pauli P, in closed form.

    Because P*P = I, the exponential series splits into the cos and sin series:
        exp(-i t P) = cos(t) I - i sin(t) P.
    Section 3 checks this against a brute-force series expansion.
    """
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * np.asarray(P)


def Rx(t):
    return rot(X, t)


def Ry(t):
    return rot(Y, t)


def Rz(t):
    return rot(Z, t)


SHOW = [("H", H), ("X", X), ("Y", Y), ("Z", Z), ("S", S), ("T", T),
        (r"$R_x(\pi/3)$", Rx(np.pi / 3))]

fig, axes = plt.subplots(2, len(SHOW), figsize=(12.4, 4.1))
for j, (name, M) in enumerate(SHOW):
    for i, part in enumerate(("re", "im")):
        im = grid.matrix(axes[i, j], M, part=part, cbar=False, vmax=1.0,
                         fmt="{:+.2f}")
        axes[i, j].tick_params(labelsize=7.5)
        if j:
            axes[i, j].set_yticklabels([])
    axes[0, j].set_title(name, loc="center", fontsize=10, pad=6)

axes[0, 0].set_ylabel("real part", fontsize=9.5)
axes[1, 0].set_ylabel("imaginary part", fontsize=9.5)
cb = fig.colorbar(im, ax=axes, fraction=0.020, pad=0.012)
cb.outline.set_visible(False)
cb.set_ticks([-1, 0, 1])
cb.ax.tick_params(labelsize=7.5, color=style.MUTED, labelcolor=style.MUTED)
fig.suptitle("The standard single-qubit gate set, entry by entry",
             x=0.005, ha="left", fontsize=11.5)
plt.show()'''))

# ------------------------------------------------------------- exponentials
cells.append(md(r"""## 3. Rotations are matrix exponentials

Here is where the "gate = rotation" claim gets its algebra. Define

$$R_{\hat n}(\theta) \;=\; \exp\!\left(-\,\frac{i\theta}{2}\,\hat n \cdot \vec\sigma\right),
\qquad \hat n\cdot\vec\sigma = n_x X + n_y Y + n_z Z.$$

The matrix exponential means exactly what the scalar one does — the power series
$\sum_k M^k/k!$. It looks intimidating and then immediately isn't, because
$(\hat n\cdot\vec\sigma)^2 = I$ for any unit $\hat n$. So the even powers are all
$I$ and the odd powers are all $\hat n\cdot\vec\sigma$, the series splits into the
cosine series and the sine series, and

$$\boxed{\;R_{\hat n}(\theta) = \cos\tfrac{\theta}{2}\,I \;-\; i\sin\tfrac{\theta}{2}\,(\hat n\cdot\vec\sigma)\;}$$

Two things to notice before we verify it.

**The half-angle is not a typo.** $R_{\hat n}(2\pi) = -I$: a full $360°$ rotation
of the Bloch sphere returns the state to itself *times $-1$*. You need $720°$ to
get back to the same matrix. That factor of two is the difference between the
sphere of states and the group $SU(2)$ that acts on it, and it is real physics —
Figure 4 will walk into it.

**Why $\exp$ at all?** Because rotations compose by adding angles, and $\exp$ is
the function that turns addition into multiplication. $R_{\hat n}(\alpha)R_{\hat
n}(\beta) = R_{\hat n}(\alpha+\beta)$ falls out for free.

Let's not take the closed form on trust — compute the exponential the dumb way
and compare."""))

cells.append(code(r'''def expm_series(M, terms=80):
    """exp(M) by brute-force Taylor series. Slow, obvious, no library help."""
    M = np.asarray(M, dtype=complex)
    out = np.zeros_like(M)
    term = np.eye(M.shape[0], dtype=complex)
    for k in range(terms):
        out = out + term
        term = term @ M / (k + 1)
    return out


print("Rx(theta) == exp(-i theta X / 2)?   (series vs closed form)")
print(f"{'theta/pi':>9} {'max |series - closed|':>23} {'unitary?':>10}")
for t in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
    theta = t * np.pi
    series = expm_series(-1j * theta * X / 2)
    closed = Rx(theta)
    print(f"{t:9.2f} {np.abs(series - closed).max():23.2e} "
          f"{str(is_unitary(closed)):>10}")

print("\nThe same for a tilted axis n = (1,0,1)/sqrt(2):")
n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
n_sigma = n[0] * X + n[1] * Y + n[2] * Z
for t in [0.5, 1.0, 1.5]:
    theta = t * np.pi
    series = expm_series(-1j * theta * n_sigma / 2)
    closed = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * n_sigma
    print(f"  theta = {t:.2f} pi   max diff = {np.abs(series - closed).max():.2e}")

print("\nAngles add:  Rx(a) Rx(b) == Rx(a+b)?",
      np.allclose(Rx(0.7) @ Rx(1.3), Rx(2.0)))
print("Half-angle is real:  Rx(2 pi) =")
print(np.round(Rx(2 * np.pi).real, 12))
print("...which is -I, not I. A full turn costs a global phase of -1.")'''))

cells.append(md(r"""### The Pauli gates are half-turns, up to phase

$$R_x(\pi) = \cos\tfrac{\pi}{2}I - i\sin\tfrac{\pi}{2}X = -iX.$$

So $X$ *is* a $180°$ rotation about $\hat x$, wearing a global phase of $-i$ that
no measurement can see. Same story for $Y$, $Z$, and — with the tilted axis — for
$H$. Let's confirm the whole family at once."""))

cells.append(code(r'''def same_up_to_phase(A, B, tol=1e-10):
    """True if A = e^{i phi} B for some real phi.

    The overlap Tr(A-dagger B) equals e^{-i phi} ||B||^2 when the two differ
    only by a phase, so its argument hands us phi directly. (Comparing the
    largest entry of each matrix instead is tempting and wrong: |1| and
    |e^{i pi/4}| are the same number in exact arithmetic but not in floating
    point, so the two matrices can pick different reference entries.)
    """
    A, B = np.asarray(A), np.asarray(B)
    overlap = np.trace(dag(A) @ B)
    if abs(overlap) < tol:
        return False
    return np.allclose(A * np.exp(1j * np.angle(overlap)), B, atol=tol)


n_h = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)


def rot_axis(nvec, theta):
    """R_n(theta) = cos(theta/2) I - i sin(theta/2) (n . sigma)."""
    nvec = np.asarray(nvec, dtype=float)
    nvec = nvec / np.linalg.norm(nvec)
    ns = nvec[0] * X + nvec[1] * Y + nvec[2] * Z
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * ns


claims = [("X", X, Rx(np.pi), "180 deg about x"),
          ("Y", Y, Ry(np.pi), "180 deg about y"),
          ("Z", Z, Rz(np.pi), "180 deg about z"),
          ("S", S, Rz(np.pi / 2), " 90 deg about z"),
          ("T", T, Rz(np.pi / 4), " 45 deg about z"),
          ("H", H, rot_axis(n_h, np.pi), "180 deg about (x+z)/sqrt(2)")]

print(f"{'gate':>5}  {'is this rotation':<32} {'same up to global phase?':>25}")
for name, G, R, desc in claims:
    print(f"{name:>5}  {desc:<32} {str(same_up_to_phase(G, R)):>25}")

print("\nThe leftover phases (gate = e^{i phi} x rotation), which nothing"
      " can measure:")
for name, G, R, _ in claims:
    phi = -np.angle(np.trace(dag(G) @ R))
    print(f"  {name}: e^(i phi) with phi = {phi/np.pi:+.4f} pi"
          f"   -> {np.exp(1j*phi).real:+.4f}{np.exp(1j*phi).imag:+.4f}j")'''))

# ----------------------------------------------------------- bloch machinery
cells.append(md(r"""## 4. From matrix to rotation, explicitly

A02 built the Bloch vector of a state as the expectation values of the three
Paulis:

$$\vec r(\psi) = \big(\langle\psi|X|\psi\rangle,\ \langle\psi|Y|\psi\rangle,\ \langle\psi|Z|\psi\rangle\big),$$

three real numbers on the unit sphere. Now ask what a gate does to $\vec r$. After
the gate the state is $U|\psi\rangle$, so

$$r'_i = \langle\psi|U^\dagger P_i U|\psi\rangle.$$

Expand $U^\dagger P_i U$ in the Pauli basis — legal, because $\{I,X,Y,Z\}$ spans
everything — and the $I$ component vanishes because $\mathrm{Tr}(U^\dagger P_i U) =
\mathrm{Tr}(P_i) = 0$. What's left is a *linear map on the three real components*:

$$r'_i = \sum_j R_{ij}\, r_j, \qquad R_{ij} = \tfrac12 \mathrm{Tr}\!\left(P_i\, U P_j U^\dagger\right).$$

$R$ is a real $3\times 3$ matrix, and it is a **rotation**: $R^\top R = I$ and
$\det R = +1$. That is the claim of this notebook, and the function below is the
proof you can run.

Note $R$ eats the global phase: $U$ and $e^{i\varphi}U$ give the same $R$, because
the phase appears once as $e^{i\varphi}$ and once as $e^{-i\varphi}$."""))

cells.append(code(r'''def bloch_vector(state):
    """r = (<X>, <Y>, <Z>) for a normalized 2-vector."""
    s = np.asarray(state, dtype=complex)
    return np.array([np.vdot(s, P @ s).real for P in (X, Y, Z)])


def bloch_rotation(U):
    """The real 3x3 matrix R with r' = R r.  R_ij = Tr(P_i U P_j U*)/2."""
    P = (X, Y, Z)
    return np.array([[0.5 * np.trace(P[i] @ U @ P[j] @ dag(U)).real
                      for j in range(3)] for i in range(3)])


def axis_angle(U):
    """Recover (n_hat, theta) with U = e^{i phi} R_n(theta).

    Divide out det(U) to land in SU(2), then read off cos(theta/2) from the
    trace and n from Tr(P_k U)/(-2i sin(theta/2)).
    """
    V = np.asarray(U) / np.sqrt(np.linalg.det(U))     # now det V = 1
    c = np.trace(V).real / 2
    if c < 0:                                          # fix the sqrt branch so
        V, c = -V, -c                                  # theta lands in [0, pi]
    theta = 2 * np.arccos(np.clip(c, -1.0, 1.0))
    s = np.sin(theta / 2)
    if abs(s) < 1e-12:
        return np.array([0.0, 0.0, 1.0]), 0.0
    nvec = np.array([(1j * np.trace(P @ V) / (2 * s)).real for P in (X, Y, Z)])
    return nvec / np.linalg.norm(nvec), theta


print(f"{'gate':>14} {'axis n_hat':>26} {'angle':>10}   R orthogonal?  det R")
named = [("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T),
         ("Rx(pi/3)", Rx(np.pi / 3)), ("HT", H @ T)]
for name, M in named:
    nv, th = axis_angle(M)
    R = bloch_rotation(M)
    orth = np.allclose(R.T @ R, np.eye(3), atol=1e-10)
    print(f"{name:>14} {np.array2string(np.round(nv, 3), separator=','):>26} "
          f"{th/np.pi:8.3f} pi   {str(orth):>10}  {np.linalg.det(R):+.6f}")'''))

cells.append(md(r"""Every single gate: an axis, an angle, an orthogonal matrix with determinant
$+1$. Not one of them is anything but a rotation.

`H` is the line to stare at. Its axis is $(0.707, 0, 0.707)$ — the diagonal
between $\hat x$ and $\hat z$ — and its angle is exactly $\pi$.

### Figure 3 — the axes, drawn

Four gates, four axes. In each panel the thick coloured rod is $\hat n$ drawn
through the sphere, the dashed ring is the **orbit** that the starting state is
dragged around, and the two arrows are before (grey) and after (orange).

The Hadamard panel is the payoff. $H$ is routinely described as "the gate that
puts a qubit into superposition" or "a 50/50 beam splitter", and both descriptions
mislead — they suggest something probabilistic and irreversible. What $H$ actually
is: a **half-turn about the tilted axis $(\hat x + \hat z)/\sqrt2$**. That axis
sits exactly halfway between the $|0\rangle$ pole and the $|+\rangle$ equator
point, so a half-turn about it swaps those two — which is precisely
$H|0\rangle = |+\rangle$ and $H|+\rangle = |0\rangle$. And $H^2 = I$ is now not a
fact to memorise but a thing you can see: two half-turns is a full turn."""))

cells.append(code(r'''def rodrigues(nvec, t, v):
    """Rotate vector v about unit axis n by angle t (Rodrigues' formula)."""
    nvec = np.asarray(nvec, float) / np.linalg.norm(nvec)
    v = np.asarray(v, float)
    return (v * np.cos(t) + np.cross(nvec, v) * np.sin(t)
            + nvec * np.dot(nvec, v) * (1 - np.cos(t)))


def bloch_frame(ax, labels=True, zoom=1.5):
    """A Bloch sphere drawn as an outline rather than a translucent ball.

    Two reasons. Visually, an outline keeps 16 small panels from turning into
    mud. Practically, a shaded surface is a smooth gradient across thousands of
    pixels and roughly doubles the PNG - and this repo has a size budget, since
    every notebook ships with its outputs committed.

    The rim is the sphere's silhouette: the great circle perpendicular to the
    viewing direction (elev=18, azim=32, matching qviz.bloch's fixed camera).
    """
    bloch.sphere(ax, labels=labels, wire=False, alpha=0.0)
    e, a = np.deg2rad(18.0), np.deg2rad(32.0)
    view = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    u1 = np.cross(view, [0.0, 0.0, 1.0])
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(view, u1)
    t = np.linspace(0, 2 * np.pi, 160)
    p = np.outer(np.cos(t), u1) + np.outer(np.sin(t), u2)
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color=style.AXIS, lw=1.1, zorder=1)
    # 3D axes leave a wide default margin around the cube; zooming fills the
    # panel, which is also what pulls the six ket labels apart into legibility.
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    return ax


panels = [
    (r"$R_x(\pi/2)$ — axis $\hat{x}$", Rx(np.pi / 2), np.array([0., 0., 1.])),
    (r"$R_y(\pi/2)$ — axis $\hat{y}$", Ry(np.pi / 2), np.array([0., 0., 1.])),
    (r"$R_z(\pi/2)$ — axis $\hat{z}$", Rz(np.pi / 2), np.array([1., 0., 0.])),
    (r"$H$ — axis $(\hat{x}+\hat{z})/\sqrt{2}$", H, np.array([0., 0., 1.])),
]

fig, axes = grid.frames(4, ncols=4, panel=(3.0, 3.20), projection="3d")
for ax, (title, U, r0) in zip(axes, panels):
    bloch_frame(ax, zoom=1.42)
    nv, th = axis_angle(U)

    # the rotation axis, drawn as a rod through the whole sphere
    ax.plot(*np.array([-1.22 * nv, 1.22 * nv]).T, color=style.VIOLET, lw=3.0,
            zorder=4, solid_capstyle="round")
    ax.scatter(*(1.22 * nv), color=style.VIOLET, s=18, depthshade=False,
               zorder=5)

    # the full orbit of r0 about that axis
    ts = np.linspace(0, 2 * np.pi, 160)
    orbit = np.array([rodrigues(nv, t, r0) for t in ts])
    ax.plot(orbit[:, 0], orbit[:, 1], orbit[:, 2], color=style.MUTED, lw=1.0,
            ls=(0, (3, 3)), zorder=3)

    # the arc actually traversed, plus before / after arrows
    arc = np.array([rodrigues(nv, t, r0) for t in np.linspace(0, th, 60)])
    bloch.path(ax, arc, color=style.BLUE, lw=2.6)
    bloch.vector(ax, r0, color=style.MUTED, lw=1.8)
    bloch.vector(ax, rodrigues(nv, th, r0), color=style.ORANGE, lw=2.6)
    bloch.label(ax, title + f"\nangle = {th/np.pi:.2f}" + r"$\pi$", y=0.07,
                size=9.0)

fig.suptitle("Every gate is a rotation: axis (violet), orbit (dashed), "
             "start (grey) -> end (orange)", x=0.005, ha="left", fontsize=11.5)
plt.show()

print("H swaps the |0> pole and the |+> equator point:")
for nm, st in [("|0>", np.array([1, 0], dtype=complex)),
               ("|+>", np.array([1, 1], dtype=complex) / np.sqrt(2))]:
    print(f"  r({nm}) = {np.round(bloch_vector(st), 6)}"
          f"   ->   r(H{nm}) = {np.round(bloch_vector(H @ st), 6)}")
print("H @ H == I ->", np.allclose(H @ H, I2), " (two half-turns is a full turn)")'''))

# ------------------------------------------------------------ signature fig
cells.append(md(r"""### Figure 4 — the signature figure: a rotation, frame by frame

A notebook committed with its outputs cannot animate, so this repo shows
evolution as a **frame grid with a fading trail** — the tail dims towards the
past, which reads as direction of travel in a still image.

Sixteen frames of $|0\rangle$ under $R_x(\theta)$, $\theta$ stepping by $\pi/8$.
The orange arrow is the state; the blue trail is where it has been; the dashed
ring is the full orbit it is committed to. Watch it leave the north pole, sweep
down the back of the sphere through $|{-}i\rangle$, hit $|1\rangle$ at
$\theta=\pi$ — that is the $X$ gate — and climb back up the front.

Underneath each panel is $P(1)$, and it traces $\sin^2(\theta/2)$: at
$\theta = \pi/2$ the state is on the equator and the qubit is a genuine 50/50
superposition; at $\theta = \pi$ it has flipped outright. A single knob, turned
continuously, moving between "no gate" and "$X$ gate" through everything in
between. This is what a real device actually does — a gate is a pulse of a
certain duration, and $\theta$ is how long you left it on."""))

cells.append(code(r'''ket0 = np.array([1, 0], dtype=complex)
thetas = np.arange(16) * np.pi / 8

# fine sampling for a smooth trail, then sliced per frame
fine = np.linspace(0, thetas[-1], 481)
trail = np.array([bloch_vector(Rx(t) @ ket0) for t in fine])
full_orbit = np.array([bloch_vector(Rx(t) @ ket0)
                       for t in np.linspace(0, 2 * np.pi, 200)])

fig, axes = grid.frames(16, ncols=4, panel=(2.15, 2.15), projection="3d")
for k, (ax, th) in enumerate(zip(axes, thetas)):
    bloch_frame(ax, labels=False, zoom=1.62)
    ax.plot(full_orbit[:, 0], full_orbit[:, 1], full_orbit[:, 2],
            color=style.AXIS, lw=0.7, ls=(0, (3, 3)), zorder=3)
    upto = trail[fine <= th + 1e-9]
    if len(upto) > 1:
        bloch.path(ax, upto, color=style.BLUE, lw=2.2)
    st = Rx(th) @ ket0
    bloch.vector(ax, bloch_vector(st), color=style.ORANGE, lw=2.4)
    p1 = abs(st[1]) ** 2
    lab = r"$\theta=0$" if k == 0 else rf"$\theta={k}\pi/8$"
    bloch.label(ax, lab + f"\nP(1) = {p1:.2f}", y=0.10, size=8.5)

fig.suptitle(r"$|0\rangle$ under $R_x(\theta)$ — sixteen frames, fading trail",
             x=0.005, ha="left", fontsize=11.5)
plt.show()

print("P(1) after Rx(theta) vs the analytic sin^2(theta/2):")
for th in thetas[::4]:
    st = Rx(th) @ ket0
    print(f"  theta = {th/np.pi:4.2f} pi   P(1) = {abs(st[1])**2:.6f}"
          f"   sin^2(theta/2) = {np.sin(th/2)**2:.6f}")'''))

cells.append(md(r"""The seventeenth frame would be $\theta = 2\pi$, and the arrow would be back
exactly where it started — but the *matrix* would be $R_x(2\pi) = -I$, not $I$.
The sphere completed one loop; the state vector picked up a minus sign. The
picture genuinely cannot show you this, and Section 7 is about when it matters."""))

# --------------------------------------------------------------- composition
cells.append(md(r"""## 5. Composition: circuits are matrix products

Apply $A$ then $B$ and you get $B A |\psi\rangle$ — **right to left**, the opposite
of the left-to-right order a circuit diagram is drawn in. This trips up everyone
once. A circuit drawn as

```
|psi> --[ A ]--[ B ]--[ C ]-->
```

is the matrix $C B A$.

Two claims worth checking rather than assuming:

1. the product of unitaries is unitary — so any circuit, however long, is itself
   a single gate;
2. $R(BA) = R(B)\,R(A)$ — composing gates composes their Bloch rotations, in the
   same order. The map from $2\times2$ unitaries to $3\times3$ rotations respects
   multiplication, which is what makes "gate = rotation" a *structural* statement
   rather than a coincidence that happens to hold gate by gate."""))

cells.append(code(r'''circuit = [("H", H), ("T", T), ("H", H), ("S", S), ("Rx(0.7)", Rx(0.7))]

U = I2.copy()
for name, G in circuit:
    U = G @ U                 # left-multiply: later gates go on the LEFT
print("circuit  |psi> -> " + " -> ".join(n for n, _ in circuit))
print("single equivalent matrix U =")
print(np.round(U, 4))
print("\nunitary? ", is_unitary(U))
nv, th = axis_angle(U)
print(f"...and therefore a rotation: axis {np.round(nv, 4)}, "
      f"angle {th/np.pi:.4f} pi")

# 1. sequential application really equals one matrix multiply
for _ in range(5):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = v / np.linalg.norm(v)
    step = v.copy()
    for _, G in circuit:
        step = G @ step
    assert np.allclose(step, U @ v)
print("\nsequential application == one matrix product: verified on 5 random states")

# 2. the homomorphism R(BA) = R(B) R(A)
print("\nR(BA) == R(B) R(A)?")
for na, A in [("H", H), ("T", T), ("Rx(0.7)", Rx(0.7))]:
    for nb, B in [("S", S), ("Ry(1.1)", Ry(1.1))]:
        ok = np.allclose(bloch_rotation(B @ A),
                         bloch_rotation(B) @ bloch_rotation(A), atol=1e-12)
        print(f"  A={na:<8} B={nb:<8} -> {ok}")'''))

# ---------------------------------------------------------- non-commutativity
cells.append(md(r"""### Figure 5 — order matters (except when it doesn't)

Matrix multiplication does not commute, and neither do rotations: turn a book
$90°$ about one axis then another, swap the order, and it ends up facing
differently. The commutator

$$[A,B] = AB - BA$$

measures the failure. The heatmap below is $\|[A,B]\|_F$ over the standard gate
set — pale means the two gates can be freely reordered, dark means they cannot.

The diagonal is zero (everything commutes with itself). The $Z$/$S$/$T$ block is
zero because they are all rotations about the *same* axis, and rotations about a
shared axis just add their angles. Everything else is dark.

But read the two Bloch panels carefully, because there is a trap. **A non-zero
commutator does not automatically mean a visible difference.** $X$ and $Z$
anticommute, $XZ = -ZX$, so their commutator has norm 4 — the largest in the grid
— and yet $XZ$ and $ZX$ differ only by $-1$, a global phase. They are the *same
rotation of the sphere*. $R_x(\pi/2)$ and $R_z(\pi/2)$ are the honest example: the
two orders land on genuinely different points.

(The two Bloch panels start from a *generic* state rather than $|0\rangle$. The
north pole lies on the $z$ axis, so $R_z$ would leave it exactly where it is and
one of the two steps would be invisible — true, but a weaker demonstration.)"""))

cells.append(code(r'''cnames = ["X", "Y", "Z", "H", "S", "T", r"$R_x$", r"$R_y$"]
cmats = [X, Y, Z, H, S, T, Rx(np.pi / 3), Ry(np.pi / 3)]
Cnorm = np.array([[np.linalg.norm(A @ B - B @ A) for B in cmats]
                  for A in cmats])

fig = plt.figure(figsize=(12.2, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0])
axc = fig.add_subplot(gs[0, 0])
grid.matrix(axc, Cnorm, part="abs", labels=cnames, annot=True, fmt="{:.1f}",
            hide_zeros=False)   # 0 is the interesting value here, not clutter
axc.set_title(r"$\|[A,B]\|_F$ — pale means the two commute", loc="left",
              fontsize=10)

A1, A2 = Rx(np.pi / 2), Rz(np.pi / 2)
# Start from a generic state rather than |0>: the north pole sits ON the z axis,
# so Rz would leave it fixed and one of the two steps would be invisible.
start = Ry(np.pi / 3) @ ket0
orders = [("first $R_x(\\pi/2)$, then $R_z(\\pi/2)$", (A1, A2)),
          ("first $R_z(\\pi/2)$, then $R_x(\\pi/2)$", (A2, A1))]

for i, (title, (G1, G2)) in enumerate(orders):
    ax = fig.add_subplot(gs[0, i + 1], projection="3d")
    bloch_frame(ax)
    r0 = bloch_vector(start)
    n1, t1 = axis_angle(G1)
    mid = rodrigues(n1, t1, r0)
    n2, t2 = axis_angle(G2)
    end = rodrigues(n2, t2, mid)

    seg1 = np.array([rodrigues(n1, t, r0) for t in np.linspace(0, t1, 60)])
    seg2 = np.array([rodrigues(n2, t, mid) for t in np.linspace(0, t2, 60)])
    bloch.path(ax, seg1, color=style.BLUE, lw=2.6, fade=False)
    bloch.path(ax, seg2, color=style.AQUA, lw=2.6, fade=False)
    bloch.vector(ax, r0, color=style.MUTED, lw=1.6)
    bloch.vector(ax, end, color=style.ORANGE, lw=2.8)
    coords = ", ".join(f"{c if abs(c) > 5e-3 else 0.0:+.2f}" for c in end)
    bloch.label(ax, title + f"\nends at ({coords})", y=0.07, size=9.0)

fig.suptitle("Gate order changes where you land", x=0.005, ha="left",
             fontsize=11.5)
plt.show()

print("start       ->", np.round(bloch_vector(start), 6))
print("Rx then Rz  ->", np.round(bloch_vector(A2 @ A1 @ start), 6))
print("Rz then Rx  ->", np.round(bloch_vector(A1 @ A2 @ start), 6))

print("\nThe trap: X and Z anticommute, so the commutator is as big as it gets...")
print("  ||[X,Z]||_F =", round(float(np.linalg.norm(X @ Z - Z @ X)), 6))
print("  X@Z ==  -Z@X ->", np.allclose(X @ Z, -(Z @ X)))
print("...but -1 is a global phase, so the two orders are the SAME rotation:")
print("  R(XZ) == R(ZX) ->",
      np.allclose(bloch_rotation(X @ Z), bloch_rotation(Z @ X)))
print("  r(XZ|0>) =", np.round(bloch_vector(X @ Z @ ket0), 6),
      "   r(ZX|0>) =", np.round(bloch_vector(Z @ X @ ket0), 6))'''))

# ----------------------------------------------------------------- S and T
cells.append(md(r"""## 6. $S$ and $T$: fractions of a $Z$ rotation

$Z$, $S$ and $T$ are the same gate at three different strengths:

$$Z = \mathrm{diag}(1, e^{i\pi}), \qquad S = \mathrm{diag}(1, e^{i\pi/2}), \qquad
T = \mathrm{diag}(1, e^{i\pi/4}).$$

Each multiplies the $|1\rangle$ amplitude by a phase and leaves $|0\rangle$ alone,
which on the sphere is a rotation about $\hat z$ by $\pi$, $\pi/2$, $\pi/4$. So
$S = T^2$ and $Z = S^2 = T^4$, and $T^8 = I$ — eight eighth-turns is a full turn.

$T$ matters far more than its modest angle suggests. $\{H, S, \text{CNOT}\}$ — the
Clifford group — is efficiently simulable on a classical computer (the
Gottesman–Knill theorem), so a circuit built only from those gates gives no
quantum advantage at all. Adding $T$ is what breaks out. On error-corrected
hardware $T$ is also *dramatically* more expensive than everything else, which is
why "T-count" is the currency compiler papers actually optimise.

None of that is visible in the matrix. It is a diagonal with one $45°$ phase in
it. Which is a good reminder that the interesting content of a gate set is not in
any one gate.

### Figure 6 — eight eighth-turns

Left: $T^k|+\rangle$ for $k = 0\ldots 8$, marching round the equator in $45°$
steps and closing the loop exactly.

Right: the phase piled onto the $|1\rangle$ amplitude, as a function of $k$. Three
straight lines whose slopes are $\pi/4$, $\pi/2$, $\pi$ — the rotation angles
themselves. The hollow rings are what `np.angle` actually returns: the same
quantity folded into $[0, 2\pi)$, because a phase is only ever knowable modulo
$2\pi$. The filled line is the accumulation we *know* happened; the rings are all
an experiment could ever see. Prising the unwrapped value back out of the wrapped
one is, more or less, what quantum phase estimation is for."""))

cells.append(code(r'''plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
steps = np.arange(9)

fig = plt.figure(figsize=(11.4, 4.1))
gs6 = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55])

axb = fig.add_subplot(gs6[0, 0], projection="3d")
bloch_frame(axb)
# T^k for fractional k is just Rz(k pi/4) - the continuous rotation T samples
pts = np.array([bloch_vector(Rz(k * np.pi / 4) @ plus)
                for k in np.linspace(0, 8, 240)])
bloch.path(axb, pts, color=style.BLUE, lw=2.2)
marks = np.array([bloch_vector(np.linalg.matrix_power(T, k) @ plus)
                  for k in steps])
axb.scatter(marks[:, 0], marks[:, 1], marks[:, 2], color=style.BLUE, s=26,
            depthshade=False, zorder=6, edgecolors=style.SURFACE, linewidths=0.8)
bloch.vector(axb, marks[0], color=style.MUTED, lw=1.8, label="k=0")
bloch.vector(axb, marks[3], color=style.ORANGE, lw=2.4, label="k=3")
bloch.label(axb, r"$T^k|+\rangle$: eight steps round the equator",
            y=0.06, size=9.0)


def relative_phase(G):
    """The phase a diagonal gate puts on |1> relative to |0>."""
    return float(np.angle(G[1, 1]) - np.angle(G[0, 0]))


axp = fig.add_subplot(gs6[0, 1])
for name, G, col, tex in [("T", T, style.BLUE, r"$\pi/4$"),
                          ("S", S, style.ORANGE, r"$\pi/2$"),
                          ("Z", Z, style.AQUA, r"$\pi$")]:
    lam = relative_phase(G)
    order = int(round(2 * np.pi / lam))          # applications to return to I
    kk = np.arange(order + 1)
    # accumulated (unwrapped) phase after k applications is exactly k * lam
    axp.plot(kk, kk * lam / np.pi, marker="o", ms=4.5, color=col,
             label=f"{name}   (step = " + tex + f",  order {order})")
    # what np.angle actually reports: the same thing folded into [0, 2 pi)
    wrapped = np.array([np.angle((np.linalg.matrix_power(G, int(k))
                                  @ plus)[1]) % (2 * np.pi) for k in kk])
    axp.plot(kk, wrapped / np.pi, ls="none", marker="o", ms=9,
             mfc="none", mec=col, mew=1.1, zorder=4)
axp.axhline(2.0, color=style.MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
axp.text(6.0, 2.06, r"$2\pi$ — back to the identity", fontsize=8,
         color=style.MUTED, va="bottom", ha="right")
axp.set_xlabel("k  (gate applied k times)")
axp.set_ylabel(r"phase on $|1\rangle$  $/\ \pi$")
axp.set_title("Phase accumulates linearly; the slope IS the rotation angle",
              loc="left", fontsize=10)
axp.set_xticks(steps)
axp.set_xlim(-0.35, 8.6)
axp.set_ylim(-0.35, 3.05)
axp.legend(loc="upper center", ncols=3, fontsize=8.0)
grid.annotate(axp, "hollow rings: what np.angle actually reports —\n"
                   "the same phase, folded back into $[0, 2\\pi)$",
              xy=(8, 0.0), xytext=(3.15, 0.42))
plt.show()

print("T^2 == S ->", np.allclose(np.linalg.matrix_power(T, 2), S))
print("T^4 == Z ->", np.allclose(np.linalg.matrix_power(T, 4), Z))
print("T^8 == I ->", np.allclose(np.linalg.matrix_power(T, 8), I2))
print("\nT^8 =")
print(np.round(np.linalg.matrix_power(T, 8), 12))'''))

cells.append(md(r"""Note that $T^8 = I$ **exactly** — not $-I$. $T = e^{-i\pi/8}R_z(\pi/4)$ carries a
global phase, and eight of those phases multiply to $e^{-i\pi} = -1$, which
cancels the $-1$ from $R_z(2\pi)$. Two conventions, two minus signs, one identity.
That is the last friendly thing global phase will do for us."""))

# ----------------------------------------------------------- honest limits
cells.append(md(r"""## 7. Honest limits — what the sphere hides

The Bloch picture is exact for one pure qubit and it is the best mental model in
the subject. It still hides three things, and one of them is load-bearing.

### Global phase is invisible, until the gate is controlled

$R_z(\theta) = \mathrm{diag}(e^{-i\theta/2}, e^{i\theta/2})$ and the phase gate
$P(\theta) = \mathrm{diag}(1, e^{i\theta})$ differ by exactly $e^{-i\theta/2}$ — a
global phase. They produce **the same rotation** of the Bloch sphere, and no
measurement on a single qubit can tell them apart. Most texts treat them as
interchangeable, and for a lone qubit they are.

They are not interchangeable as *controlled* gates. Making a gate controlled means
"apply $U$ only on the branch where the control is 1" — and on that branch the
global phase of $U$ is no longer global. It attaches to one branch and not the
other, which makes it a **relative** phase of the two-qubit state, which is
physical, measurable, and the entire mechanism behind phase kickback (A08),
quantum phase estimation, and Shor's algorithm.

The figure makes it concrete: same single-qubit rotation, visibly different
two-qubit operators.

*(Endianness note, per the repo convention: little-endian, so index $i$ of the
statevector maps to a bitstring with qubit 0 as the rightmost character, and the
leftmost factor of a Kronecker product is qubit 1.)*"""))

cells.append(code(r'''def P_gate(theta):
    """Phase gate diag(1, e^{i theta}) - Rz up to a global phase."""
    return np.array([[1, 0], [0, np.exp(1j * theta)]], dtype=complex)


def controlled(U):
    """Control on qubit 0, target on qubit 1, little-endian.

    Index i has bit0 = i & 1 (qubit 0) and bit1 = (i >> 1) & 1 (qubit 1), so
    qubit 1 is the LEFT Kronecker factor.
    """
    p0 = np.array([[1, 0], [0, 0]], dtype=complex)     # |0><0| on qubit 0
    p1 = np.array([[0, 0], [0, 1]], dtype=complex)     # |1><1| on qubit 0
    return np.kron(I2, p0) + np.kron(U, p1)


theta = np.pi / 2
print("Single qubit: identical rotations of the sphere.")
print("  R(Rz) == R(P) ->",
      np.allclose(bloch_rotation(Rz(theta)), bloch_rotation(P_gate(theta))))
print("  Rz and P differ only by a global phase ->",
      same_up_to_phase(Rz(theta), P_gate(theta)))
print("\nTwo qubits: NOT the same operator.")
CRZ, CP = controlled(Rz(theta)), controlled(P_gate(theta))
print("  controlled versions equal up to global phase ->",
      same_up_to_phase(CRZ, CP))
print("  diag(CRz) =", np.round(np.diag(CRZ), 4))
print("  diag(CP)  =", np.round(np.diag(CP), 4))'''))

cells.append(code(r'''pp = np.kron(plus, plus)          # |+>|+>, all four amplitudes 1/2

fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
for j, (name, M) in enumerate([(r"controlled-$R_z(\pi/2)$", CRZ),
                               (r"controlled-$P(\pi/2)$", CP)]):
    grid.matrix(axes[0, j], M, part="phase", annot=False)
    axes[0, j].set_title(name + "  —  phase of each entry", loc="left",
                         fontsize=10)
    out = M @ pp
    grid.amp_bars(axes[1, j], out, labels=["00", "01", "10", "11"], ylim=0.62)
    axes[1, j].set_title(name + r"  applied to  $|{+}{+}\rangle$", loc="left",
                         fontsize=10)
fig.suptitle("Same Bloch rotation, different two-qubit operator",
             x=0.005, ha="left", fontsize=11.5)
plt.show()

print("amplitudes after controlled-Rz:", np.round(CRZ @ pp, 4))
print("amplitudes after controlled-P :", np.round(CP @ pp, 4))
print("\nBoth states have identical measurement probabilities...")
print("  P(CRz) =", np.round(np.abs(CRZ @ pp) ** 2, 4))
print("  P(CP)  =", np.round(np.abs(CP @ pp) ** 2, 4))
print("...but the phase patterns differ, and A08 turns exactly that into an answer.")'''))

cells.append(md(r"""The bar heights are identical; the hues and clock hands are not. Two states that
no measurement in this basis distinguishes — which is precisely the situation A01
warned about, and precisely what a Hadamard on the control turns into a
measurable difference.

### The other two things the sphere hides

- **A frame grid is not an animation.** Figure 4 samples a continuous path at
  sixteen points. If a rotation went round more than once between frames, the
  fading trail would look identical to a single pass — the picture cannot
  distinguish $\theta$ from $\theta + 2\pi$, though the *matrix* can (that is the
  $-I$ again). Any conclusion you draw about "how far it turned" comes from the
  algebra, not the picture.
- **One qubit only.** Every visualisation in this notebook is a single arrow in a
  ball. Two entangled qubits have no such picture: each qubit's Bloch vector
  shrinks to the origin while the *pair* is in a perfectly definite state, so
  drawing two spheres shows two blank balls and hides all the information. A05
  and A12 confront this properly. Do not carry this notebook's intuition into
  multi-qubit land unexamined — it is the single most common way people go wrong.

## Checkpoint"""))

cells.append(code(r'''ALL = [("I", I2), ("X", X), ("Y", Y), ("Z", Z), ("H", H), ("S", S), ("T", T),
       ("Rx", Rx(0.7)), ("Ry", Ry(1.3)), ("Rz", Rz(2.1)),
       ("P", P_gate(0.4)), ("circuit", U)]

# 1. Every gate is unitary, and its inverse is its dagger.
for name, M in ALL:
    assert np.allclose(dag(M) @ M, I2), name
    assert np.allclose(M @ dag(M), I2), name
    assert np.isclose(abs(np.linalg.det(M)), 1.0), name

# 2. Unitaries preserve the norm of every state, so probabilities still sum to 1.
for _ in range(50):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = v / np.linalg.norm(v)
    for _, M in ALL:
        assert np.isclose(np.linalg.norm(M @ v), 1.0)

# 3. Pauli algebra.
assert np.allclose(X @ X, I2) and np.allclose(Y @ Y, I2) and np.allclose(Z @ Z, I2)
assert np.allclose(X @ Y, 1j * Z)
assert np.allclose(Y @ Z, 1j * X)
assert np.allclose(Z @ X, 1j * Y)
for A, B in [(X, Y), (Y, Z), (Z, X)]:
    assert np.allclose(A @ B + B @ A, 0)

# 4. The closed form really is the matrix exponential.
for t in rng.uniform(0, 4 * np.pi, 12):
    assert np.allclose(Rx(t), expm_series(-1j * t * X / 2), atol=1e-12)
    assert np.allclose(Ry(t), expm_series(-1j * t * Y / 2), atol=1e-12)
    assert np.allclose(Rz(t), expm_series(-1j * t * Z / 2), atol=1e-12)

# 5. Half-angle: a full 2 pi turn is -I, and 4 pi is +I.
assert np.allclose(Rx(2 * np.pi), -I2)
assert np.allclose(Rx(4 * np.pi), I2)

# 6. Angles add along a fixed axis.
for a, b in rng.uniform(-3, 3, size=(8, 2)):
    assert np.allclose(Ry(a) @ Ry(b), Ry(a + b))

# 7. The named gates are the rotations we claimed, up to global phase.
assert same_up_to_phase(X, Rx(np.pi))
assert same_up_to_phase(Y, Ry(np.pi))
assert same_up_to_phase(Z, Rz(np.pi))
assert same_up_to_phase(S, Rz(np.pi / 2))
assert same_up_to_phase(T, Rz(np.pi / 4))
assert same_up_to_phase(H, rot_axis([1, 0, 1], np.pi))

# 8. EVERY single-qubit unitary, including 200 random ones, is a Bloch rotation:
#    R is orthogonal with determinant +1, and it reproduces the axis-angle form.
for _ in range(200):
    M = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, Rr = np.linalg.qr(M)                      # Haar-ish random unitary
    Q = Q * (np.diag(Rr) / np.abs(np.diag(Rr)))
    assert is_unitary(Q, tol=1e-10)
    Rb = bloch_rotation(Q)
    assert np.allclose(Rb.T @ Rb, np.eye(3), atol=1e-10)
    assert np.isclose(np.linalg.det(Rb), 1.0, atol=1e-10)
    nv, th = axis_angle(Q)
    assert same_up_to_phase(Q, rot_axis(nv, th), tol=1e-9)

# 9. Composition is a homomorphism: R(BA) = R(B) R(A).
for _ in range(20):
    a, b = rng.uniform(0, 2 * np.pi, 2)
    A, B = rot_axis(rng.normal(size=3), a), rot_axis(rng.normal(size=3), b)
    assert np.allclose(bloch_rotation(B @ A),
                       bloch_rotation(B) @ bloch_rotation(A), atol=1e-10)

# 10. Gates move the state exactly as the rotation moves the Bloch vector.
for _ in range(20):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    v = v / np.linalg.norm(v)
    G = rot_axis(rng.normal(size=3), rng.uniform(0, 2 * np.pi))
    assert np.allclose(bloch_vector(G @ v), bloch_rotation(G) @ bloch_vector(v),
                       atol=1e-10)

# 11. S and T are fractional Z rotations.
assert np.allclose(T @ T, S)
assert np.allclose(np.linalg.matrix_power(T, 4), Z)
assert np.allclose(np.linalg.matrix_power(T, 8), I2)
assert np.allclose(H @ H, I2)

# 12. X and Z anticommute, so XZ and ZX are the same rotation despite [X,Z] != 0.
assert not np.allclose(X @ Z, Z @ X)
assert np.allclose(bloch_rotation(X @ Z), bloch_rotation(Z @ X))

# 13. Rz and P are the same rotation but different controlled gates.
for t in rng.uniform(0.2, 3.0, 5):
    assert np.allclose(bloch_rotation(Rz(t)), bloch_rotation(P_gate(t)),
                       atol=1e-12)
    assert same_up_to_phase(Rz(t), P_gate(t))
    assert not same_up_to_phase(controlled(Rz(t)), controlled(P_gate(t)))

print("A03 checkpoint passed.")'''))

cells.append(md(r"""---

**Next:** [A04 — Measurement and the Born Rule](A04_Measurement.ipynb). Every gate
so far has been reversible, deterministic and length-preserving. Measurement is
none of those things — it is the one operation in the theory that is not a
rotation, and it is where the probabilities finally cash out."""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = notebook_metadata()

if __name__ == "__main__":
    import pathlib

    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)
