"""Generator for A05 - Tensor Products and Multi-Qubit States.

This is the notebook where the repo's endianness convention is established and
justified, so the whole of Track A can match Qiskit index-for-index. Structure
follows gen_A01.py (the golden reference): on-ramp -> derivation in visible
code -> substantive figures -> honest limits -> Checkpoint.
"""

import nbformat as nbf

NB_ID = "A05"
TITLE = "Tensor Products and Multi-Qubit States"
OUT = f"notebooks/{NB_ID}_Tensor_Products.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = []

# NOTE: plain r-strings for markdown, never f-strings. LaTeX is full of braces
# (\tfrac{1}{2}, \mathbb{C}) and an f-string reads them as interpolation slots.
cells.append(md(r"""# A05 — Tensor Products and Multi-Qubit States

**Track A — under the hood.** Pure NumPy. Qiskit is not imported anywhere in this
notebook; every number here is one you can derive with a pen.

---

## The one-sentence version

Two qubits are not two separate pairs of amplitudes. They are **one** list of four
amplitudes — and that innocent-looking upgrade is where entanglement, exponential
cost, and every indexing bug you will ever write all come from.

## Why one list and not two

A single qubit is $\alpha|0\rangle + \beta|1\rangle$: two complex numbers, one per
label. Put two qubits side by side and the labels you can see are `00`, `01`, `10`,
`11` — four of them. So the state is four complex numbers:

$$|\psi\rangle = c_{00}|00\rangle + c_{01}|01\rangle + c_{10}|10\rangle + c_{11}|11\rangle,
\qquad \sum |c|^2 = 1.$$

Sometimes those four numbers factor into "what qubit 0 is doing" times "what qubit 1
is doing". Sometimes — and this is the whole story — **they don't**. A state that
does not factor is *entangled*, and there is no way to describe it as two separate
qubits. That is why we are forced to carry the joint list.

Three things follow, and this notebook does each in turn:

1. The operation that glues descriptions together is the **tensor (Kronecker)
   product**, and it is completely mechanical.
2. The list has length $2^n$. Not $2n$. $2^n$. We will plot exactly where your
   laptop dies.
3. Gluing requires choosing **which end of the bitstring qubit 0 lives on**, and
   choosing wrong makes your simulator silently disagree with every textbook,
   every SDK, and every other notebook in this repo. Section 3 is the most
   important part of this file.

Let's build it."""))

cells.append(code(r'''from qviz import backends, grid, style

from functools import reduce

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

style.use()

SEED = backends.seed_for("A05")
rng = np.random.default_rng(SEED)
print("seed:", SEED)

# The single-qubit vocabulary from A01/A03, rebuilt here so this notebook stands
# alone. Nothing is imported from a helper that does quantum mechanics.
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)

KET0 = np.array([1, 0], dtype=complex)
KET1 = np.array([0, 1], dtype=complex)
PLUS = np.array([1, 1], dtype=complex) / np.sqrt(2)
MINUS = np.array([1, -1], dtype=complex) / np.sqrt(2)'''))

# ------------------------------------------------------------------ kron
cells.append(md(r"""## 1. The Kronecker product

The rule is one line. If $u$ and $v$ are the amplitude vectors of two independent
systems, the joint system's amplitude for "$u$ is in state $a$ **and** $v$ is in
state $b$" is the product $u_a v_b$:

$$(u \otimes v)_{(a,b)} = u_a\, v_b.$$

Probabilities of independent events multiply; amplitudes of independent systems
multiply for the same reason. The only real content is how we flatten the pair
$(a,b)$ into a single integer index. NumPy's `np.kron` makes the choice

$$(u \otimes v)_{2a+b} = u_a\, v_b,$$

so the **left** factor supplies the **high** bit and the right factor the low bit.
Hold on to that sentence — section 3 is entirely about it.

For matrices the same rule applies to both indices at once:

$$(A \otimes B)_{2i+k,\;2j+l} = A_{ij} B_{kl},$$

which is why $A \otimes B$ looks like a $2\times2$ grid of copies of $B$, each copy
scaled by the corresponding entry of $A$. Nested, recursive, and completely
mechanical."""))

cells.append(code(r'''A = np.array([[1., 2.],
              [3., 4.]])
B = np.array([[5., 6.],
              [7., 8.]])
K = np.kron(A, B)

print("A (2x2) kron B (2x2) ->", K.shape)
print(K.astype(int))
print()
print("block (0,0) is A[0,0]*B =", (A[0, 0] * B).astype(int).tolist())
print("block (1,0) is A[1,0]*B =", (A[1, 0] * B).astype(int).tolist())'''))

# ---------------------------------------------------------------- figure 1
cells.append(md(r"""### Figure 1 — the block structure, drawn

Every cell of $A \otimes B$ is tinted by **which entry of $A$ owns it**. The
$4\times4$ result is visibly four copies of $B$, and if you tensored a third matrix
on, each of those sixteen cells would open up into another copy. That recursion is
the whole reason $n$ qubits cost $2^n$: each new factor subdivides every cell that
already exists."""))

cells.append(code(r'''fig = plt.figure(figsize=(11.6, 3.35))
gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.75, 1.9])
axA, axB, axS, axK = [fig.add_subplot(gs[0, j]) for j in range(4)]

def tint(a):
    """Map an entry of A (1..4) onto the sequential ramp."""
    return style.SEQ(0.10 + 0.58 * (a - 1.0) / 3.0)

def dark(a):
    return (a - 1.0) / 3.0 > 0.55

def cells_2x2(ax, M, colour, dark_fn=None, fmt="{:.0f}", fs=11):
    for r in range(2):
        for c in range(2):
            ax.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=colour(r, c),
                                   edgecolor=style.SURFACE, lw=1.6))
            hot = dark_fn(r, c) if dark_fn is not None else False
            ax.text(c, r, fmt.format(M[r, c]), ha="center", va="center",
                    fontsize=fs, color="white" if hot else style.INK)
    ax.set_xlim(-.55, 1.55)
    ax.set_ylim(1.55, -.55)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

# --- A, tinted by its own value; B, neutral
cells_2x2(axA, A, lambda r, c: tint(A[r, c]), lambda r, c: dark(A[r, c]))
axA.set_title("$A$", loc="left", fontsize=11)
cells_2x2(axB, B, lambda r, c: style.GRID)
axB.set_title("$B$", loc="left", fontsize=11)

# --- schematic: four blocks, labelled
for r in range(2):
    for c in range(2):
        axS.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=tint(A[r, c]),
                                edgecolor=style.SURFACE, lw=2.4))
        # faint 2x2 subdivision: the copy of B hiding inside each block.
        # Drawn with a gap in the middle so it does not strike through the label.
        for x0, x1 in [(c - .5, c - .32), (c + .32, c + .5)]:
            axS.plot([x0, x1], [r, r], color=style.SURFACE, lw=0.9, alpha=0.85)
        for y0, y1 in [(r - .5, r - .22), (r + .22, r + .5)]:
            axS.plot([c, c], [y0, y1], color=style.SURFACE, lw=0.9, alpha=0.85)
        axS.text(c, r, rf"$a_{{{r}{c}}}\,B$", ha="center", va="center", fontsize=13,
                 color="white" if dark(A[r, c]) else style.INK)
axS.set_xlim(-.55, 1.55)
axS.set_ylim(1.55, -.55)
axS.set_aspect("equal")
axS.set_xticks([])
axS.set_yticks([])
axS.grid(False)
for sp in axS.spines.values():
    sp.set_visible(False)
axS.set_title(r"$A \otimes B$  as blocks", loc="left", fontsize=11)

# --- the numbers
labs = ["00", "01", "10", "11"]
for r in range(4):
    for c in range(4):
        a = A[r // 2, c // 2]
        axK.add_patch(Rectangle((c - .5, r - .5), 1, 1, facecolor=tint(a),
                                edgecolor=style.SURFACE, lw=1.0))
        axK.text(c, r, f"{K[r, c]:.0f}", ha="center", va="center", fontsize=8.5,
                 color="white" if dark(a) else style.INK)
for t in (1.5,):
    axK.plot([-.5, 3.5], [t, t], color=style.INK, lw=1.6)
    axK.plot([t, t], [-.5, 3.5], color=style.INK, lw=1.6)
axK.set_xlim(-.55, 3.55)
axK.set_ylim(3.55, -.55)
axK.set_aspect("equal")
axK.set_xticks(range(4))
axK.set_yticks(range(4))
axK.set_xticklabels(labs, fontsize=8)
axK.set_yticklabels(labs, fontsize=8)
axK.grid(False)
for sp in axK.spines.values():
    sp.set_visible(False)
axK.set_title(r"$A \otimes B$  numerically", loc="left", fontsize=11)

fig.suptitle("The Kronecker product is a matrix of scaled copies",
             x=0.005, ha="left", fontsize=11.5)
plt.show()'''))

cells.append(md(r"""Row and column labels on the right-hand panel are already
two-bit strings: the tensor product hands you the product basis for free. The heavy
cross marks the block boundary — inside each quadrant sits an untouched copy of
$B$.

## 2. Dimension: $2^n$, and only $2^n$

Each `kron` multiplies the length. One qubit: 2. Two: 4. Ten: 1024. Fifty:
about $1.1\times10^{15}$.

There is no way around this for a *general* state, and it is worth being precise
about why. $n$ qubits have $2^n$ distinguishable classical configurations; a pure
quantum state assigns an amplitude to each. Any description shorter than $2^n$
numbers is a description of a *special* state, not of the whole space. (Special
states are extremely useful — tensor-network simulators live on them — but they
are a restriction, not a free lunch.)"""))

cells.append(code(r'''def register(*qubit_states):
    """Combine single-qubit states into one register vector.

    Arguments are given qubit 0 FIRST - the way you talk - and reversed
    internally, because np.kron gives the LEFT factor the HIGH bit.  Section 3
    derives this; for now, note the [::-1].
    """
    return reduce(np.kron, qubit_states[::-1])


for k in range(1, 6):
    v = register(*([PLUS] * k))
    print(f"{k} qubit(s): length {v.size:>2}  = 2**{k},   norm = {np.linalg.norm(v):.6f}")

psi2 = register(PLUS, KET0)          # qubit 0 in |+>, qubit 1 in |0>
print("\n|+> on qubit 0, |0> on qubit 1:")
print(grid.show_state(psi2))'''))

# ------------------------------------------------------------ ENDIANNESS
cells.append(md(r"""## 3. Endianness — the section that matters

Here is the trap, stated as plainly as possible.

A statevector is a flat array. Index `5` of a 3-qubit register is *some* physical
configuration of three qubits — but **which one?** Writing $5 = 101_2$ does not
answer the question, because it does not say which end of `101` is qubit 0.

Two conventions exist, and they are both defensible:

| convention | index of `(q0, q1, q2)` | qubit 0 is... |
|---|---|---|
| **big-endian** (most textbooks, `np.kron(q0, q1, q2)`) | $4q_0 + 2q_1 + q_2$ | the **left**most character |
| **little-endian** (Qiskit, and this repo) | $q_0 + 2q_1 + 4q_2$ | the **right**most character |

**This repo is little-endian everywhere.** Statevector index $i$ corresponds to the
bitstring $b_{n-1}\ldots b_1 b_0$ with

$$i \;=\; \sum_{k=0}^{n-1} b_k\, 2^k ,$$

so qubit 0 is the **least significant bit**, printed rightmost. Applying $X$ to
qubit 0 of a 2-qubit register lands on index 1, which we print as $|01\rangle$.

### Why we chose the "wrong-looking" one

Because Qiskit did, and the whole point of this repo is that Track B can end each
notebook with

```python
assert np.allclose(my_state, Statevector(qc).data)
```

If Track A were big-endian, that assertion would fail on every asymmetric state —
and worse, it would *pass* on symmetric ones, so the bug would hide until the
interesting notebook.

### The consequence, derived

Take the little-endian index formula and ask what it means for `kron`. NumPy gives

$$(u \otimes v)_{2a+b} = u_a v_b,$$

so the **left** factor of a `kron` controls the **high** bit of the index. But in
our convention the high bit is qubit $n-1$, not qubit 0. Therefore the leftmost
factor in the chain must be **qubit $n-1$**, and qubit 0 must come **last**:

$$|\psi\rangle \;=\; |q_{n-1}\rangle \otimes \cdots \otimes |q_1\rangle \otimes |q_0\rangle .$$

The naive, read-it-left-to-right `np.kron(q0, q1)` is precisely the big-endian
convention. **It is the single most common way a from-scratch simulator goes
silently wrong.**

The same argument applies to operators. To act with a $2\times2$ unitary $U$ on
qubit $k$ and leave the rest alone, tensor $U$ into position $k$ of a chain of
identities, written in **reversed** qubit order:

$$U^{(k)} \;=\; \underbrace{I \otimes \cdots \otimes I}_{n-1-k}
\;\otimes\; U \;\otimes\; \underbrace{I \otimes \cdots \otimes I}_{k}.$$

That is the helper every later Track A notebook uses."""))

cells.append(code(r'''def op_on(U, k, n):
    """Embed the 2x2 operator ``U`` on qubit ``k`` of an ``n``-qubit register.

    LITTLE-ENDIAN: qubit 0 is the least significant bit, so it must be the
    RIGHTMOST factor of the Kronecker chain.  Build the list in qubit order
    (index = qubit number), then reverse it before folding.

    Returns a (2**n, 2**n) complex matrix.
    """
    if not 0 <= k < n:
        raise ValueError(f"qubit {k} out of range for {n} qubits")
    ops = [I2] * n
    ops[k] = np.asarray(U, dtype=complex)
    return reduce(np.kron, ops[::-1])          # [q_{n-1}, ..., q_1, q_0]


# The reversal is easy to fumble, so here is a completely independent
# construction straight from the index formula i = sum_k b_k 2**k.  No kron at
# all: for each basis index j, read bit k, and scatter U's column into place.
def op_on_by_index(U, k, n):
    """Same operator, built by explicit bit surgery. Used to verify op_on."""
    U = np.asarray(U, dtype=complex)
    D = 2 ** n
    M = np.zeros((D, D), dtype=complex)
    for j in range(D):
        bj = (j >> k) & 1                       # value of qubit k in basis state j
        rest = j & ~(1 << k)                    # all the other qubits, untouched
        for bi in (0, 1):
            M[rest | (bi << k), j] = U[bi, bj]
    return M


for n in (1, 2, 3, 4):
    for k in range(n):
        for U in (X, H, S, Z):
            assert np.allclose(op_on(U, k, n), op_on_by_index(U, k, n))
print("op_on agrees with the index-formula construction for n = 1..4, all k, four gates.")'''))

cells.append(md(r"""Now the hand-check. For two qubits, $X$ on qubit 0 should map
$|00\rangle \to |01\rangle$, i.e. index $0 \to 1$; and $X$ on qubit 1 should map
index $0 \to 2$. Write the matrices out and look at them."""))

cells.append(code(r'''X0 = op_on(X, 0, 2)      # little-endian: expect kron(I, X)
X1 = op_on(X, 1, 2)      # little-endian: expect kron(X, I)

hand_X0 = np.array([[0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=complex)

print("op_on(X, 0, 2) ==")
print(X0.real.astype(int))
print("matches the hand-written matrix:", np.allclose(X0, hand_X0))
print("equals np.kron(I, X) :", np.allclose(X0, np.kron(I2, X)))
print("equals np.kron(X, I) :", np.allclose(X0, np.kron(X, I2)), " <- the naive reading")
print()

e00 = np.zeros(4, dtype=complex)
e00[0] = 1
print("X on qubit 0 sends |00> to index", int(np.argmax(np.abs(X0 @ e00))),
      "  printed |01>   (CORRECT, matches Qiskit)")
print("X on qubit 1 sends |00> to index", int(np.argmax(np.abs(X1 @ e00))),
      "  printed |10>")
print()
print("The naive big-endian build, np.kron(X, I2), would send |00> to index",
      int(np.argmax(np.abs(np.kron(X, I2) @ e00))),
      " -- wrong qubit, no error message.")'''))

# ---------------------------------------------------------------- figure 2
cells.append(md(r"""### Figure 2 — the Rosetta table

Left: the same eight statevector indices, spelled out under both conventions. Both
columns write the register in the standard physics order $|q_2 q_1 q_0\rangle$, so
they are directly comparable — and the four highlighted rows are the ones where the
two conventions describe **different physical states while sharing an index**.
Those rows are where a big-endian simulator silently disagrees with Qiskit.

Right: the concrete case. One gate, one qubit, two answers."""))

cells.append(code(r'''n = 3
bits = [format(i, "03b") for i in range(2 ** n)]

fig = plt.figure(figsize=(11.4, 4.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], hspace=0.42, wspace=0.10)
axT = fig.add_subplot(gs[:, 0])
axW = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[1, 1])

# ---- the table
axT.set_axis_off()
axT.set_xlim(0, 1)
axT.set_ylim(0, 1)
xi, xb, xl, xn = 0.09, 0.36, 0.68, 0.90
top, dy = 0.80, 0.093

axT.text(0.0, 0.955, "One index, two readings", fontsize=11.5, color=style.INK,
         weight="medium", va="baseline")
axT.text(0.0, 0.905, "both columns written  " + grid.ket("q_2 q_1 q_0"),
         fontsize=8.5, color=style.MUTED, va="baseline")
for x, lab, col in [(xi, "index $i$", style.INK_2),
                    (xb, "BIG-endian\n(naive kron)", style.RED),
                    (xl, "little-endian\n(Qiskit, this repo)", style.BLUE)]:
    axT.text(x, top + 0.035, lab, fontsize=9, color=col, ha="center",
             va="bottom", weight="medium", linespacing=1.35)
axT.plot([0.0, 1.0], [top + 0.022, top + 0.022], color=style.AXIS, lw=1.0)

for i, b in enumerate(bits):
    y = top - 0.028 - i * dy
    big, little = b[::-1], b            # big-endian reads i's bits q0-first
    differ = big != little
    if differ:
        axT.add_patch(Rectangle((0.0, y - 0.030), 1.0, dy * 0.86,
                                facecolor=style.SEQ2(0.10), edgecolor="none",
                                zorder=0))
    axT.text(xi, y, str(i), fontsize=9.5, ha="center", color=style.INK_2)
    axT.text(xb, y, grid.ket(big), fontsize=11, ha="center",
             color=style.RED if differ else style.MUTED)
    axT.text(xl, y, grid.ket(little), fontsize=11, ha="center",
             color=style.BLUE if differ else style.MUTED)
    axT.text(xn, y, r"$\neq$" if differ else r"$=$", fontsize=10, ha="center",
             color=style.INK_2 if differ else style.MUTED)
axT.text(0.0, top - 0.028 - 8 * dy - 0.030,
         "shaded rows: same index, different physical state",
         fontsize=8.5, color=style.INK_2, va="top")

# ---- the concrete trap
wrong = np.kron(X, I2) @ e00        # naive left-to-right kron  -> big-endian
right = op_on(X, 0, 2) @ e00        # this repo's convention    -> little-endian

for ax, vec, head, col in [(axW, wrong, "np.kron(X, I) @ |00>   BIG-endian", style.RED),
                           (axR, right, "op_on(X, 0, 2) @ |00>   little-endian", style.BLUE)]:
    grid.amp_bars(ax, vec, labels=["00", "01", "10", "11"], ylim=1.28, hands=False)
    ax.set_title(head, loc="left", fontsize=9, color=col)
    ax.set_ylabel("|amp|")
    hit = int(np.argmax(np.abs(vec)))
    ax.annotate("X hit qubit " + ("1" if hit == 2 else "0"), xy=(hit, 1.02),
                xytext=(hit + 0.35, 1.19), fontsize=8.5, color=col, ha="left",
                va="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=0.9, shrinkA=2,
                                shrinkB=3))
axR.set_xlabel("Same code intent, different qubit moved. No exception is raised.",
               fontsize=8.5, color=style.INK_2)
plt.show()'''))

cells.append(md(r"""The bug this prevents is nasty precisely because it is quiet.
Every symmetric state — $|00\rangle$, $|{+}{+}\rangle$, the Bell state
$(|00\rangle+|11\rangle)/\sqrt2$ — is *invariant* under swapping the convention, so
a big-endian simulator passes all the tests you write first. It fails on the first
asymmetric circuit, which is usually the first interesting one."""))

# ---------------------------------------------------------------- figure 3
cells.append(md(r"""### Figure 3 — reading a 3-qubit amplitude plot

The signature plot from A01, now over $2^3 = 8$ basis states: bar height is
$|$amplitude$|$, bar colour and clock hand are the phase. The ruler underneath says
which character of the label is which qubit — this is the part people get wrong.

The state on the bottom row is built by $H$ on qubit 0, $H$ on qubit 2, then $S$ on
qubit 0. **Qubit 1 is never touched**, so the middle character of every populated
label stays `0`. Look for that: it is the endianness convention showing up as a
visible pattern rather than an assertion."""))

cells.append(code(r'''psi_x = op_on(X, 0, 3) @ np.eye(8, dtype=complex)[:, 0]

psi = np.eye(8, dtype=complex)[:, 0]
for U, k in [(H, 0), (H, 2), (S, 0)]:
    psi = op_on(U, k, 3) @ psi

fig = plt.figure(figsize=(9.6, 6.15))
gs = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1.05], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
axr = fig.add_subplot(gs[2])

grid.amp_bars(ax1, psi_x, ylim=1.25)
ax1.set_title("X on qubit 0 of  " + grid.ket("000") + "   ->  index 1", loc="left",
              fontsize=10)
grid.annotate(ax1, "index 1, not index 4", xy=(1, 1.02), xytext=(1.6, 1.12))

grid.amp_bars(ax2, psi, ylim=0.72)
ax2.set_title("H on qubit 0, H on qubit 2, S on qubit 0", loc="left", fontsize=10)
ax2.text(2.55, 0.635, "every populated label has 0 in the middle:\n"
                      "qubit 1 was never touched",
         fontsize=9, color=style.INK_2, ha="center", va="center", linespacing=1.45)

# ---- the ruler
axr.set_axis_off()
axr.set_xlim(0, 10)
axr.set_ylim(-0.75, 1.0)
xs = [4.15, 4.95, 5.75]
names = ["qubit 2", "qubit 1", "qubit 0"]
weights = [r"$\times\,4$", r"$\times\,2$", r"$\times\,1$"]
axr.text(3.55, 0.42, r"$|$", fontsize=22, ha="center", va="center", color=style.INK)
axr.text(6.35, 0.42, r"$\rangle$", fontsize=22, ha="center", va="center",
         color=style.INK)
for j, (x, nm, w) in enumerate(zip(xs, names, weights)):
    lead = j == 2
    axr.add_patch(Rectangle((x - 0.33, 0.42 - 0.26), 0.66, 0.52,
                            facecolor=style.SEQ2(0.16) if lead else style.GRID,
                            edgecolor=style.ORANGE if lead else style.AXIS,
                            lw=1.5 if lead else 0.9))
    axr.text(x, 0.42, f"$b_{2 - j}$", fontsize=13, ha="center", va="center",
             color=style.INK)
    axr.text(x, -0.02, nm, fontsize=8.5, ha="center", va="top",
             color=style.ORANGE if lead else style.INK_2)
    axr.text(x, -0.26, w, fontsize=8, ha="center", va="top", color=style.MUTED)
axr.annotate("", xy=(6.6, 0.42), xytext=(7.05, 0.42),
             arrowprops=dict(arrowstyle="<|-", color=style.ORANGE, lw=1.3,
                             mutation_scale=11))
axr.text(7.15, 0.42, "qubit 0 is the RIGHTMOST character\n(least significant bit)",
         fontsize=9, va="center", color=style.ORANGE, linespacing=1.4)
axr.text(0.15, 0.42, "index  $=\\;4b_2 + 2b_1 + b_0$", fontsize=10.5, va="center",
         color=style.INK)
axr.text(0.15, -0.10, "little-endian, the whole repo", fontsize=8.5, va="center",
         color=style.MUTED)
plt.show()

print("populated labels:", [format(i, "03b") for i in range(8) if abs(psi[i]) > 1e-9])
print("phases / pi     :",
      [round(float(np.angle(psi[i]) / np.pi), 3) for i in range(8) if abs(psi[i]) > 1e-9])'''))

# ---------------------------------------------------------------- figure 4
cells.append(md(r"""## 4. Where your laptop dies

$2^n$ complex numbers at `complex128` is $16 \cdot 2^n$ bytes. That is the entire
calculation, and it is worth plotting because the exponent makes intuition useless:
going from a comfortable 25 qubits to a merely-ambitious 35 is not "a bit more
memory", it is a thousand times more.

The plot below is log-scaled on $y$. A straight line on a log axis is exponential
growth — and this line is *perfectly* straight, all the way off the top."""))

cells.append(code(r'''GIB = 1024 ** 3
ns = np.arange(1, 61)
byts = 16.0 * 2.0 ** ns          # complex128 = 16 bytes per amplitude

fig, ax = plt.subplots(figsize=(9.6, 4.7))
ax.fill_between(ns, 8 * GIB, 1e21, color=style.RED, alpha=0.045, zorder=0)
ax.semilogy(ns, byts, color=style.BLUE, lw=2.2, zorder=3)

refs = [(8 * GIB, "8 GB  - a normal laptop", style.RED),
        (64 * GIB, "64 GB  - a big workstation", style.ORANGE),
        (1024 ** 5, "1 PB  - petabyte-class supercomputer", style.INK_2)]
for y, lab, col in refs:
    ax.axhline(y, color=col, lw=1.0, ls=(0, (5, 4)), zorder=2)
    ax.text(1.2, y * 2.0, lab, fontsize=8.5, color=col, va="bottom", ha="left")
    n_cross = np.log2(y / 16.0)
    ax.plot([n_cross], [y], marker="o", ms=6.5, color=col, zorder=5,
            markeredgecolor=style.SURFACE, markeredgewidth=1.1)

ax.annotate("29 qubits fills 8 GB.\n30 does not fit at all.",
            xy=(29.0, 8 * GIB), xytext=(31.5, 3e6), fontsize=9, color=style.RED,
            ha="left", va="center", linespacing=1.4,
            arrowprops=dict(arrowstyle="-", color=style.RED, lw=1.0, shrinkA=3,
                            shrinkB=4))
for n_mark, txt in [(20, "20 q\n16 MB"), (40, "40 q\n16 TB"), (50, "50 q\n16 PB"),
                    (60, "60 q\n16 EB")]:
    y = 16.0 * 2.0 ** n_mark
    ax.plot([n_mark], [y], marker="o", ms=5.0, color=style.BLUE, zorder=5,
            markeredgecolor=style.SURFACE, markeredgewidth=1.0)
    ax.text(n_mark - 0.9, y * 3.5, txt, fontsize=8, color=style.BLUE, ha="right",
            va="bottom", linespacing=1.3)

ax.set_xlim(0, 62)
ax.set_ylim(10, 1e20)
ax.set_xlabel("number of qubits $n$")
ax.set_ylabel("statevector memory (bytes, complex128)")
ax.set_xticks(range(0, 61, 10))
ax.set_title("A statevector is $16 \\cdot 2^n$ bytes. That is the whole problem.",
             loc="left")
ax.grid(axis="y")
plt.show()

for k in (10, 20, 30, 40, 50):
    print(f"n = {k:>2}:  {2**k:>16,d} amplitudes   {16 * 2.0**k / GIB:>22,.3f} GB")'''))

cells.append(md(r"""Two honest caveats on this plot, because it is often quoted
badly:

- It is the cost of storing a **general** state. Circuits with limited entanglement
  can be simulated far past 50 qubits with tensor networks, and stabilizer circuits
  (Clifford-only) run on thousands of qubits in polynomial time. The wall is real,
  but it is a wall around *one* method.
- It says nothing about whether the quantum computer is *useful*. 60 qubits of
  noisy hardware is not automatically beyond a laptop; A07 and the C-series come
  back to this.

## 5. Marginals — what one qubit looks like on its own

Given the joint state, the distribution of a single qubit $k$ is obtained by
**summing the joint probabilities over everything else**:

$$P(b_k = 1) = \sum_{i \,:\, \text{bit } k \text{ of } i = 1} |\psi_i|^2 .$$

That is a *marginal*, and the little-endian convention makes it a one-line bit
test. The interesting question is the converse: do the marginals determine the
joint distribution? For independent (product) states, yes — the joint factorises.
For entangled states, **no**, and that failure is exactly what entanglement is."""))

cells.append(code(r'''def marginal(psi, k, n):
    """P(qubit k = 0), P(qubit k = 1) for a little-endian statevector."""
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(2 ** n)
    b = (idx >> k) & 1
    return np.array([p[b == 0].sum(), p[b == 1].sum()])


prod = register(PLUS, PLUS)                                  # |+> on both qubits
bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)    # (|00> + |11>)/sqrt(2)

for name, st in [("|+>|+>", prod), ("Bell ", bell)]:
    m0, m1 = marginal(st, 0, 2), marginal(st, 1, 2)
    print(f"{name}  P(q0)= {np.round(m0, 4)}   P(q1)= {np.round(m1, 4)}")
print("\nIdentical marginals -- and yet:")
print("np.allclose(prod, bell) ->", np.allclose(prod, bell))'''))

# ---------------------------------------------------------------- figure 5
cells.append(md(r"""### Figure 5 — marginals throw information away

Bars are the true joint distribution; the black ticks are what you would predict
from the two marginals alone, assuming independence. On the left they land exactly
on the bars. On the right they miss every one of them.

Both states have the *same* single-qubit marginals — a fair coin on each qubit. The
difference lives entirely in the correlations, which no per-qubit picture can
show."""))

cells.append(code(r'''fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), sharey=True)
for ax, (name, st) in zip(axes, [("product:  " + grid.ket("+") + grid.ket("+"), prod),
                                 ("entangled:  Bell state", bell)]):
    m0, m1 = marginal(st, 0, 2), marginal(st, 1, 2)
    joint = np.abs(st) ** 2
    # little-endian: index = 2*b1 + b0, so the outer product is m1 (rows) x m0 (cols)
    predicted = np.outer(m1, m0).ravel()
    grid.prob_bars(ax, joint, analytic=predicted, labels=["00", "01", "10", "11"],
                   ymax_pad=1.45)
    ax.set_title(name, loc="left", fontsize=10)
    # prob_bars labels its artists "measured"/"analytic"; matplotlib returns
    # lines before patches, so map by NAME rather than by position.
    rename = {"measured": "true joint", "analytic": r"$P(q_1)\times P(q_0)$"}
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h, [rename[x] for x in lab], loc="upper center", ncol=2, fontsize=8)
axes[1].set_ylabel("")
fig.suptitle("Same marginals, different states - independence fails on the right",
             x=0.005, ha="left", fontsize=11.5)
plt.show()

print("Bell:  predicted from marginals =", np.round(np.outer(marginal(bell, 1, 2),
                                                             marginal(bell, 0, 2)).ravel(), 4))
print("Bell:  actual joint             =", np.round(np.abs(bell) ** 2, 4))'''))

# ------------------------------------------------------- product vs entangled
cells.append(md(r"""## 6. Product states, and states that refuse to factor

Now the main event. A two-qubit state is a **product state** if there exist
single-qubit states $|a\rangle, |b\rangle$ with

$$|\psi\rangle = |b\rangle \otimes |a\rangle \quad\text{(qubit 1)} \otimes \text{(qubit 0)}.$$

Write the four amplitudes as a $2\times2$ matrix $C$ with rows indexed by $b_1$ and
columns by $b_0$ — which, because index $=2b_1+b_0$, is literally
`psi.reshape(2, 2)`. Then the product condition says

$$C_{b_1 b_0} = b_{b_1}\, a_{b_0}, \qquad\text{i.e.}\qquad C = \mathbf{b}\,\mathbf{a}^{T}.$$

An outer product of two vectors is exactly a **rank-1 matrix**. So:

$$\boxed{\ |\psi\rangle \text{ is a product state} \iff \operatorname{rank} C = 1\ }$$

Try it on the Bell state $C = \tfrac{1}{\sqrt2}\begin{pmatrix}1 & 0\\ 0 & 1\end{pmatrix}$.
That is $\tfrac{1}{\sqrt2}$ times the identity — rank 2. No pair of single-qubit
states produces it. If it did, we would need $b_0 a_1 = 0$, so $b_0 = 0$ or
$a_1 = 0$; either kills one of the diagonal entries, which are both nonzero.
Contradiction. **The Bell state is not two qubits with individual states.** It is
one object.

The graded version of "rank" is the list of **singular values** of $C$. One nonzero
singular value = product. Two = entangled, and how *unequal* they are measures how
weakly. That list is the Schmidt spectrum, and its entropy is the entanglement
entropy — B18's subject. We are getting an honest preview for free, out of a
`reshape` and one call to `svd`."""))

cells.append(code(r'''def amp_matrix(psi, n=2, k=0):
    """Reshape a statevector into the amplitude matrix across the cut {k} | rest.

    For k = 0 this is just psi.reshape(2**(n-1), 2): rows are the high qubits,
    columns are qubit 0 - which is exactly the little-endian index split.
    """
    if k != 0:
        raise NotImplementedError("only the qubit-0 cut is needed here")
    return np.asarray(psi).reshape(2 ** (n - 1), 2)


def schmidt(psi, n=2):
    """Singular values of the amplitude matrix. Rank 1 <=> product state."""
    return np.linalg.svd(amp_matrix(psi, n), compute_uv=False)


t = 0.20 * np.pi
partial = np.array([np.cos(t), 0, 0, np.sin(t)], dtype=complex)

trio = [("product\n" + r"$|0\rangle_1 |+\rangle_0$", register(PLUS, KET0)),
        ("partly entangled\n" + r"$\cos t\,|00\rangle + \sin t\,|11\rangle$", partial),
        ("maximally entangled\n" + r"Bell $(|00\rangle + |11\rangle)/\sqrt{2}$", bell)]

for name, st in trio:
    sv = schmidt(st)
    print(f"{name.splitlines()[0]:>20}  singular values = {np.round(sv, 4)}   "
          f"rank = {np.linalg.matrix_rank(amp_matrix(st), tol=1e-9)}")'''))

# ---------------------------------------------------------------- figure 6
cells.append(md(r"""### Figure 6 — rank 1 versus rank 2, visually

Top row: the amplitude matrix $C$. Bottom row: its singular values.

In the left panel the two rows of $C$ are proportional (one is zero, the other
isn't) — that is what rank 1 *looks* like, and it is why a single singular value
carries everything. Move right and the second singular value climbs out of the
floor. It is not a threshold or a heuristic; it is a linear-algebra fact about
whether an outer-product factorisation exists."""))

cells.append(code(r'''fig, axes = plt.subplots(2, 3, figsize=(10.4, 5.5),
                         gridspec_kw={"height_ratios": [1.35, 1.0]})
for j, (name, st) in enumerate(trio):
    C = amp_matrix(st)
    grid.matrix(axes[0, j], C, part="re", labels=[r"$b_0=0$", r"$b_0=1$"],
                cbar=False, annot=True, vmax=1.0, fmt="{:+.3f}")
    axes[0, j].set_yticklabels([r"$b_1=0$", r"$b_1=1$"], fontsize=8)
    axes[0, j].set_title(name, loc="left", fontsize=9.5, linespacing=1.45)

    sv = schmidt(st)
    rank = int(np.sum(sv > 1e-9))
    cols = [style.BLUE, style.ORANGE]
    axes[1, j].bar([0, 1], sv, width=0.55, color=cols, edgecolor=style.SURFACE,
                   linewidth=1.2, zorder=3)
    for i, s in enumerate(sv):
        axes[1, j].text(i, s + 0.035, f"{s:.3f}", ha="center", va="bottom",
                        fontsize=8.5, color=cols[i])
    axes[1, j].axhline(0, color=style.AXIS, lw=0.9)
    axes[1, j].set_xticks([0, 1])
    axes[1, j].set_xticklabels([r"$\sigma_1$", r"$\sigma_2$"], fontsize=10)
    axes[1, j].set_ylim(0, 1.28)
    axes[1, j].set_xlim(-0.6, 1.6)
    axes[1, j].set_ylabel("singular value" if j == 0 else "")
    axes[1, j].set_title(f"rank {rank}  ->  " + ("PRODUCT" if rank == 1 else "ENTANGLED"),
                         loc="left", fontsize=9.5,
                         color=style.BLUE if rank == 1 else style.ORANGE)
fig.suptitle("Reshape the amplitudes into a matrix; the rank answers the question",
             x=0.005, ha="left", fontsize=11.5)
plt.show()'''))

cells.append(md(r"""The same test works for any bipartition of any number of
qubits: reshape into a matrix whose rows index one side of the cut and whose
columns index the other, then look at the singular values. A single nonzero value
means the two halves factorise; several means they do not. Nothing about it is
special to two qubits — the $2\times2$ case is just the one you can check by
eye."""))

# ---------------------------------------------------------------- limits
cells.append(md(r"""## Honest limits — what the bar chart does not tell you

The amplitude plot is a **complete** description of a pure state: every complex
number is on the page, height and hue. Completeness is not the same as
readability, and it is definitely not the same as insight.

- **It stops being readable at about five qubits.** 32 bars still works; 128 is a
  texture; 1024 is a smear. Past that the plot is decoration.
- **It does not show entanglement at a glance.** This is the big one. A product
  state and a maximally entangled state can produce plots with identical
  *character* — same spread, same busyness — because entanglement is a property of
  the *correlations*, not of any individual amplitude. The rank/singular-value
  test from Figure 6 sees it instantly; your eye does not.
- **It is per-basis.** The same state in a different basis looks completely
  different, so "concentrated" or "spread out" are statements about the basis as
  much as the state.
- **Pure states only.** A statistical mixture cannot be written as one amplitude
  vector at all; that needs a density matrix (A12).

Figure 7 makes the second point concrete."""))

cells.append(code(r'''def haar_state(n, rng):
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return v / np.linalg.norm(v)


def haar_qubit(rng):
    v = rng.normal(size=2) + 1j * rng.normal(size=2)
    return v / np.linalg.norm(v)


n5 = 5
prod5 = register(*[haar_qubit(rng) for _ in range(n5)])   # product by construction
ent5 = haar_state(n5, rng)                                # generically entangled
big7 = haar_state(7, rng)

fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))
for ax, (name, st, nq) in zip(axes, [
        ("5 qubits, PRODUCT state", prod5, 5),
        ("5 qubits, ENTANGLED state", ent5, 5),
        ("7 qubits - 128 bars, unreadable", big7, 7)]):
    # edge=False past ~64 bars: a 1.2pt white bar edge is wider than the bar
    # itself and erases the data.
    grid.amp_bars(ax, st, n_qubits=nq, edge=nq <= 5)
    ax.set_title(name, loc="left", fontsize=9.5)
    if nq == 7:
        ticks = [0, 31, 63, 95, 127]
    else:
        ticks = list(range(0, 2 ** nq, 2))      # every 2nd label; all 32 collide
    ax.set_xticks(ticks)
    ax.set_xticklabels([grid.ket(format(i, f"0{nq}b")) for i in ticks],
                       fontsize=6.5, rotation=90)
    if ax is not axes[0]:
        ax.set_ylabel("")
fig.suptitle("The first two panels look alike. Only one factorises.",
             x=0.005, ha="left", fontsize=11.5)
plt.show()

for name, st in [("product ", prod5), ("entangled", ent5)]:
    sv = schmidt(st, n=5)
    print(f"{name}: cut {{q0}} | rest -> singular values {np.round(sv, 5)}   "
          f"rank {int(np.sum(sv > 1e-9))}")'''))

cells.append(md(r"""The two 5-qubit panels are visually interchangeable. The
numbers underneath are not: the product state has a second singular value of
numerical zero across the cut, the random state does not. That gap is the whole
difference between "five qubits" and "one 32-dimensional object".

## Checkpoint"""))

cells.append(code(r'''E = np.eye(2, dtype=complex)

# 1. Dimension is 2**n, and register() preserves the norm.
for k in range(1, 8):
    v = register(*[haar_qubit(rng) for _ in range(k)])
    assert v.size == 2 ** k
    assert np.isclose(np.linalg.norm(v), 1.0)

# 2. op_on matches the independent index-formula construction everywhere.
for n in range(1, 6):
    for k in range(n):
        for U in (X, Z, H, S):
            assert np.allclose(op_on(U, k, n), op_on_by_index(U, k, n))

# 3. Hand-checked little-endian placements.
assert np.allclose(op_on(X, 0, 2), np.kron(I2, X))          # qubit 0 -> RIGHTmost
assert np.allclose(op_on(X, 1, 2), np.kron(X, I2))
assert np.allclose(op_on(H, 1, 3), reduce(np.kron, [I2, H, I2]))
assert np.allclose(op_on(S, 2, 3), reduce(np.kron, [S, I2, I2]))
assert np.allclose(op_on(X, 0, 2), hand_X0)

# 4. THE endianness assertion: X on qubit 0 of 2 qubits gives index 1, not 2.
e0 = np.eye(4, dtype=complex)[:, 0]
assert int(np.argmax(np.abs(op_on(X, 0, 2) @ e0))) == 1        # |01>
assert int(np.argmax(np.abs(op_on(X, 1, 2) @ e0))) == 2        # |10>
# ...and the naive big-endian build really does disagree.
assert int(np.argmax(np.abs(np.kron(X, I2) @ e0))) == 2
assert not np.allclose(op_on(X, 0, 2), np.kron(X, I2))
# 3 qubits: X on qubit k must light up index 2**k.
for k in range(3):
    e = np.eye(8, dtype=complex)[:, 0]
    assert int(np.argmax(np.abs(op_on(X, k, 3) @ e))) == 2 ** k

# 5. register() puts qubit 0 last in the kron chain.
a, b, c = haar_qubit(rng), haar_qubit(rng), haar_qubit(rng)
assert np.allclose(register(a, b), np.kron(b, a))
assert np.allclose(register(a, b, c), reduce(np.kron, [c, b, a]))

# 6. op_on is unitary and preserves the norm of any state.
for n in (2, 3, 4):
    for k in range(n):
        for U in (X, H, S):
            M = op_on(U, k, n)
            assert np.allclose(M.conj().T @ M, np.eye(2 ** n))
            v = haar_state(n, rng)
            assert np.isclose(np.linalg.norm(M @ v), 1.0)

# 7. Operators on DIFFERENT qubits commute; on the same qubit generally not.
assert np.allclose(op_on(X, 0, 3) @ op_on(H, 2, 3), op_on(H, 2, 3) @ op_on(X, 0, 3))
assert not np.allclose(op_on(X, 0, 3) @ op_on(H, 0, 3), op_on(H, 0, 3) @ op_on(X, 0, 3))

# 8. The rank test separates product from entangled states.
assert np.linalg.matrix_rank(amp_matrix(register(PLUS, KET0)), tol=1e-9) == 1
assert np.linalg.matrix_rank(amp_matrix(register(MINUS, PLUS)), tol=1e-9) == 1
assert np.linalg.matrix_rank(amp_matrix(bell), tol=1e-9) == 2
assert np.allclose(np.sort(schmidt(bell)), [1 / np.sqrt(2)] * 2)
assert np.isclose(schmidt(partial)[0], np.cos(t)) and np.isclose(schmidt(partial)[1],
                                                                 np.sin(t))
for _ in range(20):                       # random products are always rank 1
    assert schmidt(register(haar_qubit(rng), haar_qubit(rng)))[1] < 1e-12
for _ in range(20):                       # random states essentially never are
    assert schmidt(haar_state(2, rng))[1] > 1e-6

# 9. Singular values of any amplitude matrix square-sum to the norm.
for n in (2, 3, 5):
    v = haar_state(n, rng)
    assert np.isclose(np.sum(schmidt(v, n) ** 2), 1.0)

# 10. Marginals: product states factorise, the Bell state does not.
p = register(haar_qubit(rng), haar_qubit(rng))
assert np.allclose(np.abs(p) ** 2,
                   np.outer(marginal(p, 1, 2), marginal(p, 0, 2)).ravel())
assert not np.allclose(np.abs(bell) ** 2,
                       np.outer(marginal(bell, 1, 2), marginal(bell, 0, 2)).ravel())
assert np.allclose(marginal(bell, 0, 2), [0.5, 0.5])
for n in (2, 3, 4):
    v = haar_state(n, rng)
    for k in range(n):
        assert np.isclose(marginal(v, k, n).sum(), 1.0)

# 11. Kronecker structure: the block claim from Figure 1.
assert np.allclose(np.kron(A, B)[0:2, 2:4], A[0, 1] * B)
assert np.allclose(np.kron(A, B)[2:4, 0:2], A[1, 0] * B)

print("A05 checkpoint passed.")'''))

cells.append(md(r"""---

**What to carry forward.** Three things:

1. `register(q0, q1, ...)` and `op_on(U, k, n)` — with the `[::-1]`. Every Track A
   notebook from here on uses this exact convention, and Track B's
   `np.allclose(mine, Statevector(qc).data)` assertions are only possible because
   of it.
2. $2^n$, not $2n$. The wall at ~30 qubits on a laptop is a fact about the method,
   and A07 exploits the structure to push it as far as it will go.
3. Reshape the amplitudes into a matrix and look at the rank. Rank 1 is a product;
   anything more is entanglement you cannot talk away.

**Next:** [A06 — Two-Qubit Gates and Controlled Operations](A06_Two_Qubit_Gates.ipynb).
We can now place any single-qubit gate anywhere in a register — but nothing we have
built so far can *create* a rank-2 state from a rank-1 one. Single-qubit gates
simply cannot. The next notebook builds the gates that can."""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.13.7"},
}

if __name__ == "__main__":
    import pathlib

    pathlib.Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, OUT)
    print("wrote", OUT)
