"""Build solutions/03_move_grid_sensors_qaoa_solution.ipynb (notebook 3: grid sensor placement, minimum vertex cover, QAOA).

    python tools/build_nb03.py

Notebook 3 sets the QAOA pattern that notebook 4 follows.  The reusable pieces are the three helpers defined in the notebook:
`success_of` (exact probability of the right answers), `optimise` (QAOA at depth p: best of six starts) and `draw_grid`-style
plotting.  Everything else (energy function, sampler, MinimumEigenOptimizer shortcut) comes from qff and Qiskit.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, arabic_table, callout, details, hints, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1 = """**The problem.** A power grid is a network of substations joined by lines. A sensor placed at a substation (think of a phasor measurement unit, a PMU, which reports the state of the grid many times a second) watches every line that touches that substation. Sensors cost money, so: place the *fewest* sensors so that every line is watched by at least one of them. Mathematicians call this **minimum vertex cover**. (Real placement rules can be richer; the paper and this notebook use the vertex-cover framing.)

**Who faces it.** A national grid operator, or any transmission utility that plans where to put its monitoring equipment.

""" + callout("real", """**What is real today.** The WEF/Aramco paper (Case 12, p. 26) frames sensor placement as a minimum vertex cover problem and reports neutral-atom quantum processors, with decomposition (cutting a big graph into smaller pieces), on instances of 80 qubits. The classical incumbents are mature: integer programming, local search and greedy rules are strong on graphs of this size. As far as we know, nobody has shown the quantum route beating them (see the rule in section 0).""") + """

**Why a quantum computer might help.** The number of possible placements doubles with every substation, and a quantum state can carry amplitude on all of them at once; QAOA's layers try to steer that amplitude towards good placements by interference. What is still missing: today's devices are noisy, and no one has shown the steering beats the best classical solvers at a useful size.

| Paper's method | Our teaching route |
|---|---|
| Neutral-atom processors with decomposition, 80-qubit instances (Case 12, p. 26) | QAOA on a 6-qubit simulator in Qiskit: the gate-model counterpart, chosen because every line of it can be run and read in a day |"""

S2_MD = """Six substations and seven lines, drawn as a 2×3 ladder. We number the substations 0 to 5 and write each line as a pair. A **choice** is a list of six 0/1 values `x = (x0, ..., x5)`: `x_i = 1` means "put a sensor at substation i". Six yes/no questions give 2⁶ = 64 choices, so six qubits are enough to hold one amplitude per choice."""

S2_INSTANCE = """QUBITS = 6                       # the test runner reads this
import time
import networkx as nx

edges = [(0, 1), (1, 2), (0, 3), (1, 4), (2, 5), (3, 4), (4, 5)]
G = nx.Graph()
G.add_nodes_from(range(6))
G.add_edges_from(edges)
POS = {i: (i % 3, -(i // 3)) for i in range(6)}          # the ladder: nodes 0-1-2 on top, 3-4-5 below


def draw_grid(chosen=(), title="", ax=None):
    \"\"\"Draw the grid. Substations in `chosen` get a sensor (blue); lines no sensor touches are drawn red and dashed.\"\"\"
    ax = ax or plt.subplots(figsize=(4, 2.4))[1]
    bad = [e for e in edges if not set(e) & set(chosen)]                   # lines that touch no chosen substation
    nx.draw_networkx_edges(G, POS, ax=ax, edgelist=[e for e in edges if e not in bad], width=2, edge_color="#0F62FE")
    nx.draw_networkx_edges(G, POS, ax=ax, edgelist=bad, width=2, edge_color="#FA4D56", style="dashed")
    nx.draw_networkx_nodes(G, POS, ax=ax, node_color=["#0F62FE" if i in chosen else "#E0E0E0" for i in G.nodes],
                           edgecolors="#161616", node_size=500)
    nx.draw_networkx_labels(G, POS, ax=ax, font_color="white" if len(chosen) else "black", font_size=11)
    ax.set_title(title + (f"  ({len(bad)} line(s) unwatched)" if bad else ""), fontsize=10)
    ax.set_axis_off()
    return ax


draw_grid(title="the grid: 6 substations, 7 lines");"""

S2_CL_MD = """**Classical baseline first.** There are only 64 choices, so we can simply try them all. A choice is a *cover* if no line is left unwatched. We also try a greedy rule: keep picking the substation that touches the most unwatched lines."""

S2_CLASSICAL = """# Classical baseline: always runs first.
from qff.bits import index_to_x


def is_cover(x):
    \"\"\"True if every line has a sensor at one end (x is a 0/1 list, x[i] = 1 means a sensor at substation i).\"\"\"
    return all(x[i] + x[j] >= 1 for i, j in edges)


t0 = time.perf_counter()
all_x = [tuple(index_to_x(k, QUBITS)) for k in range(2 ** QUBITS)]
covers = [x for x in all_x if is_cover(x)]
classical_min_size = min(sum(x) for x in covers)
classical_optima = [x for x in covers if sum(x) == classical_min_size]
brute_seconds = time.perf_counter() - t0
print(f"choices checked: {len(all_x)} | covers: {len(covers)} | smallest cover: {classical_min_size} sensors")
print("optimal covers (x0..x5):", classical_optima, "-> substations", [[i for i in range(6) if x[i]] for x in classical_optima])

# Greedy: repeatedly take the substation touching the most unwatched lines (ties: lowest number).
t0 = time.perf_counter()
left, greedy_nodes = list(edges), []
while left:
    degree = {i: sum(i in e for e in left) for i in range(6)}
    pick = max(degree, key=lambda i: (degree[i], -i))
    greedy_nodes.append(pick)
    left = [e for e in left if pick not in e]
greedy_seconds = time.perf_counter() - t0
print("greedy picks:", greedy_nodes, "-> size", len(greedy_nodes))

fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
draw_grid([1, 4], "{1, 4}: not a cover", ax=axes[0])
draw_grid([1, 3, 5], "{1, 3, 5}: an optimal cover", ax=axes[1])"""

S2_NOTE = """Greedy lands on an optimal cover here, which is luck of the small example: greedy has no guarantee on other graphs. Both methods answer in a blink. That is the honest starting point for everything below: **on this toy the classical methods are instant, and QAOA cannot beat instant.** The toy exists so you can see how the quantum method works."""

S3_ROADMAP = r"""### 3.0 · The map of this section

""" + callout("idea", r"""**QAOA in one sentence.** Turn the cost of every choice into a *twist* of a quantum state, *shake* the choices together so that the twists add up for some choices and cancel for others, and let a classical optimiser tune how hard to twist and shake until the good choices are the likely ones.""") + r"""

Everything below builds that sentence from two things you already know: lists of numbers (vectors) and probability. Every step has a **numbered equation**, and right after it the **code cell that implements it**, so you can watch the maths run and check it against Qiskit.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | A quantum state is a list of 64 numbers; probability = number² | (1) | state cell | |
| 3.2 | The rule becomes a cost for every choice | (2) | cost cell | **T1** |
| 3.3 | The cost becomes a Hamiltonian (an energy) | (3) | `qubo.to_ising()` | |
| 3.4 | The circuit: start, cost layer, mixer, repeat | (4) to (7) | `qaoa_ansatz`, `qaoa_by_hand` | |
| 3.5 | A worked example you can do on paper (2 substations) | (4) to (8) | mini-example cell | |
| 3.6 | Average cost and success probability | (8), (9) | `energy`, `success_of` | **T2** |
| 3.7 | Measuring, and the optimiser loop | (10) | sampling and `optimise` cells | |
| 3.8 | More layers, why it works, the penalty | (7) again | layer cells | **T3** |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it."""

S3_FLOW = """qff.plots.plot_flow(
    [("The rule", "every line watched", "source"), ("Cost C(x)", "equation 2", "move"), ("Hamiltonian", "equation 3", "store"),
     ("Circuit", "equations 4-7", "use"), ("Measure", "equations 8-9", "source")],
    loop=(4, 3, "the classical optimiser retunes the circuit's knobs (equation 10)"), title="QAOA: from rule to answer");"""

S3_STATE_MD = r"""### 3.1 · From bits to a quantum state

**Start with what you know.** A *choice* is a list of six bits $x=(x_0,\dots,x_5)$, where $x_i=1$ means "sensor at substation $i$". There are $2^6=64$ such lists. An ordinary computer holds *one* of them at a time. Six qubits hold a list of **64 numbers**, one per choice. These numbers are called **amplitudes** and written $\alpha_x$:

$$|\psi\rangle=\sum_{x}\alpha_x\,|x\rangle,\qquad p(x)=|\alpha_x|^2,\qquad \sum_x |\alpha_x|^2=1 \tag{1}$$

How to read equation (1):

- $|x\rangle$ (say "ket $x$") is just a *name tag* for the choice $x$. As a vector it is a column of 64 numbers with a single 1 at position $x$ and 0 everywhere else. For one qubit, $|0\rangle=\binom{1}{0}$ and $|1\rangle=\binom{0}{1}$.
- $\alpha_x$ is the amplitude of choice $x$. It may be negative, or even complex. Its **squared size** $|\alpha_x|^2$ is the **probability** of reading out $x$ when you **measure**.
- The 64 probabilities add up to 1, as probabilities must.

""" + callout("math", r"""**Complex numbers in 60 seconds.** An amplitude may be a complex number. All you need is one fact: $e^{i\theta}=\cos\theta+i\sin\theta$ is a point on a circle of radius 1, at angle $\theta$. Multiplying an amplitude by $e^{i\theta}$ **turns it by the angle $\theta$ and does not change its size**, so $|\alpha|^2$ stays the same. That turning angle is called the **phase**. A measurement cannot see a phase on its own, but phases decide how amplitudes *add or cancel* later (this is **interference**): two amplitudes of size $\tfrac12$ that point the same way add up to size 1; pointing opposite ways they cancel to 0.""", "Complex numbers, only what we need") + "\n\n" + callout("watch", r"""**Bit order.** Choice $x$ sits at position $k=\sum_i x_i\,2^i$ of the list, so $x_0$ is the *least* significant bit. Qiskit prints strings with qubit 0 on the **right**: the string `'110'` means $x=(0,1,1)$. Never index count strings by hand; use `qff.bits.to_x` and `index_to_x`, as the cells below do.""") + "\n\n" + arabic_table([
    ("Quantum state", "الحالة الكمومية", "I"), ("Qubit", "كيوبت", "G"), ("Amplitude", "السعة", "G"), ("Probability", "الاحتمال", "I"),
    ("Measurement", "القياس", "G"), ("Superposition", "التراكب", "G"), ("Phase", "الطور", "U"), ("Interference", "التداخل", "G"),
    ("Basis state (one choice)", "حالة القاعدة", "I"), ("Complex number", "العدد المركّب", "I")])

S3_STATE_CODE = """# Equation (1) in code: a state is a list of 64 amplitudes; the probabilities are their squared sizes.
alpha = np.zeros(2 ** QUBITS, dtype=complex)
alpha[0] = 1                                              # |000000>: all the weight on choice number 0 (no sensors)
print("all weight on one choice :", "sum of |alpha|^2 =", np.sum(np.abs(alpha) ** 2))
alpha = np.full(2 ** QUBITS, 1 / 8, dtype=complex)        # 64 equal amplitudes of 1/8
print("64 equal amplitudes of 1/8:", "p(x) = |alpha_x|^2 =", abs(alpha[5]) ** 2, "= 1/64 =", 1 / 64, "| sum =", np.sum(np.abs(alpha) ** 2))"""

S3_COST_MD = r"""### 3.2 · The cost of a choice

The rule is "every line has a sensor at one end or at both": $x_i+x_j\ge 1$ for each line $(i,j)$. We turn the rule into one number that we can make small:

$$C(x)=\underbrace{\sum_{i} x_i}_{\text{sensors}}\;+\;P\,\underbrace{\sum_{(i,j)\in E}(1-x_i)(1-x_j)}_{\text{unwatched lines}} \tag{2}$$

- $\sum_i x_i$ counts the sensors: each costs 1.
- $(1-x_i)(1-x_j)$ is 1 **only when both ends have no sensor** ($x_i=x_j=0$) and 0 otherwise, so the sum counts the **unwatched lines**.
- $P$ is the **penalty**: the price of one unwatched line. We use $P=2$ (section 3.8 shows why its size matters).
- $E$ is the set of the 7 lines.

So $C(x)$ = (number of sensors) + $P\times$(number of unwatched lines). A choice that watches every line pays no penalty, so its cost is just its number of sensors, and **the cheapest choice is the answer**. Three examples you can check by hand: no sensors at all costs $0+2\cdot 7=14$; sensors at $\{1,4\}$ cost $2+2\cdot 2=6$ (lines $(0,3)$ and $(2,5)$ are unwatched); sensors at $\{1,3,5\}$ cost $3+0=3$.

""" + arabic_table([
    ("Cost function", "دالة التكلفة", "I"), ("Penalty", "العقوبة (penalty)", "I"), ("Constraint", "القيد", "I"),
    ("Optimization", "الأمثَلة (التحسين)", "I"), ("Vertex cover", "الغطاء الرأسي", "I"), ("Graph", "الرسم البياني", "U"),
    ("Substation", "محطة تحويل", "I"), ("Transmission line", "خط نقل", "I"), ("Sensor", "مستشعر", "I")])

S3_COST_CODE = '''# Equation (2) in code, and the cost of all 64 choices.
PENALTY = 2                                                       # P in equation (2)
x_all = [index_to_x(k, QUBITS) for k in range(2 ** QUBITS)]       # all 64 choices, x_all[k] = x of basis state k


def cost_by_formula(x):
    """Equation (2): number of sensors + P * number of unwatched lines."""
    return sum(x) + PENALTY * sum((1 - x[i]) * (1 - x[j]) for i, j in edges)


examples = [("no sensors", []), ("{1, 4}", [1, 4]), ("{1, 3, 5}", [1, 3, 5])]
fig, axes = plt.subplots(1, 4, figsize=(12, 2.4))
for ax, (label, nodes) in zip(axes, examples):
    x = [1 if i in nodes else 0 for i in range(QUBITS)]
    draw_grid(nodes, f"{label}: cost {cost_by_formula(x)}", ax=ax)
levels, counts = np.unique([cost_by_formula(x) for x in x_all], return_counts=True)
axes[3].bar(levels, counts, color="#0F62FE")
axes[3].set_xlabel("cost C(x)"); axes[3].set_ylabel("number of choices"); axes[3].set_title("all 64 choices by cost", fontsize=10)
plt.tight_layout()
print("cost -> how many of the 64 choices have it:", dict(zip(levels.tolist(), counts.tolist())))'''

S3_COST_TRY = callout("try", """Add your own example to the list, say `("{0, 1}", [0, 1])`, and **predict its cost on paper first**, then run the cell. Then look at the bar chart: how many choices have the lowest possible cost? (Compare with the number of optimal covers found by the classical baseline in section 2.)""")

T1_INTRO = r"""### T1 · Warm-up: spot the unwatched lines · التمرين ١

**What you are about to build, and why.** Equation (2) charges a penalty for every *unwatched line*. Before a computer can charge it, it must be able to *find* those lines. In T1 you write the small function `uncovered(chosen)` that returns them, find a 3-substation cover by eye, and check yourself against the classical brute-force baseline (Qiskit's `QuadraticProgram` is the container that holds the problem; the next cell builds it).

**The formula you implement.** A line $(i,j)$ is unwatched when neither end has a sensor: $(1-x_i)(1-x_j)=1$. In plain Python, with $S$ the list `chosen`: $i\notin S$ **and** $j\notin S$.

""" + hints(
    r"""A line is *unwatched* when neither of its two ends has a sensor. Say it in a sentence with the words "neither" and "end", then turn the sentence into a test for one line. The red dashed lines in the pictures of 3.2 are the unwatched ones.""",
    r"""Go through every line in `edges`; keep a line only if both of its ends fail the test "is in `chosen`"; return the kept lines as a list. For `my_cover`, look for three substations that between them touch all seven lines (the two hubs are not enough). `brute_force` returns three things: the best choice, its value, and a table.""",
    r"""```python
def uncovered(chosen):
    return ______          # the lines of `edges` that no substation in `chosen` touches

my_cover = ______          # three substation numbers; no two of them neighbours
bf_x, bf_value, bf_table = ______   # the classical baseline on the QuadraticProgram
```
Test yourself: `uncovered([1, 4])` must give `[(0, 3), (2, 5)]`, and `uncovered(my_cover)` must give `[]`.""")

S3_INTRO = """### 3.2 (continued) · The problem as Qiskit sees it
Qiskit keeps "minimise this, subject to these rules" in a **QuadraticProgram**. The next cell builds one for our graph (Qiskit has a ready-made `VertexCover` class) and prints it: the objective is just the number of sensors, and every line gives one rule $x_i+x_j\\ge 1$."""

S3_HAM_MD = r"""### 3.3 · From cost to Hamiltonian

A quantum computer does not read a Python function `C(x)`. It works with **operators** that act on the state. An operator is a *matrix*: a square table of numbers that turns one state vector into another. We need a matrix whose "energy" for choice $x$ is exactly $C(x)$. Physicists call such a matrix a **Hamiltonian** (an energy operator), and a one-line trick builds it.

The **Pauli $Z$ gate** is the matrix $Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$: it leaves $|0\rangle$ alone and flips the sign of $|1\rangle$, so $Z|0\rangle=(+1)|0\rangle$ and $Z|1\rangle=(-1)|1\rangle$. We say $|0\rangle$ and $|1\rangle$ are **eigenstates** of $Z$, with **eigenvalues** $+1$ and $-1$. Now look at $\tfrac{1-Z}{2}$: its eigenvalue is $0$ on $|0\rangle$ and $1$ on $|1\rangle$, which is exactly the value of a bit. So replace every bit in equation (2) by an operator, $x_i\to\frac{1-Z_i}{2}$, and multiply out:

$$\begin{aligned}H_C&=c_0+\sum_{i}h_i\,Z_i+\sum_{(i,j)\in E}J\,Z_iZ_j,\\ h_i&=-\frac12+\frac P4\,d_i,\qquad J=\frac P4,\qquad c_0=\frac n2+\frac P4\,|E|\end{aligned} \tag{3}$$

Here $d_i$ is the number of lines that touch substation $i$ (its *degree*), $n=6$ qubits and $|E|=7$ lines. A Hamiltonian made only of $Z$ terms like this one is called an **Ising** Hamiltonian (that is why the Qiskit method is called `to_ising`). The key property of $H_C$ is that it is a **diagonal** matrix (every number off the diagonal is 0) with the costs on its diagonal:

$$H_C\,|x\rangle=C(x)\,|x\rangle \tag{3b}$$

Every choice $|x\rangle$ is an eigenstate of $H_C$ and its eigenvalue (its energy) is its cost. The cheapest choice is the state of lowest energy, the **ground state**. That is the whole point of this step: **finding the cheapest sensor placement means finding the ground state of $H_C$.**

For our grid with $P=2$: $h_i=\tfrac12$ for the four substations with 2 lines, $h_i=1$ for the two with 3 lines (substations 1 and 4), $J=\tfrac12$ for each of the 7 lines, and $c_0=3+3.5=6.5$. Notice that $c_0=6.5$ is also the *average* cost of a random choice (the $Z$ terms average out to zero).

""" + details("Show the algebra (optional)", r"""With $1-x_i\to\frac{1+Z_i}{2}$ we get $(1-x_i)(1-x_j)\to\frac{(1+Z_i)(1+Z_j)}{4}=\frac{1+Z_i+Z_j+Z_iZ_j}{4}$. Then

$$C\;\to\;\sum_i\frac{1-Z_i}{2}+\frac P4\sum_{(i,j)\in E}\bigl(1+Z_i+Z_j+Z_iZ_j\bigr).$$

Collect the terms. Constants: $\frac n2+\frac P4|E|$. Each $Z_i$ gets $-\frac12$ from the first sum, and $+\frac P4$ from every line that touches $i$, which is $d_i$ lines. Each $Z_iZ_j$ gets $\frac P4$. That is equation (3).""") + "\n\n" + callout("real", r"""You rarely write $H_C$ by hand: Qiskit's `QuadraticProgramToQubo(...).to_ising()` does it, as the next cell shows (QUBO = *quadratic unconstrained binary optimisation*: the rule has been folded into the cost as a penalty, so no constraint is left). Doing it once on paper tells you what the library is doing for you, and the check cell after it proves that equations (2) and (3) match what Qiskit produced.""") + "\n\n" + arabic_table([
    ("Hamiltonian", "مؤثر هاملتون (الهاملتوني)", "G"), ("Operator", "المؤثر", "G"), ("Eigenvalue", "القيمة الذاتية", "G"),
    ("Eigenstate", "الحالة الذاتية", "G"), ("Pauli gates (X, Y, Z)", "بوابات باولي", "G"), ("Ground state", "الحالة الأرضية", "I"),
    ("Diagonal matrix", "مصفوفة قطرية", "I"), ("Degree of a vertex", "درجة الرأس", "I")])

S3_CHECK_MD = """**Check equations (2) and (3) against Qiskit.** The cell below recomputes the cost table from equation (2) and the Hamiltonian coefficients from equation (3), and stops with an error if either disagrees with what `to_ising()` returned."""

S3_CHECK = '''# Check: equations (2) and (3) reproduce Qiskit's cost table and Hamiltonian exactly.
deg = dict(G.degree())                                                        # d_i, the number of lines at substation i
assert np.allclose(diag, [cost_by_formula(x) for x in x_all]), "equation (2) disagrees with Qiskit's cost table"
h_formula = {i: -1 / 2 + PENALTY / 4 * deg[i] for i in range(QUBITS)}         # equation (3): coefficient of Z_i
J_formula = PENALTY / 4                                                       # equation (3): coefficient of every Z_i Z_j
c0_formula = QUBITS / 2 + PENALTY / 4 * len(edges)                            # equation (3): the constant
for pauli, coeff in zip(cost_op.paulis, cost_op.coeffs):
    on = [i for i in range(QUBITS) if pauli.z[i]]                             # which qubits carry a Z in this term
    want = h_formula[on[0]] if len(on) == 1 else J_formula
    assert abs(coeff.real - want) < 1e-12, (pauli, coeff, want)
assert abs(offset - c0_formula) < 1e-12 and abs(diag.mean() - c0_formula) < 1e-12
# Equation (3b): H_C|x> = C(x)|x>, i.e. the matrix of H_C is diagonal and its diagonal holds the costs.
H_matrix = cost_op.to_matrix() + offset * np.eye(2 ** QUBITS)
assert np.allclose(H_matrix, np.diag(np.diag(H_matrix))), "H_C is not diagonal"
assert np.allclose(np.diag(H_matrix).real, diag), "diagonal of H_C differs from the cost table"
print("OK: equations (2) and (3) match Qiskit.")
print("h_i =", h_formula, "| J =", J_formula, "| c0 =", c0_formula, "(also the average cost)")'''

S3_CIRCUIT_MD = r"""### 3.4 · The QAOA circuit

The circuit has three ingredients, applied in this order to six qubits that start in $|0\rangle$.

**(a) Start: an equal mixture.** The **Hadamard gate** $H$ turns $|0\rangle$ into $\tfrac1{\sqrt2}(|0\rangle+|1\rangle)$. On six qubits it gives the same amplitude $\tfrac18$ to every one of the 64 choices, so every choice has probability $\tfrac1{64}$ and the average cost is 6.5. This is pure guessing.

$$|s\rangle=H^{\otimes 6}\,|000000\rangle=\frac1{\sqrt{2^6}}\sum_x|x\rangle \tag{4}$$

($H^{\otimes 6}$ just means "one $H$ gate on each of the six qubits".)

**(b) Cost layer: twist every choice by its cost.** For an angle $\gamma$ (a knob we will tune) apply

$$U_C(\gamma)=e^{-i\gamma H_C},\qquad\text{which acts as}\qquad |x\rangle\;\longmapsto\;e^{-i\gamma\,C(x)}\,|x\rangle \tag{5}$$

Because $H_C$ is diagonal, "the exponential of a matrix" is just the exponential of each diagonal entry. Every choice is *turned* by an angle proportional to its cost, so costly choices turn faster. No probability changes, only phases (see the picture below). In gates, each $Z_i$ term of (3) becomes one **RZ** gate on qubit $i$ and each $Z_iZ_j$ term becomes one **RZZ** gate on the pair, which is **one gate per line of the grid**: $e^{-i\gamma hZ_i}=R_Z(2\gamma h)$ and $e^{-i\gamma JZ_iZ_j}=R_{ZZ}(2\gamma J)$.

**(c) Mixer layer: let neighbouring choices talk.** For a second knob $\beta$ apply

$$U_B(\beta)=e^{-i\beta\sum_i X_i}=\bigotimes_{i}R_X(2\beta) \tag{6}$$

The **Pauli $X$ gate** flips a bit ($|0\rangle\leftrightarrow|1\rangle$), and $e^{-i\beta X}=\cos\beta\,I-i\sin\beta\,X$ (here $I$ is the "do nothing" matrix, and $\bigotimes_i R_X(2\beta)$ in (6) means one $R_X$ gate on every qubit) reads "keep with amplitude $\cos\beta$, flip with amplitude $-i\sin\beta$". So the mixer moves amplitude between choices that differ in **one** bit, that is one sensor added or removed. Where amplitudes arrive from different neighbours with different phases, they **interfere**: some add up, some cancel.

**(d) Repeat.** One cost layer plus one mixer is one **layer**. With $p$ layers the circuit has $2p$ knobs $\gamma_1,\dots,\gamma_p$ and $\beta_1,\dots,\beta_p$, and layer 1 acts first:

$$|\gamma,\beta\rangle=U_B(\beta_p)\,U_C(\gamma_p)\cdots U_B(\beta_1)\,U_C(\gamma_1)\,|s\rangle \tag{7}$$

""" + callout("idea", r"""**Why "twist, then shake" can find good answers.** The cost layer changes no probability by itself; it only writes each choice's cost into its *phase*. The mixer is what turns phase differences into probability differences: contributions that arrive in step reinforce each other, the others cancel. With well-chosen $(\gamma,\beta)$ the low-cost choices end up reinforced. For our problem we do not know good knob values in advance, so a classical optimiser searches for them (section 3.7).""") + "\n\n" + callout("watch", r"""**Knob order.** Qiskit sorts the knobs alphabetically, so a list of knob values must be **all $\beta$ first, then all $\gamma$**: for $p=1$ it is `[beta, gamma]`, for $p=2$ it is `[beta1, beta2, gamma1, gamma2]`. Many papers write $(\gamma,\beta)$ the other way round.""") + "\n\n" + arabic_table([
    ("Hadamard gate", "بوابة هادامارد", "G"), ("Quantum circuit", "الدائرة الكمومية", "U"), ("Rotation gate (RX, RZ)", "بوابة دوران", "I"),
    ("Cost layer", "طبقة التكلفة", "I"), ("Mixer layer", "طبقة الخلط", "I"), ("Ansatz (the parametrised circuit)", "الافتراض المبدئي (ansatz)", "G"),
    ("Variational", "التغايري / المتغايرة", "G"), ("QAOA", "خوارزمية الأمثلة الكمومية التقريبية", "G"), ("Circuit depth", "عمق الدائرة", "G"),
    ("Global phase", "الطور الكلي", "I")])

S3_SYMBOLS = r"""**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).

| Symbol | Meaning | In the code |
|---|---|---|
| $x=(x_0,\dots,x_5)$ | a choice: sensor at substation $i$ when $x_i=1$ | `x`, `x_all[k]`, `index_to_x(k, 6)` |
| $E$, $d_i$ | the set of 7 lines; number of lines at substation $i$ | `edges`, `G.degree` |
| $P$ | penalty for one unwatched line | `PENALTY`, `penalty=2` |
| $C(x)$ | cost of choice $x$ | `cost_by_formula(x)`, `diag[k]` |
| $Z_i,\ X_i$ | Pauli $Z$ and $X$ acting on qubit $i$ | the Pauli strings of `cost_op` |
| $H_C$, $h_i$, $J$, $c_0$ | cost Hamiltonian and its coefficients | `cost_op`, `cost_op.coeffs`, `offset` |
| $\lvert x\rangle$, $\alpha_x$, $p(x)$ | a choice as a state, its amplitude, its probability | `Statevector(...).data`, `.probabilities()` |
| $\lvert s\rangle$ | the equal mixture at the start | the first layer of H gates |
| $\gamma$, $\beta$ | cost-layer and mixer knobs | circuit parameters `γ[k]`, `β[k]` |
| $p$ | number of layers (depth) | `reps=p` |
| $\langle C\rangle$ | average cost, what the optimiser minimises | `energy(theta)` (T2), `qaoa_energy` |
| $P_{\rm opt}$ | probability of an optimal cover | `success_of(circuit, theta)` |"""

S3_BYHAND_MD = """**Read the drawing, then build the same circuit yourself.** In the circuit above, read left to right: the **H** gates are equation (4); the **RZ** and **ZZ** gates are the cost layer (5), one RZ per substation and one ZZ per line of the grid (the label shows the knob $\\gamma$ they carry); the **RX** gates are the mixer (6) and carry $\\beta$. The cell below builds equations (4) to (7) yourself with plain `h`, `rz`, `rzz` and `rx` gates (with a barrier between the stages so you can see them), and checks that the two circuits give the same state up to a global phase (a **global phase** is one and the same turn applied to *every* amplitude, which changes no probability and cannot be detected)."""

S3_BYHAND = '''# Equations (4)-(7) by hand: Hadamards, then (cost layer, mixer) p times.
from qiskit import QuantumCircuit


def qaoa_by_hand(cost_op, gammas, betas, barriers=True):
    """QAOA circuit from the equations. Z terms of equation (3) -> RZ, ZZ terms -> RZZ, mixer -> RX."""
    n = cost_op.num_qubits
    qc = QuantumCircuit(n)
    qc.h(range(n))                                                    # equation (4): the equal mixture
    for gamma, beta in zip(gammas, betas):
        if barriers:
            qc.barrier()
        for pauli, coeff in zip(cost_op.paulis, cost_op.coeffs):      # equation (5): one gate per term of equation (3)
            on = [i for i in range(n) if pauli.z[i]]
            if len(on) == 1:
                qc.rz(2 * gamma * coeff.real, on[0])                  # e^{-i gamma h Z_i}     = RZ(2 gamma h)
            elif len(on) == 2:
                qc.rzz(2 * gamma * coeff.real, on[0], on[1])          # e^{-i gamma J Z_i Z_j} = RZZ(2 gamma J)
        if barriers:
            qc.barrier()
        qc.rx(2 * beta, range(n))                                     # equation (6): the mixer, RX(2 beta) on every qubit
    return qc


beta_test, gamma_test = np.random.default_rng(SEED).uniform(0, np.pi, 2)            # any knob values will do
by_hand = qaoa_by_hand(cost_op, [gamma_test], [beta_test])
same = Statevector(by_hand).equiv(Statevector(ansatz.assign_parameters([beta_test, gamma_test])))
assert same, "the hand-built circuit should equal qaoa_ansatz up to a global phase"
print("hand-built circuit == qaoa_ansatz (same state up to a global phase):", same)
print("gates in one layer:", {g: c for g, c in by_hand.count_ops().items() if g != "barrier"}, "(7 RZZ = one per line of the grid)")
try:
    display(by_hand.draw("mpl", fold=45))
except Exception:                                                     # drawing with matplotlib needs pylatexenc; fall back to text
    print(by_hand.draw("text", fold=120))'''

S3_FIG_MD = """**Two pictures of what the layers do.** Left: the start state (equation 4) is spread evenly over all 64 choices, so its cost distribution has the shape of the bar chart in 3.2 and an average of 6.5; a state sitting on one choice has all its probability at one cost. Right: a cost layer (equation 5) turns each choice by the angle $-\\gamma\\,C(x)$. Choices with the same cost turn together, so they point the same way."""

S3_FIG = '''fig = plt.figure(figsize=(10, 3.6))
ax1 = fig.add_subplot(1, 2, 1)
one_choice = np.zeros(64); one_choice[0] = 1
qff.plots.plot_cost_histogram(diag, {"all on one choice": one_choice, "equal mixture (eq. 4)": np.full(64, 1 / 64)},
                              title="before any layer: where is the probability?", ax=ax1)
ax2 = fig.add_subplot(1, 2, 2, projection="polar")
qff.plots.plot_phases(diag, 0.3, title="cost layer, gamma = 0.3: each cost points its own way", ax=ax2)
plt.tight_layout()'''

S3_FIG_TRY = callout("try", """Change `0.3` in the polar plot to `0.0`, then to `1.0`. At $\\gamma=0$ every arrow points the same way (no twist at all, so the mixer has nothing to work with). What do you expect the average cost $\\langle C\\rangle$ to be if you apply *only* the Hadamards and the cost layer? (Answer: still 6.5, because a cost layer changes no probability.)""")

S3_MINI_MD = r"""### 3.5 · A worked example you can check on paper

Take the smallest grid there is: **two substations joined by one line**. It has only $2^2=4$ choices, so every step can be done by hand.

**Step 1: costs (equation 2, with $P=2$).** $C(x_0,x_1)=x_0+x_1+2\,(1-x_0)(1-x_1)$.

| choice $(x_0,x_1)$ | sensors | line unwatched? | cost |
|---|---|---|---|
| (0, 0) | 0 | yes | $0+2=2$ |
| (1, 0) | 1 | no | $1$ |
| (0, 1) | 1 | no | $1$ |
| (1, 1) | 2 | no | $2$ |

The best choices are $(1,0)$ and $(0,1)$: one sensor, cost 1.

**Step 2: Hamiltonian (equation 3).** Each substation touches one line, so $d_i=1$: $h_i=-\tfrac12+\tfrac24\cdot1=0$, $J=\tfrac24=\tfrac12$, $c_0=\tfrac22+\tfrac24\cdot1=\tfrac32$. Hence $H_C=\tfrac32+\tfrac12\,Z_0Z_1$. Check it: $Z_0Z_1=+1$ when the two bits are equal and $-1$ when they differ, so $H_C=2$ for $(0,0)$ and $(1,1)$ and $H_C=1$ for $(1,0)$ and $(0,1)$. It matches step 1.

**Step 3: the circuit with $p=1$.** Equation (4) puts amplitude $\tfrac12$ on all four choices. Equation (5): the constant $\tfrac32$ only adds a phase that nothing can detect, so drop it. Choices with equal bits get the phase $e^{-i\gamma/2}$, choices with different bits get $e^{+i\gamma/2}$. Equation (6) then mixes. After a short calculation (open the box below to see it) the probability of ending on a best choice is

$$P_{\rm opt}(\gamma,\beta)=\frac12\bigl(1-\sin\gamma\,\sin4\beta\bigr) \tag{M}$$

**Step 4: read it off.**

- No mixer ($\beta=0$): $P_{\rm opt}=\tfrac12$. That is random guessing: 2 of the 4 choices are best.
- $\gamma=\tfrac\pi2$ and $\beta=\tfrac{3\pi}8$: $\sin\gamma=1$ and $\sin4\beta=\sin\tfrac{3\pi}2=-1$, so $P_{\rm opt}=\tfrac12(1+1)=1$. **One layer finds a best answer for certain** on this tiny grid (half the time $(1,0)$, half the time $(0,1)$).
- $\gamma=\tfrac\pi2$ and $\beta=\tfrac\pi8$: $\sin4\beta=1$, so $P_{\rm opt}=0$. The same circuit with the wrong knob is the *worst* possible. The knobs matter.

""" + details("Show the algebra for (M) (optional)", r"""After the cost layer the state is $\tfrac12\bigl[a(|00\rangle+|11\rangle)+b(|01\rangle+|10\rangle)\bigr]$ with $a=e^{-i\gamma/2}$ and $b=e^{i\gamma/2}$.

Use the two states $|\pm\rangle=\tfrac1{\sqrt2}(|0\rangle\pm|1\rangle)$ (what a Hadamard gate produces). The $X$ gate leaves them alone up to a sign: $X|+\rangle=+|+\rangle$ and $X|-\rangle=-|-\rangle$. Then $|00\rangle+|11\rangle=|{+}{+}\rangle+|{-}{-}\rangle$ and $|01\rangle+|10\rangle=|{+}{+}\rangle-|{-}{-}\rangle$, so the state is $\cos\tfrac\gamma2\,|{+}{+}\rangle-i\sin\tfrac\gamma2\,|{-}{-}\rangle$.

The mixer $e^{-i\beta(X_0+X_1)}$ multiplies $|{+}{+}\rangle$ by $e^{-2i\beta}$ and $|{-}{-}\rangle$ by $e^{+2i\beta}$, because $X_0+X_1$ has eigenvalue $+2$ on $|{+}{+}\rangle$ and $-2$ on $|{-}{-}\rangle$. Converting back to the choices, the amplitude of $|01\rangle$ (and of $|10\rangle$) is $\tfrac12\bigl[\cos\tfrac\gamma2\,e^{-2i\beta}+i\sin\tfrac\gamma2\,e^{2i\beta}\bigr]$. Its squared size is $\tfrac14\,(1-\sin\gamma\sin4\beta)$. Adding the two equal contributions gives (M).""") + "\n\n" + callout("watch", r"""Two substations is trivially easy: a classical computer solves it with one glance. The point of the example is that **the same four steps run unchanged** for 6 substations (the rest of this notebook) or for 80 (the paper). Only the size of the tables changes.""") + "\n\nThe next cell builds this example with the same Qiskit pipeline as the 6-substation problem and compares equation (M) with Qiskit's answer at four knob settings."

S3_MINI = '''# The worked example, checked against Qiskit.
G_mini = nx.Graph([(0, 1)])                                                       # two substations, one line
qp_mini = VertexCover(G_mini).to_quadratic_program()
op_mini, offset_mini = QuadraticProgramToQubo(penalty=2).convert(qp_mini).to_ising()
diag_mini = qff.classical.diag_of(op_mini, offset_mini)
print("Hamiltonian terms:", {k: float(v) for k, v in zip(op_mini.paulis.to_labels(), op_mini.coeffs.real)}, "| constant:", offset_mini)   # {'ZZ': 0.5} 1.5
print("cost of (0,0), (1,0), (0,1), (1,1):", diag_mini.tolist())                                                       # step 1 = step 2
assert diag_mini.tolist() == [2.0, 1.0, 1.0, 2.0] and abs(offset_mini - 1.5) < 1e-12

ansatz_mini = qaoa_ansatz(op_mini, reps=1)                                         # knobs in the order [beta, gamma]


def p_opt_qiskit(beta, gamma):
    probs = Statevector(ansatz_mini.assign_parameters([beta, gamma])).probabilities()
    return float(probs[1] + probs[2])                                              # the choices (1,0) and (0,1): one sensor


def p_opt_formula(beta, gamma):
    return 0.5 * (1 - np.sin(gamma) * np.sin(4 * beta))                            # equation (M)


print(f"{'beta':>8}{'gamma':>8}{'by hand (M)':>14}{'Qiskit':>10}")
for beta, gamma in [(0.0, 1.0), (np.pi / 8, np.pi / 2), (3 * np.pi / 8, np.pi / 2), (0.3, 2.0)]:
    print(f"{beta:8.3f}{gamma:8.3f}{p_opt_formula(beta, gamma):14.4f}{p_opt_qiskit(beta, gamma):10.4f}")
    assert abs(p_opt_formula(beta, gamma) - p_opt_qiskit(beta, gamma)) < 1e-9, "equation (M) disagrees with Qiskit"
qff.plots.plot_landscape(p_opt_formula, (0, np.pi / 2), (0, np.pi), n=61, xlabel="beta (mixer knob)", ylabel="gamma (cost knob)",
                         title="P(optimal) for the 2-substation example, equation (M)", cmap="viridis",
                         marks=[(3 * np.pi / 8, np.pi / 2, "best: P = 1", "use")]);'''

S3_ENERGY_MD = r"""### 3.6 · Average cost and success probability

When the circuit has run, every choice $x$ has a probability $p(x)$ (equation 1). Two numbers summarise how good the circuit is:

$$\langle C\rangle=\sum_x p(x)\,C(x) \tag{8}$$

$$P_{\rm opt}=\sum_{x\ \text{optimal}}p(x) \tag{9}$$

- **(8)** is the *average cost* you would see if you ran the circuit many times: each cost weighted by how likely it is. **It is the number the optimiser tries to make small.** In code it is one line, `probabilities @ diag`, the dot product of the 64 probabilities with the 64 costs. (Physicists write the same number as $\langle\psi|H_C|\psi\rangle$; because $H_C$ is diagonal the two are identical, and the exact-statevector trick needs no shots.)
- **(9)** is what we care about at the end: the chance of reading out one of the **two** best covers. Random guessing gives $2/64=3.1\%$.

Two numbers to remember for this grid: the *lowest* cost is **3.0** (three sensors, no penalty) and the *average* cost of a random choice is **6.5**. A good QAOA run pushes $\langle C\rangle$ down from 6.5 towards 3. The shared helper `qff.quantum.qaoa_energy` computes (8); in T2 you write it yourself, and we also need a way to measure (9). This small helper `success_of` is reused by notebook 4.

""" + arabic_table([
    ("Expectation value (average cost)", "القيمة المتوقعة (متوسط التكلفة)", "I"), ("Success probability", "احتمال النجاح", "I"),
    ("State vector", "متجه الحالة", "I"), ("Classical optimiser", "المُحسِّن الكلاسيكي", "I"), ("Minimise", "تصغير (تقليل)", "I"),
    ("Random guessing", "التخمين العشوائي", "I")])

T2_INTRO = r"""### T2 · Core step: the average cost and the first optimisation · التمرين ٢

**What you are about to build, and why.** This is the heart of QAOA's classical side. You write the function `energy(theta)` that takes two knob values $\theta=(\beta,\gamma)$, runs the circuit, and returns the average cost of equation (8). Then you hand it to an optimiser, which keeps trying other knob values until $\langle C\rangle$ stops improving. Everything else in the notebook (more layers, the shortcut, the comparison) reuses this one idea.

**Two ways to do it.** The task cell already contains the plumbing (the function frame of `energy`, the call to the optimiser `minimize` with its start point, method and step limit, the best knobs `theta1`, the success probability `success_p1` and the printout) and leaves two blanks that carry the ideas: the state of the circuit at the knob values `theta`, and the average cost of equation (8) computed from it. Teams that want the full exercise can delete the scaffold and write everything from the Goal.

**The formula you implement:**

$$\langle C\rangle(\theta)=\sum_x p_\theta(x)\,C(x),\qquad p_\theta(x)=|\alpha_x(\theta)|^2 .$$

Put into words: fill the knobs into the circuit, get the 64 probabilities, take their dot product with the 64 costs `diag`.

""" + hints(
    r"""Blank `state`: `ansatz` has two *free* numbers in it. Which Qiskit object turns a finished circuit into a list of amplitudes, and how did `success_of` (above) put the knob values into the circuit before building it? Blank `return`: the circuit has 6 qubits, so there are 64 probabilities and 64 costs. What do you do with two lists of the same length to get one average (equation 8)? Which method of the state gives the probabilities, and which array holds the costs?""",
    r"""**`state`**: copy the way the helper `success_of` (defined above) builds the state: assign the knob values `theta` (as a list) to the parameters of `ansatz`, then wrap the result in a `Statevector`. **`return`**: equation (8) is the sum over all covers x of p(x) times C(x), where the p(x) are the probabilities of that state (`state.probabilities()`, one per cover) and the C(x) are the entries of the cost table `diag`; the sum of products of two equal-length arrays is their dot product (the `@` operator). Return it wrapped in `float(...)` so that the optimiser gets a plain Python float.""",
    r"""```python
state = Statevector(______.assign_parameters(list(______)))
return float(state.______() @ ______)
```
Self-check target: `energy([0.5, 0.5])` equals `f_shared([0.5, 0.5])`.""")

S3_LAND_MD = """**What the optimiser walked on.** The next picture is the whole $p=1$ landscape: the average cost $\\langle C\\rangle$ for every pair $(\\beta,\\gamma)$ (dark is low, which is good). The optimiser started at the magenta dot, where $\\langle C\\rangle$ is even *worse* than guessing (9.7 against 6.5), and walked downhill to the star. Notice the repeating pattern of dark valleys: more than one place is "good", and a single start can end up in a poor one. This is *your* `energy` function, evaluated on a grid."""

S3_LAND = '''# The p = 1 landscape: average cost <C> for a grid of (beta, gamma) values (your energy() from T2).
qff.plots.plot_landscape(lambda b, g: energy([b, g]), (0, np.pi), (0, np.pi), n=31, xlabel="beta (mixer knob)", ylabel="gamma (cost knob)",
                         title="average cost <C> at p = 1  (dark = low = good)",
                         marks=[(0.5, 0.5, "start", "use"), (theta1[0] % np.pi, theta1[1] % np.pi, "end: best found", "source")]);
print("start theta = [0.5, 0.5] ->", round(energy([0.5, 0.5]), 3), "| best theta =", np.round(theta1, 3), "->", round(energy(theta1), 3))'''

S3_SAMPLE_MD = r"""### 3.7 · Measuring, and the optimiser loop

**Measuring.** On a real quantum computer you never see the 64 probabilities. You **run** the circuit and **measure**, which returns *one* choice, chosen at random with probability $p(x)$. Repeat 4096 times (these repeats are called **shots**) and the frequencies approximate $p(x)$; the error shrinks like $1/\sqrt{\text{shots}}$. In the cell below we run the tuned $p=1$ circuit this way and plot the 12 most frequent choices. Two best covers are drawn in red.
"""

S3_SAMPLE_TRY = callout("try", """Change `4096` to `100` in the sampling cell (both places) and run it again. The bars get ragged, and the red bars may even disappear from the top 12. That is the price of few shots.""")

S3_LOOP_MD = r"""**The hybrid loop.** QAOA is a conversation between a quantum circuit and an ordinary program:

$$(\gamma^*,\beta^*)=\arg\min_{\gamma,\beta}\ \langle C\rangle(\gamma,\beta) \tag{10}$$

("argmin" means "the values at which the function is smallest".) The circuit gives $\langle C\rangle$ for the current knob values; a classical **optimiser** (here COBYLA, which needs no derivatives: it just asks "what is $\langle C\rangle$ here?" and uses the answers to choose where to look next) proposes better knobs; repeat. The landscape above has several valleys (**local minima**), so one starting point can get stuck in a poor one. The helper below therefore starts from $(0.5,0.5)$ **and** from five random points (fixed seed, so everybody gets the same numbers) and keeps the best. It is the helper you reuse for any QAOA problem: give it a cost Hamiltonian, the cost diagonal and a depth $p$."""

S3_LOOP_FIG = """qff.plots.plot_flow([("Knobs", "beta, gamma", "source"), ("Circuit", "equations 4-7", "use"), ("Average cost", "equation 8", "move"),
                     ("Optimiser", "COBYLA", "store")], loop=(3, 0, "new knob values"), title="the hybrid loop, repeated until <C> stops falling");"""

S3_LOOP_ARABIC = arabic_table([
    ("Measurement", "القياس", "G"), ("Shots", "عدد مرات التشغيل (shots)", "I"), ("Sampling", "أخذ العيّنات", "I"),
    ("Hybrid quantum-classical loop", "الحلقة الهجينة (كمومية وكلاسيكية)", "I"), ("Local minimum", "حدّ أدنى محلي", "I"),
    ("Cost landscape", "مشهد التكلفة", "I")])

S3_SHORTCUT_MD = """**The shortcut Qiskit offers.** `MinimumEigenOptimizer` wraps everything above: it converts the problem, builds the QAOA circuit, runs the optimiser and hands back an answer. We use it with two layers and a single start. One warning: read the probabilities from `min_eigen_solver_result.eigenstate`, not from `result.samples[...].probability`, which does not add up to 1 with this version of Qiskit's primitives."""

T3_INTRO = r"""### T3 · Stretch: how many layers? · التمرين ٣

**What you are about to build, and why.** The depth $p$ is the one real dial of QAOA: equation (7) has $2p$ knobs, and every layer adds one RZZ gate per line of the grid. More knobs should let the optimiser build a sharper interference pattern, but a deeper circuit costs more gates, and gates are what noisy hardware does badly. T3 measures both sides for $p=1,2,3$: how $P_{\rm opt}$ (equation 9) grows, and what the circuit costs on a pretend device. Then you explain the trade-off in your own words.

**What you implement.** For each $p$: $(\beta,\gamma)^*_p=\arg\min\langle C\rangle$ with $2p$ knobs (equation 10), then $P_{\rm opt}(p)$ from equation (9), then the depth and the number of two-qubit gates of the tuned circuit.

""" + hints(
    r"""You already have a function that does the hard part for *one* value of p: `optimise`. Read its docstring: what does it return? Run it for p = 1 first and print what comes back. Then think about how to repeat it for 1, 2 and 3 and keep all three answers (a dictionary with p as the key is a good home).""",
    r"""A loop over p = 1, 2, 3: call `optimise` and store its four return values in four dictionaries keyed by p; compute the success probability of the tuned circuit with `success_of`; get the pretend-device numbers with `qff.quantum.device_preview` for the tuned circuit; time the call if you want the seconds. After the loop: a bar chart of success against p with the random-guess line, and the explanation string.""",
    r"""```python
success, energy_by_p, ansatz_by_p, theta_by_p, evals_by_p, seconds_by_p, device_by_p = {}, {}, {}, {}, {}, {}, {}
for p in (1, 2, 3):
    # time the next call
    ______                  # run the optimiser for this p: fills ansatz_by_p, theta_by_p, energy_by_p, evals_by_p
    ______                  # seconds_by_p[p]
    ______                  # success[p] of the tuned circuit
    ______                  # device_by_p[p]: the tuned circuit on the pretend device
# a bar chart of success[p] with a dashed line at RANDOM_GUESS
explanation = "..."         # what the extra knobs buy, and what the extra gates cost on a real device
```""")

S3_HIST_MD = """**Before and after.** Where does the probability go? The grey bars are random guessing: most of the weight sits between costs 4 and 8. As layers are added, weight moves to the left, towards cost 3, which is the optimal cover."""

S3_HIST = '''dists = {"random guessing": np.full(2 ** QUBITS, 1 / 2 ** QUBITS)}
for p in (1, 2, 3):
    dists[f"QAOA p = {p}"] = Statevector(ansatz_by_p[p].assign_parameters(theta_by_p[p])).probabilities()
qff.plots.plot_cost_histogram(diag, dists, title="probability of each cost: guessing against QAOA with 1, 2, 3 layers")
print({name: round(float(pr @ diag), 3) for name, pr in dists.items()})          # the average cost <C> of each distribution'''

S3_WHY_MD = r"""### 3.8 · Why does it work, and when does it not?

Put the pieces together. The cost layer writes each choice's cost into its phase (5). The mixer lets neighbouring choices exchange amplitude (6), and where the phases line up the amplitudes add; where they do not, they cancel (interference). The optimiser's only job is to find the knob values $(\gamma,\beta)$ for which cheap choices line up and expensive ones cancel. More layers mean more knobs, hence a more flexible pattern: the success probability rises from 11% to 28% to 51% in the table above.

**What is known, and what is not.**

- In principle, with enough layers and the right angles, QAOA can approach the exact ground state: this is the result of Farhi, Goldstone and Gutmann, who introduced the algorithm in 2014 (arXiv:1411.4028). In practice $p$ is small, and nothing guarantees that a small $p$ finds the optimum. Here $p=3$ gives about half.
- Finding good angles is itself a hard optimisation. For larger circuits the landscape can become almost flat (**barren plateaus**), and the optimiser has nothing to follow.
- Real devices add **noise**. Each gate fails a little, deeper circuits fail more, and the failures blur the very interference the layers are building.

""" + callout("watch", r"""**This is a toy.** At $p=3$ the right answer appears about half the time *in an exact, noise-free simulation*, on 64 choices that brute force checks in microseconds. Nothing here shows that QAOA beats a classical solver. See section 4.""") + "\n\n" + arabic_table([
    ("Interference", "التداخل", "G"), ("Noise", "الضوضاء", "G"), ("Barren plateau", "الهضبة العقيمة", "G"),
    ("Quantum advantage", "الأفضلية الكمومية", "G"), ("Number of layers (depth p)", "عدد الطبقات (العمق p)", "I")])

S3_PENALTY_MD = """### 3.8 (continued) · What does the penalty do?
The penalty $P$ in equation (2) says how much an unwatched line costs. Too small and cheating pays: leaving a line unwatched would be cheaper than adding a sensor. Too large and the cost landscape gets steep and spiky, which is harder for a few layers of QAOA to navigate (the Qiskit default for this problem is 7). Below, the same problem with penalty 2 and penalty 7, two layers each: the lowest cost is the same (3), but the right answer is about 2.5 times more likely with the smaller penalty.

""" + callout("math", r"""**How small may $P$ be?** Take any choice that leaves a line unwatched and put a sensor on one end of that line. The sensor costs $+1$ and removes at least one unwatched line, which saves at least $P$. So the new choice is no more expensive when $P\ge1$, and strictly cheaper when $P>1$. Hence for $P>1$ **every cheapest choice is a true cover**, and for $P\le1$ cheating can tie or win. The first table below checks this on our grid.""")

S3_PENALTY_CHECK = '''# Is the cheapest choice always a true cover? (uses the cost formula of equation (2) with different penalties)
print(f"{'P':<6}{'lowest cost':<13}{'cheapest choices that are covers':<36}{'cheapest choices that are not'}")
for P_try in (0.5, 1, 2, 7):
    costs_try = np.array([sum(x) + P_try * sum((1 - x[i]) * (1 - x[j]) for i, j in edges) for x in x_all])
    cheapest = [x for x, c in zip(x_all, costs_try) if np.isclose(c, costs_try.min())]
    n_cover = sum(is_cover(x) for x in cheapest)
    print(f"{P_try:<6}{costs_try.min():<13.2f}{n_cover:<36}{len(cheapest) - n_cover}")'''


S3_IMPORTS_QP = """from scipy.optimize import minimize
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import VertexCover
from qiskit_optimization.converters import QuadraticProgramToQubo
from qff.bits import to_x, from_x, x_to_index

algorithm_globals.random_seed = SEED                    # before any qiskit_algorithms / qiskit_optimization solver

# Cell 1: the problem as a QuadraticProgram (Qiskit has a ready-made VertexCover class).
vc = VertexCover(G)
qp = vc.to_quadratic_program()
print(qp.prettyprint())"""

T1 = """def uncovered(chosen):
    return [e for e in edges if e[0] not in chosen and e[1] not in chosen]


print("lines left unwatched by {1, 4}:", uncovered([1, 4]))
my_cover = [1, 3, 5]                 # any 3 substations that watch every line
bf_x, bf_value, bf_table = qff.classical.brute_force(qp)
print("brute_force says:", bf_x, "with", bf_value, "sensors")"""
T1_STARTER = """# Goal: uncovered(chosen) returns the list of lines (pairs from `edges`) that no substation in `chosen` touches.
# my_cover is a list of three substation numbers that watch every line. Keep the classical answer as (bf_x, bf_value, bf_table).
raise NotImplementedError("T1: write uncovered(chosen), find a 3-node cover, check it with brute_force")"""
T1_CHECK = """assert sorted(uncovered([1, 4])) == [(0, 3), (2, 5)], "T1: {1, 4} leaves the lines (0, 3) and (2, 5) unwatched"
assert len(my_cover) == 3 and uncovered(my_cover) == [], "T1: my_cover must have 3 substations and leave no line unwatched"
assert bf_value == 3.0 and tuple(int(v) for v in bf_x) in classical_optima, "T1: brute_force should find a 3-sensor optimal cover" """

S3_QUBO = """# Cell 2: constraints -> penalty -> Ising Hamiltonian. penalty=2 is deliberate (see the penalty cell later).
qubo = QuadraticProgramToQubo(penalty=2).convert(qp)       # equation (2) with P = 2
cost_op, offset = qubo.to_ising()                           # equation (3): cost_op = the Z and ZZ terms, offset = c0
diag = qff.classical.diag_of(cost_op, offset)           # the cost of every one of the 64 states, as an array
print(f"qubits: {cost_op.num_qubits} | Pauli terms: {len(cost_op)} | constant offset: {offset}")
print("lowest cost:", diag.min(), "| average cost over all 64 choices:", diag.mean())
print("choices at the lowest cost:", ["".join(map(str, index_to_x(k, 6))) for k in np.flatnonzero(np.isclose(diag, diag.min()))])
print("first terms (Pauli strings are little-endian: the rightmost letter is qubit 0):")
for pauli, coeff in list(zip(cost_op.paulis, cost_op.coeffs))[:4]:
    print("  ", pauli, coeff.real)"""

S3_ANSATZ = """# Cell 3: the QAOA circuit for one layer (p = 1).
ansatz = qaoa_ansatz(cost_op, reps=1)                      # equations (4)-(7) with p = 1
print("parameters, in the order a list of knob values must follow:", list(ansatz.parameters))
try:
    display(ansatz.draw("mpl", fold=40))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(ansatz.draw("text", fold=120))"""

S3_SUCCESS = """# Cell 4: how likely is a right answer?
GOOD = [x_to_index(x) for x in classical_optima]         # basis-state numbers of the two optimal covers
RANDOM_GUESS = len(GOOD) / 2 ** QUBITS


def success_of(circuit, theta):
    \"\"\"Exact probability that `circuit` (knobs set to theta) ends in one of the optimal answers.\"\"\"
    probs = Statevector(circuit.assign_parameters(list(theta))).probabilities()
    return float(sum(probs[k] for k in GOOD))               # equation (9)


f_shared = qff.quantum.qaoa_energy(ansatz, diag)
theta_start = [0.5, 0.5]
print(f"random guessing: {RANDOM_GUESS:.3f}")
print(f"circuit at the starting knobs {theta_start}: <C> = {f_shared(theta_start):.3f}, success = {success_of(ansatz, theta_start):.3f}")"""

T2 = """def energy(theta):
    state = Statevector(ansatz.assign_parameters(list(theta)))
    return float(state.probabilities() @ diag)              # equation (8): sum of p(x) * C(x)


res1 = minimize(energy, x0=np.full(2, 0.5), method="COBYLA", options={"maxiter": 300})
theta1 = res1.x
success_p1 = success_of(ansatz, theta1)
print(f"p = 1: <C> went from {energy([0.5, 0.5]):.3f} to {res1.fun:.3f} in {res1.nfev} evaluations; success = {success_p1:.3f}")"""
T2_STARTER = """# Goal: energy(theta) returns the exact average cost <C> (equation 8) of the p = 1 circuit at knob values theta (a list of 2 numbers).
# Minimise it from [0.5, 0.5] with a gradient-free method (at most 300 steps). Keep: res1 (the optimiser result),
# theta1 (the best knobs) and success_p1 (the probability of an optimal cover at theta1).
# EASIER ROUTE (default): the plumbing below is written for you. Fill the blanks marked ______ , then delete the "raise" line.
# CHALLENGE ROUTE: delete the scaffold and write everything yourself from the Goal above (the hints still apply).
def energy(theta):
    state = ______    # the state of the circuit `ansatz` with its knobs set to `theta` (look at how success_of does it)
    return ______     # equation (8): the average cost over all covers, as one float


raise NotImplementedError("T2: fill the two blanks ______ in energy(theta), then delete this line")
res1 = minimize(energy, x0=np.full(2, 0.5), method="COBYLA", options={"maxiter": 300})
theta1 = res1.x
success_p1 = success_of(ansatz, theta1)
print(f"p = 1: <C> went from {energy([0.5, 0.5]):.3f} to {res1.fun:.3f} in {res1.nfev} evaluations; success = {success_p1:.3f}")"""
T2_CHECK = """assert abs(energy([0.5, 0.5]) - f_shared([0.5, 0.5])) < 1e-9, "T2: energy() should agree with qff.quantum.qaoa_energy"
assert res1.fun < 4.6, "T2: the optimiser should get <C> below 4.6 (it starts at about 6)"
assert 0.08 < success_p1 < 0.16, "T2: with one layer the success probability should be about 0.11" """

S3_SAMPLE = """# Cell 5: run the tuned circuit like a real experiment (4096 shots) and look at what comes out.
qc = ansatz.assign_parameters(theta1)
qc.measure_all()
counts = qff.quantum.sv_sampler(4096).run([qc]).result()[0].data.meas.get_counts()      # count strings are little-endian
sampled = {"".join(map(str, to_x(k))): v / 4096 for k, v in counts.items()}              # x-ordered: character i = substation i
best_two = ["".join(map(str, x)) for x in classical_optima]                              # x-ordered strings of the optimal covers
print("sampled frequency of the two optimal covers:", round(sum(sampled.get(s, 0) for s in best_two), 3))
qff.plots.plot_counts(dict(sorted(sampled.items(), key=lambda kv: -kv[1])[:12]), highlight=best_two,
                      title="p = 1, 4096 shots, 12 most frequent choices (optimal covers in red)");"""

S3_HELPER = """def optimise(cost_op, diag, p, n_random=5):
    \"\"\"QAOA at depth p: best of 1 + n_random COBYLA runs. Returns (ansatz, best theta, best <C>, number of evaluations).\"\"\"
    circuit = qaoa_ansatz(cost_op, reps=p)
    f = qff.quantum.qaoa_energy(circuit, diag)
    starts = [np.full(2 * p, 0.5)] + list(np.random.default_rng(1).uniform(0, np.pi, (n_random, 2 * p)))
    runs = [minimize(f, s, method="COBYLA", options={"maxiter": 300}) for s in starts]
    best = min(runs, key=lambda r: r.fun)                     # equation (10): the knobs with the lowest <C>
    return circuit, best.x, float(best.fun), int(sum(r.nfev for r in runs))"""

S3_SHORTCUT = """# Cell 7: the high-level route, about 5 seconds.
t0 = time.perf_counter()
qaoa = QAOA(sampler=qff.quantum.sv_sampler(4096), optimizer=COBYLA(maxiter=200), reps=2, initial_point=[0.5] * 4)
shortcut_result = MinimumEigenOptimizer(qaoa, penalty=2).solve(qp)
shortcut_seconds = time.perf_counter() - t0
eigenstate = shortcut_result.min_eigen_solver_result.eigenstate                          # {little-endian string: probability}
shortcut_success = sum(p for k, p in eigenstate.items() if tuple(to_x(k)) in classical_optima)
print("answer x:", [int(v) for v in shortcut_result.x], "| sensors:", shortcut_result.fval, "| substations:", vc.interpret(shortcut_result))
print(f"probability mass on the two optimal covers: {shortcut_success:.3f} (random guessing: {RANDOM_GUESS:.3f}) | {shortcut_seconds:.1f} s")"""

T3 = """success, energy_by_p, ansatz_by_p, theta_by_p, evals_by_p, seconds_by_p, device_by_p = {}, {}, {}, {}, {}, {}, {}
for p in (1, 2, 3):
    t0 = time.perf_counter()
    ansatz_by_p[p], theta_by_p[p], energy_by_p[p], evals_by_p[p] = optimise(cost_op, diag, p)
    seconds_by_p[p] = time.perf_counter() - t0
    success[p] = success_of(ansatz_by_p[p], theta_by_p[p])
    device_by_p[p] = qff.quantum.device_preview(ansatz_by_p[p].assign_parameters(theta_by_p[p]))
    print(f"p = {p}: <C> = {energy_by_p[p]:.3f} | success = {success[p]:.3f} | depth {device_by_p[p]['depth']}, "
          f"two-qubit gates {device_by_p[p]['two_qubit_gates']}")

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
axes[0].bar([1, 2, 3], [success[p] for p in (1, 2, 3)], color="#0F62FE")
axes[0].axhline(RANDOM_GUESS, ls="--", color="k", label="random guessing")
axes[0].set_xticks([1, 2, 3]); axes[0].set_xlabel("layers p"); axes[0].set_ylabel("probability of an optimal cover"); axes[0].legend()
axes[1].plot([1, 2, 3], [device_by_p[p]["two_qubit_gates"] for p in (1, 2, 3)], "o-", color="#FA4D56")
axes[1].set_xticks([1, 2, 3]); axes[1].set_xlabel("layers p"); axes[1].set_ylabel("two-qubit gates on a pretend device")
plt.tight_layout()

explanation = ("More layers give the optimiser more knobs, so it can build a sharper interference pattern and the optimal covers "
               "become more likely (about 11%, 28%, 51%). The price is circuit depth: every layer adds the same number of two-qubit "
               "gates, and on a real device each gate adds noise, so the benefit can be eaten by errors.")
print(explanation)"""
T3_STARTER = """# Goal: for p = 1, 2, 3 run the QAOA optimiser and fill these dictionaries (keys 1, 2, 3): success, ansatz_by_p, theta_by_p, energy_by_p,
# evals_by_p, seconds_by_p, device_by_p (circuit size on the pretend device). Plot success against p with the random-guess line,
# then write explanation = "...": why more layers help, and what they cost.
raise NotImplementedError("T3: run p = 1, 2, 3, plot success against p, explain what more layers buy and cost")"""
T3_CHECK = """assert set(success) == {1, 2, 3}, "T3: success needs entries for p = 1, 2 and 3"
assert success[2] > success[1] and success[3] > success[2], "T3: success should grow with p (it does here, from 0.11 to about 0.5)"
assert device_by_p[3]["depth"] > device_by_p[1]["depth"], "T3: deeper QAOA should give a deeper circuit"
assert isinstance(explanation, str) and len(explanation) > 60, "T3: write your explanation in a sentence or two" """

S3_PENALTY = """penalty_rows = {}
for P in (2, 7):
    op_P, off_P = QuadraticProgramToQubo(penalty=P).convert(qp).to_ising()
    diag_P = qff.classical.diag_of(op_P, off_P)
    circuit_P, theta_P, energy_P, _ = optimise(op_P, diag_P, 2)
    penalty_rows[P] = (diag_P.min(), diag_P.mean(), success_of(circuit_P, theta_P))
print(f"{'penalty':<9}{'lowest cost':<13}{'average cost':<14}{'success at p = 2'}")
for P, (lo, mean, s) in penalty_rows.items():
    print(f"{P:<9}{lo:<13.1f}{mean:<14.2f}{s:.3f}")"""

S3_IBM_AFTER = callout("real", """The preview above is the p = 3 circuit on a pretend device: the depth and two-qubit gate counts it prints are for a toy with only six qubits. A circuit this deep would probably not come out clean on today's real hardware: every two-qubit gate fails a small fraction of the time, the failures add up layer after layer, and they blur exactly the peak that more layers were meant to sharpen. That matters for our claim: the 51% above is an *ideal-simulator* number, not a promise about a real device.""")

S4_TABLE = """# Results table: classical baseline rows first.
def top_x(circuit, theta):
    \"\"\"Most likely choice of a tuned circuit, as an x list.\"\"\"
    probs = Statevector(circuit.assign_parameters(list(theta))).probabilities()
    return index_to_x(int(np.argmax(probs)), QUBITS)


rows = [("brute force (64 states)", "[0,2,4] or [1,3,5]", 3, "always", brute_seconds, "64 choices"),
        ("greedy rule", sorted(greedy_nodes), len(greedy_nodes), "always", greedy_seconds, f"{len(greedy_nodes)} picks")]
for p in (1, 2, 3):
    x = top_x(ansatz_by_p[p], theta_by_p[p])
    rows.append((f"QAOA p = {p} (best of 6 starts)", [i for i in range(6) if x[i]], sum(x), f"{success[p]:.3f}",
                 seconds_by_p[p], f"{evals_by_p[p]} circuit evaluations"))
xs = [int(v) for v in shortcut_result.x]
rows.append(("QAOA p = 2 via MinimumEigenOptimizer", [i for i in range(6) if xs[i]], sum(xs), f"{shortcut_success:.3f}",
             shortcut_seconds, "about 200 evaluations, 1 start"))
print(f"{'method':<40}{'answer (substations)':<24}{'sensors':<9}{'P(optimal)':<12}{'seconds':<9}{'work'}")
for name, nodes, size, prob, secs, work in rows:
    print(f"{name:<40}{str(nodes):<24}{size:<9}{prob:<12}{secs:<9.3f}{work}")
print("(for QAOA rows the answer is the single most likely state; at p = 1 it is a cover but not a smallest one)")"""

S4_PLOT = """fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
labels = ["guessing", "p = 1", "p = 2", "p = 3"]
values = [RANDOM_GUESS, success[1], success[2], success[3]]
axes[0].bar(labels, values, color=["#8D8D8D", "#0F62FE", "#0F62FE", "#0F62FE"])
for i, v in enumerate(values):
    axes[0].text(i, v + 0.01, f"{v:.0%}", ha="center")
axes[0].set_ylabel("probability of an optimal cover (exact simulation)")
final = Statevector(ansatz_by_p[3].assign_parameters(theta_by_p[3])).probabilities_dict()
qff.plots.plot_counts(dict(qff.quantum.top_states(final, 8)), highlight=best_two, title="p = 3: the 8 likeliest choices", ax=axes[1])
plt.tight_layout()"""

S4_SHOWS = """**What this toy shows.** A coverage rule becomes an Ising energy, and QAOA's alternating layers raise the probability of the two best covers from 3% (guessing) to 11%, 28% and 51% as layers are added (best of six starts each). The single-start shortcut run does less well than the best-of-six p = 2 run: the optimiser's starting point matters."""

S4_NOTSHOW = callout("watch", """**What this toy does not show.** That a quantum computer is useful here. 64 subsets are trivial classically, and greedy and brute force answer instantly. QAOA was simulated exactly with no noise; deeper circuits are exactly what today's devices cannot run cleanly. The paper's 80-qubit result is on different hardware with a different method, so nothing here is a statement about it.""")

S4_BEFORE_AFTER = '''# Before and after: the most likely choice at p = 1 and at p = 3, next to the true optimum.
x_p1, x_p3 = top_x(ansatz_by_p[1], theta_by_p[1]), top_x(ansatz_by_p[3], theta_by_p[3])
fig, axes = plt.subplots(1, 3, figsize=(10, 2.5))
draw_grid([i for i in range(6) if x_p1[i]], f"p = 1 most likely: {sum(x_p1)} sensors", ax=axes[0])
draw_grid([i for i in range(6) if x_p3[i]], f"p = 3 most likely: {sum(x_p3)} sensors", ax=axes[1])
draw_grid([1, 3, 5], "brute force optimum: 3 sensors", ax=axes[2])
plt.tight_layout()'''

S4_PAPER = """**Paper versus our route.** WEF/Aramco Case 12 (p. 26) frames grid sensor (PMU) placement as minimum vertex cover and reports neutral-atom processors with decomposition on 80-qubit instances. We keep the problem and use QAOA, the gate-model counterpart available in Qiskit. Vertex cover on a neutral-atom machine is solved through a related independent-set encoding, not through QAOA layers. So our results illustrate the *problem* and the *variational idea*; they are not a reproduction of the paper's experiment."""

ACC = """assert QUBITS == 6
assert classical_min_size == 3 and set(map(tuple, classical_optima)) == {(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0)}
assert abs(diag.min() - 3.0) < 1e-9
assert 0.08 < success[1] < 0.16 and success[2] > success[1] and success[3] > 0.35
assert tuple(int(v) for v in shortcut_result.x) in {(0, 1, 0, 1, 0, 1), (1, 0, 1, 0, 1, 0)}
print("p = 1, 2, 3 success:", [round(success[p], 3) for p in (1, 2, 3)], "| random guessing:", round(RANDOM_GUESS, 3))"""

INSTRUCTOR = """**Instructor note (never reaches the starter).** Expected numbers (seed 2026, best of six starts, exact simulation): success 0.114, 0.283, 0.511 for p = 1, 2, 3; <C> 4.565, 4.071, 3.708. The shortcut (one start, p = 2) puts only about 0.12 of its probability on the two optimal covers, yet its most likely answer is an optimal cover. Pitfalls: keep penalty 2; do not print `result.samples[...].probability`; assert membership in the set of two optima, not equality with one; parameter order is all β then all γ. This notebook is the QAOA pattern for notebook 4: `success_of`, `optimise` and the T1-T3 shape carry over."""


S3_MATH_ONE = '<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**\n\n**The one line to remember.** The optimiser minimises the average cost $\\langle C\\rangle=\\sum_xp(x)\\,C(x)$ (equation 8) of the circuit $|\\gamma,\\beta\\rangle=U_B(\\beta_p)U_C(\\gamma_p)\\cdots U_B(\\beta_1)U_C(\\gamma_1)|s\\rangle$ (equation 7). The cost layer writes the cost into the phase; the mixer turns phase differences into probabilities.\n\n</div>'


def main():
    """Write the solution notebook."""
    nb = new(3, "Grid sensor placement as minimum vertex cover",
             use_case="Place the fewest sensors on a power grid so that every line is watched (minimum vertex cover)",
             algorithm="QAOA (Quantum Approximate Optimization Algorithm)", qubits=6, minutes=3)
    nb.section(1)
    nb.md(S1)
    nb.section(2)
    nb.md(S2_MD)
    nb.code(S2_INSTANCE)
    nb.md(S2_CL_MD)
    nb.code(S2_CLASSICAL)
    nb.md(S2_NOTE)
    nb.section(3)
    nb.md(S3_ROADMAP)
    nb.code(S3_FLOW)
    nb.md(S3_SYMBOLS)
    nb.md(S3_MATH_ONE)
    nb.md(S3_STATE_MD)
    nb.code(S3_STATE_CODE)
    nb.md(S3_COST_MD)
    nb.code(S3_COST_CODE)
    nb.md(S3_COST_TRY)
    nb.md(S3_INTRO)
    nb.code(S3_IMPORTS_QP)
    nb.md(TOOLBOX[3][0])
    nb.code(TOOLBOX[3][1])
    nb.md(T1_INTRO)
    nb.task("T1", "show {1, 4} leaves lines unwatched, find a 3-node cover, check with brute_force", T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.md(S3_HAM_MD)
    nb.code(S3_QUBO)
    nb.md(S3_CHECK_MD)
    nb.code(S3_CHECK)
    nb.md(S3_CIRCUIT_MD)
    nb.code(S3_ANSATZ)
    nb.md(S3_BYHAND_MD)
    nb.code(S3_BYHAND)
    nb.md(S3_FIG_MD)
    nb.code(S3_FIG)
    nb.md(S3_FIG_TRY)
    nb.md(S3_MINI_MD)
    nb.code(S3_MINI)
    nb.md(S3_ENERGY_MD)
    nb.code(S3_SUCCESS)
    nb.md(T2_INTRO)
    nb.task("T2", "write the QAOA energy function and optimise it for p = 1", T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_LAND_MD)
    nb.code(S3_LAND)
    nb.md(S3_SAMPLE_MD)
    nb.code(S3_SAMPLE)
    nb.md(S3_SAMPLE_TRY)
    nb.md(S3_LOOP_MD)
    nb.code(S3_LOOP_FIG)
    nb.md(S3_LOOP_ARABIC)
    nb.code(S3_HELPER)
    nb.md(S3_SHORTCUT_MD)
    nb.code(S3_SHORTCUT)
    nb.md(T3_INTRO)
    nb.task("T3", "stretch: run p = 1, 2, 3, plot success against p, explain what more layers buy and cost", T3,
            starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_HIST_MD)
    nb.code(S3_HIST)
    nb.md(S3_WHY_MD)
    nb.md(S3_PENALTY_MD)
    nb.code(S3_PENALTY_CHECK)
    nb.code(S3_PENALTY)
    nb.md(INSTRUCTOR, tags=["solution-only"])
    nb.code("circuit = ansatz_by_p[3].assign_parameters(theta_by_p[3])        # the main circuit of this notebook: p = 3, tuned", tags=[])
    nb.md(day.IBM_MD)
    nb.code(day.IBM_PREVIEW_CODE)
    nb.md(S3_IBM_AFTER)
    nb.md(day.glossary_md([("QAOA", "خوارزمية الأمثلة الكمومية التقريبية (QAOA)", "G"), "cost function", "optimization",
                           ("Hamiltonian", "مؤثر هاملتون (الهاملتوني)", "G"), ("ansatz", "الافتراض المبدئي (ansatz)", "G"),
                           ("interference", "التداخل", "G"), "expectation value", ("penalty", "العقوبة (penalty)", "I"),
                           ("graph", "الرسم البياني", "U"), ("vertex cover", "الغطاء الرأسي", "I")]), tags=["glossary"])
    nb.section(4)
    nb.code(S4_TABLE)
    nb.code(S4_PLOT)
    nb.code(S4_BEFORE_AFTER)
    nb.md(S4_SHOWS)
    nb.md(S4_NOTSHOW)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="Place the fewest sensors on a power grid so that every line is watched.",
                counting_fact="6 substations give 64 choices to check; 1 000 give 2¹⁰⁰⁰ (a number with 302 digits). "
                              "Smart classical methods do not check them all, so the honest question is where they stop scaling.",
                single_idea="One layer rewards good covers with a phase twist, the next layer mixes the choices; repeat, and the good "
                            "answers interfere constructively.",
                ideas=["An interactive map where visitors place sensors and see uncovered lines turn red",
                       "A bar-chart race of p = 1, 2, 3", "A comic of a sensor crew deciding where to stand"],
                reuse="the grid drawing with the cover highlighted (`draw_grid`), the success-versus-p chart (3%, 11%, 28%, 51%), "
                      "and the results table")
    nb.section6(honest="at p = 3 the right answer appears about half the time, in simulation; brute force is instant here.",
                question_a="There are two best answers; which did you get and does it matter?",
                question_b="What does the penalty do?")
    path = nb.save(os.path.join(ROOT, "solutions", "03_move_grid_sensors_qaoa_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
