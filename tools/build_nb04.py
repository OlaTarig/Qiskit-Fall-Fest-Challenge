"""Build solutions/04_move_lng_routing_qaoa_solution.ipynb (notebook 4: LNG maritime inventory routing, QAOA).

    python tools/build_nb04.py

Follows the pattern of notebook 3 (tools/build_nb03.py): classical baseline first, T1 warm-up, guided Qiskit cells, T2 core step,
depth study, T3 what-if, shortcut, glossary, compare, acceptance.  `success_of` and `optimise` are the notebook 3 helpers with
two small changes: `success_of` takes the set of good answers, and `optimise` also returns every start's result (to show the spread).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, arabic_table, callout, details, hints, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1 = """**The problem.** Liquefied natural gas (LNG) travels in ships. A ship loads at a terminal, sails, and unloads at customer terminals whose storage tanks are being drained by customers every day. The planner decides *which terminal gets the next cargo, and when*, so that no tank runs dry or overflows and the ship does not waste sailing time. This is called **maritime inventory routing**: routing (where the ship goes) and inventory (how full each tank is) have to be solved together.

**Who faces it.** An LNG shipper or a gas-supply company that serves several customer terminals with a small fleet.

""" + callout("real", """**What is real today.** The WEF/Aramco paper (Case 4, p. 15) describes ExxonMobil and IBM Research formulating LNG maritime inventory routing as a vehicle routing problem with time windows in QUBO form, at the "R&D/prototype formulation and benchmarking" stage and "not yet at operational scale". The classical incumbents are strong: real instances are written as mixed-integer linear programs and solved with commercial solvers. A 2025 benchmark of a quantum-annealing hybrid on 30 such instances lost to the classical solver CPLEX, running on a laptop, on all 30. See the honesty rule in section 0.""") + """

**Why a quantum computer might help.** The number of possible schedules doubles with every yes/no decision (ship to terminal A in week 3? yes or no), and a quantum state can carry amplitude on all schedules at once; QAOA's layers try to steer that amplitude towards cheap schedules by interference. What is still missing: today's devices are noisy and small, and no one has shown the steering beats the best classical solvers at a useful size.

| Paper's method | Our teaching route |
|---|---|
| ExxonMobil and IBM Research: LNG inventory routing as vehicle routing with time windows, in QUBO form (Case 4, p. 15), prototype stage | A hand-built three-period miniature in the same spirit (QUBO), solved with QAOA on a 6-qubit simulator in Qiskit |"""

S2_MD = """One LNG carrier, two customer terminals **A** and **B**, three periods (think of them as three weeks). Each tank is used up by customers at a steady rate, and the ship can deliver one cargo per period, to one terminal at most. We write `x_A1` for "the ship delivers to terminal A in period 1", and so on. Six yes/no decisions in the order `x_A1, x_A2, x_A3, x_B1, x_B2, x_B3` give 2⁶ = 64 schedules, so six qubits hold one amplitude per schedule.

Numbers: start stock A = 2, B = 1; each tank loses 1 unit per period; a cargo adds 2 units; the target level of both tanks is 2.

The **cost** of a schedule adds three things:
- *keep each tank near its target*: for every terminal and every period, (stock after the period − target)²;
- *sailing*: +1 each time the ship switches terminal from one period to the next;
- *one ship cannot be in two places*: +8 if the ship is asked to serve both terminals in the same period.

The stock of terminal p after period t is `I_p(t) = start_p − use_p·t + cargo · (number of deliveries to p up to period t)`."""

S2_INSTANCE = """QUBITS = 6                       # the test runner reads this
import time

NAMES = ["x_A1", "x_A2", "x_A3", "x_B1", "x_B2", "x_B3"]        # variable order: index 0..5 = qubits 0..5
START = {"A": 2, "B": 1}      # stock at the start
USE = {"A": 1, "B": 1}        # use per period
CARGO = 2                     # units in one delivery
TARGET = {"A": 2, "B": 2}     # tank level we would like after every period
SAIL, CLASH = 1, 8            # cost of switching terminal; cost of serving both terminals in one period


def schedule_cost(x, start=START, cargo=CARGO, sail=SAIL):
    \"\"\"Cost of a schedule x = [x_A1, x_A2, x_A3, x_B1, x_B2, x_B3], straight from the three rules above.\"\"\"
    a, b = x[:3], x[3:]
    cost = 0
    for name, d in (("A", a), ("B", b)):
        level = start[name]                              # follow the tank period by period
        for t in range(3):
            level = level - USE[name] + cargo * d[t]
            cost += (level - TARGET[name]) ** 2
    cost += sail * sum(a[t] * b[t + 1] + b[t] * a[t + 1] for t in (0, 1))
    cost += CLASH * sum(a[t] * b[t] for t in range(3))
    return cost


def plot_stocks(x, title="", ax=None):
    \"\"\"Draw the tank levels of both terminals for schedule x, with the target line and the deliveries marked.\"\"\"
    ax = ax or plt.subplots(figsize=(4.4, 2.6))[1]
    for k, (name, colour) in enumerate((("A", "#0F62FE"), ("B", "#FA4D56"))):
        d = x[3 * k:3 * k + 3]
        levels = [START[name]]
        for t in range(3):                               # follow the tank period by period
            levels.append(levels[-1] - USE[name] + CARGO * d[t])
        ax.plot(range(4), levels, "o-", color=colour, label=f"terminal {name}")
        for t in (1, 2, 3):
            if d[t - 1]:
                ax.annotate("ship", (t, levels[t]), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8, color=colour)
    ax.axhline(2, ls="--", color="k", lw=1)
    ax.set_xticks(range(4)); ax.set_xlabel("period"); ax.set_ylabel("tank level"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)
    return ax


plot_stocks([1, 1, 1, 0, 0, 0], "schedule A, A, A: terminal B runs dry");"""

S2_CL_MD = """**Classical baseline first.** Only 64 schedules, so we try them all and keep the cheapest. We also try a *greedy dispatcher*, the rule a human scheduler might use: each period, serve the terminal whose tank would be lower after this period's use."""

S2_CLASSICAL = """# Classical baseline: always runs first.
from qff.bits import index_to_x

t0 = time.perf_counter()
table = sorted(((schedule_cost(index_to_x(k, QUBITS)), tuple(index_to_x(k, QUBITS))) for k in range(2 ** QUBITS)))
brute_seconds = time.perf_counter() - t0
classical_value = table[0][0]
classical_optima = [x for c, x in table if c == classical_value]
classical_x = list(classical_optima[0])
costs_all = np.array([c for c, x in table])
print(f"schedules checked: {len(table)} | best cost: {classical_value} | best schedule x = {classical_x}")
print("cheapest costs:", [c for c, x in table[:6]], "| worst:", costs_all.max(), "| average:", costs_all.mean())


def who(x):
    \"\"\"Schedule as a sentence: which terminal is served in each period.\"\"\"
    return ", ".join("/".join(n for n, d in (("A", x[t]), ("B", x[3 + t])) if d) or "nobody" for t in range(3))


# Greedy dispatcher: each period serve the terminal whose tank would be lower after this period's use (ties: A).
t0 = time.perf_counter()
level, greedy_x = dict(START), [0] * 6
for t in range(3):
    after = {n: level[n] - USE[n] for n in "AB"}
    pick = min("AB", key=lambda n: (after[n], n))
    greedy_x[t + (3 if pick == "B" else 0)] = 1
    level = {n: after[n] + (CARGO if n == pick else 0) for n in "AB"}
greedy_seconds = time.perf_counter() - t0
greedy_value = schedule_cost(greedy_x)
print(f"best schedule serves: {who(classical_x)} | greedy dispatcher serves: {who(greedy_x)} (cost {greedy_value})")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
plot_stocks([1, 1, 1, 0, 0, 0], f"A, A, A: cost {schedule_cost([1, 1, 1, 0, 0, 0])}", ax=axes[0])
plot_stocks(classical_x, f"{who(classical_x)}: cost {classical_value}", ax=axes[1])
plt.tight_layout()"""

S2_NOTE = """The greedy dispatcher also finds B, A, B: on this toy a simple rule is already optimal, which is luck of a small, symmetric example. Four schedules tie for second place at cost 8, so the cost landscape has a clear single winner and a crowd just behind it. Both methods answer in a blink. That is the honest starting point: **on this toy the classical methods are instant, and QAOA cannot beat instant.** The toy exists so you can see how the quantum method works."""

S2_TSP_MD = """**How big does routing alone get?** Qiskit has a ready-made `Tsp` class for the travelling-salesman problem (visit every terminal once, shortest tour). We use it only to count qubits."""

S2_TSP = """import networkx as nx
from qiskit_optimization.applications import Tsp

D3 = np.array([[0, 4, 6], [4, 0, 3], [6, 3, 0]])                     # sailing times between three terminals
D4 = np.array([[0, 4, 6, 5], [4, 0, 3, 2], [6, 3, 0, 7], [5, 2, 7, 0]])
tsp_vars = {n: Tsp(nx.from_numpy_array(D)).to_quadratic_program().get_num_vars() for n, D in ((3, D3), (4, D4))}
print("qubits for plain routing (no tanks, no time):", tsp_vars)"""

S2_TSP_NOTE = """Pure routing already needs n² qubits: 9 for three terminals, 16 for four. Adding time periods and tank levels on top is why the paper's formulation is large, and why our toy fixes **one ship and three periods** and makes the decision "does the ship serve terminal p in period t?", which costs one qubit each. Our model is **not** the ExxonMobil–IBM model; it is a three-period miniature in the same spirit."""

S3_ROADMAP = r"""### 3.0 · The map of this section

""" + callout("idea", r"""**QAOA in one sentence.** Turn the cost of every schedule into a *twist* of a quantum state, *shake* the schedules together so that twists add up for some and cancel for others, and let a classical optimiser tune how hard to twist and shake until cheap schedules are the likely ones.""") + r"""

Everything below builds that sentence from two things you already know: lists of numbers (vectors) and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**, so you can watch the maths run and check it against Qiskit. (If you did notebook 3 you will recognise steps 3.1, 3.5 and 3.7: the algorithm is the same, only the cost differs. They are repeated here so this notebook stands on its own.)

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | A quantum state is a list of 64 numbers; probability = number² | (1) | state cell | |
| 3.2 | Tank levels and the cost of a schedule | (2), (3) | `schedule_cost` | **T1** |
| 3.3 | Expanding a square into a QUBO | (4) | `add_square` | |
| 3.4 | The QUBO becomes a Hamiltonian | (5) | `qp.to_ising()` | |
| 3.5 | The circuit: start, cost layer, mixer, repeat | (6) to (9) | `qaoa_ansatz` | |
| 3.6 | A worked example you can do on paper (2 decisions) | (4) to (10) | mini-example cell | |
| 3.7 | Average cost and success probability | (10), (11) | `success_of` | **T2** |
| 3.8 | Measuring, the optimiser loop, layers, what-ifs | (12) | `optimise` | **T3** |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it."""

S3_FLOW = """qff.plots.plot_flow(
    [("Rules", "section 2", "source"), ("Cost C(x)", "equations 2-3", "move"), ("QUBO", "equation 4", "store"),
     ("Hamiltonian", "equation 5", "use"), ("Circuit", "equations 6-9", "source"), ("Measure", "equations 10-11", "move")],
    loop=(5, 4, "the classical optimiser retunes the circuit's knobs (equation 12)"), title="QAOA: from rules to a schedule");"""

S3_STATE_MD = r"""### 3.1 · From yes/no decisions to a quantum state

**Start with what you know.** A *schedule* is a list of six yes/no decisions $x=(x_{A1},x_{A2},x_{A3},x_{B1},x_{B2},x_{B3})$, where $x_{A2}=1$ means "the ship delivers to terminal A in period 2". There are $2^6=64$ schedules. An ordinary computer holds *one* of them at a time. Six qubits hold a list of **64 numbers**, one per schedule. These numbers are called **amplitudes** and written $\alpha_x$:

$$|\psi\rangle=\sum_{x}\alpha_x\,|x\rangle,\qquad p(x)=|\alpha_x|^2,\qquad \sum_x |\alpha_x|^2=1 \tag{1}$$

How to read equation (1):

- $|x\rangle$ (say "ket $x$") is just a *name tag* for the schedule $x$. As a vector it is a column of 64 numbers with a single 1 at position $x$ and 0 everywhere else. For one qubit, $|0\rangle=\binom{1}{0}$ and $|1\rangle=\binom{0}{1}$.
- $\alpha_x$ is the amplitude of schedule $x$. It may be negative, or even complex. Its **squared size** $|\alpha_x|^2$ is the **probability** of reading out $x$ when you **measure**.
- The 64 probabilities add up to 1, as probabilities must.

""" + callout("math", r"""**Complex numbers in 60 seconds.** An amplitude may be a complex number. All you need is one fact: $e^{i\theta}=\cos\theta+i\sin\theta$ is a point on a circle of radius 1, at angle $\theta$. Multiplying an amplitude by $e^{i\theta}$ **turns it by the angle $\theta$ and does not change its size**, so $|\alpha|^2$ stays the same. That turning angle is the **phase**. A measurement cannot see a phase on its own, but phases decide how amplitudes *add or cancel* later (this is **interference**): two amplitudes of size $\tfrac12$ that point the same way add up to size 1; pointing opposite ways they cancel to 0.""", "Complex numbers, only what we need") + "\n\n" + callout("watch", r"""**Bit order.** Variable number $i$ in `NAMES` is qubit $i$, and schedule $x$ sits at position $k=\sum_i x_i\,2^i$ of the list. Qiskit prints strings with qubit 0 on the **right**: the string `'101010'` means $x=(0,1,0,1,0,1)$, which is B, A, B. Never index count strings by hand; use `qff.bits.to_x` and `index_to_x`, as the cells below do.""") + "\n\n" + arabic_table([
    ("Quantum state", "الحالة الكمومية", "I"), ("Qubit", "كيوبت", "G"), ("Amplitude", "السعة", "G"), ("Probability", "الاحتمال", "I"),
    ("Measurement", "القياس", "G"), ("Superposition", "التراكب", "G"), ("Phase", "الطور", "U"), ("Interference", "التداخل", "G"),
    ("Basis state (one schedule)", "حالة القاعدة", "I"), ("Complex number", "العدد المركّب", "I")])

S3_STATE_CODE = """# Equation (1) in code: a state is a list of 64 amplitudes; the probabilities are their squared sizes.
alpha = np.zeros(2 ** QUBITS, dtype=complex)
alpha[0] = 1                                              # |000000>: all the weight on schedule number 0 (nothing delivered)
print("all weight on one schedule :", "sum of |alpha|^2 =", np.sum(np.abs(alpha) ** 2))
alpha = np.full(2 ** QUBITS, 1 / 8, dtype=complex)        # 64 equal amplitudes of 1/8
print("64 equal amplitudes of 1/8:", "p(x) = |alpha_x|^2 =", abs(alpha[5]) ** 2, "= 1/64 =", 1 / 64, "| sum =", np.sum(np.abs(alpha) ** 2))"""

S3_STOCK_MD = r"""### 3.2 · Tank levels and the cost of a schedule

Two ingredients. First, **how full is each tank?** Terminal $p$ starts with stock $s_p$, loses $u_p$ units every period, and gains $q$ units whenever the ship delivers to it. After period $t$:

$$I_p(t)=s_p-u_p\,t+q\sum_{u=1}^{t}x_{p,u} \tag{2}$$

The sum counts the deliveries to terminal $p$ up to period $t$. With our numbers ($s_A=2$, $s_B=1$, $u=1$, $q=2$) the schedule A, A, A gives $I_A=3,4,5$ and $I_B=0,-1,-2$: terminal B runs dry.

Second, **what does a schedule cost?** We add the three rules of section 2 into one number, with target level $T_p=2$, sailing price $\sigma=1$ and clash price $\kappa=8$:

$$C(x)=\underbrace{\sum_{p\in\{A,B\}}\sum_{t=1}^{3}\bigl(I_p(t)-T_p\bigr)^2}_{\text{tanks away from target}}\;+\;\underbrace{\sigma\sum_{t=1}^{2}\bigl(x_{A,t}\,x_{B,t+1}+x_{B,t}\,x_{A,t+1}\bigr)}_{\text{switching terminal}}\;+\;\underbrace{\kappa\sum_{t=1}^{3}x_{A,t}\,x_{B,t}}_{\text{one ship, two places}} \tag{3}$$

The products are easy to read: $x_{A,t}\,x_{B,t+1}=1$ only if the ship serves A in period $t$ **and** B in period $t+1$, so it had to sail. The function `schedule_cost` in section 2 is exactly equation (3). The cheapest schedule is the answer.

""" + arabic_table([
    ("Cost function", "دالة التكلفة", "I"), ("Inventory", "المخزون", "I"), ("Routing", "تحديد المسارات", "I"), ("Schedule", "الجدول الزمني", "I"),
    ("Tank / storage tank", "خزان", "I"), ("Terminal", "محطة (مرفأ)", "I"), ("Penalty", "العقوبة (penalty)", "I"), ("Optimization", "الأمثَلة (التحسين)", "I")])

S3_STOCK_CODE = '''# Equations (2) and (3) in code: the cost of all 64 schedules (schedule_cost is equation (3)).
levels_c, counts_c = np.unique(costs_all, return_counts=True)
fig, axes = plt.subplots(1, 3, figsize=(12, 2.7))
plot_stocks([1, 1, 1, 0, 0, 0], f"A, A, A: cost {schedule_cost([1, 1, 1, 0, 0, 0])}", ax=axes[0])
plot_stocks(classical_x, f"{who(classical_x)}: cost {classical_value}", ax=axes[1])
axes[2].bar(levels_c, counts_c, color="#0F62FE", width=0.8)
axes[2].set_xlabel("cost C(x)"); axes[2].set_ylabel("number of schedules"); axes[2].set_title("all 64 schedules by cost", fontsize=10)
plt.tight_layout()
print("lowest cost", costs_all.min(), "| average cost", costs_all.mean(), "| highest", costs_all.max())'''

S3_STOCK_TRY = callout("try", """Predict the cost of the schedule A, B, A on paper using equation (3) (tank levels first, then the switches), then check with `schedule_cost([1, 0, 1, 0, 1, 0])`. The bar chart shows that very few schedules are cheap: how many have the lowest cost?""")

T1_INTRO = r"""### T1 · Warm-up: tank levels · التمرين ١

**What you are about to build, and why.** Equation (3) starts from the tank levels. Before any quantum step, you should be able to compute them yourself and see *why* B, A, B beats A, A, A. In T1 you write `stock_levels(x)` (equation 2) and `cost_from_levels(x)` (the first sum of equation 3), and compare the two schedules.

**The formula you implement:** $I_p(t)=s_p-u_p\,t+q\sum_{u\le t}x_{p,u}$, and the tank cost $\sum_p\sum_t (I_p(t)-T_p)^2$.

**Do this on paper first** for A, A, A: terminal A: $2-1+2=3$, then $2-2+2\cdot2=4$, then $2-3+2\cdot3=5$. Then check with code.

""" + hints(
    r"""The list `x` has six numbers: the first three are terminal A's decisions, the last three terminal B's. For terminal A after period 2, which of those decisions count as "deliveries so far"? How many cargoes arrived? Multiply that by the cargo size and apply equation (2).""",
    r"""For each terminal, take its three decisions; for each period t = 1, 2, 3 count the deliveries up to t, and apply equation (2): start stock, minus use times t, plus cargo times the number of deliveries. Collect the three levels per terminal in a dictionary. The tank cost squares each level's distance from the target and adds everything up, over both terminals and all periods.""",
    r"""```python
def stock_levels(x, start=START, cargo=CARGO):
    out = {}
    for k, name in enumerate("AB"):
        d = ______                       # this terminal's three decisions
        out[name] = ______               # the three levels after periods 1, 2, 3 (equation 2)
    return out

def cost_from_levels(x):
    levels = ______
    return ______                        # sum of (level - target)^2 over both terminals and three periods
```
Test yourself: `stock_levels([1, 1, 1, 0, 0, 0])` is `{"A": [3, 4, 5], "B": [0, -1, -2]}`.""")

S3_QUBO_MD = r"""### 3.3 · From squares to a QUBO

The tank part of (3) is a sum of **squares**. A square of a sum of 0/1 decisions expands into constants, single terms and pairs. Write $(c+\sum_i a_ix_i)^2$ and use the fact that a 0/1 variable satisfies $x_i^2=x_i$:

$$\Bigl(c+\sum_i a_i x_i\Bigr)^2=c^2+\sum_i\bigl(2c\,a_i+a_i^2\bigr)\,x_i+\sum_{i<j}2\,a_ia_j\,x_ix_j \tag{4}$$

Adding all the squares, the sailing products and the clash products of equation (3) therefore gives one polynomial with a constant, single terms $x_i$ and pair terms $x_ix_j$ and nothing worse. A cost of this shape is called a **QUBO**: *quadratic unconstrained binary optimisation* (quadratic: at most pairs; unconstrained: no side rules, all rules are already inside the cost; binary: 0/1 variables). Qiskit keeps such a polynomial in a **QuadraticProgram** (its `minimize` method takes the constant, the single-term coefficients and the pair coefficients). The helper `add_square` below applies equation (4) for you and collects the coefficients; you use it in T2.

**A tiny example you can check by hand.** Take $c=-1$ and $a_1=a_2=2$: $(-1+2x_1+2x_2)^2$. Equation (4) gives constant $1$; linear terms $2\cdot(-1)\cdot2+2^2=0$ for each variable; one pair term $2\cdot2\cdot2=8$. So the square equals $1+8\,x_1x_2$. Check all four cases: $(0,0)\to(-1)^2=1$; $(1,0)\to1^2=1$; $(0,1)\to1$; $(1,1)\to3^2=9=1+8$. ✓

""" + callout("real", r"""A QUBO is the common input format of quantum optimisation: quantum annealers and gate-model algorithms such as QAOA both accept it. The WEF/Aramco paper's LNG formulation (Case 4) is also in QUBO form, though a much larger one than ours.""") + "\n\n" + arabic_table([
    ("QUBO", "مسألة تحسين ثنائية تربيعية غير مقيدة (QUBO)", "I"), ("Binary variable", "متغير ثنائي", "I"), ("Quadratic", "تربيعي", "I"),
    ("Polynomial", "كثيرة حدود", "I"), ("Constraint", "القيد", "I")])

S3_DEMO_CHECK = '''# Check equation (4) on the tiny example: the expanded polynomial equals the square for all four cases.
for a1, a2 in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    direct = (-1 + 2 * a1 + 2 * a2) ** 2
    expanded = demo["const"] + demo["linear"]["x_A1"] * a1 + demo["linear"]["x_A2"] * a2 + demo["quadratic"][("x_A1", "x_A2")] * a1 * a2
    print(f"x_A1={a1}, x_A2={a2}:  square = {direct}   expanded = {expanded}")
    assert direct == expanded, "equation (4) disagrees with add_square"
print("OK: add_square implements equation (4).")'''

S3_ISING_MD = r"""### 3.4 · From a QUBO to a Hamiltonian

A quantum computer works with **operators** that act on the state. An operator is a *matrix*: a square table of numbers that turns one state vector into another. We need a matrix whose "energy" for schedule $x$ is exactly $C(x)$. Physicists call it a **Hamiltonian** (an energy operator).

The **Pauli $Z$ gate** is the matrix $Z=\begin{pmatrix}1&0\\0&-1\end{pmatrix}$: it leaves $|0\rangle$ alone and flips the sign of $|1\rangle$, so $Z|0\rangle=(+1)|0\rangle$ and $Z|1\rangle=(-1)|1\rangle$. We say $|0\rangle$ and $|1\rangle$ are **eigenstates** of $Z$ with **eigenvalues** $+1$ and $-1$. Notice that $\frac{1-Z}{2}$ has eigenvalue $0$ on $|0\rangle$ and $1$ on $|1\rangle$: exactly the value of a bit. So replace every variable of the QUBO by an operator, $x_i\to\frac{1-Z_i}{2}$ and $x_ix_j\to\frac{(1-Z_i)(1-Z_j)}{4}$, and multiply out. For a QUBO $C=c_0+\sum_i l_i x_i+\sum_{i<j}q_{ij}x_ix_j$ the result is

$$\begin{aligned}H_C&=c_0'+\sum_i h_i\,Z_i+\sum_{i<j}J_{ij}\,Z_iZ_j,\\ c_0'&=c_0+\sum_i\frac{l_i}{2}+\sum_{i<j}\frac{q_{ij}}{4},\quad h_i=-\frac{l_i}{2}-\sum_{j\ne i}\frac{q_{ij}}{4},\quad J_{ij}=\frac{q_{ij}}{4}\end{aligned} \tag{5}$$

(where $q_{ij}$ with $j<i$ means $q_{ji}$). A Hamiltonian made only of $Z$ terms like this one is called an **Ising** Hamiltonian (that is why the Qiskit method is called `to_ising`). It is a **diagonal** matrix (every number off the diagonal is 0) with the costs on its diagonal: $H_C|x\rangle=C(x)\,|x\rangle$. The cheapest schedule is the state of lowest energy, the **ground state**. **Finding the best schedule means finding the ground state of $H_C$.**

**Our tiny example again.** $C=1+8\,x_1x_2$ has $c_0=1$, $l_i=0$, $q_{12}=8$. Equation (5): $c_0'=1+\frac84=3$, $h_1=h_2=-\frac84=-2$, $J_{12}=\frac84=2$. So $H_C=3-2Z_1-2Z_2+2Z_1Z_2$. Check: for $(0,0)$ both $Z=+1$: $3-2-2+2=1$ ✓. For $(1,1)$ both $Z=-1$: $3+2+2+2=9$ ✓. For $(1,0)$: $Z_1=-1,Z_2=+1$: $3+2-2-2=1$ ✓.

""" + callout("watch", r"""Nothing is lost in this step: $H_C$ has exactly the 64 costs of equation (3) on its diagonal. The code cell below checks equation (5) against what `to_ising()` returns, first on the tiny example, and after T2 on the real problem.""") + "\n\n" + arabic_table([
    ("Hamiltonian", "مؤثر هاملتون (الهاملتوني)", "G"), ("Operator", "المؤثر", "G"), ("Eigenvalue", "القيمة الذاتية", "G"),
    ("Eigenstate", "الحالة الذاتية", "G"), ("Pauli gates (X, Y, Z)", "بوابات باولي", "G"), ("Ground state", "الحالة الأرضية", "I"),
    ("Diagonal matrix", "مصفوفة قطرية", "I"), ("Matrix", "مصفوفة", "I")])

S3_ISING_CODE = '''# Equation (5) in code, checked against Qiskit's qp.to_ising() on the tiny example C = 1 + 8 x_A1 x_A2.
def ising_by_formula(qp):
    """Equation (5): constant c0', Z coefficients h_i and ZZ coefficients J_ij of the Hamiltonian of a QUBO."""
    n = qp.get_num_vars()
    lin, pairs = np.zeros(n), {}
    for i, v in qp.objective.linear.to_dict().items():
        lin[i] += v
    for (i, j), v in qp.objective.quadratic.to_dict().items():
        if i == j:
            lin[i] += v                                   # x*x = x for a 0/1 variable
        else:
            pairs[tuple(sorted((i, j)))] = pairs.get(tuple(sorted((i, j))), 0) + v
    h = -lin / 2
    for (i, j), v in pairs.items():
        h[i] -= v / 4
        h[j] -= v / 4
    return qp.objective.constant + lin.sum() / 2 + sum(pairs.values()) / 4, h, {k: v / 4 for k, v in pairs.items()}


def check_ising(qp, op, off):
    """Stop with an error if Qiskit's Hamiltonian (op, off) differs from equation (5)."""
    c0, h, J = ising_by_formula(qp)
    assert abs(off - c0) < 1e-9, "constant differs"
    for pauli, coeff in zip(op.paulis, op.coeffs):
        on = [i for i in range(op.num_qubits) if pauli.z[i]]
        want = h[on[0]] if len(on) == 1 else J[tuple(on)]
        assert abs(coeff.real - want) < 1e-9, (pauli, coeff, want)


qp_demo = QuadraticProgram()
for name in ("x_A1", "x_A2"):
    qp_demo.binary_var(name)
qp_demo.minimize(constant=demo["const"], linear=demo["linear"], quadratic=demo["quadratic"])
op_demo, off_demo = qp_demo.to_ising()
check_ising(qp_demo, op_demo, off_demo)
diag_demo = qff.classical.diag_of(op_demo, off_demo)
print("Hamiltonian terms:", {k: float(v) for k, v in zip(op_demo.paulis.to_labels(), op_demo.coeffs.real)}, "| constant:", off_demo)   # ZZ 2, Z -2, Z -2; 3
print("cost of (0,0), (1,0), (0,1), (1,1):", diag_demo.tolist())                                                   # [1, 1, 1, 9]
assert diag_demo.tolist() == [1.0, 1.0, 1.0, 9.0]
print("OK: equation (5) matches Qiskit on the tiny example.")'''

S3_CIRCUIT_MD = r"""### 3.5 · The QAOA circuit

The circuit has three ingredients, applied in this order to six qubits that start in $|0\rangle$.

**(a) Start: an equal mixture.** The **Hadamard gate** $H$ turns $|0\rangle$ into $\tfrac1{\sqrt2}(|0\rangle+|1\rangle)$. On six qubits it gives the same amplitude $\tfrac18$ to every one of the 64 schedules: every schedule has probability $\tfrac1{64}$ and the average cost is 22. This is pure guessing.

$$|s\rangle=H^{\otimes 6}\,|000000\rangle=\frac1{\sqrt{2^6}}\sum_x|x\rangle \tag{6}$$

($H^{\otimes 6}$ just means "one $H$ gate on each of the six qubits".)

**(b) Cost layer: twist every schedule by its cost.** For an angle $\gamma$ (a knob we will tune) apply

$$U_C(\gamma)=e^{-i\gamma H_C},\qquad\text{which acts as}\qquad |x\rangle\;\longmapsto\;e^{-i\gamma\,C(x)}\,|x\rangle \tag{7}$$

Because $H_C$ is diagonal, "the exponential of a matrix" is just the exponential of each diagonal entry. Every schedule is *turned* by an angle proportional to its cost, so costly schedules turn faster. No probability changes, only phases. In gates, each $Z_i$ term of (5) becomes one **RZ** gate and each $Z_iZ_j$ term one **RZZ** gate, **one per pair of decisions that the cost couples**: $e^{-i\gamma hZ_i}=R_Z(2\gamma h)$ and $e^{-i\gamma JZ_iZ_j}=R_{ZZ}(2\gamma J)$.

**(c) Mixer layer: let neighbouring schedules talk.** For a second knob $\beta$ apply

$$U_B(\beta)=e^{-i\beta\sum_i X_i}=\bigotimes_{i}R_X(2\beta) \tag{8}$$

The **Pauli $X$ gate** flips a bit ($|0\rangle\leftrightarrow|1\rangle$), and $e^{-i\beta X}=\cos\beta\,I-i\sin\beta\,X$ (here $I$ is the "do nothing" matrix, and $\bigotimes_i R_X(2\beta)$ means one $R_X$ gate on every qubit) reads "keep with amplitude $\cos\beta$, flip with amplitude $-i\sin\beta$". So the mixer moves amplitude between schedules that differ in **one** decision. Where amplitudes arrive from different neighbours with different phases, they **interfere**: some add up, some cancel.

**(d) Repeat.** One cost layer plus one mixer is one **layer**. With $p$ layers the circuit has $2p$ knobs $\gamma_1,\dots,\gamma_p$ and $\beta_1,\dots,\beta_p$, and layer 1 acts first:

$$|\gamma,\beta\rangle=U_B(\beta_p)\,U_C(\gamma_p)\cdots U_B(\beta_1)\,U_C(\gamma_1)\,|s\rangle \tag{9}$$

""" + callout("idea", r"""**Why "twist, then shake" can find cheap schedules.** The cost layer changes no probability by itself; it only writes each schedule's cost into its *phase*. The mixer turns phase differences into probability differences: contributions that arrive in step reinforce each other, the others cancel. With well-chosen $(\gamma,\beta)$ the cheap schedules end up reinforced. We do not know good knob values in advance, so a classical optimiser searches for them (section 3.8).""") + "\n\n" + callout("watch", r"""**Knob order.** Qiskit sorts the knobs alphabetically, so a list of knob values must be **all $\beta$ first, then all $\gamma$**: for $p=1$ it is `[beta, gamma]`, for $p=2$ it is `[beta1, beta2, gamma1, gamma2]`. Many papers write $(\gamma,\beta)$ the other way round.""") + "\n\n" + arabic_table([
    ("Hadamard gate", "بوابة هادامارد", "G"), ("Quantum circuit", "الدائرة الكمومية", "U"), ("Rotation gate (RX, RZ)", "بوابة دوران", "I"),
    ("Cost layer", "طبقة التكلفة", "I"), ("Mixer layer", "طبقة الخلط", "I"), ("Ansatz (the parametrised circuit)", "الافتراض المبدئي (ansatz)", "G"),
    ("Variational", "التغايري / المتغايرة", "G"), ("QAOA", "خوارزمية الأمثلة الكمومية التقريبية", "G"), ("Circuit depth", "عمق الدائرة", "G"),
    ("Global phase", "الطور الكلي", "I")])

S3_SYMBOLS = r"""**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).

| Symbol | Meaning | In the code |
|---|---|---|
| $x=(x_{A1},\dots,x_{B3})$ | a schedule: the ship serves terminal $p$ in period $t$ when $x_{p,t}=1$ | `x`, `NAMES`, `index_to_x(k, 6)` |
| $s_p,\ u_p,\ q,\ T_p$ | start stock, use per period, cargo size, target level | `START`, `USE`, `CARGO`, `TARGET` |
| $I_p(t)$ | tank level of terminal $p$ after period $t$ | `stock_levels(x)` (T1) |
| $\sigma,\ \kappa$ | price of switching terminal; price of serving both at once | `SAIL`, `CLASH` |
| $C(x)$ | cost of schedule $x$ | `schedule_cost(x)`, `diag[k]` |
| $c_0,\ l_i,\ q_{ij}$ | constant, single-term and pair coefficients of the QUBO | `coef`, `qp.objective` |
| $Z_i,\ X_i$ | Pauli $Z$ and $X$ acting on qubit $i$ | the Pauli strings of `cost_op` |
| $H_C,\ h_i,\ J_{ij},\ c_0'$ | cost Hamiltonian and its coefficients | `cost_op`, `cost_op.coeffs`, `offset` |
| $\lvert x\rangle,\ \alpha_x,\ p(x)$ | a schedule as a state, its amplitude, its probability | `Statevector(...).data`, `.probabilities()` |
| $\lvert s\rangle$ | the equal mixture at the start | the first layer of H gates |
| $\gamma,\ \beta$ | cost-layer and mixer knobs | circuit parameters `γ[k]`, `β[k]` |
| $p$ | number of layers (depth) | `reps=p` |
| $\langle C\rangle$ | average cost, what the optimiser minimises | `qaoa_energy`, `res1.fun` |
| $P_{\rm opt}$ | probability of the best schedule | `success_of(circuit, theta)` |"""

S3_MINI_MD = r"""### 3.6 · A worked example you can check on paper

Run the whole algorithm on the tiny cost from 3.3, $C=1+8\,x_1x_2$: two decisions, four schedules, costs $1,1,1,9$ (only "both yes" is expensive). With $p=1$, $\gamma$ and $\beta$:

1. **Start (6):** amplitude $\tfrac12$ on all four schedules.
2. **Cost layer (7):** each schedule gets the phase $e^{-i\gamma C(x)}$. Drop the common factor $e^{-i\gamma}$ (a **global phase**: the same turn for every amplitude, which nothing can detect). Then $(0,0),(1,0),(0,1)$ keep amplitude $\tfrac12$ and $(1,1)$ gets $\tfrac12e^{-8i\gamma}$.
3. **Mixer (8):** each qubit is kept with amplitude $c=\cos\beta$ or flipped with amplitude $-is$, where $s=\sin\beta$. The amplitude of $(1,1)$ after the mixer is the sum over where it came from: from $(1,1)$ itself ($c\cdot c$), from $(1,0)$ or $(0,1)$ (one flip: $c\cdot(-is)$ each), from $(0,0)$ (two flips: $(-is)^2=-s^2$):

$$\alpha_{11}=\tfrac12\bigl[c^2e^{-8i\gamma}-2i\,cs-s^2\bigr],\qquad P(1,1)=|\alpha_{11}|^2 ,\qquad \langle C\rangle=1+8\,P(1,1) \tag{M}$$

**Plug in numbers.** Take $\gamma=\frac{3\pi}{16}$ and $\beta=\frac\pi4$. Then $8\gamma=\frac{3\pi}2$, so $e^{-8i\gamma}=e^{-3\pi i/2}=i$; and $c=s=\frac1{\sqrt2}$ gives $c^2=s^2=cs=\frac12$. So $\alpha_{11}=\frac12\bigl[\frac i2-i-\frac12\bigr]=\frac12\bigl[-\frac12-\frac i2\bigr]$ and $P(1,1)=\frac14\bigl(\frac14+\frac14\bigr)=\frac18$. The average cost is $\langle C\rangle=1+8\cdot\frac18=2$, **down from 3** for random guessing ($1+8\cdot\frac14$). One layer, set well, already pushes probability away from the expensive schedule.

""" + callout("math", r"""**Phases repeat.** The cost layer only sees $e^{-i\gamma\,(\text{cost difference})}$. Here the difference is 8, so $\gamma$ and $\gamma+\frac{2\pi}{8}=\gamma+\frac\pi4$ give the same circuit. With integer costs the useful range of $\gamma$ is limited, which is one reason the optimiser's landscape repeats (the picture below).""") + "\n\n" + details("Why is the mixer sum like that? (optional)", r"""The mixer is a product of one-qubit matrices, $R_X(2\beta)=\begin{pmatrix}c&-is\\-is&c\end{pmatrix}$. The entry that takes qubit value $b$ to value $a$ is $c$ if $a=b$ and $-is$ if $a\ne b$. A two-qubit amplitude is the product of two such entries. So the amplitude of $(1,1)$ after the mixer is $\sum_y\langle 11|U_B|y\rangle\,\alpha_y$ with entries $c^2,\ -ics,\ -ics,\ (-is)^2$ for $y=(1,1),(1,0),(0,1),(0,0)$, and with $\alpha_y=\tfrac12$ except $\alpha_{11}=\tfrac12e^{-8i\gamma}$. That is the formula above.""") + "\n\nThe next cell builds this example with the same Qiskit pipeline as the real problem and compares equation (M) with Qiskit's answer at four knob settings."

S3_MINI = '''# The worked example, checked against Qiskit: QAOA with p = 1 on the tiny cost C = 1 + 8 x_1 x_2.
ansatz_demo = qaoa_ansatz(op_demo, reps=1)                                           # knobs in the order [beta, gamma]


def p11_qiskit(beta, gamma):
    return float(Statevector(ansatz_demo.assign_parameters([beta, gamma])).probabilities()[3])      # schedule (1, 1)


def p11_by_hand(beta, gamma):
    c, s = np.cos(beta), np.sin(beta)
    return abs(0.5 * (c * c * np.exp(-8j * gamma) - 2j * c * s - s * s)) ** 2                          # equation (M)


print(f"{'beta':>8}{'gamma':>8}{'P(1,1) by hand':>16}{'Qiskit':>10}{'<C> = 1 + 8 P':>16}")
for beta, gamma in [(0.0, 0.7), (np.pi / 4, 3 * np.pi / 16), (np.pi / 4, np.pi / 16), (0.3, 1.1)]:
    ph, pq = p11_by_hand(beta, gamma), p11_qiskit(beta, gamma)
    print(f"{beta:8.3f}{gamma:8.3f}{ph:16.4f}{pq:10.4f}{1 + 8 * pq:16.4f}")
    assert abs(ph - pq) < 1e-9, "equation (M) disagrees with Qiskit"
assert abs(p11_qiskit(np.pi / 4, 3 * np.pi / 16) - 1 / 8) < 1e-9                                      # the hand calculation
qff.plots.plot_landscape(lambda b, g: 1 + 8 * p11_by_hand(b, g), (0, np.pi / 2), (0, np.pi / 4), n=61, xlabel="beta (mixer knob)",
                         ylabel="gamma (cost knob)", title="average cost <C> for the tiny example (dark = low)",
                         marks=[(np.pi / 4, 3 * np.pi / 16, "hand-checked point: <C> = 2", "use")]);'''

S3_SUCCESS_MD = r"""### 3.7 · Average cost and success probability

When the circuit has run, every schedule $x$ has a probability $p(x)$ (equation 1). Two numbers summarise how good the circuit is:

$$\langle C\rangle=\sum_x p(x)\,C(x) \tag{10}$$

$$P_{\rm opt}=\sum_{x\ \text{optimal}}p(x) \tag{11}$$

- **(10)** is the *average cost* you would see if you ran the circuit many times: each cost weighted by how likely it is. **It is what the optimiser tries to make small.** In code it is one line, `probabilities @ diag`: the dot product of the 64 probabilities with the 64 costs. (Physicists write the same number as $\langle\psi|H_C|\psi\rangle$; since $H_C$ is diagonal the two are identical, and the exact-statevector trick needs no shots.)
- **(11)** is what we care about at the end: the chance of reading out the best schedule. Random guessing gives $1/64=1.6\%$.

Two numbers to remember: the *lowest* cost is **5** and the *average* cost of a random schedule is **22**. A good QAOA run pushes $\langle C\rangle$ down from 22 towards 5. The shared helper `qff.quantum.qaoa_energy` computes (10); `success_of` below computes (11).

""" + arabic_table([
    ("Expectation value (average cost)", "القيمة المتوقعة (متوسط التكلفة)", "I"), ("Success probability", "احتمال النجاح", "I"),
    ("State vector", "متجه الحالة", "I"), ("Classical optimiser", "المُحسِّن الكلاسيكي", "I"), ("Random guessing", "التخمين العشوائي", "I")])

T2_INTRO = r"""### T2 · Core step: build the problem and run QAOA · التمرين ٢

**What you are about to build, and why.** This is the heart of the notebook. You turn the *rules* into a Qiskit problem: `build_qp` adds one square per terminal and period (the tank rule, equation 4), one product for every possible switch and one for every clash (equation 3), and hands the polynomial to a `QuadraticProgram`. A note on the numbers: the single start $(0.1,0.1)$ of this task can land lower than the "about 8%" quoted later in 3.8, because that figure is the best of six starting points; the Self-check accepts 2% to 15%. Then you convert it to a Hamiltonian (equation 5), build the $p=1$ circuit (6 to 9) and let an optimiser minimise $\langle C\rangle$ (10).

**Two ways to do it.** The task cell already contains the plumbing (the loops over terminals, periods and neighbouring periods, the delivery terms of each square, the two switching products, the `QuadraticProgram` with its six binary variables, the conversion to a Hamiltonian, the $p=1$ circuit and the call to the optimiser) and leaves two blanks that carry the ideas: the constant part $c$ of each square (the tank rule, equation 4), and the clash product of equation (3). Teams that want the full exercise can delete the scaffold and write everything from the Goal.

**The formula you implement** is equation (3), term by term:

$$C(x)=\sum_{p,t}\Bigl(\underbrace{s_p-u_pt-T_p}_{c}+\sum_{u\le t}q\,x_{p,u}\Bigr)^2+\sigma\sum_{t}(x_{A,t}x_{B,t+1}+x_{B,t}x_{A,t+1})+\kappa\sum_t x_{A,t}x_{B,t}$$

Each squared term is exactly the shape $(c+\sum a_ix_i)^2$ that `add_square(coef, c, terms)` expects: the constant $c=s_p-u_pt-T_p$ and the terms $a_i=q$ for each delivery up to period $t$.

""" + hints(
    r"""Blank `add_square`: a square has the shape $(c+\sum a_ix_i)^2$ and the cell already gives you the terms $a_ix_i$ (the `deliveries`). What is the constant $c$ for terminal `name` after period `t`: how much is in the tank at the start, how much has been used up to period `t`, and what is the level you are aiming for? Blank `add_pair`: the clash penalises two decisions together. Which two decisions are they (look at the indices in equation 3), and what weight multiplies them? How do the two sailing lines just above call `add_pair`?""",
    r"""**The square**: the constant is $c=s_p-u_pt-T_p$ of equation (4): the start stock of terminal `name`, minus the use per period times the number of periods so far (`t`), minus the target. The three tables are indexed by the terminal name (`start[name]`, `USE[name]`, `TARGET[name]`; note that `start` is the argument of `build_qp`, not the constant `START`). **The clash**: one call of `add_pair(coef, first_variable, second_variable, weight)`, exactly like the two sailing lines above; the variable names look like `f"x_A{t}"` and `f"x_B{t}"`, and a clash means both terminals are served in the same period `t`, with the weight `clash` (the argument of `build_qp`).""",
    r"""```python
add_square(coef, ______[name] - USE[name] * ______ - ______[name], deliveries)
add_pair(coef, f"x_A{______}", f"x_B{______}", ______)
```""")

S3_CHECK_MD = """**Check equations (5) and (6) to (9) on the real problem.** The first part compares the Hamiltonian that `to_ising()` returned for *your* `qp` with equation (5). The second builds the circuit one gate at a time (a plain `h`, then `rz`, `rzz` and `rx` gates for equations 6 to 8) and checks that it gives the same state as `qaoa_ansatz` up to a **global phase** (one and the same turn applied to *every* amplitude, which changes no probability)."""

S3_CHECK = '''# Check: equation (5) on the real problem, and equations (6)-(9) built gate by gate.
check_ising(qp, cost_op, offset)
print("OK: equation (5) matches Qiskit's Hamiltonian for the real problem.")
print("terms:", len(cost_op), "| Z terms:", sum(1 for p_ in cost_op.paulis if p_.z.sum() == 1), "| ZZ terms:", sum(1 for p_ in cost_op.paulis if p_.z.sum() == 2))

from qiskit import QuantumCircuit


def qaoa_by_hand(cost_op, gammas, betas, barriers=True):
    """QAOA circuit from the equations. Z terms of equation (5) -> RZ, ZZ terms -> RZZ, mixer -> RX."""
    n = cost_op.num_qubits
    qc = QuantumCircuit(n)
    qc.h(range(n))                                                    # equation (6): the equal mixture
    for gamma, beta in zip(gammas, betas):
        if barriers:
            qc.barrier()
        for pauli, coeff in zip(cost_op.paulis, cost_op.coeffs):      # equation (7): one gate per term of equation (5)
            on = [i for i in range(n) if pauli.z[i]]
            if len(on) == 1:
                qc.rz(2 * gamma * coeff.real, on[0])                  # e^{-i gamma h Z_i}     = RZ(2 gamma h)
            elif len(on) == 2:
                qc.rzz(2 * gamma * coeff.real, on[0], on[1])          # e^{-i gamma J Z_i Z_j} = RZZ(2 gamma J)
        if barriers:
            qc.barrier()
        qc.rx(2 * beta, range(n))                                     # equation (8): the mixer, RX(2 beta) on every qubit
    return qc


beta_test, gamma_test = np.random.default_rng(SEED).uniform(0, 1, 2)
by_hand = qaoa_by_hand(cost_op, [gamma_test], [beta_test])
same = Statevector(by_hand).equiv(Statevector(ansatz.assign_parameters([beta_test, gamma_test])))
assert same, "the hand-built circuit should equal qaoa_ansatz up to a global phase"
print("hand-built circuit == qaoa_ansatz (same state up to a global phase):", same)
print("gates in one layer:", {g: c for g, c in by_hand.count_ops().items() if g != "barrier"})'''

S3_ANSATZ_MD = r"""**Read the drawing.** In the circuit below, read left to right: the **H** gates are equation (6); the **RZ** and **ZZ** gates are the cost layer (7), one RZ per decision and one ZZ for every pair of decisions the cost couples (the label shows the knob $\gamma$ they carry); the **RX** gates are the mixer (8) and carry $\beta$. Compare it with the hand-built circuit in the check cell above."""

S3_LAND_MD = r"""**What the optimiser walked on.** The next picture is the whole $p=1$ landscape: the average cost $\langle C\rangle$ for every pair $(\beta,\gamma)$ (dark is low, which is good), computed with the same Qiskit circuit you built in T2. The magenta dot is the starting point $(0.1,0.1)$ and the star is where the optimiser ended. All the structure is at small $\gamma$: the costs are large numbers (up to 47), so even a small $\gamma$ turns the expensive schedules by a lot, and the picture repeats rapidly as $\gamma$ grows. Rough, narrow valleys like these are why one start is not enough."""

S3_LAND = '''# The p = 1 landscape: average cost <C> for a grid of (beta, gamma) values.
f_energy = qff.quantum.qaoa_energy(ansatz, diag)
qff.plots.plot_landscape(lambda b, g: f_energy([b, g]), (0, np.pi), (0, 0.8), n=41, xlabel="beta (mixer knob)", ylabel="gamma (cost knob)",
                         title="average cost <C> at p = 1  (dark = low = good)",
                         marks=[(0.1, 0.1, "start", "use"), (theta1[0] % np.pi, theta1[1] % np.pi, "end: best found", "source")]);
print("start theta = [0.1, 0.1] ->", round(f_energy([0.1, 0.1]), 3), "| best theta =", np.round(theta1, 3), "->", round(f_energy(theta1), 3),
      "| random guessing average:", round(diag.mean(), 3))'''

S3_SAMPLE_MD = r"""### 3.8 · Measuring, the optimiser loop, layers

**Measuring.** On a real quantum computer you never see the 64 probabilities. You **run** the circuit and **measure**, which returns *one* schedule, chosen at random with probability $p(x)$. Repeat 4096 times (the repeats are called **shots**) and the frequencies approximate $p(x)$; the error shrinks like $1/\sqrt{\text{shots}}$. In the cell below we run the tuned $p=1$ circuit this way and plot the 12 most frequent schedules. The best schedule is drawn in red."""

S3_SAMPLE_TRY = callout("try", """Change `4096` to `100` in the sampling cell (both places) and run it again. The bars get ragged and the red bar may vanish from the top 12. That is the price of few shots.""")

S3_LOOP_MD = r"""**The hybrid loop.** QAOA is a conversation between a quantum circuit and an ordinary program:

$$(\gamma^*,\beta^*)=\arg\min_{\gamma,\beta}\ \langle C\rangle(\gamma,\beta) \tag{12}$$

("argmin" means "the values at which the function is smallest".) The circuit gives $\langle C
angle$ for the current knob values; a classical **optimiser** (here COBYLA, which needs no derivatives: it just asks "what is $\langle C
angle$ here?" and uses the answers to choose where to look next) proposes better knobs; repeat. The landscape above has several valleys (**local minima**), so one starting point can get stuck in a poor one. This cost is also rougher than the one in notebook 3: its coefficients range from 1 to 24. The helper below therefore starts at $(0.1,0.1)$ **and** from five random points (fixed seed, so everybody gets the same numbers) and keeps the best, for $p=1,2,3$. It also returns every start's result so you can see the spread."""

S3_LOOP_FIG = """qff.plots.plot_flow([("Knobs", "beta, gamma", "source"), ("Circuit", "equations 6-9", "use"), ("Average cost", "equation 10", "move"),
                     ("Optimiser", "COBYLA", "store")], loop=(3, 0, "new knob values"), title="the hybrid loop, repeated until <C> stops falling");"""

S3_LOOP_ARABIC = arabic_table([
    ("Measurement", "القياس", "G"), ("Shots", "عدد مرات التشغيل (shots)", "I"), ("Sampling", "أخذ العيّنات", "I"),
    ("Hybrid quantum-classical loop", "الحلقة الهجينة (كمومية وكلاسيكية)", "I"), ("Local minimum", "حدّ أدنى محلي", "I"),
    ("Cost landscape", "مشهد التكلفة", "I")])

S3_HIST_MD = """**Before and after.** Where does the probability go? The grey bars are random guessing, spread over costs 5 to 47. With QAOA the weight moves towards the cheap end, but only part of the way: this problem is harder for QAOA than the one in notebook 3."""

S3_HIST = '''dists = {"random guessing": np.full(2 ** QUBITS, 1 / 2 ** QUBITS)}
for p in (1, 2, 3):
    dists[f"QAOA p = {p}"] = Statevector(ansatz_by_p[p].assign_parameters(theta_by_p[p])).probabilities()
qff.plots.plot_cost_histogram(diag, dists, title="probability of each cost: guessing against QAOA with 1, 2, 3 layers")
print({name: round(float(pr @ diag), 3) for name, pr in dists.items()})          # the average cost <C> of each distribution'''

S3_WHY_MD = r"""**Why does it work, and when does it not?** The cost layer writes each schedule's cost into its phase (7); the mixer lets neighbouring schedules exchange amplitude (8), and where the phases line up the amplitudes add, elsewhere they cancel. The optimiser looks for the knobs for which cheap schedules line up and expensive ones cancel. More layers mean more knobs, hence a more flexible pattern, **in principle**. In practice the knobs have to be *found*, and that is itself a hard optimisation: with $2p=6$ knobs and a rough landscape, six starts are not enough, which is why $p=3$ does not beat $p=2$ above.

What is known, and what is not:

- In principle, with enough layers and the right angles, QAOA can approach the exact ground state; the algorithm was introduced by Farhi, Goldstone and Gutmann in 2014 (arXiv:1411.4028). At small $p$ nothing guarantees it.
- For larger circuits the landscape can become almost flat (**barren plateaus**) and the optimiser has nothing to follow.
- Real devices add **noise**; deeper circuits fail more and the failures blur the very interference the layers build.

""" + arabic_table([("Interference", "التداخل", "G"), ("Noise", "الضوضاء", "G"), ("Barren plateau", "الهضبة العقيمة", "G"),
                    ("Quantum advantage", "الأفضلية الكمومية", "G"), ("Number of layers (depth p)", "عدد الطبقات (العمق p)", "I")])

T3_INTRO = r"""### T3 · Stretch: what if the instance changes? · التمرين ٣

**What you are about to build, and why.** A planner never solves one instance. Fuel prices, ship speed or customer demand change, and the problem has to be rebuilt and solved again. In T3 you write `what_if(p, **changes)`: it rebuilds the cost with different inputs (start stocks, cargo size or sailing price), finds the new best schedule **classically**, runs QAOA on the new Hamiltonian, and reports how well QAOA did compared with guessing. It tests whether you really understand which input enters which term of equations (2) and (3).

**What you implement.** The same chain as in T2 with other numbers: $C_{\text{new}}(x)$ from (3), $H_C$ from (5), the knobs from (12), and then $P_{\rm opt}$ from (11) **summed over every optimal schedule**, because a change may produce ties. Guessing is $\text{(number of optima)}/64$.

""" + hints(
    r"""You already have `build_qp(start=..., cargo=..., sail=..., clash=...)` from T2: calling it with changed values gives the changed problem. What do you need after that to find the best schedule *without* any quantum step? And what do you need to run QAOA on the new problem? (Think of what T2 did after `build_qp`.)""",
    r"""Inside `what_if`: build the changed problem; find its best schedule and *all* tied optima by brute force; convert to a Hamiltonian and its diagonal; run `optimise` with that Hamiltonian; the success is the probability of *any* optimal schedule (use `success_of` with the list of their state numbers); return the dictionary with the five keys. Call it twice and write the explanation.""",
    r"""```python
def what_if(p=2, **changes):
    qp2 = ______                # the changed problem
    new_x, new_value, new_table = ______   # classical answer
    optima = ______             # every schedule that ties for the best
    op2, off2 = ______
    diag2 = ______
    circuit2, theta2, _, _, _ = ______     # QAOA on the new Hamiltonian
    return {}                   # x, value, how_many_optima, success, guessing
new = ______                    # the start stocks swapped
heavier_sailing = ______        # sailing price 6
explanation = "..."
```""")



S3_IMPORTS = """from scipy.optimize import minimize
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import Statevector
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qff.bits import to_x, from_x, x_to_index

algorithm_globals.random_seed = SEED                    # before any qiskit_algorithms / qiskit_optimization solver"""

T1 = """def stock_levels(x, start=START, cargo=CARGO):
    out = {}
    for k, name in enumerate("AB"):
        d = x[3 * k:3 * k + 3]
        out[name] = [start[name] - USE[name] * t + cargo * sum(d[:t]) for t in (1, 2, 3)]
    return out


def cost_from_levels(x):
    \"\"\"Tank part of the cost only: sum of (level - target)^2 over both terminals and three periods.\"\"\"
    levels = stock_levels(x)
    return sum((lv - TARGET[n]) ** 2 for n in "AB" for lv in levels[n])


aaa, bab = [1, 1, 1, 0, 0, 0], [0, 1, 0, 1, 0, 1]
print("A, A, A: levels", stock_levels(aaa), "| tank cost", cost_from_levels(aaa), "| full cost", schedule_cost(aaa))
print("B, A, B: levels", stock_levels(bab), "| tank cost", cost_from_levels(bab), "| full cost", schedule_cost(bab))"""
T1_STARTER = """# Goal: stock_levels(x) returns {"A": [level after period 1, 2, 3], "B": [...]} (equation 2);
# cost_from_levels(x) returns the tank part of the cost (first sum of equation 3); x is the whole six-number schedule, not the levels.
# Print the level tables and the costs of A,A,A = [1,1,1,0,0,0] and B,A,B = [0,1,0,1,0,1] (work A,A,A out on paper first).
raise NotImplementedError("T1: write stock_levels and cost_from_levels; compare A,A,A with B,A,B")"""
T1_CHECK = """assert stock_levels([1, 1, 1, 0, 0, 0]) == {"A": [3, 4, 5], "B": [0, -1, -2]}, "T1: for A,A,A terminal A fills up 3, 4, 5 and B runs down 0, -1, -2"
assert stock_levels([0, 1, 0, 1, 0, 1]) == {"A": [1, 2, 1], "B": [2, 1, 2]}, "T1: for B,A,B the levels should stay close to the target 2"
assert cost_from_levels([1, 1, 1, 0, 0, 0]) == 43 == schedule_cost([1, 1, 1, 0, 0, 0]), "T1: A,A,A costs 43 (no sailing, no clash)"
assert cost_from_levels([0, 1, 0, 1, 0, 1]) == 3 and schedule_cost([0, 1, 0, 1, 0, 1]) == 5, "T1: B,A,B: tank cost 3, plus 2 for the two switches = 5" """

S3_ADD_SQUARE = """# Cell 1: a helper that turns "(c + sum a_i x_i)^2" into QUBO coefficients (provided; read it, then use it in T2).
def new_coefficients():
    \"\"\"Empty cost: a constant, a dict of linear terms, a dict of quadratic terms.\"\"\"
    return {"const": 0.0, "linear": {}, "quadratic": {}}


def add_pair(coef, i, j, weight):
    \"\"\"Add weight * x_i * x_j (for i == j this is weight * x_i, because x*x = x for a 0/1 variable).\"\"\"
    if i == j:
        coef["linear"][i] = coef["linear"].get(i, 0) + weight
    else:
        key = tuple(sorted((i, j)))
        coef["quadratic"][key] = coef["quadratic"].get(key, 0) + weight


def add_square(coef, c, terms):
    \"\"\"Add (c + sum of a * x_name for (name, a) in terms) squared to the coefficients.\"\"\"
    coef["const"] += c * c
    for name, a in terms:
        coef["linear"][name] = coef["linear"].get(name, 0) + 2 * c * a
    for name_i, a in terms:
        for name_j, b in terms:
            add_pair(coef, name_i, name_j, a * b)


demo = new_coefficients()
add_square(demo, -1, [("x_A1", 2), ("x_A2", 2)])         # (-1 + 2 x_A1 + 2 x_A2)^2
print(demo)"""

S3_SUCCESS = """# Cell 2: how likely is a right answer? (the same helper as notebook 3, with the good answers passed in)
GOOD = [x_to_index(x) for x in classical_optima]         # basis-state numbers of the best schedule(s)
RANDOM_GUESS = len(GOOD) / 2 ** QUBITS


def success_of(circuit, theta, good=GOOD):
    \"\"\"Exact probability that `circuit` (knobs set to theta) ends in one of the optimal answers.\"\"\"
    probs = Statevector(circuit.assign_parameters(list(theta))).probabilities()
    return float(sum(probs[k] for k in good))            # equation (11): P_opt = sum of p(x) over the optimal x


print(f"random guessing: {RANDOM_GUESS:.3f} | basis-state number of the best schedule: {GOOD} "
      f"(count string {from_x(classical_x)}, which reads {classical_x} once reversed by to_x)")"""

T2 = """def build_qp(start=START, cargo=CARGO, sail=SAIL, clash=CLASH):
    coef = new_coefficients()
    for name in "AB":
        for t in (1, 2, 3):
            deliveries = [(f"x_{name}{u}", cargo) for u in range(1, t + 1)]
            add_square(coef, start[name] - USE[name] * t - TARGET[name], deliveries)
    for t in (1, 2):
        add_pair(coef, f"x_A{t}", f"x_B{t + 1}", sail)
        add_pair(coef, f"x_B{t}", f"x_A{t + 1}", sail)
    for t in (1, 2, 3):
        add_pair(coef, f"x_A{t}", f"x_B{t}", clash)
    qp = QuadraticProgram("lng")
    for n in NAMES:
        qp.binary_var(n)
    qp.minimize(constant=coef["const"], linear=coef["linear"], quadratic=coef["quadratic"])
    return qp


qp = build_qp()
print(qp.prettyprint())
cost_op, offset = qp.to_ising()
diag = qff.classical.diag_of(cost_op, offset)
print(f"qubits: {cost_op.num_qubits} | Pauli terms: {len(cost_op)} | constant offset: {offset}")
print(f"lowest cost {diag.min()} | average cost {diag.mean()} | best state number {int(np.argmin(diag))}")

ansatz = qaoa_ansatz(cost_op, reps=1)
res1 = minimize(qff.quantum.qaoa_energy(ansatz, diag), x0=np.full(2, 0.1), method="COBYLA", options={"maxiter": 300})
theta1 = res1.x
success_p1 = success_of(ansatz, theta1)
print(f"p = 1 from the start 0.1: <C> = {res1.fun:.2f} in {res1.nfev} evaluations; success = {success_p1:.3f}")"""
T2_STARTER = """# Goal: build_qp(start=START, cargo=CARGO, sail=SAIL, clash=CLASH) returns the QuadraticProgram of equation (3) with the six binary
# variables named in NAMES. Then convert it to a Hamiltonian: qp, cost_op, offset, diag; build the p = 1 circuit `ansatz`
# and minimise the average cost from the start (0.1, 0.1). Keep res1, theta1 (its best knobs) and success_p1.
# EASIER ROUTE (default): the plumbing below is written for you. Fill the blanks marked ______ , then delete the "raise" line.
# CHALLENGE ROUTE: delete the scaffold and write everything yourself from the Goal above (the hints still apply).
def build_qp(start=START, cargo=CARGO, sail=SAIL, clash=CLASH):
    coef = new_coefficients()
    for name in "AB":
        for t in (1, 2, 3):
            deliveries = [(f"x_{name}{u}", cargo) for u in range(1, t + 1)]
            add_square(coef, ______, deliveries)    # the constant part of the square for terminal `name` after period `t`: start stock minus use so far minus target (equation 4)
    for t in (1, 2):
        add_pair(coef, f"x_A{t}", f"x_B{t + 1}", sail)
        add_pair(coef, f"x_B{t}", f"x_A{t + 1}", sail)
    for t in (1, 2, 3):
        ______    # one add_pair call for the clash term of equation (3): both terminals served in the same period t, weight `clash`
    qp = QuadraticProgram("lng")
    for n in NAMES:
        qp.binary_var(n)
    qp.minimize(constant=coef["const"], linear=coef["linear"], quadratic=coef["quadratic"])
    return qp


raise NotImplementedError("T2: fill the two blanks ______ in build_qp, then delete this line")
qp = build_qp()
print(qp.prettyprint())
cost_op, offset = qp.to_ising()
diag = qff.classical.diag_of(cost_op, offset)
print(f"qubits: {cost_op.num_qubits} | Pauli terms: {len(cost_op)} | constant offset: {offset}")
print(f"lowest cost {diag.min()} | average cost {diag.mean()} | best state number {int(np.argmin(diag))}")

ansatz = qaoa_ansatz(cost_op, reps=1)
res1 = minimize(qff.quantum.qaoa_energy(ansatz, diag), x0=np.full(2, 0.1), method="COBYLA", options={"maxiter": 300})
theta1 = res1.x
success_p1 = success_of(ansatz, theta1)
print(f"p = 1 from the start 0.1: <C> = {res1.fun:.2f} in {res1.nfev} evaluations; success = {success_p1:.3f}")"""
T2_CHECK = """assert qp.get_num_vars() == 6 and cost_op.num_qubits == 6, "T2: six binary variables, six qubits"
assert all(abs(qp.objective.evaluate(list(x)) - schedule_cost(list(x))) < 1e-9 for x in
           (tuple(index_to_x(k, 6)) for k in range(64))), "T2: your qp must give the same cost as schedule_cost for all 64 schedules"
assert abs(diag.min() - 5.0) < 1e-9 and abs(diag.mean() - 22.0) < 1e-9, "T2: lowest cost 5, average cost 22"
assert int(np.argmin(diag)) == x_to_index([0, 1, 0, 1, 0, 1]), "T2: the lowest state should be B, A, B"
assert 0.02 < success_p1 < 0.15, "T2: with one layer the success probability is small but above guessing (1.6%)" """

S3_ANSATZ = """# Cell 3: the QAOA circuit for one layer (p = 1).
print("parameters, in the order a list of knob values must follow:", list(ansatz.parameters))
try:
    display(ansatz.draw("mpl", fold=40))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(ansatz.draw("text", fold=120))"""

S3_SAMPLE = """# Cell 4: run the tuned p = 1 circuit like a real experiment (4096 shots) and look at what comes out.
qc = ansatz.assign_parameters(theta1)
qc.measure_all()
counts = qff.quantum.sv_sampler(4096).run([qc]).result()[0].data.meas.get_counts()      # count strings are little-endian
sampled = {"".join(map(str, to_x(k))): v / 4096 for k, v in counts.items()}              # x-ordered: character i = x_i
best_str = ["".join(map(str, x)) for x in classical_optima]
print("sampled frequency of the best schedule:", round(sum(sampled.get(s, 0) for s in best_str), 3), "| guessing:", round(RANDOM_GUESS, 3))
qff.plots.plot_counts(dict(sorted(sampled.items(), key=lambda kv: -kv[1])[:12]), highlight=best_str,
                      title="p = 1, 4096 shots, 12 most frequent schedules (best in red)");"""

S3_HELPER = """def optimise(cost_op, diag, p, n_random=5):
    \"\"\"QAOA at depth p: best of 1 + n_random COBYLA runs -> (ansatz, best theta, best <C>, evaluations, list of all runs).\"\"\"
    circuit = qaoa_ansatz(cost_op, reps=p)
    f = qff.quantum.qaoa_energy(circuit, diag)
    starts = [np.full(2 * p, 0.1)] + list(np.random.default_rng(1).uniform(0, 1, (n_random, 2 * p)))
    runs = [minimize(f, s, method="COBYLA", options={"maxiter": 300}) for s in starts]
    best = min(runs, key=lambda r: r.fun)
    return circuit, best.x, float(best.fun), int(sum(r.nfev for r in runs)), runs


success, energy_by_p, ansatz_by_p, theta_by_p, evals_by_p, seconds_by_p, device_by_p = {}, {}, {}, {}, {}, {}, {}
first_start, spread = {}, {}
for p in (1, 2, 3):
    t0 = time.perf_counter()
    ansatz_by_p[p], theta_by_p[p], energy_by_p[p], evals_by_p[p], runs = optimise(cost_op, diag, p)
    seconds_by_p[p] = time.perf_counter() - t0
    success[p] = success_of(ansatz_by_p[p], theta_by_p[p])
    spread[p] = [success_of(ansatz_by_p[p], r.x) for r in runs]                       # success of each of the six starts
    first_start[p] = (float(runs[0].fun), spread[p][0])                                # the start at 0.1 only
    device_by_p[p] = qff.quantum.device_preview(ansatz_by_p[p].assign_parameters(theta_by_p[p]))
    print(f"p = {p}: best of six <C> = {energy_by_p[p]:.2f}, success = {success[p]:.3f} | first start only: <C> = {first_start[p][0]:.2f}, "
          f"success = {first_start[p][1]:.3f} | depth {device_by_p[p]['depth']}, two-qubit gates {device_by_p[p]['two_qubit_gates']}")
    print("          success of the six starts:", [round(s, 3) for s in spread[p]])"""

S3_DEPTH_MD = """Read the numbers carefully. From p = 1 to p = 2 the best-of-six success rises (about 8% to 15%). **From p = 2 to p = 3 it does not rise** (about 12%). A deeper circuit can do at least as well as a shallower one in principle, so the shortfall is the optimiser's fault, not the circuit's: with 2p = 6 knobs on a rough landscape, six starts and 300 evaluations each are not enough to find the best settings. Look at the spread of the six starts: it is wide. Keep this in mind when you present: more layers are not automatically better in practice."""

S3_SHORTCUT_MD = """**The shortcut Qiskit offers.** `MinimumEigenOptimizer` wraps everything above: it takes the `QuadraticProgram`, builds the QAOA circuit, runs the optimiser and returns an answer. Two layers, a single start. Read the probabilities from `min_eigen_solver_result.eigenstate`, not from `result.samples[...].probability`, which does not add up to 1 with this version of Qiskit's primitives."""

S3_SHORTCUT = """# Cell 5: the high-level route, under 10 seconds.
t0 = time.perf_counter()
qaoa = QAOA(sampler=qff.quantum.sv_sampler(4096), optimizer=COBYLA(maxiter=200), reps=2, initial_point=[0.1] * 4)
shortcut_result = MinimumEigenOptimizer(qaoa).solve(qp)
shortcut_seconds = time.perf_counter() - t0
eigenstate = shortcut_result.min_eigen_solver_result.eigenstate                          # {little-endian string: probability}
shortcut_success = sum(p for k, p in eigenstate.items() if tuple(to_x(k)) in classical_optima)
print("answer x:", [int(v) for v in shortcut_result.x], "| cost:", shortcut_result.fval, "| serves:", who([int(v) for v in shortcut_result.x]))
print(f"probability mass on the best schedule: {shortcut_success:.3f} (random guessing: {RANDOM_GUESS:.3f}) | {shortcut_seconds:.1f} s")"""

T3 = """def what_if(p=2, **changes):
    \"\"\"Change the instance (start=, cargo=, sail=), find the new best schedule by brute force, run QAOA at depth p.\"\"\"
    qp2 = build_qp(**changes)
    new_x, new_value, new_table = qff.classical.brute_force(qp2)
    optima = [r[0] for r in new_table if r[1] == new_value]
    op2, off2 = qp2.to_ising()
    diag2 = qff.classical.diag_of(op2, off2)
    circuit2, theta2, _, _, _ = optimise(op2, diag2, p)
    return {"x": new_x, "value": new_value, "how_many_optima": len(optima),
            "success": success_of(circuit2, theta2, [x_to_index(o) for o in optima]), "guessing": len(optima) / 64}


new = what_if(start={"A": 1, "B": 2})                       # swap the start stocks
heavier_sailing = what_if(sail=6)                          # make switching terminals expensive
for label, r in (("start stocks (1, 2)", new), ("sailing cost 6", heavier_sailing)):
    print(f"{label}: best {who(r['x'])}, cost {r['value']:.0f}, {r['how_many_optima']} optimal schedule(s) | QAOA p = 2 success {r['success']:.3f} "
          f"(guessing {r['guessing']:.3f})")
explanation = ("With the start stocks swapped the picture is mirrored: the best schedule becomes A, B, A at the same cost 5, and QAOA does "
               "about as well as before. With sailing cost 6 the cheap alternating schedule no longer wins: the best cost rises to 11 and two "
               "schedules tie (one of them even leaves a period empty), so success is the sum over both.")
print(explanation)"""
T3_STARTER = """# Goal: what_if(p=2, **changes) rebuilds the problem with changed inputs (start=, cargo=, sail=), finds its best schedule(s) classically,
# runs QAOA at depth p, and returns a dict with keys "x", "value", "how_many_optima", "success" (probability of ANY optimal schedule),
# "guessing" (= how_many_optima / 64). Keep new = what_if(start={"A": 1, "B": 2}) and heavier_sailing = what_if(sail=6).
# Then write explanation = "...": what changed in the best schedule, and what happened to QAOA's success?
raise NotImplementedError("T3: change the start stocks or the sailing cost, report the new optimum and QAOA's success")"""
T3_CHECK = """assert new["x"] == [1, 0, 1, 0, 1, 0] and new["value"] == 5, "T3: with start stocks (1, 2) the mirror image A, B, A is best, cost 5"
assert new["success"] > 2 * new["guessing"], "T3: QAOA should still beat random guessing by a clear margin"
assert heavier_sailing["value"] == 11 and heavier_sailing["how_many_optima"] == 2, "T3: sailing cost 6 gives best cost 11 with two optimal schedules"
assert isinstance(explanation, str) and len(explanation) > 60, "T3: write your explanation in a sentence or two" """

S3_IBM_AFTER = callout("real", """The preview above is the p = 3 circuit on a pretend device: the depth and two-qubit gate counts are for a toy with six qubits. A circuit this deep would probably not come out clean on today's real hardware: each two-qubit gate fails a small fraction of the time, the failures add up layer after layer, and they blur the peak that the layers were meant to sharpen. That matters for our claim: our 8% to 15% figures are *ideal-simulator* numbers, not a promise about a real device.""")

S4_TABLE = """# Results table: classical baseline rows first.
rows = [("brute force (64 schedules)", who(classical_x), classical_value, "always", brute_seconds, "64 cost evaluations"),
        ("greedy dispatcher", who(greedy_x), greedy_value, "always", greedy_seconds, "3 decisions")]
for p in (1, 2, 3):
    probs = Statevector(ansatz_by_p[p].assign_parameters(theta_by_p[p])).probabilities()
    x = index_to_x(int(np.argmax(probs)), QUBITS)
    rows.append((f"QAOA p = {p} (best of 6 starts)", who(x), schedule_cost(x), f"{success[p]:.3f}", seconds_by_p[p],
                 f"{evals_by_p[p]} circuit evaluations"))
xs = [int(v) for v in shortcut_result.x]
rows.append(("QAOA p = 2 via MinimumEigenOptimizer", who(xs), schedule_cost(xs), f"{shortcut_success:.3f}", shortcut_seconds,
             "about 200 evaluations, 1 start"))
print(f"{'method':<40}{'serves (periods 1-3)':<24}{'cost':<6}{'P(best)':<9}{'seconds':<9}{'work'}")
for name, serves, cost, prob, secs, work in rows:
    print(f"{name:<40}{serves:<24}{cost:<6}{prob:<9}{secs:<9.3f}{work}")
print("(for QAOA rows the answer is the single most likely schedule; its cost may be higher than 5)")"""

S4_PLOT = """fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
labels = ["guessing", "p = 1", "p = 2", "p = 3"]
values = [RANDOM_GUESS, success[1], success[2], success[3]]
axes[0].bar(labels, values, color=["#8D8D8D", "#0F62FE", "#0F62FE", "#0F62FE"])
for i, v in enumerate(values):
    axes[0].text(i, v + 0.005, f"{v:.1%}", ha="center")
axes[0].set_ylabel("probability of the best schedule (exact simulation)")
plot_stocks(classical_x, f"the best schedule ({who(classical_x)}): cost {classical_value}", ax=axes[1])
plt.tight_layout()"""

S4_SHOWS = """**What this toy shows.** Tank levels, sailing and the one-ship rule fit in one quadratic cost, and QAOA raises the probability of the best schedule from 1.6% (guessing) to about 8% at one layer and 15% at two (best of six starts). Three layers did *not* beat two (12%): the optimiser, not the circuit, is the limit here."""

S4_NOTSHOW = callout("watch", """**What this toy does not show.** That a quantum computer is useful. About one run in eight finds the best schedule, which is still a wrong answer most of the time, on a problem a spreadsheet solves instantly. Real maritime inventory routing has many ships, ports and weeks, and is a linear mixed-integer program where classical solvers are very strong; a 2025 benchmark of a quantum-annealing hybrid on 30 such instances lost to CPLEX on a laptop on all 30. QAOA was also simulated exactly with no noise.""")

S4_BEFORE_AFTER = '''# Before and after: the most likely schedule at p = 1 and at p = 3, next to the true optimum.
x_p1, x_p3 = (index_to_x(int(np.argmax(Statevector(ansatz_by_p[p].assign_parameters(theta_by_p[p])).probabilities())), QUBITS) for p in (1, 3))
fig, axes = plt.subplots(1, 3, figsize=(12, 2.7))
plot_stocks(x_p1, f"p = 1 most likely: {who(x_p1)} (cost {schedule_cost(x_p1)})", ax=axes[0])
plot_stocks(x_p3, f"p = 3 most likely: {who(x_p3)} (cost {schedule_cost(x_p3)})", ax=axes[1])
plot_stocks(classical_x, f"brute force optimum: {who(classical_x)} (cost {classical_value})", ax=axes[2])
plt.tight_layout()'''

S4_PAPER = """**Paper versus our route.** WEF/Aramco Case 4 (p. 15) describes ExxonMobil and IBM Research formulating LNG maritime inventory routing as vehicle routing with time windows in QUBO form, at the "R&D/prototype formulation and benchmarking" stage, "not yet at operational scale". We use the same family (QUBO plus QAOA) on a hand-built miniature; Qiskit's `Tsp` class appears only to count qubits. Our results illustrate the *problem* and the *variational idea*; they are not a reproduction of the paper's work."""

ACC = """assert QUBITS == 6 and cost_op.num_qubits == 6
assert classical_x == [0, 1, 0, 1, 0, 1] and classical_value == 5
assert tsp_vars == {3: 9, 4: 16}
assert abs(diag.mean() - 22.0) < 1e-9 and abs(diag.min() - 5.0) < 1e-9
assert 0.05 < success[1] < 0.12 and success[2] > 0.10
print("p = 1, 2, 3 success:", [round(success[p], 3) for p in (1, 2, 3)], "| random guessing:", round(RANDOM_GUESS, 3))"""

INSTRUCTOR = """**Instructor note (never reaches the starter).** Expected numbers (seed 2026, starts: 0.1 plus five from `default_rng(1).uniform(0, 1)`, exact simulation). Best of six: success 0.083, 0.147, 0.123 and <C> 13.44, 11.40, 11.74 for p = 1, 2, 3. First start only: <C> 18.14, 13.42, 11.74 and success 0.032, 0.090, 0.123. Random guessing 1/64 = 0.016. p = 3 does not beat p = 2: keep that in the story (optimiser limit). What-ifs: start (1, 2) mirrors the problem (best A, B, A, cost 5, success about 0.146 at p = 2); cargo 3 changes the optimum to cost 5 at x = 010100 and makes the landscape harder; sailing 6 gives best cost 11, two optima, success 0.118. Shortcut (reps 2, one start at 0.1) returns the optimum but puts only about 0.05 of probability on it. Pitfalls: the optimum is state number 42 (count string `101010`), which is x = `010101`; use `qff.bits`. Do not describe this as the ExxonMobil-IBM model. Report best-of-six and show the spread. This notebook reuses notebook 3's `success_of`, `optimise` and T1-T3 shape."""


S3_MATH_ONE = '<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**\n\n**The one line to remember.** Rules become one cost, $C(x)=\\sum(I_p(t)-T_p)^2+\\sigma(\\ldots)+\\kappa(\\ldots)$ (equation 3), which becomes the Hamiltonian $H_C$ (equation 5); QAOA then minimises $\\langle C\\rangle=\\sum_xp(x)C(x)$ (equation 10) over the knobs $(\\gamma,\\beta)$.\n\n</div>'


def main():
    """Write the solution notebook."""
    nb = new(4, "LNG maritime inventory routing",
             use_case="Decide which customer terminal an LNG ship serves in each period so that no tank strays from its target (maritime inventory routing)",
             algorithm="QAOA (Quantum Approximate Optimization Algorithm)", qubits=6, minutes=3)
    nb.section(1)
    nb.md(S1)
    nb.section(2)
    nb.md(S2_MD)
    nb.code(S2_INSTANCE)
    nb.md(S2_CL_MD)
    nb.code(S2_CLASSICAL)
    nb.md(S2_NOTE)
    nb.md(S2_TSP_MD)
    nb.code(S2_TSP)
    nb.md(S2_TSP_NOTE)
    nb.section(3)
    nb.md(S3_ROADMAP)
    nb.code(S3_FLOW)
    nb.md(S3_SYMBOLS)
    nb.md(S3_MATH_ONE)
    nb.md(S3_STATE_MD)
    nb.code(S3_STATE_CODE)
    nb.md(S3_STOCK_MD)
    nb.code(S3_STOCK_CODE)
    nb.md(S3_STOCK_TRY)
    nb.code(S3_IMPORTS)
    nb.md(TOOLBOX[4][0])
    nb.code(TOOLBOX[4][1])
    nb.md(T1_INTRO)
    nb.task("T1", "work out the tank levels by hand, write stock_levels, compare A,A,A with B,A,B", T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.md(S3_QUBO_MD)
    nb.code(S3_ADD_SQUARE)
    nb.code(S3_DEMO_CHECK)
    nb.md(S3_ISING_MD)
    nb.code(S3_ISING_CODE)
    nb.md(S3_CIRCUIT_MD)
    nb.md(S3_MINI_MD)
    nb.code(S3_MINI)
    nb.md(S3_SUCCESS_MD)
    nb.code(S3_SUCCESS)
    nb.md(T2_INTRO)
    nb.task("T2", "build the QuadraticProgram from the stock formula, convert it, run QAOA at p = 1", T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_CHECK_MD)
    nb.code(S3_CHECK)
    nb.md(S3_ANSATZ_MD)
    nb.code(S3_ANSATZ)
    nb.md(S3_LAND_MD)
    nb.code(S3_LAND)
    nb.md(S3_SAMPLE_MD)
    nb.code(S3_SAMPLE)
    nb.md(S3_SAMPLE_TRY)
    nb.md(S3_LOOP_MD)
    nb.code(S3_LOOP_FIG)
    nb.md(S3_LOOP_ARABIC)
    nb.code(S3_HELPER)
    nb.md(S3_HIST_MD)
    nb.code(S3_HIST)
    nb.md(S3_DEPTH_MD)
    nb.md(S3_WHY_MD)
    nb.md(S3_SHORTCUT_MD)
    nb.code(S3_SHORTCUT)
    nb.md(T3_INTRO)
    nb.task("T3", "stretch: change the start stocks or the sailing cost, report the new optimum and how QAOA's success moves", T3,
            starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(INSTRUCTOR, tags=["solution-only"])
    nb.code("circuit = ansatz_by_p[3].assign_parameters(theta_by_p[3])        # the main circuit of this notebook: p = 3, tuned")
    nb.md(day.IBM_MD)
    nb.code(day.IBM_PREVIEW_CODE)
    nb.md(S3_IBM_AFTER)
    nb.md(day.glossary_md([("QAOA", "خوارزمية الأمثلة الكمومية التقريبية (QAOA)", "G"), "cost function", "optimization",
                           ("Hamiltonian", "مؤثر هاملتون (الهاملتوني)", "G"), ("ansatz", "الافتراض المبدئي (ansatz)", "G"),
                           ("interference", "التداخل", "G"), "expectation value",
                           ("routing", "تحديد المسارات", "I"), ("inventory", "المخزون", "I"),
                           ("QUBO", "مسألة تحسين ثنائية تربيعية غير مقيدة (QUBO)", "I")]), tags=["glossary"])
    nb.section(4)
    nb.code(S4_TABLE)
    nb.code(S4_PLOT)
    nb.code(S4_BEFORE_AFTER)
    nb.md(S4_SHOWS)
    nb.md(S4_NOTSHOW)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="Decide where an LNG ship goes each period so that no customer tank runs dry or overflows, without wasting sailing time.",
                counting_fact="6 yes/no decisions give 64 schedules to check; a fleet over a year has more possible schedules than there are atoms "
                              "in the universe. Smart classical methods do not check them all, so the honest question is where they stop scaling.",
                single_idea="One layer rewards cheap schedules with a phase twist, the next layer mixes the schedules; repeat, and the cheap "
                            "answers interfere constructively.",
                ideas=["A tank-level animation for any schedule the visitor types in",
                       "A captain's-log story that follows the ship through the three periods",
                       "A \"beat the quantum sample\" game: the visitor picks a schedule and competes with a QAOA sample"],
                reuse="the stock-level plot of the best schedule (`plot_stocks`), the success table for p = 1, 2, 3 and the spread of the six starts")
    nb.section6(honest="QAOA found the best schedule in roughly one shot in eight; the classical solver is exact and instant.",
                question_a="Why did three layers not beat two?",
                question_b="What would change with two ships?")
    path = nb.save(os.path.join(ROOT, "solutions", "04_move_lng_routing_qaoa_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
