"""Build solutions/05_move_unit_commitment_grover_solution.ipynb (mentors and builders only).

    python tools/build_nb05.py
    python tools/make_starters.py 5
    python tools/run_tests.py 5

Notebook 5 sets the Grover adaptive search (GAS) pattern that notebook 9 follows: instance function -> classical baseline ->
Grover by hand (to see one step) -> the adaptive loop in words -> GroverOptimizer behind one helper (`solve_gas`) ->
reliability over ten seeds -> value-register experiment.  Keep the cell order if you copy it.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, arabic_table, callout, details, hints, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1 = """**The problem.** A power system has to meet demand every hour, and the operator decides which power stations to switch on. Switching a unit on costs money, and a unit that is on must run. Too little capacity and the lights go out; too much and money is wasted. This is *unit commitment*: for each unit and each hour, on or off. Four units means 2⁴ = 16 plans for one hour; 26 units over 24 hours means 2⁶²⁴ plans.

**Who faces it.** A national grid operator, or a utility that owns a fleet of generating units, solves this every day for the next day.

""" + callout("real", """**What is real today.** WEF/Aramco *Quantum for Energy and Utilities* (April 2026), Case 13 (p. 27): IonQ and Oak Ridge National Laboratory, under the US DOE GRID-Q effort, ran a hybrid formulation for 26 generators over 24 hours on a 36-qubit trapped-ion system, using a layered variational method. The classical incumbent is mixed-integer programming in commercial solvers, which handles real systems with ramp limits, minimum up-times and reserves every day. (The paper's 76% survey figure belongs to "large-scale optimization of high-voltage grid flows", not to unit commitment.)""") + """

**Why a quantum computer might help.** Grover's algorithm finds a marked item among N in about √N steps instead of N, and wrapping it in a loop that keeps lowering a cost threshold turns "search" into "optimise". What is still missing: the quadratic speed-up only pays off on error-corrected hardware, and the circuit must compute the cost inside itself.

| Paper's method | Our teaching route |
|---|---|
| Layered variational method on a 36-qubit trapped-ion system, 26 generators over 24 hours (Case 13) | Grover adaptive search with Qiskit's `GroverOptimizer` on 4 generators for one hour. It builds directly on the Grover lecture and has a ready Qiskit class. This is a teaching choice, not the paper's method. |"""

S2_MD = """One peak hour, four generating units, demand **D = 5** (units of 100 MW). Unit *i* has capacity cᵢ, a start cost sᵢ, and a binary switch uᵢ (1 = on).

| Unit | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Capacity cᵢ | 3 | 2 | 2 | 1 |
| Start cost sᵢ | 4 | 3 | 2 | 2 |

We minimise the start cost plus a penalty λ = 2 for every squared unit of mismatch with demand:

**cost(u) = Σ sᵢ uᵢ + λ (Σ cᵢ uᵢ − D)²**

Expanded, this is the QUBO `24 u0u1 + 24 u0u2 + 12 u0u3 + 16 u1u2 + 8 u1u3 + 8 u2u3 − 38 u0 − 29 u1 − 30 u2 − 16 u3 + 50`. The demand penalty is written by hand so you can see it. (The numbers are invented for teaching.)

Qubit budget: **4 decision qubits** (one per unit) **+ 7 value qubits** (the circuit has to hold the cost inside itself) = **11 qubits**. (Qiskit adds one more flag qubit inside the oracle, so the circuit it runs shows 12; we count 4 + 7.)"""

S2_INSTANCE = """QUBITS = 11                      # [KEEP] the test runner reads this
DECISION_QUBITS = 4
VALUE_QUBITS = 7
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qff.bits import index_to_x

def plan(x):
    \"\"\"Unit switches as a string in unit order: [1, 0, 1, 0] -> '1010' (u0 first).\"\"\"
    return "".join(str(int(v)) for v in x)

CAPACITY = [3, 2, 2, 1]
START_COST = [4, 3, 2, 2]
LAMBDA = 2

def unit_commitment_qp(demand=5):
    \"\"\"cost(u) = sum s_i u_i + LAMBDA * (sum c_i u_i - demand)^2, expanded by hand into linear and quadratic terms.\"\"\"
    qp = QuadraticProgram("unit_commitment")
    for i in range(4):
        qp.binary_var(f"u{i}")
    linear = {f"u{i}": START_COST[i] + LAMBDA * (CAPACITY[i] ** 2 - 2 * demand * CAPACITY[i]) for i in range(4)}
    quadratic = {(f"u{i}", f"u{j}"): 2 * LAMBDA * CAPACITY[i] * CAPACITY[j] for i in range(4) for j in range(i + 1, 4)}
    qp.minimize(constant=LAMBDA * demand ** 2, linear=linear, quadratic=quadratic)
    return qp

DEMAND = 5
qp = unit_commitment_qp(DEMAND)
print(qp.prettyprint())"""

S2_CLASSICAL = """# Classical baseline: always runs first.
import time
t0 = time.perf_counter()
classical_x, classical_value, table = qff.classical.brute_force(qp)
classical_seconds = time.perf_counter() - t0
print(f"brute force over {2 ** DECISION_QUBITS} states: best u = {plan(classical_x)}, cost {classical_value:g}")
print("cheapest five:", [(plan(x), v) for x, v, _ in table[:5]], "| worst:", (plan(table[-1][0]), table[-1][1]))

# A second baseline, "merit order": switch units on from cheapest start cost per MW until demand is met.
order = sorted(range(4), key=lambda i: START_COST[i] / CAPACITY[i])
merit_x, supplied = [0, 0, 0, 0], 0
for i in order:
    if supplied >= DEMAND:
        break
    merit_x[i], supplied = 1, supplied + CAPACITY[i]
merit_value = qp.objective.evaluate(merit_x)
print(f"merit order {order}: u = {plan(merit_x)}, supplies {supplied}, cost {merit_value:g}")"""

S2_NOTE = """Merit order happens to find the optimum on this tiny instance. It is a rule of thumb, not a guarantee: it ignores how overshoot and start costs trade off, and it fails on harder instances. The point of the baseline is that **16 states are checked in microseconds**."""

S3_ROADMAP = r"""### 3.0 · The map of this section

""" + callout("idea", r"""**Grover adaptive search in one sentence.** Give every plan an amplitude, mark the plans that are cheaper than the best one found so far by flipping the sign of their amplitude, let a second trick (inversion about the average) turn those sign flips into *larger* amplitudes for the marked plans, measure, and repeat with a lower bar until nothing cheaper exists.""") + r"""

There is **no knob-tuning** here (unlike QAOA or VQE). The circuit is fixed; what is cleverly arranged is how amplitudes move. We build it from two things you already know: lists of numbers (vectors) and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | The cost of each plan, and a state with equal amplitude on all 16 plans | (1), (2) | `cost`, `by_hand` | **T1** |
| 3.2 | The oracle: flip the sign of every plan cheaper than the bar | (3) | `DiagonalGate` | |
| 3.3 | Inversion about the average, and one full Grover step | (4), (5) | `grover_operator` | |
| 3.4 | A rotation picture: how many steps are best? | (6), (7) | `p_marked_curve` | **T2** |
| 3.5 | The adaptive loop, and Qiskit's `GroverOptimizer` | (7) | `solve_gas` | |
| 3.6 | The value register: why the circuit needs 7 extra qubits | (8) | `reliability` | **T3** |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it."""

S3_FLOW = """qff.plots.plot_flow(
    [("Pick a bar y", "cost of a random plan", "source"), ("Oracle", "mark plans cheaper than y", "move"), ("Grover steps", "random number", "store"),
     ("Measure", "one plan", "use"), ("Cheaper?", "then lower y", "source")],
    loop=(4, 1, "repeat; stop after several failures in a row"), title="Grover adaptive search");"""

S3_COSTSTATE_MD = r"""### 3.1 · The cost of every plan, and a state to search in

A **plan** is a list of four on/off switches $u=(u_0,u_1,u_2,u_3)$, one per unit, so there are $2^4=16$ plans. The cost of a plan is the one from section 2: the start costs of the units that are on, plus a penalty $\lambda=2$ for every squared unit of mismatch with the demand $D=5$:

$$\text{cost}(u)=\sum_i s_i\,u_i+\lambda\Bigl(\sum_i c_i\,u_i-D\Bigr)^2 \tag{1}$$

For example $u=(1,0,1,0)$: units 0 and 2 are on, start cost $4+2=6$, supply $3+2=5=D$, so no penalty: cost $6$. For $u=(0,0,0,0)$: nothing on, supply 0, penalty $2\cdot(0-5)^2=50$: cost $50$. Task T1 fills this table for all 16 plans.

**The state.** Four qubits hold a list of $2^4=16$ numbers called **amplitudes** $a_x$, one per plan; $|a_x|^2$ is the **probability** of reading out plan $x$ when you **measure**, and the probabilities add up to 1. The **Hadamard gate** $H$ puts one qubit into an equal mixture of $|0\rangle$ and $|1\rangle$; one $H$ on each of the four qubits (written $H^{\otimes4}$) gives every plan the same amplitude:

$$|s\rangle=H^{\otimes4}|0000\rangle=\frac1{\sqrt{16}}\sum_x|x\rangle,\qquad a_x=\frac14\ \text{for every plan},\qquad \sum_x a_x^2=16\cdot\frac1{16}=1 \tag{2}$$

($|x\rangle$ is just a *name tag* for plan $x$: a column of 16 numbers with a single 1 at position $x$.) This is pure guessing: each plan has probability $\frac1{16}$. Position $k$ of the list is the plan whose switch $u_i$ is bit $i$ of $k$ (so $k=5=0101_2$ is $u_0=1,u_2=1$, the plan `1010`).

""" + callout("watch", r"""**Amplitudes can be negative.** Probabilities cannot, but amplitudes can: only their *squares* are probabilities. A minus sign is invisible to a measurement of that state, but it changes what later gates do. The whole trick of Grover's algorithm is built on a minus sign.""") + "\n\n" + arabic_table([
    ("Unit commitment", "التزام الوحدات (جدولة تشغيل المولدات)", "I"), ("Generating unit", "وحدة توليد", "I"), ("Plan", "الخطة", "I"),
    ("Amplitude", "السعة", "G"), ("Probability", "الاحتمال", "I"), ("Measurement", "القياس", "G"), ("Hadamard gate", "بوابة هادامارد", "G"),
    ("Cost function", "دالة التكلفة", "I"), ("Superposition", "التراكب", "G")])

T1_INTRO = r"""### T1 · Warm-up: the cost table · التمرين ١

**What you are about to build, and why.** Grover's oracle (next) will mark *every plan cheaper than a bar*. To understand it, you need the cost of all 16 plans in front of you. In T1 you fill the list `cost` (one integer per position $k=0..15$), and the list of plans whose cost is below 8 (the plans the oracle would mark at the bar $y=8$).

**The formula you implement:** equation (1) for every plan. `qp.objective.evaluate(x)` computes it for you given the switches $x$; your job is to produce the 16 switch lists from the positions $k$ (bit $i$ of $k$ is unit $i$) and collect the results.

""" + hints(
    r"""You need to visit the 16 positions k = 0, ..., 15. For each one you need the four switches as a list, with bit i of k as unit i: the helper `index_to_x(k, 4)`, imported in section 2, does exactly that (it returns the list $[x_0,x_1,x_2,x_3]$ for position `k`). Then ask the problem for the cost of that list.""",
    r"""A list comprehension over the 16 positions builds `cost`: turn each position into its switch list, evaluate the objective of the problem on it, and round to an integer. For the marked plans, go through the positions again and keep the plan string of every position whose cost is below the bar.""",
    r"""```python
cost = ______              # 16 integers: the cost of plan k, for k = 0..15
marked_thr8 = ______       # the plan strings whose cost is below 8
```
Test yourself: `cost[5]` (the plan `1010`) is 6, and four plans are cheaper than 8.""")

S3_ORACLE_MD = r"""### 3.2 · The oracle: flip the sign of every cheap plan

Pick a **bar** $y$. A plan is **marked** if its cost is below $y$. The **oracle** $O$ is a rule that flips the sign of the amplitude of every marked plan and leaves the others alone:

$$O\,|x\rangle=\begin{cases}-|x\rangle&\text{if }\text{cost}(x)<y\ \ (\text{marked})\\+|x\rangle&\text{otherwise}\end{cases} \tag{3}$$

As a matrix, $O$ is *diagonal*: a $16\times16$ table with $\pm1$ on its diagonal and zeros elsewhere. In Qiskit that is a `DiagonalGate` built from the list of $\pm1$ values. A sign flip changes **no probability** ($(-a)^2=a^2$), so right after the oracle a measurement still sees every plan equally likely. The information "which plans are cheap" is now stored in the *signs* where a measurement cannot see it, and the next step turns it into amplitude.

**Careful.** This oracle looks at the cost table directly, which a real algorithm cannot do (the real oracle computes costs *inside* the circuit, section 3.6). Here it is for intuition only.

""" + arabic_table([("Oracle", "العرّاف (oracle)", "I"), ("Marked plan", "الخطة الموسومة", "I"), ("Phase / sign flip", "قلب الإشارة (الطور)", "I"),
                    ("Diagonal matrix", "مصفوفة قطرية", "I"), ("Threshold (the bar)", "العتبة", "I")])

S3_REFLECT_MD = r"""### 3.3 · Inversion about the average

The second ingredient is a **reflection about the average**. Let $\bar a$ be the average of the 16 amplitudes. The step replaces every amplitude by

$$a_x\ \longmapsto\ 2\bar a-a_x \tag{4}$$

In words: *mirror each amplitude around the average.* An amplitude that is *far below* the average (a marked plan, after its sign flip) lands *far above* it; an amplitude close to the average barely moves. (As a matrix this is $D=2|s\rangle\langle s|-I$: twice the projection onto the equal state, minus the identity.) One full **Grover step** is "oracle, then inversion about the average":

$$G=D\,O \tag{5}$$

**A worked example you can check on paper: 4 plans, 1 of them marked.** Amplitudes start at $\frac12$ each. The oracle flips the marked one: $(\frac12,\frac12,\frac12,-\frac12)$. The average is $\bar a=\frac14(\frac12+\frac12+\frac12-\frac12)=\frac14$. Inversion (4): the three unmarked become $2\cdot\frac14-\frac12=0$ and the marked one becomes $2\cdot\frac14+\frac12=1$. Result $(0,0,0,1)$: **one step finds the marked plan with certainty.** (Check: $0^2+0^2+0^2+1^2=1$ ✓.) The cell below does this arithmetic with numpy, then the same for our 16 plans.

""" + callout("idea", r"""**Why this amplifies.** The sign flip pulls the marked amplitudes *below* the average; the mirror pushes them *above* it by about the same distance, while the average itself moves only a little. Net effect: marked plans grow, unmarked plans shrink. Repeat, and the probability of a marked plan keeps growing, up to a point (next section).""") + "\n\n" + arabic_table([
    ("Grover's algorithm", "خوارزمية جروفر", "G"), ("Amplitude amplification", "تضخيم السعة", "I"), ("Reflection / inversion about the mean", "الانعكاس (القلب) حول المتوسط", "I"),
    ("Average (mean)", "المتوسط", "I"), ("Interference", "التداخل", "G"), ("Quantum speedup (quadratic)", "تسريع تربيعي", "U")])

S3_BYHAND = '''# Equations (2)-(5) with plain numpy: sign flip, then inversion about the average. No quantum library.
def grover_by_hand(cost, bar, steps=3):
    """Amplitudes of the 16 plans after each action. Returns (dict name -> amplitudes, list of P(marked) after 0..steps steps, marked positions)."""
    c = np.array(cost)
    marked = np.flatnonzero(c < bar)
    a = np.full(len(c), 1 / np.sqrt(len(c)))                          # equation (2): equal amplitudes
    stages, curve = {"start": a.copy()}, [float((a[marked] ** 2).sum())]
    for step in range(1, steps + 1):
        a = a.copy()
        a[marked] *= -1                                               # equation (3): the oracle flips the sign of marked plans
        stages[f"step {step}: oracle"] = a.copy()
        a = 2 * a.mean() - a                                          # equation (4): inversion about the average
        stages[f"step {step}: invert"] = a.copy()
        curve.append(float((a[marked] ** 2).sum()))
    return stages, curve, marked


# the 4-plan example from the text: one plan marked
tiny = np.full(4, 0.5)
tiny[3] *= -1
print("4 plans, 1 marked: after the oracle", tiny, "-> after inversion about the average", 2 * tiny.mean() - tiny)
assert np.allclose(2 * tiny.mean() - tiny, [0, 0, 0, 1])

stages8, by_hand_thr8, marked8 = grover_by_hand(cost, 8)
print("16 plans, bar 8 (4 marked): P(marked) after 0, 1, 2, 3 steps =", np.round(by_hand_thr8, 3))
sel = {k: v for k, v in stages8.items() if k in ("start", "step 1: oracle", "step 1: invert", "step 2: oracle", "step 2: invert")}
qff.plots.plot_amplitude_stages(sel, highlight=marked8, labels=("not marked", "marked"),
                                suptitle="bar 8: four plans are marked (magenta). The oracle flips their sign, the inversion lifts them")'''

S3_BYHAND_TRY = callout("try", """Change the bar `8` to `7` in the last cell (so only one plan, `1010`, is marked) and rerun it with `steps=3`. After the oracle the marked amplitude is slightly negative; after the inversion it is much bigger than the others. **Predict** which step gives the highest P(marked) before you look at the printed curve.""")

S3_ROT_MD = r"""### 3.4 · Why it is a rotation, and how many steps are best

Let $N=16$ be the number of plans and $M$ the number of marked ones. Group the amplitudes into two classes, marked and not marked. Initially the state is a mix with $P(\text{marked})=M/N$; write this as $\sin^2\theta=M/N$. Each Grover step $G=DO$ **rotates** the state by the angle $2\theta$ in the plane of the two classes, towards "all marked". After $k$ steps the state sits at angle $(2k+1)\theta$, so

$$P(\text{marked after }k\text{ steps})=\sin^2\bigl((2k+1)\theta\bigr),\qquad \sin\theta=\sqrt{M/N} \tag{6}$$

Two things follow. First, the probability **rises, peaks and falls again**, because rotating past "all marked" turns the state back. Second, the number of steps that gets closest to the peak is about

$$k^*\approx\frac\pi{4\theta}-\frac12\ \approx\ \frac\pi4\sqrt{\frac NM}\quad(\text{for small }M/N) \tag{7}$$

For $M=4$, $N=16$: $\sin\theta=\frac12$, so $\theta=\frac\pi6$; step 1 reaches $\sin^2\frac\pi2=1$, and step 2 gives $\sin^2\frac{5\pi}6=\frac14$ again (the state rotated *past* the target). For $M=1$: $\sin\theta=\frac14$, $\theta\approx0.253$; $k=3$ gives $\sin^2(7\theta)=0.961$ and is the best. **The best number of steps depends on $M$, which the search does not know.** That is why Grover *adaptive* search draws the number of steps at random (section 3.5). Compared with a classical search that checks about $N/2$ plans on average, about $\sqrt N$ steps is the **quadratic speed-up**: $4$ steps instead of $16$ looks small here, but $10^6$ steps instead of $10^{12}$ does not.

""" + arabic_table([("Rotation", "الدوران", "I"), ("Quadratic speed-up", "تسريع تربيعي", "U"), ("Number of steps", "عدد الخطوات", "I")])

S3_GEOM = '''# Equations (6) and (7) as a picture: each Grover step turns the state by 2*theta.
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
qff.plots.plot_grover_geometry(np.arcsin(np.sqrt(4 / 16)), steps=2, ax=axes[0])
axes[0].set_title("4 of 16 marked: theta = 30 deg; step 1 hits, step 2 overshoots", fontsize=8)
qff.plots.plot_grover_geometry(np.arcsin(np.sqrt(1 / 16)), steps=3, ax=axes[1])
axes[1].set_title("1 of 16 marked: theta = 14.5 deg; best at step 3", fontsize=8)
plt.tight_layout()'''

T2_INTRO = r"""### T2 · Core step: the oracle and the Grover step in Qiskit · التمرين ٢

**What you are about to build, and why.** The numpy cell above showed what a Grover step does to the amplitudes. In T2 you build the same thing with Qiskit gates, as a function `p_marked_curve(threshold)`: it builds the oracle (equation 3) as a `DiagonalGate`, the full step $G=DO$ (equation 5) with Qiskit's `grover_operator`, starts from $H^{\otimes4}$ (equation 2), applies $G$ zero, one, two, … times, and records $P(\text{marked})$ after each. You will then see that Qiskit's circuit gives the same numbers as your pencil-and-paper arithmetic and as equation (6).

**The formulas you implement:** (2), (3), (5), and $P(\text{marked})=\sum_{k\in\text{marked}}|a_k|^2$ (read from `Statevector(qc).probabilities()`).

""" + hints(
    r"""Break it into pieces. (1) Which positions are marked? (2) The oracle is a circuit on 4 qubits that contains one `DiagonalGate`, which takes a list of 16 numbers (which value for a marked state, which for the others?). (3) Qiskit turns an oracle circuit into the full Grover step with a ready-made function. (4) To apply the step k times, start from the equal mixture and repeat.""",
    r"""Inside `p_marked_curve`: find the marked positions; build the oracle circuit with one `DiagonalGate`; get the step with `grover_operator`; then for each number of steps from 0 to `max_steps` build a circuit (Hadamard on every qubit, then the step applied that many times with `compose`), take the probabilities of its `Statevector`, and add up the marked ones. Return the list of probabilities and the step.""",
    r"""```python
def p_marked_curve(threshold, max_steps=4):
    marked = ______            # the marked positions
    oracle = ______            # a 4-qubit circuit with the diagonal gate of equation (3)
    G = ______                 # the full Grover step
    curve = []
    for steps in range(max_steps + 1):
        qc = ______            # equal mixture, then the step applied `steps` times
        curve.append(______)   # probability of the marked positions, as a float
    return curve, G
```
Self-check targets: with the bar at 8 the curve starts at 0.25, reaches 1.0 after one step and falls back to 0.25 after two.""")

S3_T2_CHECK_MD = """**Check against pencil-and-paper arithmetic and equation (6).** The cell below compares your Qiskit curves with the numpy arithmetic of 3.3 and with $\\sin^2((2k+1)\\theta)$, for both bars, and then plots them."""

S3_T2_CHECK = '''# Check: Qiskit's curves = the numpy arithmetic = equation (6).
for bar, qiskit_curve in ((8, p_marked_thr8), (7, p_marked_thr7)):
    _, hand_curve, marked_k = grover_by_hand(cost, bar, steps=4)
    theta = np.arcsin(np.sqrt(len(marked_k) / 16))
    formula = [np.sin((2 * k + 1) * theta) ** 2 for k in range(5)]
    assert np.allclose(qiskit_curve, hand_curve) and np.allclose(qiskit_curve, formula), f"bar {bar}: the three curves should agree"
    print(f"bar {bar}: {len(marked_k)} plan(s) marked, theta = {np.degrees(theta):.1f} deg | Qiskit {np.round(qiskit_curve, 3)} | formula (6) {np.round(formula, 3)}")
print("OK: Qiskit, pencil and paper, and equation (6) all give the same numbers.")
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, curve, label, M in ((axes[0], p_marked_thr8, "bar 8: 4 plans marked", 4), (axes[1], p_marked_thr7, "bar 7: 1 plan marked", 1)):
    fine = np.linspace(0, 4, 200)
    ax.plot(fine, np.sin((2 * fine + 1) * np.arcsin(np.sqrt(M / 16))) ** 2, color="#C6C6C6", label="equation (6), any k")
    ax.plot(range(len(curve)), curve, "o", color="#0F62FE", label="Qiskit")
    ax.set_title(label); ax.set_xlabel("Grover steps"); ax.set_xticks(range(len(curve))); ax.grid(alpha=0.3); ax.legend(fontsize=8)
axes[0].set_ylabel("P(marked plan)")
plt.tight_layout(); plt.show()
print("The best number of steps depends on how many plans are marked, and the loop does not know that in advance.")
print("That is why Grover adaptive search draws the number of steps at random.")'''

S3_LOOP_MD = r"""### 3.5 · The adaptive loop, and Qiskit's `GroverOptimizer`

Grover's algorithm needs an oracle that marks the answers we want. For optimisation we do not know the answers in advance, so **Grover adaptive search** builds the oracle from a bar $y$: *mark every plan that costs less than $y$*. The loop, in words:

```
pick a random plan; let y = its cost
repeat:
    mark every plan with cost < y            (the oracle, computed *inside* the circuit)
    run Grover with a random number of steps  (we do not know how many plans are marked)
    measure one plan
    if its cost < y:  y = that cost           (the bar goes down)
    else:             count a failure
stop after several failures in a row
```

When the bar is low, few plans are marked ($M$ small) and many steps are best; when the bar is high, many plans are marked and few steps are best (equation 7). Because $M$ is unknown, the number of steps is drawn at random each round; this idea of searching with an unknown number of marked items goes back to Boyer, Brassard, Høyer and Tapp (1998), and its use for finding a minimum to Dürr and Høyer (1996). Qiskit's `GroverOptimizer` follows the same pattern for polynomial binary problems (it cites Gilliam, Woerner and Gonciulea). Each round that finds a cheaper plan lowers the bar, so the search finishes when *no plan is cheaper*: the best plan.

**The expected cost of the idea.** A classical program checks the 16 plans one by one: 16 lookups. The loop above needs *several* rounds, each of a few Grover steps, each step an oracle that computes the cost inside the circuit. On 16 plans this is **slower** than brute force; the speed-up is about $\sqrt N$ per round, and it only pays when $N$ is astronomically large and the hardware is error-corrected.

""" + arabic_table([("Grover adaptive search", "البحث التكيفي بخوارزمية جروفر", "I"), ("Threshold (the bar)", "العتبة", "I"), ("Random number of steps", "عدد عشوائي من الخطوات", "I"),
                    ("Optimization", "الأمثَلة (التحسين)", "I"), ("Value register", "سجل القيمة", "I")])

S3_GAS_MD = """**The real thing: `GroverOptimizer`.** Qiskit's `GroverOptimizer` runs the loop above. It builds the oracle *without* peeking: it computes the cost of every plan inside the circuit, in a **value register** of binary digits, subtracts the bar, and marks the plans whose result is negative. That register is why the circuit has 11 qubits, not 4. The next section explains how it works and how it fails.

Three things to know:
- `num_value_qubits` is the width of the value register. It must hold the signed range of (cost − bar). **Too few and the answers are silently wrong.**
- `num_iterations` is the number of *consecutive rounds without improvement* before it stops. It is not the number of Grover steps.
- Costs must be integers. Both seeds matter: `algorithm_globals.random_seed` and the sampler's `seed`.

We wrap it in one helper, `solve_gas`, and a tiny recording sampler so we can look at the circuits it really runs."""

S3_REG_MD = r"""### 3.6 · The value register: why 7 extra qubits, and how it fails

The oracle must decide "is $\text{cost}(x)<y$?" for all plans *at once*, inside the circuit. It does this with binary arithmetic on extra qubits, the **value register**: it computes the difference $\text{cost}(x)-y$ in the register and looks at the **sign bit** (the top qubit): 1 means negative, i.e. marked. Negative numbers are stored in **two's complement**: with $v$ bits the register can hold the whole numbers

$$-2^{\,v-1},\ \dots,\ -1,\ 0,\ 1,\ \dots,\ 2^{\,v-1}-1 \tag{8}$$

and anything outside this range **wraps around**. A tiny example, $v=3$: the range is $-4..3$, the bit patterns $100,101,110,111$ mean $-4,-3,-2,-1$, so the value $+5$ (which does not fit) is stored as $101$ and read as $-3$. A *positive* difference has turned into a *negative* one: an expensive plan looks cheap, and the oracle **marks the wrong plan**.

For our problem costs run from 6 to 50, and the bar $y$ is always one of the costs, so $\text{cost}-y$ can be as low as $6-50=-44$ and as high as $50-6=+44$. Equation (8) says 6 bits cover only $-32..31$ (**too few**); 7 bits cover $-64..63$ (enough). This is a simplified picture of what Qiskit's oracle does (its arithmetic circuit is more elaborate), but the overflow rule is the same. The cell below checks it and shows the failure: with 6 bits and a bar of 12, the *worst* plan (`0000`, cost 50, difference $+38$) wraps to $-26$ and is marked.

""" + callout("watch", r"""**Silent failure.** With too few value qubits the routine still reports success and returns a wrong plan, sometimes even `0000` at cost 50. Nothing warns you. **Always check a quantum optimiser's answer against something you can compute classically.**""") + "\n\n" + arabic_table([
    ("Value register", "سجل القيمة", "I"), ("Two's complement", "المتمم الثنائي", "I"), ("Sign bit", "بت الإشارة", "I"), ("Overflow", "فيض (تجاوز المدى)", "I"),
    ("Qubit", "كيوبت", "G"), ("Binary number", "عدد ثنائي", "I")])

S3_REG_CODE = '''# Equation (8): what a v-bit two's complement register can hold, and what overflow does to the oracle.
def wrap(value, bits):
    """The number a `bits`-bit two's complement register shows when asked to hold `value`."""
    return (value + 2 ** (bits - 1)) % 2 ** bits - 2 ** (bits - 1)


print("3-bit example: the value 5 is read as", wrap(5, 3), "| the value -4 is read as", wrap(-4, 3), "| the value 3 is read as", wrap(3, 3))
lo_needed, hi_needed = min(cost) - max(cost), max(cost) - min(cost)               # range of (cost - bar) when the bar is one of the costs
print(f"costs run from {min(cost)} to {max(cost)}, so cost - bar ranges over [{lo_needed}, {hi_needed}]")
print(f"{'value qubits':>13}{'register holds':>20}{'enough?':>10}")
for v in (5, 6, 7, 8):
    lo, hi = -2 ** (v - 1), 2 ** (v - 1) - 1
    print(f"{v:>13}{f'{lo} .. {hi}':>20}{str(lo <= lo_needed and hi_needed <= hi):>10}")


def bits_needed(costs):
    """Smallest register width that holds every cost - bar, with the bar one of the costs (equation 8)."""
    lo, hi = min(costs) - max(costs), max(costs) - min(costs)
    v = 1
    while not (-2 ** (v - 1) <= lo and hi <= 2 ** (v - 1) - 1):
        v += 1
    return v


assert bits_needed(cost) == 7
bar = 12
for v in (6, 7):
    wrongly_marked = [plan(index_to_x(k, 4)) for k in range(16) if cost[k] >= bar and wrap(cost[k] - bar, v) < 0]
    print(f"bar {bar}, {v} value qubits: plans marked although they are NOT cheaper than the bar: {wrongly_marked or 'none'}")
fig, ax = plt.subplots(figsize=(7, 2.6))
for i, v in enumerate((5, 6, 7, 8)):
    ax.barh(i, 2 ** v - 1, left=-2 ** (v - 1), color="#0F62FE" if (-2 ** (v - 1) <= lo_needed and hi_needed <= 2 ** (v - 1) - 1) else "#FA4D56")
ax.axvspan(lo_needed, hi_needed, color="#24A148", alpha=0.25, label="range needed: cost - bar")
ax.set_yticks(range(4)); ax.set_yticklabels([f"{v} value qubits" for v in (5, 6, 7, 8)]); ax.set_xlabel("whole numbers the register can hold")
ax.legend(fontsize=8, loc="lower right"); plt.tight_layout()'''

S3_REG_TRY = callout("try", """Change `bar = 12` to `bar = 7`, `20` or `30` and rerun the last cell. For which bars does the 6-qubit register mislabel a plan? Why does the mislabelling happen for the *worst* plans (high cost) and not the cheap ones?""")

T3_INTRO = r"""### T3 · Stretch: the width of the value register, and a new demand · التمرين ٣

**What you are about to build, and why.** The failure of a narrow value register is the most important lesson of this notebook, and T3 lets you see it yourself. You run Grover adaptive search with 6, 7 and 8 value qubits on the same instance (same seed), and explain in one sentence *why* the width matters, using equation (8) and the range of $\text{cost}-y$. Then you change the demand to 6 (a new instance), solve it classically first, and solve it with Grover adaptive search using enough value qubits. You may use the helper `bits_needed` from the cell above to decide how many.

**The formulas you use:** the register holds $-2^{v-1}..2^{v-1}-1$ (equation 8); the oracle needs $[\min\text{cost}-\max\text{cost},\ \max\text{cost}-\min\text{cost}]$.

""" + hints(
    r"""Look at the table in 3.6: which widths hold the range -44 to 44 and which do not? You do not need new maths: you need three calls of the helper `solve_gas(problem, value_qubits, seed)`, which returns an object with the chosen plan and its cost. Remember that `plan(...)` turns a list of switches into a string.""",
    r"""Structure: for 6, 7 and 8 value qubits, solve with seed 2 and store the plan string and the cost; write one sentence using equation (8); build the instance for demand 6 with `unit_commitment_qp`; solve it classically first with `brute_force`; then solve it with `solve_gas` using a register wide enough for the new range (`bits_needed(costs)` helps: it takes a *list of costs*, so for the new instance take the costs of all plans from the table that `brute_force` returns, and remember that a bigger demand can need more bits than the 7 of the main example).""",
    r"""```python
wide = {}
for nv in (6, 7, 8):
    ______                     # solve with nv value qubits; wide[nv] = (plan string, cost)
explanation = "..."            # one sentence, using the range of cost - bar
qp6 = ______                   # the instance for demand 6
x6, v6, _ = ______             # classical answer first
r6 = ______                    # Grover adaptive search with enough value qubits
```""")


T1 = """cost = [int(round(qp.objective.evaluate(index_to_x(k, DECISION_QUBITS)))) for k in range(2 ** DECISION_QUBITS)]
marked_thr8 = [plan(index_to_x(k, DECISION_QUBITS)) for k in range(16) if cost[k] < 8]"""
T1_STARTER = """# Goal: `cost` holds one integer per position k = 0..15 (bit i of k is unit i); `marked_thr8` holds the plans (strings like "1100") with cost below 8.
cost = []
marked_thr8 = []
raise NotImplementedError("T1: fill the cost table and the marked plans")"""
T1_CHECK = """assert len(cost) == 16, "T1: the table needs one cost for each of the 16 plans"
assert cost == [50, 12, 21, 7, 20, 6, 7, 17, 34, 8, 13, 11, 12, 10, 7, 29], "T1: a cost is off; check bit i of k is unit i"
assert sorted(marked_thr8) == sorted(["1100", "1010", "0110", "0111"]), "T1: four plans are cheaper than 8: 1100, 1010, 0110, 0111"
print("marked for threshold 8:", marked_thr8)"""

T2 = """def p_marked_curve(threshold, max_steps=4):
    \"\"\"P(marked) after 0..max_steps Grover steps, where marked means cost < threshold.\"\"\"
    marked = [k for k in range(16) if cost[k] < threshold]
    oracle = QuantumCircuit(DECISION_QUBITS)
    oracle.append(DiagonalGate([-1 if c < threshold else 1 for c in cost]), range(DECISION_QUBITS))
    G = grover_operator(oracle)
    curve = []
    for steps in range(max_steps + 1):
        qc = QuantumCircuit(DECISION_QUBITS)
        qc.h(range(DECISION_QUBITS))
        for _ in range(steps):
            qc.compose(G, inplace=True)
        curve.append(float(Statevector(qc).probabilities()[marked].sum()))
    return curve, G

p_marked_thr8, G8 = p_marked_curve(8)
p_marked_thr7, G7 = p_marked_curve(7)
best_steps_thr7 = int(np.argmax(p_marked_thr7))"""
T2_STARTER = """def p_marked_curve(threshold, max_steps=4):
    # Goal: p_marked_curve(threshold, max_steps=4) returns (the list of P(marked) after 0..max_steps Grover steps, the Grover step G),
    # where "marked" means cost < threshold, using the oracle (equation 3) and the step (equation 5) of section 3.
    raise NotImplementedError("T2: build the phase oracle and the Grover operator")

p_marked_thr8, G8 = p_marked_curve(8)
p_marked_thr7, G7 = p_marked_curve(7)
best_steps_thr7 = int(np.argmax(p_marked_thr7))"""
T2_CHECK = """assert abs(p_marked_thr8[0] - 0.25) < 1e-6, "T2: before any step, 4 of 16 plans are marked, so P = 0.25"
assert abs(p_marked_thr8[1] - 1.0) < 1e-6, "T2: with 4 of 16 marked, one Grover step should give P = 1"
assert abs(p_marked_thr8[2] - 0.25) < 1e-6, "T2: a second step undoes it (P = 0.25 again)"
assert best_steps_thr7 == 3, "T2: with one marked plan the best number of steps is 3"
print("threshold 8:", np.round(p_marked_thr8, 3))
print("threshold 7:", np.round(p_marked_thr7, 3), "-> best steps:", best_steps_thr7)"""

S3_PLOT = """fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, curve, label in ((axes[0], p_marked_thr8, "threshold 8: 4 plans marked"), (axes[1], p_marked_thr7, "threshold 7: 1 plan marked")):
    ax.plot(range(len(curve)), curve, "o-", color="#0F62FE")
    ax.set_title(label); ax.set_xlabel("Grover steps"); ax.set_xticks(range(len(curve))); ax.grid(alpha=0.3)
axes[0].set_ylabel("P(marked plan)")
plt.tight_layout(); plt.show()
print("The best number of steps depends on how many plans are marked, and the loop does not know that in advance.")
print("That is why Grover adaptive search draws the number of steps at random.")"""

S3_GAS_CODE = """from qiskit.primitives import StatevectorSampler
from qiskit_algorithms.utils import algorithm_globals
from qiskit_optimization.algorithms import GroverOptimizer

class RecordingSampler(StatevectorSampler):
    \"\"\"A StatevectorSampler that keeps the circuits it is asked to run (so we can inspect them).\"\"\"
    def __init__(self, **kw):
        super().__init__(**kw)
        self.circuits = []
    def run(self, pubs, **kw):
        self.circuits.extend(p if isinstance(p, QuantumCircuit) else p[0] for p in pubs)
        return super().run(pubs, **kw)

def solve_gas(problem, value_qubits, seed, iterations=15, sampler=None):
    \"\"\"One Grover adaptive search run. Sets BOTH seeds, then calls GroverOptimizer.solve.\"\"\"
    algorithm_globals.random_seed = seed
    sampler = sampler or StatevectorSampler(seed=seed)
    return GroverOptimizer(num_value_qubits=value_qubits, num_iterations=iterations, sampler=sampler).solve(problem)

recorder = RecordingSampler(seed=SEED)
t0 = time.perf_counter()
gas_result = solve_gas(qp, VALUE_QUBITS, SEED, sampler=recorder)
gas_seconds = time.perf_counter() - t0
gas_circuits = len(gas_result.operation_counts)
print(f"x = {plan([int(v) for v in gas_result.x])}, cost = {gas_result.fval:g}, status = {gas_result.status}")
print(f"decision qubits {gas_result.n_input_qubits}, value qubits {gas_result.n_output_qubits}, circuits run {gas_circuits}, {gas_seconds:.1f} s")
circuit = max(recorder.circuits, key=lambda c: c.depth())        # the deepest circuit it ran: used by the IBM preview below
print(f"deepest circuit it ran: {circuit.num_qubits} qubits, depth {circuit.decompose(reps=2).depth()} after two decompositions")"""

S3_REL_MD = """### Reliability, and the value-register lesson

One run can be lucky. We repeat for seeds 1 to 10 with a 7-qubit value register, then for seeds 1 to 3 with only **6** (those runs are slow, so we run fewer). The costs run from 6 to 50, and the oracle works with (cost − threshold), whose signed range needs 7 bits. Six bits overflow."""

S3_REL_CODE = """# Takes about 1 to 1.5 minutes in total (13 solves; the 6-qubit solves are slow because they wander for 50 to 150 circuits). Both seeds are set inside solve_gas.
def reliability(value_qubits, seeds=range(1, 11)):
    \"\"\"Run GAS once per seed; return the list of (seed, x as string, cost, status).\"\"\"
    rows = []
    for s in seeds:
        r = solve_gas(qp, value_qubits, s)
        rows.append((s, plan([int(v) for v in r.x]), r.fval, str(r.status).split(".")[-1]))
    return rows

gas_rows = {}
for nv, seeds in ((7, range(1, 11)), (6, range(1, 4))):
    gas_rows[nv] = reliability(nv, seeds)
    hits = sum(1 for _, xs, v, _ in gas_rows[nv] if xs == plan(classical_x) and v == classical_value)
    print(f"{nv} value qubits: optimum found in {hits} of {len(gas_rows[nv])} seeds")
    print("   ", [(s, xs, int(v)) for s, xs, v, _ in gas_rows[nv]])
gas_hits_7bits = sum(1 for _, xs, v, _ in gas_rows[7] if xs == plan(classical_x) and v == classical_value)
gas_hits_6bits = sum(1 for _, xs, v, _ in gas_rows[6] if xs == plan(classical_x) and v == classical_value)
print("status reported with 6 value qubits:", sorted({st for *_, st in gas_rows[6]}), "<- it still says SUCCESS")"""

S3_REL_NOTE = """With 6 value qubits the routine reports success and returns a wrong plan, sometimes even `0000` at cost 50. Nothing warns you. **Always check a quantum optimiser's answer against something you can compute.**"""

T3 = """# Seed 2 for all three widths (the 6- and 7-qubit rows are already in the reliability table above; only 8 is new).
wide = {nv: (xs, v) for nv in (6, 7) for s, xs, v, _ in gas_rows[nv] if s == 2}
r8 = solve_gas(qp, 8, 2)
wide[8] = (plan(r8.x), r8.fval)
for nv in (6, 7, 8):
    print(f"{nv} value qubits, seed 2: u = {wide[nv][0]}, cost {wide[nv][1]:g}")
# Why does the width matter?  The oracle subtracts the threshold: the range of (cost - threshold) is about -44 .. +44,
# which needs 7 bits in two's complement (6 bits cover only -32 .. 31).  Write one sentence below.
explanation = "Costs reach 50, so cost - threshold spans about -44 to 44; 6 bits (-32..31) overflow and flip signs."

# Now change the demand to 6 and solve again, classical first.  The range grows, so use 8 value qubits.
qp6 = unit_commitment_qp(6)
x6, v6, _ = qff.classical.brute_force(qp6)
r6 = solve_gas(qp6, 8, SEED)
print(f"demand 6: classical u = {plan(x6)}, cost {v6:g}; GAS (8 value qubits) u = {plan(r6.x)}, cost {r6.fval:g}")"""
T3_STARTER = """# Goal: wide[nv] = (plan string, cost) from solve_gas on the demand-5 instance with nv = 6, 7, 8 value qubits (seed 2);
# explanation = one sentence on why the value register needs 7 bits here; then build the demand-6 instance qp6 (the problem with demand 6),
# solve it classically first (x6 = best plan, v6 = its cost), and then with Grover adaptive search (r6 = the result object of solve_gas), and compare.
# The Self-check reads wide, explanation, qp6, x6, v6 and r6.
wide = {}
explanation = ""
qp6, x6, v6, r6 = None, None, None, None
raise NotImplementedError("T3: try 6, 7 and 8 value qubits, then change the demand")"""
T3_CHECK = """assert set(wide) == {6, 7, 8}, "T3: run 6, 7 and 8 value qubits"
assert len(explanation.strip()) > 20, "T3: write one sentence about the value register"
assert v6 == qp6.objective.evaluate(x6), "T3: the classical answer for demand 6 should be the best plan"
assert r6.fval == v6, "T3: with enough value qubits GAS should match the classical optimum for demand 6" """

S3_NOTE_SOL = """[solution-only] Instructor notes. (1) The 8-qubit solve is a 12-qubit simulation and is the slowest cell in T3. (2) If a team's 7-qubit run at SEED fails, that is not a bug in their code; reliability is 10 of 10 only with `num_iterations=15`. (3) With 6 value qubits the table uses seeds 1-3 only, to keep the notebook under 3 minutes (a full ten-seed run, measured separately, found the optimum in 2 of 10 and took about 140 s); the plan's headline is 2 of 10. (4) For demand 6 the classical optimum is checked in the cell itself; do not hard-code it."""

S4_RESULTS = """# Results table: classical baselines first.
rows = [
    ("brute force", plan(classical_x), classical_value, f"{2 ** DECISION_QUBITS} states", f"{classical_seconds * 1000:.2f} ms"),
    ("merit order", plan(merit_x), merit_value, "4 steps", "-"),
    (f"Grover adaptive, {VALUE_QUBITS} value qubits", plan([int(v) for v in gas_result.x]), gas_result.fval,
     f"{gas_circuits} circuits ({QUBITS} qubits)", f"{gas_seconds:.1f} s"),
]
print(f"{'method':<40}{'plan':<8}{'cost':<7}{'states / circuits':<26}{'time'}")
for m, plan_str, v, work, sec in rows:
    print(f"{m:<40}{plan_str:<8}{v:<7g}{work:<26}{sec}")
print(f"\\nReliability: {gas_hits_7bits}/10 seeds with 7 value qubits, {gas_hits_6bits}/3 seeds with 6.")

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.bar(["6 value qubits", "7 value qubits"], [gas_hits_6bits / 3, gas_hits_7bits / 10], color=["#FA4D56", "#0F62FE"])
ax.set_ylim(0, 1); ax.set_ylabel("share of seeds that found the optimum"); ax.set_title("Width of the value register")
plt.tight_layout(); plt.show()"""

S4_SHOWS = """**What this toy shows.** Grover's amplification can be wrapped in a loop that keeps lowering a threshold until nothing cheaper exists. On 16 schedules it finds the optimum every time with a wide enough value register, and it fails silently when the register is too narrow."""

S4_NOTSHOW = callout("watch", """**What this toy does not show.** The quantum circuit computes the cost inside itself, which took 7 extra qubits and deep arithmetic for a 4-unit problem; the loop above needed 16 or more circuit runs where brute force needed 16 lookups. Grover's speed-up is quadratic at best (about √N instead of N checks) and needs error-corrected hardware at useful sizes. Real unit commitment has ramp limits, minimum up-times, reserves and 24 or more hours, and classical mixed-integer solvers handle it daily. Nothing here is a demonstration of quantum advantage.""")

S4_PAPER = """**Paper versus our route.** WEF/Aramco Case 13 (p. 27): IonQ and Oak Ridge National Laboratory, under the US DOE GRID-Q effort, ran a hybrid formulation for 26 generators over 24 hours on a 36-qubit trapped-ion system, using a layered variational method on hardware. We chose Grover adaptive search instead because it builds directly on the Grover lecture and has a ready Qiskit class. This notebook is our teaching route, not the paper's method, and it ran on a simulator."""

ACC = """assert QUBITS == 11 and DECISION_QUBITS == 4
assert classical_x == [1, 0, 1, 0] and classical_value == 6
assert abs(p_marked_thr8[1] - 1.0) < 1e-6 and abs(p_marked_thr8[2] - 0.25) < 1e-6
assert int(np.argmax(p_marked_thr7)) == 3 and abs(p_marked_thr7[3] - 0.961) < 0.01
assert list(map(int, gas_result.x)) == [1, 0, 1, 0] and gas_result.fval == 6
assert gas_hits_7bits >= 9 and gas_hits_6bits < 3, f"hits: 7 bits {gas_hits_7bits}, 6 bits {gas_hits_6bits}"
print("acceptance passed:", dict(gas_hits_7bits=gas_hits_7bits, gas_hits_6bits=gas_hits_6bits, circuits=gas_circuits))"""

IBM_NOTE = callout("real", """The preview above is the deepest circuit `GroverOptimizer` ran (4 decision qubits, 7 value qubits and the oracle qubit). A circuit with thousands of two-qubit gates would **not** survive today's noisy devices: every gate adds a little error and the answer would drown in noise. Grover's quadratic speed-up needs error-corrected hardware, and that is why our claim in section 4 stays modest.""")

LOOP_FIX = "from qiskit.circuit.library import DiagonalGate, grover_operator\nfrom qiskit.quantum_info import Statevector"


S3_SYMBOLS = '**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).\n\n| Symbol | Meaning | In the code |\n|---|---|---|\n| $x=(u_0,\\dots,u_3)$ | a plan: unit $i$ on when $u_i=1$ | `plan(x)`, `index_to_x(k, 4)` |\n| $\\text{cost}(x),\\ y$ | cost of a plan (equation 1); the bar (threshold) | `cost[k]`, `threshold` |\n| $N,\\ M$ | number of plans (16), number of marked plans | `len(cost)`, `len(marked)` |\n| $\\lvert s\\rangle,\\ a_x$ | equal mixture, amplitude of plan $x$ | `by_hand`, `Statevector(...).data` |\n| $O,\\ D,\\ G=DO$ | oracle (sign flip), inversion about the average, one Grover step | `DiagonalGate`, `grover_operator` |\n| $\\theta,\\ k$ | rotation angle per half step, number of Grover steps | `np.arcsin(np.sqrt(M / N))`, `steps` |\n| $P(\\text{marked})$ | probability of reading a marked plan | `p_marked_curve` |\n| $v$ | width of the value register (qubits) | `num_value_qubits`, `bits_needed` |'

S3_MATH_ONE = '<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**\n\n**The one line to remember.** After $k$ Grover steps $P(\\text{marked})=\\sin^2\\bigl((2k+1)\\theta\\bigr)$ with $\\sin\\theta=\\sqrt{M/N}$ (equation 6): each step *rotates* the state by $2\\theta$, so the best $k$ is about $\\frac\\pi4\\sqrt{N/M}$ and too many steps make things worse.\n\n</div>'

S3_DRAW = '# The Grover step as a circuit (the oracle you built in T2, then the reflection about the equal mixture), for the bar 8.\ntry:\n    display(G8.decompose().draw("mpl", fold=40))\nexcept Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text\n    print(G8.decompose().draw("text", fold=120))'


S3_COST_FIG = """# The 16 costs as a picture: the plans the oracle will mark (cost below 8) are the low teal bars.
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.bar(range(16), cost, color=["#007D79" if c < 8 else "#8D8D8D" for c in cost])
ax.axhline(8, color="#FA4D56", ls="--", lw=1, label="threshold 8: plans below it are marked")
ax.set_xticks(range(16)); ax.set_xticklabels([plan(index_to_x(k, DECISION_QUBITS)) for k in range(16)], rotation=90, fontsize=7)
ax.set_xlabel("plan (one bit per unit)"); ax.set_ylabel("cost"); ax.legend(fontsize=8)
plt.tight_layout()
assert sum(c < 8 for c in cost) == len(marked_thr8) == 4, "the bars below the dashed line are exactly the marked plans"
print("OK: four teal bars sit below the threshold, and they are the four marked plans.")"""

S3_EQ1_CHECK = """# Equation (1): cost(u) = sum_i s_i u_i + lambda (sum_i c_i u_i - D)^2, recomputed for all 16 plans with the notebook's own constants.
DEMAND = 5
cost_by_formula = [sum(START_COST[i] * u[i] for i in range(4)) + LAMBDA * (sum(CAPACITY[i] * u[i] for i in range(4)) - DEMAND) ** 2
                   for u in (index_to_x(k, DECISION_QUBITS) for k in range(16))]
assert cost_by_formula == cost, "equation (1) disagrees with the cost table of T1"
print("OK: equation (1) reproduces the cost table for all 16 plans.")"""


def main():
    """Write the solution notebook."""
    nb = new(5, "Unit commitment: which power stations run?", use_case="Choose which generating units to switch on to meet demand at least cost",
             algorithm="Grover adaptive search (Qiskit `GroverOptimizer`)", qubits=11, minutes=3)
    nb.section(1)
    nb.md(S1)
    nb.section(2)
    nb.md(S2_MD)
    nb.code(S2_INSTANCE)
    nb.code(S2_CLASSICAL)
    nb.md(S2_NOTE)
    nb.section(3)
    nb.md(S3_ROADMAP)
    nb.code(S3_FLOW)
    nb.md(S3_SYMBOLS)
    nb.md(S3_MATH_ONE)
    nb.md(S3_COSTSTATE_MD)
    nb.code(LOOP_FIX)
    nb.md(TOOLBOX[5][0])
    nb.code(TOOLBOX[5][1])
    nb.md(T1_INTRO)
    nb.task("T1", "fill the 16-row cost table and find the plans cheaper than 8", T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.code(S3_COST_FIG)
    nb.code(S3_EQ1_CHECK)
    nb.md(S3_ORACLE_MD)
    nb.md(S3_REFLECT_MD)
    nb.code(S3_BYHAND)
    nb.md(S3_BYHAND_TRY)
    nb.md(S3_ROT_MD)
    nb.code(S3_GEOM)
    nb.md(T2_INTRO)
    nb.task("T2", "build the phase oracle and the Grover operator, find the best number of steps", T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_T2_CHECK_MD)
    nb.code(S3_T2_CHECK)
    nb.code(S3_DRAW)
    nb.md(S3_LOOP_MD)
    nb.md(S3_GAS_MD)
    nb.code(S3_GAS_CODE)
    nb.md(S3_REG_MD)
    nb.code(S3_REG_CODE)
    nb.md(S3_REG_TRY)
    nb.md(S3_REL_MD)
    nb.code(S3_REL_CODE)
    nb.md(S3_REL_NOTE)
    nb.md(T3_INTRO)
    nb.task("T3", "try 6, 7 and 8 value qubits, explain the failures, then change the demand to 6", T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_NOTE_SOL, tags=["solution-only"])
    nb.md(day.IBM_MD)
    nb.code(day.IBM_PREVIEW_CODE)
    nb.md(IBM_NOTE)
    nb.md(day.glossary_md([("Grover's algorithm", "خوارزمية جروفر", "G"), "oracle", "amplitude amplification", "optimization", ("threshold", "العتبة", "I"), ("value register", "سجل القيمة", "I"), ("two's complement", "المتمم الثنائي", "I")]), tags=["glossary"])
    nb.section(4)
    nb.code(S4_RESULTS)
    nb.md(S4_SHOWS)
    nb.md(S4_NOTSHOW)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="Which power stations should run tonight, at least cost, while still meeting demand?",
                counting_fact="4 units give 16 on/off plans; 26 units over 24 hours give 2⁶²⁴ plans",
                single_idea="mark every plan cheaper than your best so far, amplify the marked ones, measure, repeat with a lower bar",
                ideas=["A \"limbo bar\" animation where the cost threshold drops each round", "A control-room game: switch units on and off against the clock, then race the circuit",
                       "A physical card-trick version of amplitude amplification (flip the marked cards, reflect, repeat)"],
                reuse="the P(marked)-versus-steps plot, the 6-versus-7 value-qubit failure table")
    nb.section6(honest="It found the cheapest plan 10 times out of 10 on 16 options, using 11 simulated qubits; a loop over 16 options is faster.",
                question_a="Why does a second Grover step make things worse?",
                question_b="What happens with too few value qubits?")
    path = nb.save(os.path.join(ROOT, "solutions", "05_move_unit_commitment_grover_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
