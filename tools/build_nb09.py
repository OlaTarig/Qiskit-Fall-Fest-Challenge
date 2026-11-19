"""Build solutions/09_use_desalination_grover_solution.ipynb (mentors and builders only).

    python tools/build_nb09.py
    python tools/make_starters.py 9
    python tools/run_tests.py 9

Notebook 9 follows notebook 5 cell for cell (instance -> classical baseline -> Grover by hand -> the loop in words ->
GroverOptimizer behind `solve_gas` -> reliability -> value-register experiment).  What is new: the rule "exactly three slots" is a real
linear constraint that GroverOptimizer turns into a penalty, and the formulation is ours, not the paper's.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, arabic_table, callout, details, hints, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1 = """**The problem.** A reverse-osmosis (RO) desalination train turns seawater into fresh water by pushing it through membranes, and that takes a great deal of electricity. The plant must make a certain amount of water today, so it has to choose *when* to run. Electricity costs different amounts in different hours, cheap when the sun is high and dear in the evening peak. And every cold start of a train costs extra. Run in the cheap hours and keep running, and the bill falls. With 5 time slots there are 2⁵ = 32 possible plans; hourly slots over a day for ten trains give 2²⁴⁰.

**Who faces it.** A desalination utility, or the operator of a large water-treatment plant that buys power on a time-of-day tariff.

""" + callout("real", """**What is real today.** WEF/Aramco *Quantum for Energy and Utilities* (April 2026) lists "optimize desalination and water treatment" as a use case in its water survey (65% of respondents, Figure 8, p. 30). It gives **no case study, no hardware run and no method** for it. Today plants are scheduled with classical optimisation (mixed-integer programming, rule-based controllers), not with quantum computers.""") + """

**Why a quantum computer might help.** Scheduling is a search over a huge number of on/off plans, and Grover's algorithm finds a marked item among N in about √N steps instead of N. Wrapped in a loop that keeps lowering a cost threshold, it turns "search" into "optimise". What is still missing: the quadratic speed-up only pays off on error-corrected hardware, and the circuit must compute the cost inside itself.

| Paper's method | Our teaching route |
|---|---|
| None. The paper lists the use case (Figure 8, p. 30) but names no formulation and no algorithm. | Grover adaptive search with Qiskit's `GroverOptimizer` on one train and 5 time slots. **The formulation (tariff, adjacency saving, "exactly three slots") and the choice of method are ours: a teaching choice, not the paper's.** |"""

S2_MD = """One reverse-osmosis train must run in **exactly 3 of 5 time slots** today. Slot *t* has an electricity price pₜ (a toy number: cheap at solar noon, dear at the edges). Running in two back-to-back slots avoids a restart, which saves 1.

| Slot t | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| Price pₜ | 4 | 2 | 1 | 2 | 5 |

With rₜ = 1 if the train runs in slot *t*:

**minimise  Σ pₜ rₜ − Σ rₜ rₜ₊₁   subject to   Σ rₜ = 3**

The second sum counts back-to-back pairs. (All numbers are invented for teaching; the formulation is ours.)

Unlike notebook 5, the rule "exactly three" is written as a real **constraint**. `GroverOptimizer` cannot take constraints directly, so it turns the rule into a **penalty** P·(Σ rₜ − 3)² added to the cost. We give it `penalty=4`. We use the same conversion, `QuadraticProgramToQubo(penalty=4)`, to get the cost table for the by-hand Grover step below.

Qubit budget: **5 decision qubits** (one per slot) **+ 7 value qubits** (the circuit has to hold the cost inside itself) = **12 qubits**. (Qiskit adds one more flag qubit inside the oracle, so the circuit it runs shows 13; we count 5 + 7.)"""

S2_INSTANCE = """QUBITS = 12                      # [KEEP] the test runner reads this
DECISION_QUBITS = 5
VALUE_QUBITS = 7
PENALTY = 4
ITERATIONS = 8                   # GroverOptimizer stops after this many rounds in a row without improvement
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qff.bits import index_to_x

def plan(x):
    \"\"\"Run/stop flags as a string in slot order: [0, 1, 1, 1, 0] -> '01110' (slot 0 first).\"\"\"
    return "".join(str(int(v)) for v in x)

TARIFF = [4, 2, 1, 2, 5]

def desalination_qp(tariff=TARIFF):
    \"\"\"minimise sum tariff_t r_t - sum r_t r_(t+1)  subject to  sum r_t == 3.\"\"\"
    qp = QuadraticProgram("desalination")
    for t in range(5):
        qp.binary_var(f"r{t}")
    qp.minimize(linear={f"r{t}": tariff[t] for t in range(5)}, quadratic={(f"r{t}", f"r{t + 1}"): -1 for t in range(4)})
    qp.linear_constraint({f"r{t}": 1 for t in range(5)}, "==", 3, "run_three_slots")
    return qp

qp = desalination_qp()
print(qp.prettyprint())
qubo = QuadraticProgramToQubo(penalty=PENALTY).convert(qp)   # the same conversion GroverOptimizer does internally
print("QUBO variables:", qubo.get_num_vars(), "(no slack qubits: the rule is an equality)")"""

S2_CLASSICAL = """# Classical baseline: always runs first.
import time
t0 = time.perf_counter()
classical_x, classical_value, table = qff.classical.brute_force(qp)
classical_seconds = time.perf_counter() - t0
feasible = [(x, v) for x, v, ok in table if ok]
n_feasible = len(feasible)
print(f"brute force over {2 ** DECISION_QUBITS} states, {n_feasible} feasible: best r = {plan(classical_x)}, cost {classical_value:g}")
print("cheapest five feasible plans:", [(plan(x), v) for x, v in feasible[:5]])
print("cheapest five of all 32, rule ignored:", [(plan(x), v) for x, v, _ in table[:5]], "<- two-slot plans look cheaper: that is why the rule needs a penalty")

# A second baseline, greedy: run in the three cheapest slots.
greedy_slots = sorted(range(5), key=lambda t: (TARIFF[t], t))[:3]
greedy_x = [1 if t in greedy_slots else 0 for t in range(5)]
greedy_value = qp.objective.evaluate(greedy_x)
print(f"greedy (three cheapest slots {sorted(greedy_slots)}): r = {plan(greedy_x)}, cost {greedy_value:g}")"""

S2_NOTE = """Greedy happens to find the optimum on this tiny instance: the three cheapest slots are also back to back. In T3 you change the tariff so that they are not. The point of the baseline is that **32 states are checked in microseconds**."""

S3_ROADMAP = r"""### 3.0 · The map of this section

""" + callout("idea", r"""**Grover adaptive search in one sentence.** Give every plan an amplitude, mark the plans that are cheaper than the best one found so far by flipping the sign of their amplitude, let a second trick (inversion about the average) turn those sign flips into *larger* amplitudes for the marked plans, measure, and repeat with a lower bar until nothing cheaper exists.""") + r"""

There is **no knob-tuning** here (unlike QAOA or VQE). The circuit is fixed; what is cleverly arranged is how amplitudes move. We build it from two things you already know: lists of numbers (vectors) and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**. (If you did notebook 5 you will recognise 3.2 to 3.5: the algorithm is the same, only the problem differs. They are repeated here so this notebook stands on its own. What is new is the **penalty** in 3.1.)

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | The rule "exactly three slots" becomes a penalty; the cost of each of the 32 plans; the equal-amplitude state | (1), (2), (3) | `qubo`, `cost` | **T1** |
| 3.2 | The oracle: flip the sign of every plan cheaper than the bar | (4) | `DiagonalGate` | |
| 3.3 | Inversion about the average, and one full Grover step | (5), (6) | `grover_operator` | |
| 3.4 | A rotation picture: how many steps are best? | (7), (8) | `p_marked_curve` | **T2** |
| 3.5 | The adaptive loop, and Qiskit's `GroverOptimizer` | (8) | `solve_gas` | |
| 3.6 | The value register: why the circuit needs 7 extra qubits | (10) | `reliability` | **T3** |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it."""

S3_FLOW = """qff.plots.plot_flow(
    [("Rule to penalty", "equation 2", "source"), ("Pick a bar y", "cost of a random plan", "move"), ("Oracle", "mark plans cheaper than y", "store"),
     ("Grover steps", "random number", "use"), ("Measure", "one plan", "source"), ("Cheaper?", "then lower y", "move")],
    loop=(5, 2, "repeat; stop after several failures in a row"), title="Grover adaptive search with a constraint");"""

S3_PENALTY_MD = r"""### 3.1 · The rule becomes a penalty, and a state to search in

A **plan** is a list of five on/off flags $r=(r_0,\dots,r_4)$ ($r_t=1$: the train runs in slot $t$), so there are $2^5=32$ plans. The objective from section 2 adds the electricity price of the slots we run and subtracts 1 for every pair of back-to-back slots (no restart), and the rule says we must run in **exactly three** slots:

$$f(r)=\sum_{t}p_t\,r_t-\sum_{t=0}^{3}r_t\,r_{t+1}\qquad\text{subject to}\qquad\sum_t r_t=3 \tag{1}$$

Grover's search has no place for a side rule, so the rule is folded into the cost as a **penalty**: a term that is 0 for a plan that obeys the rule and positive for one that breaks it,

$$\text{cost}(r)=f(r)+P\Bigl(\sum_t r_t-3\Bigr)^2,\qquad P=4 \tag{2}$$

The result is a **QUBO** (*quadratic unconstrained binary optimisation*: at most pairs of variables, no side rules left, all variables 0/1). Qiskit builds it with `QuadraticProgramToQubo(penalty=4)`, the same conversion `GroverOptimizer` uses inside. **Examples you can check by hand:** the plan $01110$ (slots 1, 2, 3) pays prices $2+1+2=5$, saves $2$ for two back-to-back pairs, obeys the rule: cost $5-2=3$. The plan $00000$ runs nowhere: prices 0, no savings, but $4\cdot(0-3)^2=36$ of penalty: cost $36$, the most expensive plan. The plan $01100$ runs in two slots: $3-1=2$, plus $4\cdot(2-3)^2=4$: cost $6$.

**How big must $P$ be?** If $P$ is too small, breaking the rule can be cheaper than obeying it. The cell below tries $P=1,2,4,8$: with $P=1$ an illegal two-slot plan *ties* the best legal plan at cost 3 (so the search may return it); with $P\ge2$ the cheapest plan is legal.

**The state.** Five qubits hold a list of $2^5=32$ numbers called **amplitudes** $a_x$, one per plan; $|a_x|^2$ is the **probability** of reading out plan $x$ when you **measure**, and the probabilities add up to 1. The **Hadamard gate** $H$ puts one qubit into an equal mixture of $|0\rangle$ and $|1\rangle$; one $H$ on each of the five qubits ($H^{\otimes5}$) gives every plan the same amplitude:

$$|s\rangle=H^{\otimes5}|00000\rangle=\frac1{\sqrt{32}}\sum_x|x\rangle,\qquad a_x=\frac1{\sqrt{32}}\ \text{for every plan},\qquad \sum_x a_x^2=32\cdot\frac1{32}=1 \tag{3}$$

($|x\rangle$ is a *name tag* for plan $x$: a column of 32 numbers with a single 1 at position $x$.) This is pure guessing: each plan has probability $\frac1{32}$. Position $k$ of the list is the plan whose flag $r_t$ is bit $t$ of $k$ (so $k=14=01110_2$ read from the right is $r_1=r_2=r_3=1$: the plan `01110`).

""" + callout("watch", r"""**Amplitudes can be negative.** Probabilities cannot, but amplitudes can: only their *squares* are probabilities. A minus sign is invisible to a measurement of that state, but it changes what later gates do. The whole trick of Grover's algorithm is built on a minus sign.""") + "\n\n" + arabic_table([
    ("Penalty", "العقوبة (penalty)", "I"), ("Constraint", "القيد", "I"), ("QUBO", "مسألة تحسين ثنائية تربيعية غير مقيدة (QUBO)", "I"),
    ("Time slot", "الفترة الزمنية", "I"), ("Tariff (electricity price)", "التعرفة (سعر الكهرباء)", "I"), ("Plan", "الخطة", "I"),
    ("Amplitude", "السعة", "G"), ("Probability", "الاحتمال", "I"), ("Measurement", "القياس", "G"), ("Hadamard gate", "بوابة هادامارد", "G"),
    ("Superposition", "التراكب", "G"), ("Desalination", "تحلية المياه", "I")])

S3_PENALTY_CODE = '''# Equation (2): how big must the penalty be? The cheapest plan of the converted problem, for several P.
def f_objective(r):
    """Equation (1): prices of the slots that run, minus 1 for every back-to-back pair."""
    return sum(TARIFF[t] * r[t] for t in range(5)) - sum(r[t] * r[t + 1] for t in range(4))


def cost_by_formula(r, P=PENALTY):
    """Equation (2): objective plus P * (number of running slots - 3)^2."""
    return f_objective(r) + P * (sum(r) - 3) ** 2


print(f"{'P':>3}{'lowest cost':>13}{'cheapest plan':>15}{'runs in':>10}{'legal?':>8}")
for P_try in (1, 2, 4, 8):
    q = QuadraticProgramToQubo(penalty=P_try).convert(qp)
    x_best, v_best, _ = qff.classical.brute_force(q)
    print(f"{P_try:>3}{v_best:>13g}{plan(x_best):>15}{sum(x_best):>7} slots{str(bool(qp.is_feasible(x_best))):>8}")
print("(with P = 1 the plan 01100 breaks the rule yet ties the best legal plan, 01110, at cost 3)")'''

T1_INTRO = r"""### T1 · Warm-up: the cost table · التمرين ١

**What you are about to build, and why.** Grover's oracle (next) will mark *every plan cheaper than a bar*. To understand it, you need the cost of all 32 plans in front of you, **with the penalty**. In T1 you fill the list `cost` (one integer per position $k=0..31$, taken from the converted problem `qubo`), the list of *legal* plans (exactly three slots) with their costs, cheapest first, and the list of plans whose cost is below 6 (the plans the oracle would mark at the bar $y=6$).

**The formula you implement:** equation (2) for every plan. `qubo.objective.evaluate(x)` computes it for you given the flags $x$; your job is to produce the 32 flag lists from the positions $k$ (bit $t$ of $k$ is slot $t$) and collect the results.

""" + hints(
    r"""You need to visit the 32 positions k = 0, ..., 31. For each one you need the five flags as a list, with bit t of k as slot t: the helper `index_to_x(k, 5)`, imported in section 2, does exactly that (it returns the list $[r_0,\dots,r_4]$ for position `k`). Which problem must you evaluate: the original `qp` or the converted `qubo`? Think about whether the penalty should be inside.""",
    r"""A list comprehension over the 32 positions builds `cost`: turn the position into its flag list, evaluate the *converted* problem on it, round to an integer. For the legal plans, go through the positions again, keep those whose flags add up to 3 as pairs (plan string, cost), and sort by cost. For the marked plans keep the plan strings whose cost is below the bar.""",
    r"""```python
cost = ______              # 32 integers: the penalised cost of plan k
feasible_plans = ______    # (plan string, cost) for the plans with exactly three slots, cheapest first
marked_thr6 = ______       # the plan strings whose cost is below 6
```
Test yourself: costs run from 3 to 36, there are 10 legal plans, and two plans are cheaper than 6.""")

S3_CHECK = '''# Check: equation (2) reproduces the cost table of the converted problem for all 32 plans.
from qff.bits import x_to_index

assert cost == [cost_by_formula(index_to_x(k, DECISION_QUBITS)) for k in range(32)], "equation (2) disagrees with the QUBO"
print("OK: equation (2) matches Qiskit's QUBO for all 32 plans.")
print("cost of 01110, 00000 and 01100:", [cost[x_to_index(r)] for r in ([0, 1, 1, 1, 0], [0, 0, 0, 0, 0], [0, 1, 1, 0, 0])], "(by hand: 3, 36, 6)")'''

S3_ORACLE_MD = r"""### 3.2 · The oracle: flip the sign of every cheap plan

Pick a **bar** $y$. A plan is **marked** if its cost is below $y$. The **oracle** $O$ is a rule that flips the sign of the amplitude of every marked plan and leaves the others alone:

$$O\,|x\rangle=\begin{cases}-|x\rangle&\text{if }\text{cost}(x)<y\ \ (\text{marked})\\+|x\rangle&\text{otherwise}\end{cases} \tag{4}$$

As a matrix, $O$ is *diagonal*: a $32\times32$ table with $\pm1$ on its diagonal and zeros elsewhere. In Qiskit that is a `DiagonalGate` built from the list of $\pm1$ values. A sign flip changes **no probability** ($(-a)^2=a^2$), so right after the oracle a measurement still sees every plan equally likely. The information "which plans are cheap" is now stored in the *signs*, where a measurement cannot see it, and the next step turns it into amplitude.

**Careful.** This oracle looks at the cost table directly, which a real algorithm cannot do (the real oracle computes costs *inside* the circuit, section 3.6). Here it is for intuition only.

""" + arabic_table([("Oracle", "العرّاف (oracle)", "I"), ("Marked plan", "الخطة الموسومة", "I"), ("Phase / sign flip", "قلب الإشارة (الطور)", "I"),
                    ("Diagonal matrix", "مصفوفة قطرية", "I"), ("Threshold (the bar)", "العتبة", "I")])

S3_REFLECT_MD = r"""### 3.3 · Inversion about the average

The second ingredient is a **reflection about the average**. Let $\bar a$ be the average of the 32 amplitudes. The step replaces every amplitude by

$$a_x\ \longmapsto\ 2\bar a-a_x \tag{5}$$

In words: *mirror each amplitude around the average.* An amplitude that is *far below* the average (a marked plan, after its sign flip) lands *far above* it; an amplitude close to the average barely moves. (As a matrix this is $D=2|s\rangle\langle s|-I$: twice the projection onto the equal state, minus the identity.) One full **Grover step** is "oracle, then inversion about the average":

$$G=D\,O \tag{6}$$

**A worked example you can check on paper: 4 plans, 1 of them marked.** Amplitudes start at $\frac12$ each. The oracle flips the marked one: $(\frac12,\frac12,\frac12,-\frac12)$. The average is $\bar a=\frac14(\frac12+\frac12+\frac12-\frac12)=\frac14$. Inversion (5): the three unmarked become $2\cdot\frac14-\frac12=0$ and the marked one becomes $2\cdot\frac14+\frac12=1$. Result $(0,0,0,1)$: **one step finds the marked plan with certainty.** (Check: $0^2+0^2+0^2+1^2=1$ ✓.) The cell below does this arithmetic with numpy, then the same for our 32 plans.

""" + callout("idea", r"""**Why this amplifies.** The sign flip pulls the marked amplitudes *below* the average; the mirror pushes them *above* it by about the same distance, while the average itself moves only a little. Net effect: marked plans grow, unmarked plans shrink. Repeat, and the probability of a marked plan keeps growing, up to a point (next section).""") + "\n\n" + arabic_table([
    ("Grover's algorithm", "خوارزمية جروفر", "G"), ("Amplitude amplification", "تضخيم السعة", "I"), ("Reflection / inversion about the mean", "الانعكاس (القلب) حول المتوسط", "I"),
    ("Average (mean)", "المتوسط", "I"), ("Interference", "التداخل", "G"), ("Quantum speedup (quadratic)", "تسريع تربيعي", "U")])

S3_BYHAND = '''# Equations (3)-(6) with plain numpy: sign flip, then inversion about the average. No quantum library.
def grover_by_hand(cost, bar, steps=3):
    """Amplitudes of the plans after each action. Returns (dict name -> amplitudes, list of P(marked) after 0..steps steps, marked positions)."""
    c = np.array(cost)
    marked = np.flatnonzero(c < bar)
    a = np.full(len(c), 1 / np.sqrt(len(c)))                          # equation (3): equal amplitudes
    stages, curve = {"start": a.copy()}, [float((a[marked] ** 2).sum())]
    for step in range(1, steps + 1):
        a = a.copy()
        a[marked] *= -1                                               # equation (4): the oracle flips the sign of marked plans
        stages[f"step {step}: oracle"] = a.copy()
        a = 2 * a.mean() - a                                          # equation (5): inversion about the average
        stages[f"step {step}: invert"] = a.copy()
        curve.append(float((a[marked] ** 2).sum()))
    return stages, curve, marked


# the 4-plan example from the text: one plan marked
tiny = np.full(4, 0.5)
tiny[3] *= -1
print("4 plans, 1 marked: after the oracle", tiny, "-> after inversion about the average", 2 * tiny.mean() - tiny)
assert np.allclose(2 * tiny.mean() - tiny, [0, 0, 0, 1])

stages6, by_hand_thr6, marked6 = grover_by_hand(cost, 6)
print("32 plans, bar 6 (2 marked): P(marked) after 0, 1, 2, 3 steps =", np.round(by_hand_thr6, 3))
sel = {k: v for k, v in stages6.items() if k in ("start", "step 1: oracle", "step 1: invert", "step 2: oracle", "step 2: invert")}
qff.plots.plot_amplitude_stages(sel, highlight=marked6, labels=("not marked", "marked"),
                                suptitle="bar 6: two plans are marked (magenta). The oracle flips their sign, the inversion lifts them")'''

S3_BYHAND_TRY = callout("try", """Change the bar `6` to `5` in the last cell (so only one plan, `01110`, is marked) and rerun it with `steps=5`. After the oracle the marked amplitude is slightly negative; after the inversion it is much bigger than the others. **Predict** which step gives the highest P(marked) before you look at the printed curve (hint: the next section gives a formula).""")

S3_ROT_MD = r"""### 3.4 · Why it is a rotation, and how many steps are best

Let $N=32$ be the number of plans and $M$ the number of marked ones. Group the amplitudes into two classes, marked and not marked. Initially $P(\text{marked})=M/N$; write this as $\sin^2\theta=M/N$. Each Grover step $G=DO$ **rotates** the state by the angle $2\theta$ in the plane of the two classes, towards "all marked". After $k$ steps the state sits at angle $(2k+1)\theta$, so

$$P(\text{marked after }k\text{ steps})=\sin^2\bigl((2k+1)\theta\bigr),\qquad \sin\theta=\sqrt{M/N} \tag{7}$$

Two things follow. First, the probability **rises, peaks and falls again**, because rotating past "all marked" turns the state back. Second, the number of steps that gets closest to the peak is about

$$k^*\approx\frac\pi{4\theta}-\frac12\ \approx\ \frac\pi4\sqrt{\frac NM}\quad(\text{for small }M/N) \tag{8}$$

For the bar 6, $M=2$ plans are marked, $\sin\theta=\sqrt{2/32}=\frac14$, $\theta\approx0.253$: step 1 reaches $\sin^2(3\theta)=0.473$, step 3 reaches $\sin^2(7\theta)=0.961$ (the best), and step 4 falls back to $0.582$. For the bar 5, $M=1$: $\sin\theta=\frac1{\sqrt{32}}$, $\theta\approx0.178$, and the best is step 4 ($\sin^2(9\theta)\approx0.999$). **The best number of steps depends on $M$, which the search does not know.** That is why Grover *adaptive* search draws the number of steps at random (section 3.5). Compared with a classical search that checks about $N/2$ plans on average, about $\sqrt N$ steps is the **quadratic speed-up**: 4 steps instead of 32 looks small here, but $10^6$ steps instead of $10^{12}$ does not.

""" + arabic_table([("Rotation", "الدوران", "I"), ("Quadratic speed-up", "تسريع تربيعي", "U"), ("Number of steps", "عدد الخطوات", "I")])

S3_GEOM = '''# Equations (7) and (8) as a picture: each Grover step turns the state by 2*theta.
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
qff.plots.plot_grover_geometry(np.arcsin(np.sqrt(2 / 32)), steps=3, ax=axes[0])
axes[0].set_title("2 of 32 marked: theta = 14.5 deg; best at step 3", fontsize=8)
qff.plots.plot_grover_geometry(np.arcsin(np.sqrt(1 / 32)), steps=4, ax=axes[1])
axes[1].set_title("1 of 32 marked: theta = 10.2 deg; best at step 4", fontsize=8)
plt.tight_layout()'''

T2_INTRO = r"""### T2 · Core step: the oracle and the Grover step in Qiskit · التمرين ٢

**What you are about to build, and why.** The numpy cell above showed what a Grover step does to the amplitudes. In T2 you build the same thing with Qiskit gates, as a function `p_marked_curve(threshold)`: it builds the oracle (equation 4) as a `DiagonalGate`, the full step $G=DO$ (equation 6) with Qiskit's `grover_operator`, starts from $H^{\otimes5}$ (equation 3), applies $G$ zero, one, two, … times, and records $P(\text{marked})$ after each. You will then see that Qiskit's circuit gives the same numbers as your pencil-and-paper arithmetic and as equation (7).

**The formulas you implement:** (3), (4), (6), and $P(\text{marked})=\sum_{k\in\text{marked}}|a_k|^2$ (read from `Statevector(qc).probabilities()`).

""" + hints(
    r"""Break it into pieces. (1) Which positions are marked? (2) The oracle is a circuit on 5 qubits with one `DiagonalGate`, which takes a list of 32 numbers. (3) Qiskit has a ready-made function that turns an oracle circuit into the full Grover step. (4) To apply it k times, start from the equal mixture and repeat.""",
    r"""Inside `p_marked_curve`: find the marked positions; build the oracle circuit with one `DiagonalGate`; get the step with `grover_operator`; then for each number of steps from 0 to `max_steps` build a circuit (Hadamard on every qubit, then the step applied that many times with `compose`), take the probabilities of its `Statevector`, and add up the marked ones. Return the list and the step.""",
    r"""```python
def p_marked_curve(threshold, max_steps=4):
    marked = ______            # the marked positions
    oracle = ______            # a 5-qubit circuit with the diagonal gate of equation (4)
    G = ______                 # the full Grover step
    curve = []
    for steps in range(max_steps + 1):
        qc = ______            # equal mixture, then the step applied `steps` times
        curve.append(______)   # probability of the marked positions, as a float
    return curve, G
```
Self-check targets: with the bar at 6, P(marked) starts at 0.0625 and is best after 3 steps (about 0.96).""")

S3_T2_CHECK_MD = """**Check against pencil-and-paper arithmetic and equation (7).** The cell below compares your Qiskit curves with the numpy arithmetic of 3.3 and with $\\sin^2((2k+1)\\theta)$, for both bars, and then plots them."""

S3_T2_CHECK = '''# Check: Qiskit's curves = the numpy arithmetic = equation (7).
for bar, qiskit_curve in ((6, p_marked), (5, p_marked_thr5)):
    _, hand_curve, marked_k = grover_by_hand(cost, bar, steps=4)
    theta = np.arcsin(np.sqrt(len(marked_k) / 32))
    formula = [np.sin((2 * k + 1) * theta) ** 2 for k in range(5)]
    assert np.allclose(qiskit_curve, hand_curve) and np.allclose(qiskit_curve, formula), f"bar {bar}: the three curves should agree"
    print(f"bar {bar}: {len(marked_k)} plan(s) marked, theta = {np.degrees(theta):.1f} deg | Qiskit {np.round(qiskit_curve, 3)} | formula (7) {np.round(formula, 3)}")
print("OK: Qiskit, pencil and paper, and equation (7) all give the same numbers.")
fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, curve, label, M in ((axes[0], p_marked, "bar 6: 2 plans marked", 2), (axes[1], p_marked_thr5, "bar 5: 1 plan marked", 1)):
    fine = np.linspace(0, 4, 200)
    ax.plot(fine, np.sin((2 * fine + 1) * np.arcsin(np.sqrt(M / 32))) ** 2, color="#C6C6C6", label="equation (7), any k")
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

When the bar is low, few plans are marked ($M$ small) and many steps are best; when the bar is high, many plans are marked and few steps are best (equation 8). Because $M$ is unknown, the number of steps is drawn at random each round; this idea of searching with an unknown number of marked items goes back to Boyer, Brassard, Høyer and Tapp (1998), and its use for finding a minimum to Dürr and Høyer (1996). Qiskit's `GroverOptimizer` follows the same pattern for polynomial binary problems (it cites Gilliam, Woerner and Gonciulea). Each round that finds a cheaper plan lowers the bar, so the search ends when *no plan is cheaper*: the best plan.

**The expected cost of the idea.** A classical program checks the 32 plans one by one: 32 lookups, microseconds. The loop above needs *several* rounds, each of a few Grover steps, each step an oracle that computes the cost inside the circuit. On 32 plans this is **slower** than brute force; the speed-up is about $\sqrt N$ per round, and it only pays when $N$ is astronomically large and the hardware is error-corrected.

""" + arabic_table([("Grover adaptive search", "البحث التكيفي بخوارزمية جروفر", "I"), ("Threshold (the bar)", "العتبة", "I"), ("Random number of steps", "عدد عشوائي من الخطوات", "I"),
                    ("Optimization", "الأمثَلة (التحسين)", "I"), ("Value register", "سجل القيمة", "I")])

S3_GAS_MD = """**The real thing: `GroverOptimizer`.** Qiskit's `GroverOptimizer` runs the loop above. It first turns the constraint into the penalty of equation (2) (`penalty=4`). Then it builds the oracle *without* peeking: it computes the cost of every plan inside the circuit, in a **value register** of binary digits, subtracts the bar, and marks the plans whose result is negative. That register is why the circuit has 12 qubits, not 5. The next section explains how it works and how it fails.

Three things to know:
- `num_value_qubits` is the width of the value register. It must hold the signed range of (cost − bar). **Too few and the answers are silently wrong.**
- `num_iterations` is the number of *consecutive rounds without improvement* before it stops. It is not the number of Grover steps.
- Costs and the penalty must be integers, and the penalty must be large enough (see 3.1: with `penalty=1` an illegal two-slot plan ties the best legal plan, so the search may return it). Both seeds matter: `algorithm_globals.random_seed` and the sampler's `seed`.

We wrap it in one helper, `solve_gas`, and a tiny recording sampler so we can look at the circuits it really runs."""

S3_REG_MD = r"""### 3.6 · The value register: why 7 extra qubits, and how it fails

The oracle must decide "is $\text{cost}(x)<y$?" for all plans *at once*, inside the circuit. It does this with binary arithmetic on extra qubits, the **value register**: it computes the difference $\text{cost}(x)-y$ in the register and looks at the **sign bit** (the top qubit): 1 means negative, i.e. marked. Negative numbers are stored in **two's complement**: with $v$ bits the register can hold the whole numbers

$$-2^{\,v-1},\ \dots,\ -1,\ 0,\ 1,\ \dots,\ 2^{\,v-1}-1 \tag{10}$$

and anything outside this range **wraps around**. A tiny example, $v=3$: the range is $-4..3$, the bit patterns $100,101,110,111$ mean $-4,-3,-2,-1$, so the value $+5$ (which does not fit) is stored as $101$ and read as $-3$. A *positive* difference has turned into a *negative* one: an expensive plan looks cheap, and the oracle **marks the wrong plan**.

For our problem the penalised costs run from 3 to 36, and the bar $y$ is always one of the costs, so $\text{cost}-y$ can be as low as $3-36=-33$ and as high as $36-3=+33$. Equation (10) says 6 bits cover only $-32..31$ (**one short on each side**); 7 bits cover $-64..63$ (enough). This is a simplified picture of what Qiskit's oracle does (its arithmetic circuit is more elaborate), but the overflow rule is the same. The cell below checks it and shows the failure: with 6 bits and a bar of 3 (the optimum already found), the *worst* plan (`00000`, cost 36, difference $+33$) wraps to $-31$ and is marked, although it is the most expensive plan of all.

""" + callout("watch", r"""**Silent failure.** With too few value qubits the routine still returns an answer, often reports success, and the plan can be wrong. Nothing warns you. **Always check a quantum optimiser's answer against something you can compute classically.**""") + "\n\n" + arabic_table([
    ("Value register", "سجل القيمة", "I"), ("Two's complement", "المتمم الثنائي", "I"), ("Sign bit", "بت الإشارة", "I"), ("Overflow", "فيض (تجاوز المدى)", "I"),
    ("Qubit", "كيوبت", "G"), ("Binary number", "عدد ثنائي", "I")])

S3_REG_CODE = '''# Equation (10): what a v-bit two's complement register can hold, and what overflow does to the oracle.
def wrap(value, bits):
    """The number a `bits`-bit two's complement register shows when asked to hold `value`."""
    return (value + 2 ** (bits - 1)) % 2 ** bits - 2 ** (bits - 1)


print("3-bit example: the value 5 is read as", wrap(5, 3), "| the value -4 is read as", wrap(-4, 3), "| the value 3 is read as", wrap(3, 3))
lo_needed, hi_needed = min(cost) - max(cost), max(cost) - min(cost)               # range of (cost - bar) when the bar is one of the costs
print(f"penalised costs run from {min(cost)} to {max(cost)}, so cost - bar ranges over [{lo_needed}, {hi_needed}]")
print(f"{'value qubits':>13}{'register holds':>20}{'enough?':>10}")
for v in (5, 6, 7, 8):
    lo, hi = -2 ** (v - 1), 2 ** (v - 1) - 1
    print(f"{v:>13}{f'{lo} .. {hi}':>20}{str(lo <= lo_needed and hi_needed <= hi):>10}")


def bits_needed(costs):
    """Smallest register width that holds every cost - bar, with the bar one of the costs (equation 10)."""
    lo, hi = min(costs) - max(costs), max(costs) - min(costs)
    v = 1
    while not (-2 ** (v - 1) <= lo and hi <= 2 ** (v - 1) - 1):
        v += 1
    return v


assert bits_needed(cost) == 7
bar = 3
for v in (6, 7):
    wrongly_marked = [(plan(index_to_x(k, DECISION_QUBITS)), cost[k]) for k in range(32) if cost[k] >= bar and wrap(cost[k] - bar, v) < 0]
    print(f"bar {bar}, {v} value qubits: plans marked although they are NOT cheaper than the bar: {wrongly_marked or 'none'}")
fig, ax = plt.subplots(figsize=(7, 2.6))
for i, v in enumerate((5, 6, 7, 8)):
    ax.barh(i, 2 ** v - 1, left=-2 ** (v - 1), color="#0F62FE" if (-2 ** (v - 1) <= lo_needed and hi_needed <= 2 ** (v - 1) - 1) else "#FA4D56")
ax.axvspan(lo_needed, hi_needed, color="#24A148", alpha=0.25, label="range needed: cost - bar")
ax.set_yticks(range(4)); ax.set_yticklabels([f"{v} value qubits" for v in (5, 6, 7, 8)]); ax.set_xlabel("whole numbers the register can hold")
ax.legend(fontsize=8, loc="lower right"); plt.tight_layout()'''

S3_REG_TRY = callout("try", """Change `bar = 3` to `bar = 4`, `6` or `10` and rerun the last cell. For which bars does the 6-qubit register mislabel a plan? Why does the mislabelling happen only for the *very worst* plans (cost far above the bar) and never for the cheap ones? Predict the largest bar for which `00000` is still wrongly marked.""")

T3_INTRO = r"""### T3 · Stretch: the width of the value register, and a new tariff · التمرين ٣

**What you are about to build, and why.** The failure of a narrow value register is the most important lesson of this notebook, and T3 lets you see it yourself. You run Grover adaptive search with 6, 7 and 8 value qubits on the same instance (same seed), and explain in one sentence *why* the width matters, using equation (10) and the range of $\text{cost}-y$. Then you change the tariff so that the three cheapest slots are *not* the best plan, solve the new instance classically first, then with the greedy rule, then with Grover adaptive search.

**The formulas you use:** the register holds $-2^{v-1}..2^{v-1}-1$ (equation 10); the cost is equation (2) with the new prices; the greedy rule runs in the three cheapest slots.

""" + hints(
    r"""Look at the table in 3.6: which widths hold the range -33 to 33 and which do not? You do not need new maths: you need three calls of the helper `solve_gas(problem, value_qubits, seed)`. Remember `plan(...)` turns a list of flags into a string. For the new tariff, `desalination_qp(tariff)` builds the problem for any list of five prices.""",
    r"""Structure: for 6, 7 and 8 value qubits, solve with seed 1 and store the plan string and the cost; write one sentence using equation (10); for tariff [2, 3, 1, 3, 2] build the problem; solve it classically first; build the greedy plan (the three cheapest slots) and evaluate it; then solve with `solve_gas` using a register wide enough (the width of the main example, `VALUE_QUBITS`, is still enough here; to check it yourself, `bits_needed(costs)` takes a *list of costs* and must be given the *penalised* costs of the NEW problem, computed as in T1 from the converted problem, not the plain objective values in the table that `brute_force` returns).""",
    r"""```python
wide = {}
for nv in (6, 7, 8):
    ______                   # solve with nv value qubits; wide[nv] = (plan string, cost)
explanation = "..."          # one sentence, using the range of cost - bar
tariff2 = [2, 3, 1, 3, 2]
qp2 = ______                 # the new instance
x2, v2, _ = ______           # classical answer first
greedy2 = ______             # flags of the three cheapest slots
greedy_v2 = ______           # its cost
r2 = ______                  # Grover adaptive search with a wide enough register
```""")


T1 = """cost = [int(round(qubo.objective.evaluate(index_to_x(k, DECISION_QUBITS)))) for k in range(2 ** DECISION_QUBITS)]
feasible_plans = sorted(((plan(index_to_x(k, DECISION_QUBITS)), cost[k]) for k in range(32) if sum(index_to_x(k, DECISION_QUBITS)) == 3), key=lambda p: p[1])
marked_thr6 = [plan(index_to_x(k, DECISION_QUBITS)) for k in range(32) if cost[k] < 6]"""
T1_STARTER = """# Goal: `cost` holds one integer per position k = 0..31 (the QUBO value, penalty included; bit t of k is slot t);
# `feasible_plans` holds (plan string like "01110", cost) for the plans with exactly 3 slots, cheapest first;
# `marked_thr6` holds the plan strings with cost below 6.
cost = []
feasible_plans = []
marked_thr6 = []
raise NotImplementedError("T1: fill the cost table, the feasible plans and the marked plans")"""
T1_CHECK = """assert len(cost) == 32, "T1: the table needs one cost for each of the 32 plans"
assert min(cost) == 3 and max(cost) == 36, "T1: costs should run from 3 to 36; did you use the QUBO (with the penalty)?"
assert len(feasible_plans) == 10, "T1: choosing 3 slots out of 5 gives 10 feasible plans"
assert feasible_plans[0] == ("01110", 3), "T1: the cheapest feasible plan is 01110 at cost 3"
assert sorted(marked_thr6) == sorted(["01110", "11100"]), "T1: two plans are cheaper than 6: 01110 and 11100"
print("feasible plans:", feasible_plans)
print("marked for threshold 6:", marked_thr6)"""

T2 = """def p_marked_curve(threshold, max_steps=4):
    \"\"\"P(marked) after 0..max_steps Grover steps, where marked means cost < threshold.\"\"\"
    marked = [k for k in range(32) if cost[k] < threshold]
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

p_marked, G6 = p_marked_curve(6)
p_marked_thr5, G5 = p_marked_curve(5)
best_steps = int(np.argmax(p_marked))"""
T2_STARTER = """def p_marked_curve(threshold, max_steps=4):
    # Goal: p_marked_curve(threshold, max_steps=4) returns (the list of P(marked) after 0..max_steps Grover steps, the Grover step G),
    # where "marked" means cost < threshold, using the oracle (equation 4) and the step (equation 6) of section 3.
    raise NotImplementedError("T2: build the phase oracle and the Grover operator")

p_marked, G6 = p_marked_curve(6)
p_marked_thr5, G5 = p_marked_curve(5)
best_steps = int(np.argmax(p_marked))"""
T2_CHECK = """assert abs(p_marked[0] - 2 / 32) < 1e-6, "T2: before any step, 2 of 32 plans are marked, so P = 2/32 = 0.0625"
assert abs(p_marked[1] - 0.473) < 0.01, "T2: after one step P should be about 0.473"
assert best_steps == 3 and p_marked[3] > 0.9, "T2: with 2 marked plans out of 32 the best number of steps is 3 (P about 0.96)"
assert p_marked[4] < p_marked[3], "T2: one step too many and the probability drops again"
print("threshold 6:", np.round(p_marked, 3), "-> best steps:", best_steps)
print("threshold 5:", np.round(p_marked_thr5, 3), "-> best steps:", int(np.argmax(p_marked_thr5)))"""

S3_PLOT = """fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
for ax, curve, label in ((axes[0], p_marked, "threshold 6: 2 plans marked"), (axes[1], p_marked_thr5, "threshold 5: 1 plan marked")):
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

def solve_gas(problem, value_qubits, seed, iterations=ITERATIONS, sampler=None):
    \"\"\"One Grover adaptive search run on a constrained problem. Sets BOTH seeds, then calls GroverOptimizer.solve.\"\"\"
    algorithm_globals.random_seed = seed
    sampler = sampler or StatevectorSampler(seed=seed)
    return GroverOptimizer(num_value_qubits=value_qubits, num_iterations=iterations, sampler=sampler, penalty=PENALTY).solve(problem)

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

One run can be lucky. We repeat for seeds 1 to 5 with a 7-qubit value register, then for seeds 1 to 3 with only **6**. The penalised costs run from 3 to 36, and the oracle works with (cost − threshold), whose signed range needs 7 bits. Six bits overflow."""

S3_REL_CODE = """# Takes about 1 to 1.5 minutes in total (8 solves; the 6-qubit solves are slow because they wander for 50 to 130 circuits).
# We use few seeds to keep the notebook fast. Both seeds are set inside solve_gas.
def reliability(value_qubits, seeds):
    \"\"\"Run GAS once per seed; return the list of (seed, x as string, cost, status).\"\"\"
    rows = []
    for s in seeds:
        r = solve_gas(qp, value_qubits, s)
        rows.append((s, plan([int(v) for v in r.x]), r.fval, str(r.status).split(".")[-1]))
    return rows

gas_rows = {}
for nv, seeds in ((7, range(1, 6)), (6, range(1, 4))):
    gas_rows[nv] = reliability(nv, seeds)
    hits = sum(1 for _, xs, v, _ in gas_rows[nv] if xs == plan(classical_x) and v == classical_value)
    print(f"{nv} value qubits: optimum found in {hits} of {len(gas_rows[nv])} seeds")
    print("   ", [(s, xs, int(v), st) for s, xs, v, st in gas_rows[nv]])
gas_hits_7bits = sum(1 for _, xs, v, _ in gas_rows[7] if xs == plan(classical_x) and v == classical_value)
gas_hits_6bits = sum(1 for _, xs, v, _ in gas_rows[6] if xs == plan(classical_x) and v == classical_value)"""

S3_REL_NOTE = """With 6 value qubits the routine returns a wrong or even infeasible plan for most seeds; sometimes it reports `SUCCESS` for a plan that is not the cheapest. Nothing warns you. **Always check a quantum optimiser's answer against something you can compute.**"""

T3 = """# Part 1: seed 1 for all three widths (the 6- and 7-qubit rows are already in the reliability table above; only 8 is new).
wide = {nv: (xs, v) for nv in (6, 7) for s, xs, v, _ in gas_rows[nv] if s == 1}
r8 = solve_gas(qp, 8, 1)
wide[8] = (plan(r8.x), r8.fval)
for nv in (6, 7, 8):
    print(f"{nv} value qubits, seed 1: r = {wide[nv][0]}, cost {wide[nv][1]:g}")
# Why does the width matter?  The oracle subtracts the threshold: with penalised costs up to 36 the range of (cost - threshold)
# is about -33 .. +33, which needs 7 bits in two's complement (6 bits cover only -32 .. 31).  Write one sentence below.
explanation = "Penalised costs reach 36, so cost - threshold spans about -33 to 33; 6 bits (-32..31) can overflow and flip signs."

# Part 2: a tariff where the three cheapest slots are not the best plan.  Classical first, then greedy, then Grover.
tariff2 = [2, 3, 1, 3, 2]
qp2 = desalination_qp(tariff2)
x2, v2, _ = qff.classical.brute_force(qp2)
slots2 = sorted(range(5), key=lambda t: (tariff2[t], t))[:3]
greedy2 = [1 if t in slots2 else 0 for t in range(5)]
greedy_v2 = qp2.objective.evaluate(greedy2)
r2 = solve_gas(qp2, VALUE_QUBITS, SEED)
print(f"tariff {tariff2}: brute force {plan(x2)} cost {v2:g} | greedy {plan(greedy2)} cost {greedy_v2:g} | Grover {plan(r2.x)} cost {r2.fval:g}")"""
T3_STARTER = """# Goal: wide[nv] = (plan string, cost) from solve_gas on the original instance with nv = 6, 7, 8 value qubits (seed 1);
# explanation = one sentence on why the value register needs 7 bits here; then for tariff2 = [2, 3, 1, 3, 2]: the new problem qp2,
# its classical optimum (x2, v2), the greedy plan's cost greedy_v2, and the Grover adaptive result r2.
wide = {}
explanation = ""
x2, v2, greedy_v2, r2 = None, None, None, None
raise NotImplementedError("T3: try 6, 7 and 8 value qubits, then change the tariff")"""
T3_CHECK = """assert set(wide) == {6, 7, 8}, "T3: run 6, 7 and 8 value qubits"
assert len(explanation.strip()) > 20, "T3: write one sentence about the value register"
assert v2 == qp2.objective.evaluate(x2), "T3: the classical answer for the new tariff should be the best plan"
assert greedy_v2 > v2, "T3: with this tariff greedy (three cheapest slots) should be worse than the optimum"
assert r2.fval == v2, "T3: with enough value qubits GAS should match the classical optimum" """

S3_NOTE_SOL = """[solution-only] Instructor notes. (1) The 8-qubit solve is a 13-qubit simulation. (2) In the reliability table the 6-qubit rows are only seeds 1 to 3 (measured: 1 of 3 correct) to keep the notebook near two minutes; a full ten-seed run at num_iterations=8, measured separately, found the optimum in 4 of 10 with 6 value qubits and 10 of 10 with 7. (3) `num_iterations` is 8 here rather than 15 to save time; with 15 the 7-qubit solves took up to 12 s each. (4) For the new tariff the plans are: brute force 11100 at cost 4 (00111 ties at 4), greedy 10101 at cost 5; do not hard-code them in a team's answer. (5) The plan's suggested tariff [1, 5, 1, 5, 1] does not work: greedy and optimum tie at 3."""

S4_RESULTS = """# Results table: classical baselines first.
rows = [
    ("brute force", plan(classical_x), classical_value, f"{2 ** DECISION_QUBITS} states", f"{classical_seconds * 1000:.2f} ms"),
    ("greedy (three cheapest slots)", plan(greedy_x), greedy_value, "5 lookups", "-"),
    (f"Grover adaptive, {VALUE_QUBITS} value qubits", plan([int(v) for v in gas_result.x]), gas_result.fval,
     f"{gas_circuits} circuits ({QUBITS} qubits)", f"{gas_seconds:.1f} s"),
]
print(f"{'method':<40}{'plan':<8}{'cost':<7}{'states / circuits':<26}{'time'}")
for method, plan_str, value, work, secs in rows:
    print(f"{method:<40}{plan_str:<8}{value:<7g}{work:<26}{secs}")
print(f"\\nReliability: {gas_hits_7bits}/5 seeds with 7 value qubits, {gas_hits_6bits}/3 seeds with 6.")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
colors = ["#0F62FE" if x == 1 else "#C6C6C6" for x in classical_x]
axes[0].bar(range(5), TARIFF, color=colors)
axes[0].set_xlabel("time slot"); axes[0].set_ylabel("price"); axes[0].set_title("Tariff; blue = slots the best plan runs in")
axes[1].bar(["6 value qubits", "7 value qubits"], [gas_hits_6bits / 3, gas_hits_7bits / 5], color=["#FA4D56", "#0F62FE"])
axes[1].set_ylim(0, 1); axes[1].set_ylabel("share of seeds that found the optimum"); axes[1].set_title("Width of the value register")
plt.tight_layout(); plt.show()"""

S4_SHOWS = """**What this toy shows.** The same Grover loop as notebook 5 solves a different kind of rule ("run exactly three times") once the rule is turned into a penalty, and the search needs no structure in the problem. With a wide enough value register it finds the optimum on 32 plans, and with too narrow a register it fails silently."""

S4_NOTSHOW = callout("watch", """**What this toy does not show.** There are only 32 plans, and a loop over them takes microseconds; the quantum run needed 12 simulated qubits and several circuit runs. A real plant has many trains, membranes, tanks, energy contracts and hourly prices over weeks, and is scheduled today with classical optimisation. Grover's speed-up is quadratic at best and applies to unstructured search on error-corrected hardware, while classical solvers exploit structure and usually do far better than brute force. Nothing here is a demonstration of quantum advantage.""")

S4_PAPER = """**Paper versus our route.** The WEF/Aramco paper lists "optimize desalination and water treatment" as a use case (65% in the water survey, Figure 8, p. 30). It gives no case study and no method. **The formulation in this notebook and the choice of Grover adaptive search are ours: a teaching choice.** It ran on a simulator."""

ACC = """assert QUBITS == 12 and DECISION_QUBITS == 5
assert classical_x == [0, 1, 1, 1, 0] and classical_value == 3 and n_feasible == 10
assert int(np.argmax(p_marked)) == 3 and p_marked[3] > 0.9
assert list(map(int, gas_result.x)) == [0, 1, 1, 1, 0] and gas_result.fval == 3
assert gas_hits_7bits >= 4 and gas_hits_6bits < 3, f"hits: 7 bits {gas_hits_7bits}, 6 bits {gas_hits_6bits}"
print("acceptance passed:", dict(gas_hits_7bits=gas_hits_7bits, gas_hits_6bits=gas_hits_6bits, circuits=gas_circuits))"""

IBM_NOTE = callout("real", """The preview above is the deepest circuit `GroverOptimizer` ran (5 decision qubits, 7 value qubits and the oracle qubit). A circuit with thousands of two-qubit gates would **not** survive today's noisy devices: every gate adds a little error and the answer would drown in noise. Grover's quadratic speed-up needs error-corrected hardware, and that is why our claim in section 4 stays modest.""")

LOOP_FIX = "from qiskit.circuit.library import DiagonalGate, grover_operator\nfrom qiskit.quantum_info import Statevector"


S3_SYMBOLS = '**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).\n\n| Symbol | Meaning | In the code |\n|---|---|---|\n| $r=(r_0,\\dots,r_4)$ | a plan: the train runs in slot $t$ when $r_t=1$ | `plan(x)`, `index_to_x(k, 5)` |\n| $p_t,\\ f(r)$ | price of slot $t$; objective of equation (1) | `TARIFF`, `f_objective` |\n| $P,\\ \\text{cost}(r)$ | penalty weight; cost with penalty (equation 2) | `PENALTY`, `cost[k]` |\n| $y$ | the bar (threshold) of the search | `threshold` |\n| $N,\\ M$ | number of plans (32), number of marked plans | `len(cost)`, `len(marked)` |\n| $\\lvert s\\rangle,\\ a_x$ | equal mixture, amplitude of plan $x$ | `by_hand`, `Statevector(...).data` |\n| $O,\\ D,\\ G=DO$ | oracle, inversion about the average, one Grover step | `DiagonalGate`, `grover_operator` |\n| $\\theta,\\ k$ | rotation angle, number of steps | `np.arcsin(np.sqrt(M / N))`, `steps` |\n| $v$ | width of the value register (qubits) | `num_value_qubits`, `bits_needed` |'

S3_MATH_ONE = '<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**\n\n**The one line to remember.** The rule "exactly three slots" becomes a penalty, $\\text{cost}(r)=f(r)+P(\\sum_tr_t-3)^2$ (equation 2), and then Grover\'s rotation does the rest: $P(\\text{marked})=\\sin^2((2k+1)\\theta)$ (equation 7).\n\n</div>'

S3_DRAW = '# The Grover step as a circuit (the oracle you built in T2, then the reflection about the equal mixture), for the bar 6.\ntry:\n    display(G6.decompose().draw("mpl", fold=40))\nexcept Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text\n    print(G6.decompose().draw("text", fold=120))'


S3_COST_FIG = """# The 32 costs as a picture: teal bars are below the threshold 6 (the plans the oracle will mark); a black outline means the plan obeys the rule.
fig, ax = plt.subplots(figsize=(9, 2.8))
legal = [sum(index_to_x(k, DECISION_QUBITS)) == 3 for k in range(32)]
ax.bar(range(32), cost, color=["#007D79" if c < 6 else "#8D8D8D" for c in cost], edgecolor=["black" if ok else "none" for ok in legal], linewidth=1.2)
ax.axhline(6, color="#FA4D56", ls="--", lw=1, label="threshold 6: plans below it are marked")
ax.set_xticks(range(32)); ax.set_xticklabels([plan(index_to_x(k, DECISION_QUBITS)) for k in range(32)], rotation=90, fontsize=6)
ax.set_xlabel("plan (one bit per time slot); black outline = exactly 3 slots, the 10 legal plans"); ax.set_ylabel("cost"); ax.legend(fontsize=8)
plt.tight_layout()
assert sum(legal) == 10 and sum(c < 6 for c in cost) == len(marked_thr6) == 2, "10 legal plans; two plans below the threshold"
print("OK: 10 plans obey the rule (outlined) and two plans are below the threshold; the penalty makes every rule-breaking plan expensive.")"""


def main():
    """Write the solution notebook."""
    nb = new(9, "Desalination scheduling: when should the plant run?", use_case="Choose which 3 of 5 time slots a desalination train runs in, at the lowest electricity bill",
             algorithm="Grover adaptive search (Qiskit `GroverOptimizer`)", qubits=12, minutes=3)
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
    nb.md(S3_PENALTY_MD)
    nb.code(LOOP_FIX)
    nb.code(S3_PENALTY_CODE)
    nb.md(TOOLBOX[9][0])
    nb.code(TOOLBOX[9][1])
    nb.md(T1_INTRO)
    nb.task("T1", "fill the 32-row cost table, list the 10 feasible plans, find the plans cheaper than 6", T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.code(S3_CHECK)
    nb.code(S3_COST_FIG)
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
    nb.task("T3", "try 6, 7 and 8 value qubits, explain the failures, then change the tariff", T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_NOTE_SOL, tags=["solution-only"])
    nb.md(day.IBM_MD)
    nb.code(day.IBM_PREVIEW_CODE)
    nb.md(IBM_NOTE)
    nb.md(day.glossary_md([("Grover's algorithm", "خوارزمية جروفر", "G"), "oracle", "amplitude amplification", "optimization", ("threshold", "العتبة", "I"), ("penalty", "العقوبة (penalty)", "I"), ("value register", "سجل القيمة", "I"), ("two's complement", "المتمم الثنائي", "I"),
                           ("desalination", "تحلية المياه", "I"), ("scheduling", "الجدولة", "I")]), tags=["glossary"])
    nb.section(4)
    nb.code(S4_RESULTS)
    nb.md(S4_SHOWS)
    nb.md(S4_NOTSHOW)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="When should a desalination plant run its pumps so the water is made at the lowest electricity cost?",
                counting_fact="5 time slots give 32 plans; 24 hours for 10 trains give 2²⁴⁰ plans",
                single_idea="mark every plan cheaper than your best so far, amplify the marked ones, measure, repeat with a lower bar",
                ideas=["A day-clock interactive: the visitor picks three slots and watches the bill, then races the circuit",
                       "A water-drop character who travels through the day looking for the cheap hours",
                       "A physical demo with cups for amplitudes: flip the marked cups, reflect, repeat"],
                reuse="the tariff bar chart with the best plan highlighted, the P(marked)-versus-steps curve, the 6-versus-7 value-qubit failure table")
    nb.section6(honest="It found the cheapest plan on 32 options using 12 simulated qubits; a loop over 32 options is faster, and nobody has shown a quantum advantage here.",
                question_a="Why is the rule a penalty, and how big must it be?",
                question_b="Is Grover the natural choice when the problem has structure?")
    path = nb.save(os.path.join(ROOT, "solutions", "09_use_desalination_grover_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
