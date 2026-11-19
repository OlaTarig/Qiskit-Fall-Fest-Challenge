"""Build solutions/07_store_hydrogen_qpe_solution.ipynb (notebook 7: hydrogen electrolysis catalyst, quantum phase estimation on H2).

    python tools/build_nb07.py

No variational loop: QPE reads the energy of H2 off the phase of exp(-iHt). The notebook builds the evaluation register,
the controlled-U powers and the inverse QFT, then a Trotterised version and the qubit count for chemical accuracy.
Reusable pieces defined in the notebook: `ham`, `qpe_run` (T2) and the precision table `E_qpe`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1_MD_05 = r"""**The problem.** Splitting water with electricity makes hydrogen, a fuel that can be stored for months. The speed and the cost depend on the catalyst: a surface where hydrogen atoms stick just strongly enough, then leave as H₂. To compare candidate catalysts, chemists need the energies of hydrogen on and off the surface, to a precision of about 1.6 milli-Hartree (mHa, "chemical accuracy"). Energies are in Hartree (Ha) throughout.

**Who faces it.** A hydrogen producer, an electrolyser manufacturer, a materials research lab. The grid operator is the customer: hydrogen is one way to store surplus power.

<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

**What is real today.** The WEF and Aramco paper lists catalyst design for hydrogen electrolysis and storage as a quantum-computing use case (Table 1, p. 9; the storage survey in Figure 7, p. 24 shows 54%). It describes modelling hydrogen interactions with durable, sustainable catalysts as something being *explored*. It reports no result, no hardware size and no method. Today this is done classically, with density functional theory on a computer cluster, which is approximate for the hardest electronic cases.

</div>

**Why a quantum computer might help.** The energy of a molecule is the lowest eigenvalue of a matrix whose size doubles with every orbital, and quantum phase estimation reads an eigenvalue out as a binary number. What is still missing is an error-corrected machine: QPE circuits are long.

| Paper's method | Our teaching route |
|---|---|
| None named (the paper lists the use case, not an algorithm) | QPE on the hydrogen molecule H₂, because it is in the Part 2 lectures and is the method researchers expect to use once fault-tolerant hardware exists |

<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**Be clear about one thing.** H₂ is the *product* a catalyst releases. It is not a catalyst. Our toy computes the energy of the smallest molecule in the problem; a real catalyst surface has many metal atoms and is far beyond this size.

</div>
"""

S2_MD_07 = r"""Our toy: the hydrogen molecule H₂ at its bond length of 0.735 Å (STO-3G, the smallest standard basis). Two electrons in two orbitals become a Hamiltonian on **2 qubits**, shipped as a small file so you need no chemistry software. QPE adds *m* evaluation qubits: with m = 5 the circuit has **7 qubits**.

Energies in Hartree (Ha). Total energy = the eigenvalue of the qubit Hamiltonian + `shift` (the nuclear repulsion, a plain number).
"""

S2_CODE_08 = r"""QUBITS = 7                      # 2 system qubits + m = 5 evaluation qubits (the test runner reads this)
from qiskit import QuantumCircuit
ham = qff.chem.load_hamiltonian(f"{DATA}/h2_parity_0.735.json")
print(ham.name, "|", ham.num_qubits, "system qubits | shift =", round(ham.shift, 6), "Ha")
for label, coeff in zip(ham.op.paulis, ham.op.coeffs):
    print(f"  {coeff.real:+.6f}  {label}")
"""

S2_MD_09 = r"""**Classical baseline (runs first).** A 4×4 matrix: diagonalise it exactly, and compare with Hartree–Fock, the simple one-configuration answer.
"""

S2_CODE_10 = r"""H = ham.op.to_matrix()                                   # 4 x 4
eigenvalues = np.linalg.eigvalsh(H)
e_exact_el = eigenvalues[0]                              # electronic energy (no shift)
e_exact_total = e_exact_el + ham.shift
e_hf_total = ham.e_hf
print("all four eigenvalues  :", np.round(eigenvalues, 6))
print(f"exact ground energy   : {e_exact_el:.6f} Ha electronic, {e_exact_total:.6f} Ha total")
print(f"Hartree-Fock (file)   : {e_hf_total:.6f} Ha total")
print(f"HF is above exact by  : {(e_hf_total - e_exact_total) * 1000:.1f} mHa   (chemical accuracy is 1.6 mHa)")
hf_vec = np.zeros(4); hf_vec[1] = 1                      # basis state |01>: X on system qubit 0
w, v = np.linalg.eigh(H)
overlap = abs(v[:, 0] @ hf_vec) ** 2                     # squared overlap = probability of finding the ground state
print(f"overlap (squared) of the Hartree-Fock start state with the true ground state: {overlap:.4f}")
"""

S3_MD_12 = r"""### 3.0 · The map of this section

<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**QPE in one sentence.** Turn the energy of a molecule into a *phase* (an angle), let a register of qubits record that angle in binary using a trick called phase kickback, and read the angle, hence the energy, off a measurement: each extra qubit adds one binary digit of precision. There is **no knob-tuning**: the circuit is fixed, and the maths is about phases adding up.

</div>

Everything below builds that sentence from things you already know: lists of numbers (vectors), angles and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**, so you can watch the maths run and check it against Qiskit.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | An energy becomes a phase (and why the window matters) | (1), (2) | `phis`, `window` | |
| 3.2 | Phase kickback: the evaluation qubits record the phase | (3) | `state_eval` | |
| 3.3 | The inverse Fourier transform reads the bits; the bit-order trap | (4) | `phase_from_counts` | **T1** |
| 3.4 | QPE on H₂: why the answer jumps, and the start state | (5) | `qpe_run`, check cell | **T2** |
| 3.5 | What hardware would need: Trotter steps | (6) | `circuit_trotter` | |
| 3.6 | How many qubits for chemical accuracy? | (7) | `needed_m` | **T3** |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it.
"""

S3_CODE_13 = r"""qff.plots.plot_flow(
    [("Start state", "close to the ground state", "source"), ("Hadamards", "on m evaluation qubits", "move"), ("Controlled U", "U, U^2, U^4, ...", "store"),
     ("Inverse QFT", "turns phase into bits", "use"), ("Measure", "read m bits", "source")],
    title="Quantum phase estimation: a fixed circuit, no tuning loop");
"""

S3_MD_14 = r"""**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).

| Symbol | Meaning | In the code |
|---|---|---|
| $U,\ \varphi$ | unitary built from $H$; its phase (fraction of a turn) | `U`, `phi` |
| $E,\ E_{\min},\ W$ | energy; bottom and width of the energy window | `E_el`, `E_MIN`, `W` |
| $m,\ M=2^m$ | number of evaluation qubits; number of outcomes | `m`, `2 ** m` |
| $\lvert y\rangle$ | an outcome of the evaluation register | the count strings |
| $A_k,\ P(k)$ | amplitude and probability of outcome $k$ | `phase_from_counts`, `p_circuit` |
| $F_M(\delta)$ | the QPE response curve, $\frac{\sin^2(\pi M\delta)}{M^2\sin^2(\pi\delta)}$ | `F(delta, M)` |
| $c_n,\ \lvert c_n\rvert^2$ | eigenstate $n$ in the start state; its overlap | `weights`, `overlap` |
| $\varepsilon$ | target precision (chemical accuracy, 1.6 mHa) | `needed_m` |
"""

S3_MD_15 = r"""<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**

**The one line to remember.** The energy is read as a phase, $E=E_{\min}+\varphi\,W$ (equation 2), to $m$ binary digits: the grid step is $W/2^m$, so $m\ge\log_2(W/\varepsilon)$ (equation 6). Everything else is about where $\varphi$ falls between grid points.

</div>
"""

S3_MD_16 = r"""### 3.1 · From an energy to a phase

**Eigenstates and phases.** A **unitary** $U$ is a reversible operation (a matrix that rotates states without changing their length). Some special states, its **eigenstates** $|u\rangle$, come back unchanged except for a **phase**, an angle written as $e^{2\pi i\varphi}$ ($e^{i\theta}=\cos\theta+i\sin\theta$ is a point on a circle of radius 1, at angle $\theta$; multiplying an amplitude by it *turns it* by $\theta$ and changes no probability):

$$U\,|u\rangle=e^{2\pi i\varphi}\,|u\rangle,\qquad 0\le\varphi<1 \tag{1}$$

The number $\varphi$ ("a fraction of a full turn") is what **quantum phase estimation (QPE)** measures. To use it for chemistry we build $U$ *from the Hamiltonian* $H$ (the energy operator) so that **the energy sets the phase**. The eigenvalues of $H$ are the possible energies $E$ and its eigenvectors are the states that have them. If all eigenvalues lie in a window $[E_{\min},E_{\min}+W)$:

$$U=e^{\,i\,(H-E_{\min})\,2\pi/W}\qquad\Longrightarrow\qquad \varphi=\frac{E-E_{\min}}{W},\qquad E=E_{\min}+\varphi\,W \tag{2}$$

The window matters because a phase is an *angle*: $\varphi$ and $\varphi+1$ are the same turn. An energy outside the window **wraps around** (aliases) and is read as a different energy. For H₂ all four eigenvalues lie in $[-2.0,0.0)$, so $E_{\min}=-2.0$ and $W=2.0$. Example with the numbers below: the ground-state energy $-1.8573$ Ha gives $\varphi=\frac{-1.8573-(-2)}{2}=0.0714$, a small angle. (The energies are the *electronic* ones, without the constant `shift`.)

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Quantum phase estimation (QPE) | تقدير الطور الكمومي (QPE) | G | Unitary matrix | مصفوفة عكوسة/وحدوية | G |
| Phase | الطور | U | Hamiltonian | مؤثر هاملتون (الهاملتوني) | G |
| Eigenvalue | القيمة الذاتية | G | Energy window | نافذة الطاقة | I |
| Eigenstate | الحالة الذاتية | G | Aliasing (wrap-around) | الالتفاف الطوري | I |
"""

S3_CODE_17 = r"""# Equations (1)-(2) in code: the four energies of H2 as phases on a circle, and the weight Hartree-Fock puts on each.
window_min, window_width = -2.0, 2.0                              # the window [-2, 0) holds all four eigenvalues
phis = (eigenvalues - window_min) / window_width                  # equation (2): phase of each eigenstate
weights = np.abs(v.T @ hf_vec) ** 2                               # how much of each eigenstate is in the Hartree-Fock start state
print(f"{'eigenvalue (Ha)':>16}{'phase':>9}{'weight in HF':>14}")
for e, ph, wt in zip(eigenvalues, phis, weights):
    print(f"{e:>16.5f}{ph:>9.4f}{wt:>14.4f}")
assert np.all((phis >= 0) & (phis < 1)), "an eigenvalue is outside the window: it would alias"
fig = plt.figure(figsize=(4.2, 4.2))
ax = fig.add_subplot(111, projection="polar")
for k in range(32):
    ax.plot([2 * np.pi * k / 32] * 2, [0.97, 1.0], color="#C6C6C6", lw=1)               # the grid of a 5-qubit register
for e, ph, wt in zip(eigenvalues, phis, weights):
    ax.annotate("", xy=(2 * np.pi * ph, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#0F62FE" if wt > 0.5 else "#8D8D8D", lw=2))
    ax.text(2 * np.pi * ph, 1.22, f"{e:.2f} Ha", ha="center", fontsize=8)
ax.set_ylim(0, 1.4); ax.set_yticks([]); ax.set_title("energy = angle of a phase (grey ticks: 32-step grid)", fontsize=9, pad=12)
plt.tight_layout()
"""

S3_MD_18 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**The window must contain the spectrum.** Try `h2_parity_0.5.json` (end of section 3): its lowest eigenvalue is $-2.11$, *below* the window, so its phase wraps to $\varphi\approx0.95$ and QPE reports an energy near $-0.1$, a wrong answer with no warning. Moving `E_MIN` down brings it back.

</div>
"""

S3_MD_19 = r"""### 3.2 · Phase kickback: the evaluation register records the phase

Take **one extra qubit**, a *control* qubit, in the equal mixture $\frac1{\sqrt2}(|0\rangle+|1\rangle)$ (what a Hadamard gate makes from $|0\rangle$), and apply $U$ to the system **only when the control is $|1\rangle$**. If the system is in an eigenstate (1), the system is unchanged, but the control picks up the phase, **kicked back** onto it:

$$\tfrac1{\sqrt2}\bigl(|0\rangle+|1\rangle\bigr)|u\rangle\ \longrightarrow\ \tfrac1{\sqrt2}\bigl(|0\rangle+e^{2\pi i\varphi}|1\rangle\bigr)|u\rangle$$

Now use **$m$ control qubits**, the **evaluation register**, and give control qubit $k$ the gate $U^{2^k}$ (that is, $U$ applied $2^k$ times: $U,U^2,U^4,\dots$). Control $k$ then carries the phase $e^{2\pi i\,2^k\varphi}$. Writing a number $y$ in binary as $y=\sum_ky_k2^k$, the $2^m$ amplitudes of the register together are

$$\frac1{\sqrt{2^m}}\sum_{y=0}^{2^m-1}e^{2\pi i\,\varphi\,y}\,|y\rangle \tag{3}$$

A spiral: the amplitude of $|y\rangle$ is a unit arrow at angle $2\pi\varphi y$. The register has a *signature* that depends on $\varphi$: for $\varphi=\frac38$ the arrows turn by $135^\circ$ per step. The next step turns this signature into a readable number.

**A worked example you can check on paper.** One evaluation qubit ($m=1$), $U=\text{diag}(1,-1)$ acting on $|1\rangle$, so $e^{2\pi i\varphi}=-1$, $\varphi=\tfrac12$. The register is $\frac1{\sqrt2}(|0\rangle+e^{i\pi}|1\rangle)=\frac1{\sqrt2}(|0\rangle-|1\rangle)$. A Hadamard gate turns this into $|1\rangle$, so the single measured bit is $1$, which reads as $\varphi=0.1_2=\frac12$ ✓. The same recipe with more qubits gives more binary digits.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Phase kickback | ارتداد الطور | I | Hadamard gate | بوابة هادامارد | G |
| Evaluation register | سجل التقدير | I | Superposition | التراكب | G |
| Control qubit | كيوبت التحكم | I | Controlled gate | بوابة متحكَّم بها | I |
"""

S3_MD_20 = r"""### 3.3 · The inverse Fourier transform reads the bits

The state (3) is a spiral. The **quantum Fourier transform** (QFT) is the quantum version of the discrete Fourier transform, the tool that finds the *frequency* of a wave; the **inverse QFT** does the reverse, and here it turns the spiral of speed $\varphi$ into a single outcome $|k\rangle$ when $\varphi=k/2^m$ exactly. The amplitude of outcome $k$ after the inverse QFT is

$$A_k=\frac1{2^m}\sum_{y=0}^{2^m-1}e^{2\pi i\,(\varphi-k/2^m)\,y} \tag{4}$$

If $\varphi=k_0/2^m$, the sum for $k=k_0$ is $\frac1{2^m}\cdot2^m=1$ (all arrows point the same way) and for every other $k$ it is 0 (arrows cancel in a full circle): **the register reads $k_0$ with certainty, and $\varphi=k_0/2^m$ is its binary fraction with $m$ digits.** Check with $\varphi=\frac38=0.011_2$ and $m=3$: outcome $k=3=011_2$ with probability 1. The cell below does this with numpy.

<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**Bit order: the trap.** Qiskit's `phase_estimation` leaves out the final qubit swaps of the QFT. If evaluation qubit $i$ is measured into classical bit $i$, the printed count string has qubit 0 on the **right**, and the bits of $k$ come out **reversed**: for $\varphi=\frac38$ ($k=011_2$) you will see `'110'`. Always read the string reversed, or use `qff.bits.phase_from_counts`. Task T1 makes you meet this trap on a case where you know the answer.

</div>

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Quantum Fourier transform | تحويل فورييه الكمومي | G | Measurement | القياس | G |
| Inverse QFT | تحويل فورييه الكمومي العكسي | I | Probability | الاحتمال | I |
| Binary fraction | كسر ثنائي | I | Bit order | ترتيب البتات | I |
"""

S3_CODE_21 = r"""# Equations (3)-(4) with plain numpy, no quantum library: phi = 3/8 with m = 3 evaluation qubits.
m_demo, phi_demo = 3, 3 / 8
M_demo = 2 ** m_demo
y = np.arange(M_demo)
register = np.exp(2j * np.pi * phi_demo * y) / np.sqrt(M_demo)                        # equation (3): a spiral of unit arrows
A = np.array([(np.exp(2j * np.pi * (phi_demo - k / M_demo) * y).sum()) / M_demo for k in range(M_demo)])   # equation (4)
print("outcome probabilities |A_k|^2 for k = 0..7:", np.round(np.abs(A) ** 2, 6))
assert np.isclose(abs(A[3]) ** 2, 1.0), "phi = 3/8 = 0.011 in binary: the register must read k = 3 with certainty"
fig = plt.figure(figsize=(8, 3.4))
ax1 = fig.add_subplot(1, 2, 1, projection="polar")
for yy in y:
    ax1.annotate("", xy=(np.angle(register[yy]), 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#0F62FE", lw=1.5))
    ax1.text(np.angle(register[yy]), 1.2, f"y={yy}", ha="center", fontsize=7)
ax1.set_ylim(0, 1.4); ax1.set_yticks([]); ax1.set_title("equation (3): the spiral for phi = 3/8", fontsize=9, pad=12)
ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(range(M_demo), np.abs(A) ** 2, color="#0F62FE")
ax2.set_xlabel("outcome k"); ax2.set_ylabel("probability"); ax2.set_title("equation (4): all weight on k = 3", fontsize=9)
plt.tight_layout()
"""

S3_MD_22 = r"""<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**Each qubit is one more binary digit.** With $m$ evaluation qubits the register can show $2^m$ different readings, like a ruler with $2^m$ marks laid over one full turn. One more qubit doubles the number of marks, which adds one binary digit to $\varphi$, just as one more decimal digit divides a ruler's smallest division by ten. Translated to energy, the window $W$ is cut into $2^m$ steps of size $W/2^m$.

</div>
"""

S3_CODE_23 = r"""from qiskit.circuit.library import phase_estimation, PhaseGate, UnitaryGate
m_w = 3
warm = QuantumCircuit(m_w + 1, m_w)
warm.x(m_w)                                              # system qubit in |1>, the eigenstate of the phase gate
warm.compose(phase_estimation(m_w, PhaseGate(2 * np.pi * 0.375)), inplace=True)
warm.measure(range(m_w), range(m_w))
counts_warm = qff.quantum.sv_sampler(4096).run([warm]).result()[0].data.c.get_counts()
print(counts_warm)
"""

S3_MD_26 = r"""### T1 · Warm-up: read the phase, and meet the bit-order trap · التمرين ١

**What you are about to build, and why.** The gate `PhaseGate(2π·0.375)` multiplies $|1\rangle$ by exactly the phase $e^{2\pi i\cdot3/8}$, so $\varphi=\frac38=0.011_2$, and with 3 evaluation qubits QPE must return that number. The cell above already ran the circuit and printed the counts. In T1 you take the most frequent count string and read it **two ways**: as printed, and reversed (qubit 0 is the rightmost character). Only one of them is $0.375$. This tiny experiment saves hours later: every QPE result in this notebook goes through the same reading step.

**The formula you implement:** read the bits of $k$ in the right order: $\varphi=k/2^m$, with $k$ taken from the *reversed* string.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Look at the printed counts: one string appears (almost) every time. In the numpy example above the answer was k = 3 = 011 in binary. Is the string you see equal to 011, or to something else made of the same digits? Which character of the printed string belongs to qubit 0?

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Pick the most frequent string from the counts. Read it as printed (a binary number divided by 2 to the power of the number of evaluation qubits), then read it reversed (Python can reverse a string with a slice) and compare. Finally call the helper in `qff.bits` that returns the phase and its frequency.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
warmup_raw_string = ______      # the most frequent count string
phase_direct = ______           # as printed
phase_reversed = ______         # reversed
warmup_phase, _ = ______        # the helper from qff.bits
print("as printed:", phase_direct, "| reversed:", phase_reversed, "| helper:", warmup_phase)
```
Test yourself: the printed string is `'110'`; one reading gives 0.75, the other 0.375.

</details>
"""

T1 = r"""warmup_raw_string = max(counts_warm, key=counts_warm.get)
phase_direct = int(warmup_raw_string, 2) / 2 ** m_w               # reading the string as printed
phase_reversed = int(warmup_raw_string[::-1], 2) / 2 ** m_w       # reading it reversed (qubit 0 is the rightmost character)
warmup_phase, _ = qff.bits.phase_from_counts(counts_warm, m_w)    # the helper
print("as printed:", warmup_raw_string, "->", phase_direct, "| reversed ->", phase_reversed, "| helper ->", warmup_phase)
print("The gate phase is 0.375, so the right reading is the reversed one.")
"""

T1_STARTER = r"""# Goal: read the warm-up phase two ways and decide which is right. Keep warmup_raw_string (the most frequent count string),
# phase_direct (read as printed), phase_reversed (read reversed) and warmup_phase (from the helper in qff.bits).
raise NotImplementedError("T1: fill this in")
"""

T1_CHECK = r"""assert warmup_raw_string == "110", "T1: the raw count string should be '110'"
assert abs(phase_reversed - 0.375) < 1e-12, "T1: reversed, 110 should read as 011 = 3/8 = 0.375"
assert abs(phase_direct - 0.375) > 0.1, "T1: reading the string directly should give the wrong phase (0.75)"
assert abs(warmup_phase - 0.375) < 1e-12, "T1: the helper should return 0.375"
print("T1 ok: always reverse the string, or use qff.bits.phase_from_counts")
"""

S3_MD_29 = r"""### 3.4 · QPE on H₂: why the answer jumps, and why the start state matters

**The real thing.** We build $U$ of equation (2) exactly as a $4\times4$ matrix with `scipy.linalg.expm` and wrap it in a gate. The system register (2 qubits) holds the molecule; the evaluation register comes first, so the system's qubit 0 is circuit qubit $m$, and `qc.x(m)` prepares Hartree-Fock. In T2 you assemble this for $m=5$.

**What the result should look like.** In general $\varphi$ is *not* a fraction $k/2^m$; it falls between grid points. Then equation (4) gives, for the amplitude of outcome $k$, a geometric sum, and the probability of reading $k$ is

$$P(k)=F_{M}\!\left(\varphi-\frac kM\right),\qquad F_M(\delta)=\frac{\sin^2(\pi M\delta)}{M^2\sin^2(\pi\delta)},\qquad M=2^m \tag{5}$$

$F_M(0)=1$ (exact fraction) and $F_M$ falls quickly away from 0. For H₂'s ground-state phase $\varphi=0.07136$: with $m=5$ ($M=32$) the nearest grid point is $k=2$ ($0.0625$), $\delta=0.00886$ and $P\approx0.76$ (0.75 once the 1.2% of excited-state weight below is included); with $m=6$ ($M=64$) the nearest is $k=5$ ($0.0781$), $\delta=-0.00676$ and $P\approx0.52$ (0.51 with the mixture). **That is why the precision ladder jumps**: one more qubit halves the grid step, but if the true phase sits nearly half way between two points, the best outcome only gets about half of the shots (never less than $4/\pi^2\approx0.4$).

**The start state.** Our system starts in Hartree-Fock, not in the exact ground state. Write it as a mix of eigenstates, $\sum_nc_n|E_n\rangle$. The register is a *mixture* of the answers for each eigenstate, weighted by the **overlap** $|c_n|^2$:

$$P(k)=\sum_n|c_n|^2\,F_M\!\left(\varphi_n-\frac kM\right) \tag{5'}$$

For H₂ the ground state carries $|c_0|^2=0.9876$ of Hartree-Fock (computed in section 2), so the right answer is the likely one; for a stretched molecule the overlap is smaller, and the probability spreads over several energies (T3).

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Overlap | التراكب (التداخل الجزئي) بين حالتين | I | Grid step | خطوة الشبكة | I |
| Start state | الحالة الابتدائية | I | Evaluation qubits | كيوبتات التقدير | I |
| Hartree-Fock | هارتري-فوك | I | Ground state | الحالة الأرضية | I |
"""

S3_MD_30 = r"""### T2 · Core step: write the QPE function and read the energy · التمرين ٢

**What you are about to build, and why.** This is the heart of the notebook, and you write it as a *function* because later cells (the precision ladder, the Trotter version, the stretched molecule) call it again and again. `qpe_run(h, m)` assembles the full circuit of the map above for the Hamiltonian `h` with $m$ evaluation qubits: (1) the unitary $U=e^{i(H-E_{\min})2\pi/W}$ as a matrix (equation 2), unless the caller supplies a ready-made gate; (2) the Hartree-Fock preparation on the system qubit; (3) Qiskit's `phase_estimation(m, gate)` (Hadamards, controlled $U^{2^k}$, inverse QFT); (4) the measurement of the $m$ evaluation qubits; then it reads the phase from the counts and converts it to an energy with $E=E_{\min}+\varphi W$. It returns the energy, the frequency of the top outcome, the counts and the circuit.

**The formulas you implement:** (2) for $U$ and for $E$; the circuit has $m+2$ qubits ($m$ evaluation + 2 system). Run it for $m=5$.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Break it down. (1) U: a matrix exponential of the Hamiltonian, scaled as in equation (2); the function should build it itself unless a gate is passed in. (2) The circuit has the evaluation qubits first and the 2 system qubits after them; where is system qubit 0? (3) Qiskit has a ready-made circuit for phase estimation; it needs the number of evaluation qubits and the gate. (4) Measure the evaluation qubits into classical bits.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps inside `qpe_run`: if no gate was given, build `U` with `scipy.linalg.expm` and wrap it in a `UnitaryGate`; create a circuit with m + 2 qubits and m classical bits; flip the system qubit that Hartree-Fock needs (remember the offset m, and that `h.hf_x_qubits` lists it); `compose` the `phase_estimation` circuit into it; measure qubits 0..m-1; run the sampler with `shots` shots and read the counts of the register `c`; convert the phase with the helper you used in T1 and then to an energy with equation (2). Return the four things listed above.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
def qpe_run(h, m, unitary=None, shots=4096):
    if unitary is None:
        unitary = ______   # equation (2), as a gate
    qc = ______            # m + 2 qubits, m classical bits
    ______                 # Hartree-Fock on system qubit 0
    ______                 # the phase estimation circuit (Hadamards, controlled U^(2^k), inverse QFT)
    ______                 # measure the evaluation qubits
    counts = ______        # run the sampler
    phi, freq = ______     # the phase and its frequency, bit order handled
    return ______, freq, counts, qc    # the electronic energy
```
Self-check target: with m = 5 the top outcome gives -1.875 Ha (electronic).

</details>
"""

T2 = r'''import scipy.linalg as sla
E_MIN, W = -2.0, 2.0                                              # the window [E_MIN, E_MIN + W) holds every eigenvalue
m = 5


def qpe_run(h, m, unitary=None, shots=4096):
    """QPE for Hamiltonian h with m evaluation qubits -> (electronic energy, top frequency, counts, circuit)."""
    if unitary is None:                                           # the exact U of equation (2), as a gate
        unitary = UnitaryGate(sla.expm(1j * (h.op.to_matrix() - E_MIN * np.eye(4)) * 2 * np.pi / W), label="U")
    qc = QuantumCircuit(m + 2, m)
    qc.x(m + h.hf_x_qubits[0])                                    # Hartree-Fock on system qubit 0 (circuit qubit m)
    qc.compose(phase_estimation(m, unitary), inplace=True)
    qc.measure(range(m), range(m))
    counts = qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()
    phi, freq = qff.bits.phase_from_counts(counts, m)
    return E_MIN + phi * W, freq, counts, qc


E_el, freq, counts, circuit = qpe_run(ham, m)
phi = (E_el - E_MIN) / W
print(f"m = {m}: phase {phi}, frequency {freq:.2f}, electronic energy {E_el:.4f} Ha, total {E_el + ham.shift:.4f} Ha")
print(f"error versus exact: {(E_el - e_exact_el) * 1000:+.1f} mHa;  resolution W/2^m = {W / 2 ** m * 1000:.1f} mHa")
'''

T2_STARTER = r"""# Goal: qpe_run(h, m, unitary=None, shots=4096) runs quantum phase estimation for the Hamiltonian h with m evaluation
# qubits (m + 2 qubits in all, Hartree-Fock start) and returns (electronic energy, frequency of the top outcome, counts, circuit).
# If `unitary` is given it replaces the exact U (used later for the Trotter version). U comes from equation (2).
E_MIN, W = -2.0, 2.0
m = 5
raise NotImplementedError("T2: write qpe_run and run it for H2 with m = 5; set E_el, counts and circuit")
"""

T2_CHECK = r"""assert circuit.num_qubits == 7, "T2: with m = 5 the circuit should have 7 qubits (5 evaluation + 2 system)"
assert sum(counts.values()) == 4096, "T2: counts should add up to 4096 shots"
assert abs(E_el - (-1.875)) < 1e-9, "T2: with m = 5 the top outcome should give E = -1.875 Ha (electronic)"
print("T2 ok")
"""

S3_MD_33 = r"""**Check equation (5') against the circuit.** The cell below takes the *exact* outcome probabilities of your circuit (no shots) and compares them with the formula $P(k)=\sum_n|c_n|^2F_M(\varphi_n-k/M)$ for all 32 outcomes, then draws both.
"""

S3_CODE_34 = r'''# Check: the exact outcome probabilities of your QPE circuit equal equation (5'), for m = 5 and m = 6.
from qiskit.quantum_info import Statevector


def F(delta, M):
    """Equation (5): sin^2(pi M delta) / (M^2 sin^2(pi delta)), equal to 1 at delta = 0."""
    delta = np.asarray(delta, dtype=float)
    out = np.ones_like(delta)
    nz = np.abs(np.sin(np.pi * delta)) > 1e-12
    out[nz] = np.sin(np.pi * M * delta[nz]) ** 2 / (M ** 2 * np.sin(np.pi * delta[nz]) ** 2)
    return out


def bit_reversed(r, bits):
    """The integer whose binary digits are those of r, written backwards (the QPE bit-order trap)."""
    return int(format(r, f"0{bits}b")[::-1], 2)


def qpe_exact_probabilities(h, mm):
    """Outcome probabilities p[k] of your QPE circuit for m = mm, without shots, indexed by k = phase * 2^m."""
    qc = qpe_run(h, mm)[3].remove_final_measurements(inplace=False)
    raw = Statevector(qc).probabilities(list(range(mm)))                  # probabilities of the evaluation register
    p = np.zeros(2 ** mm)
    for r, pr in enumerate(raw):
        p[bit_reversed(r, mm)] = pr                                         # undo the bit order
    return p


fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
for ax, mm in zip(axes, (5, 6)):
    M = 2 ** mm
    p_circuit = qpe_exact_probabilities(ham, mm)
    p_formula = np.array([(weights * F(phis - k / M, M)).sum() for k in range(M)])           # equation (5')
    assert np.allclose(p_circuit, p_formula, atol=1e-9), "the circuit should follow equation (5')"
    k_top = int(np.argmax(p_circuit))
    print(f"m = {mm}: top outcome k = {k_top} (energy {E_MIN + k_top / M * W:.4f} Ha), probability {p_circuit[k_top]:.3f} from the circuit, {p_formula[k_top]:.3f} from the formula")
    ax.bar(range(M), p_circuit, color="#0F62FE", label="circuit (exact probabilities)")
    ax.plot(range(M), p_formula, "o", color="#D02670", ms=3, label="equation (5')")
    ax.axvline(M * phis[0], color="k", ls=":", label="true phase of the ground state")
    ax.set_xlabel("outcome k"); ax.set_title(f"m = {mm}: grid step {W / M * 1000:.1f} mHa", fontsize=9); ax.legend(fontsize=7)
axes[0].set_ylabel("probability")
plt.tight_layout()
print("OK: equation (5') reproduces the circuit exactly.")
'''

S3_CODE_35 = r"""# The QPE circuit your function builds, drawn for m = 3 so that it stays readable (Hadamards, controlled U^(2^k), inverse QFT, measurement).
try:
    display(qpe_run(ham, 3)[3].draw("mpl", fold=40))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(qpe_run(ham, 3)[3].draw("text", fold=120))
"""

S3_CODE_36 = r"""# Histogram of the evaluation register, x axis relabelled in Hartree (every outcome is reversed first).
probs_el = {}
for s, c in counts.items():
    e = E_MIN + (int(s[::-1], 2) / 2 ** m) * W
    probs_el[f"{e:.3f}"] = probs_el.get(f"{e:.3f}", 0) + c / 4096
order = sorted(probs_el, key=float)
plt.figure(figsize=(8, 3))
plt.bar(order, [probs_el[k] for k in order], color="#0F62FE")
plt.xticks(rotation=70, fontsize=7); plt.ylabel("probability"); plt.xlabel("electronic energy read from the qubits (Ha)")
plt.title(f"m = {m}: exact energy is {e_exact_el:.4f} Ha, grid step {W / 2 ** m:.4f} Ha"); plt.tight_layout(); plt.show()
"""

S3_MD_37 = r"""**Precision ladder.** Each extra evaluation qubit halves the grid step W/2ᵐ. Note the jumps: the true phase falls *between* grid points, so the answer is the nearest point only with some probability (equation 5), and at m = 6 it is only about one shot in two. That is expected, not a bug.
"""

S3_MD_38 = r"""<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Precision | الدقة | I | Binary digit | خانة ثنائية | I |
| Grid point | نقطة الشبكة | I |  |  |  |
"""

S3_CODE_39 = r"""E_qpe, F_qpe = {}, {}
print(f"{'m':>2} {'qubits':>6} {'top freq':>8} {'E (Ha)':>9} {'error (mHa)':>11} {'step W/2^m (mHa)':>17}")
for mm in (4, 5, 6, 7):
    E_qpe[mm], F_qpe[mm] = qpe_run(ham, mm)[:2]
    print(f"{mm:>2} {mm + 2:>6} {F_qpe[mm]:>8.2f} {E_qpe[mm]:>9.4f} {(E_qpe[mm] - e_exact_el) * 1000:>+11.1f} {W / 2 ** mm * 1000:>17.1f}")
# Equation (5) in the table above, as a picture: the top outcome is never further than one grid step from the exact energy.
assert all(abs(E_qpe[mm] - e_exact_el) <= W / 2 ** mm + 1e-9 for mm in E_qpe), "the top outcome should be within one grid step of the exact energy"
ms = sorted(E_qpe)
fig, ax = plt.subplots(figsize=(5.5, 2.8))
ax.bar([str(mm) for mm in ms], [abs(E_qpe[mm] - e_exact_el) * 1000 for mm in ms], color="#0F62FE", label="|error| of the top outcome")
ax.plot([str(mm) for mm in ms], [W / 2 ** mm * 1000 for mm in ms], "o--", color="#FA4D56", label="grid step W / 2^m")
ax.axhline(1.6, color="green", ls=":", label="chemical accuracy (1.6 mHa)")
ax.set_yscale("log"); ax.set_xlabel("evaluation qubits m"); ax.set_ylabel("mHa"); ax.legend(fontsize=7)
plt.tight_layout()
print("OK: for every m the error is within one grid step W / 2^m.")
"""

S3_MD_40 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

**Predict, then run.** Equation (5) lets you forecast a row of the table before you compute it. For $m=7$ ($M=128$): which grid point $k$ is nearest to the ground-state phase $\varphi_0=0.07136$, what is $\delta=\varphi_0-k/M$, and what does $F_M(\delta)$ give? Work it out with a line of numpy, then change `(4, 5, 6, 7)` in the cell above to `(4, 5, 6, 7, 8)` and compare the "top freq" column with your forecast.

</div>
"""

S3_MD_41 = r"""### 3.5 · What hardware would need: Trotter steps

So far $U$ was an exact matrix, which a simulator can handle and a real computer cannot. On hardware $U=e^{-iHt}$ is built from short gate sequences by **Trotterisation**: split $H=\sum_kA_k$ into its Pauli terms and use

$$e^{-i(A_1+A_2+\dots)t}\ \approx\ \Bigl(e^{-iA_1t/r}\,e^{-iA_2t/r}\cdots\Bigr)^{r} \tag{6}$$

The approximation improves as the number of steps $r$ grows (and with a symmetric ordering, "order 2", which is what we use) but costs more gates. Below: one Trotter step of order 2, twice (`reps=2`), as the gate that the evaluation qubits control, then transpiled to basic gates. Look at the CX count: it is the number that decides whether a device can run this.

(`PauliEvolutionGate(op, time=t)` is $e^{-i\,t\,op}$, so we use `time = −2π/W` to match our $U$.)
"""

S3_MD_42 = r"""<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Trotterisation | تقريب تروتر (Trotterisation) | I | Circuit depth | عمق الدائرة | G |
| Trotter step | خطوة تروتر | I | CNOT gate | بوابة النفي المتحكّم بها | G |
| Gate count | عدد البوابات | I |  |  |  |
"""

S3_CODE_43 = r"""from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp
from qiskit import transpile
op_shifted = ham.op - E_MIN * SparsePauliOp("II")
# Equation (6): U is approximated by r = 2 symmetric Trotter steps over the Pauli terms of H.
trotter_gate = PauliEvolutionGate(op_shifted, time=-2 * np.pi / W, synthesis=SuzukiTrotter(order=2, reps=2))
E_t, f_t, cnt_t, circuit_trotter = qpe_run(ham, m, unitary=trotter_gate)        # your QPE function with the Trotterised U
basic = ["cx", "rz", "sx", "x"]
t_exact = transpile(circuit, basis_gates=basic, optimization_level=1)
t_trot = transpile(circuit_trotter, basis_gates=basic, optimization_level=1)
cx_trotter = t_trot.count_ops().get("cx", 0)
print(f"Trotterised, m = {m}: same top outcome? energy {E_t:.4f} Ha, frequency {f_t:.2f}")
print(f"CX gates: {cx_trotter}, depth {t_trot.depth()}")
print(f"exact-matrix version transpiled the same way: depth {t_exact.depth()} (a 4x4 unitary is cheap to simulate but not free on hardware)")
"""

S3_CODE_44 = r"""# How does the cost grow with m? Qiskit builds the gate U^(2^k) by multiplying the evolution TIME by 2^k (same Trotter steps).
assert abs(trotter_gate.power(4).params[0] - 4 * trotter_gate.params[0]) < 1e-12, "U^4 should lengthen the evolution time by a factor of 4"
print("U^4 as a Qiskit gate: time", trotter_gate.power(4).params[0], "= 4 x", trotter_gate.params[0], "(the evolution is lengthened, not repeated)")
print(f"{'m':>3}{'qubits':>8}{'CX gates':>10}{'depth':>8}")
for mm in (2, 3, 4, 5, 6):
    gate_m = PauliEvolutionGate(op_shifted, time=-2 * np.pi / W, synthesis=SuzukiTrotter(order=2, reps=2))
    qc_m = qpe_run(ham, mm, unitary=gate_m)[3]
    t_m = transpile(qc_m, basis_gates=basic, optimization_level=1)
    print(f"{mm:>3}{mm + 2:>8}{t_m.count_ops().get('cx', 0):>10}{t_m.depth():>8}")
print("Here the gate count grows slowly with m, but the Trotter error grows with the evolution time (a longer time with the same r).")
print("In the textbook construction U^(2^k) is U applied 2^k times: 2^m - 1 applications in all, so the circuit length doubles with every extra qubit.")
"""

S3_MD_45 = r"""**Cross-check with the library.** Qiskit's `HamiltonianPhaseEstimation` does the same job with its own scaling, so its number is close to ours but not identical; do not expect equality.
"""

S3_CODE_46 = r"""from qiskit_algorithms import HamiltonianPhaseEstimation
hf_prep = QuantumCircuit(2); hf_prep.x(0)
hpe = HamiltonianPhaseEstimation(6, qff.quantum.sv_sampler(4096)).estimate(ham.op, state_preparation=hf_prep, bound=2.0)
e_library = hpe.most_likely_eigenvalue.real
print(f"HamiltonianPhaseEstimation (6 evaluation qubits, bound 2.0): {e_library:.4f} Ha   (exact {e_exact_el:.4f})")
"""

S3_MD_47 = r"""### 3.6 · How many qubits for chemical accuracy?

QPE has a resolution: with $m$ evaluation qubits the energy comes out on a grid of step $W/2^m$. How fine must the grid be, and what happens when the molecule is stretched? That is task T3.
"""

S3_MD_48 = r"""### T3 · Stretch: how many qubits for chemical accuracy, and a stretched molecule · التمرين ٣

**What you are about to build, and why.** Equation (2) says the grid step is $W/2^m$. In T3 you turn this into a design rule: *how many evaluation qubits are needed so that the grid step is below chemical accuracy* (1.6 mHa)? Then you stretch the molecule (bond length 1.6 Å instead of 0.735 Å), where Hartree-Fock is a worse start state, and watch the overlap fall and QPE's top outcome become less likely: equation (5') in action.

**The formulas you implement:**

$$\frac W{2^m}\le\varepsilon\quad\Longleftrightarrow\quad m\ \ge\ \log_2\frac W\varepsilon \tag{7}$$

and the overlap $|c_0|^2=|\langle E_0|HF\rangle|^2$ from the eigenvectors of the stretched molecule.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Solve W / 2^m <= epsilon for m: take the base-2 logarithm of both sides. epsilon = 1.6 mHa is 0.0016 Ha and W = 2. Since m must be a whole number, which way do you round? For the stretched molecule, load the 1.6 Angstrom file and compare its overlap and its QPE result with the short molecule.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps: the design rule gives `needed_m` (use the `math` module); load the 1.6 Angstrom Hamiltonian; diagonalise its matrix and take the squared overlap of the Hartree-Fock vector with the ground-state eigenvector; run your `qpe_run` on it with 7 evaluation qubits to get the energy and the top frequency.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
import math
needed_m = ______              # equation (7)
ham_far = ______               # the stretched molecule (bond length 1.6 A)
w_far, v_far = ______          # eigenvalues and eigenvectors of its matrix
overlap_far = ______           # squared overlap of Hartree-Fock with the ground state
E_far, F_far = ______          # QPE with 7 evaluation qubits on the stretched molecule
```
Self-check targets: needed_m is 11, the overlap is smaller than for the short molecule, and the top outcome is less likely.

</details>
"""

T3 = r"""import math
needed_m = math.ceil(math.log2(W / 0.0016))                       # equation (7): step W / 2^m <= 1.6 mHa  <=>  m >= log2(W / eps)
print(f"For a step of 1.6 mHa with a window of {W} Ha we need m >= {needed_m} evaluation qubits, so {needed_m + 2} qubits in total.")
ham_far = qff.chem.load_hamiltonian(f"{DATA}/h2_parity_1.6.json")      # stretched molecule: Hartree-Fock is a worse start
w_far, v_far = np.linalg.eigh(ham_far.op.to_matrix())
overlap_far = abs(v_far[:, 0] @ hf_vec) ** 2
E_far, F_far = qpe_run(ham_far, 7)[:2]
print(f"R = 1.6 A: HF overlap {overlap_far:.3f} (was {overlap:.3f}); QPE m = 7 gives {E_far:.4f} Ha with top frequency {F_far:.2f}; exact {w_far[0]:.4f} Ha")
print("A worse start state spreads the probability over several energies; the window must also still contain the whole spectrum.")
"""

T3_STARTER = r"""# Goal: needed_m = the number of evaluation qubits for a grid step below 1.6 mHa (equation 7); then stretch the molecule to 1.6 A:
# ham_far, overlap_far (squared overlap of Hartree-Fock with its ground state), E_far and F_far (QPE energy and top frequency at m = 7).
raise NotImplementedError("T3: fill this in")
"""

T3_CHECK = r"""assert needed_m == 11, "T3: W / 2^m <= 0.0016 with W = 2 needs m = 11"
assert overlap_far < overlap, "T3: the stretched molecule should have a smaller overlap with Hartree-Fock"
assert F_far < F_qpe[7], "T3: with a worse start state the top outcome should be less likely"
print("T3 ok")
"""

S3_MD_51 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

Load `h2_parity_0.5.json` instead of the file you used above and rerun your QPE function. Its lowest eigenvalue is −2.11, outside our window [−2.0, 0.0), so the energy aliases and the answer is wrong. Move `E_MIN` down and watch it come back. The window must contain the spectrum.

</div>
"""

S3_MD_52 = r"""[solution-only] Instructor notes. (1) Expected values: m = 4 and 5 both give -1.875 (error -17.7 mHa), m = 6 gives -1.84375 (+13.5 mHa, top frequency about 0.51), m = 7 gives -1.859375 (-2.1 mHa). (2) The Trotterised circuit gives the same top outcome as the exact one at m = 5 (frequency about 0.76) with 252 CX gates. (3) At R = 1.6 the squared HF overlap is 0.84, so QPE at m = 7 shows a lower top frequency (about 0.81 against 0.93 at R = 0.735).
"""

S3_IBM_NOTE = r"""The next cell does the offline part: it transpiles your circuit for a *pretend* device on this computer and reports its size.
"""

S3_CODE_54 = r"""preview = qff.quantum.device_preview(circuit)      # `circuit` = the main circuit of this notebook
print(f"depth {preview['depth']}, two-qubit gates {preview['two_qubit_gates']}")
"""

S3_MD_55 = r"""<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

A circuit this deep would not survive today's noisy devices: the preview above is only m = 5 with an exact 4×4 unitary, and a real Trotterised version has hundreds of two-qubit gates (see the CX count earlier). Every two-qubit gate adds error, so the energy would be lost in noise. That is why QPE waits for error-corrected machines, and why we claim only what the noiseless simulation shows.

</div>
"""

S4_CODE_58 = r"""# Results table: classical baseline row first.
print(f"{'method':<34}{'energy (Ha, total)':>20}{'error vs exact (mHa)':>22}   cost")
print(f"{'exact diagonalisation (4x4)':<34}{e_exact_total:>20.6f}{0.0:>22.1f}   one 4x4 matrix, milliseconds")
print(f"{'Hartree-Fock':<34}{e_hf_total:>20.6f}{(e_hf_total - e_exact_total) * 1000:>22.1f}   one configuration")
for mm in (4, 5, 6, 7):
    print(f"{'QPE, m = ' + str(mm) + ' (exact U)':<34}{E_qpe[mm] + ham.shift:>20.6f}{(E_qpe[mm] - e_exact_el) * 1000:>22.1f}   {mm + 2} qubits, 4096 shots")
print(f"{'QPE, m = 5, Trotter U':<34}{E_t + ham.shift:>20.6f}{(E_t - e_exact_el) * 1000:>22.1f}   {cx_trotter} CX gates")
ms = [4, 5, 6, 7]
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].semilogy(ms, [W / 2 ** k * 1000 for k in ms], "o-", color="#0F62FE", label="grid step W/2^m")
ax[0].semilogy(ms, [abs(E_qpe[k] - e_exact_el) * 1000 for k in ms], "s--", color="#FA4D56", label="actual error")
ax[0].axhline(1.6, color="grey", ls=":", label="chemical accuracy 1.6 mHa")
ax[0].set_xlabel("evaluation qubits m"); ax[0].set_ylabel("mHa"); ax[0].legend(fontsize=7); ax[0].set_xticks(ms)
ax[1].bar(["m=4", "m=5", "m=6", "m=7"], [F_qpe[k] for k in ms], color="#0F62FE")
ax[1].set_ylabel("probability of the top outcome"); ax[1].set_ylim(0, 1)
plt.tight_layout(); plt.show()
"""

S4_MD_59 = r"""**What this toy shows.** QPE turns an energy into a binary number: each extra evaluation qubit halves the uncertainty, and a good starting state (98.8% overlap) makes the right answer the likely one. With 7 evaluation qubits we land within about 2 mHa of the exact H₂ energy, in a noiseless simulation. The error does not fall smoothly, because the true phase falls between grid points.
"""

S4_MD_60 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**What this toy does not show.** In the textbook construction each extra qubit doubles the number of times U must be applied (in the Qiskit version used here the evolution is lengthened instead, so the gate count grows slowly but the Trotter error grows): the Trotterised m = 5 circuit already needs about 250 two-qubit gates, far beyond today's noisy devices, and chemical accuracy would need m ≥ 11 with this window. QPE is an algorithm for error-corrected machines, so we ran nothing on hardware. H₂ takes a laptop milliseconds to solve exactly. And H₂ is not a catalyst: a real hydrogen-evolution catalyst surface involves many metal atoms and is far beyond this size. No classically benchmarked quantum advantage exists for this use case.

</div>
"""

S4_MD_61 = r"""**Paper versus our route.** The WEF/Aramco paper lists catalyst design for hydrogen electrolysis and storage as a use case (Table 1, p. 9; 54% in the storage survey, Figure 7, p. 24) and names no method. QPE is our teaching choice because it is in the Part 2 lectures and is the method researchers expect to use once fault-tolerant hardware exists.
"""

ACC = r"""assert QUBITS == 7
assert abs(e_exact_el - (-1.857275)) < 1e-5 and abs(e_exact_el + ham.shift - (-1.137306)) < 1e-5
assert warmup_raw_string == "110" and abs(warmup_phase - 0.375) < 1e-12
assert abs(E_qpe[5] - (-1.875)) < 1e-9 and abs(E_qpe[7] - (-1.859375)) < 1e-9
assert abs(E_qpe[7] - e_exact_el) < 0.0157
assert cx_trotter > 100
print("acceptance ok")
"""

# Toolbox cells: taken from tools/toolbox.py when it has an entry for notebook 7, else the copy below (from the uploaded notebook).
_TB_MD = r"""### 🧰 Toolbox: how the calls you will need are written · صندوق الأدوات

No internet? No problem. The cell under this table runs each call once on a **small toy that has nothing to do with the task**, so you can see the exact spelling, the order of the arguments, and what comes back. Run it, read the printed lines, and come back to it whenever a hint says "a method of ...". *What* to build, and with *which numbers*, is your job.

| To do | How it is written | What you get |
|---|---|---|
| a circuit with 7 qubits and 5 classical bits | `qc = QuantumCircuit(7, 5)` | the bits are named `c` |
| $X$ on one qubit | `qc.x(5)` | the qubit is set to $\lvert1\rangle$ |
| put the QPE circuit in | `qc.compose(phase_estimation(m, gate), inplace=True)` | added at the end; it uses qubits $0..m-1$ for evaluation, the rest for the system |
| measure chosen qubits | `qc.measure(range(m), range(m))` | qubit $i$ into classical bit $i$ |
| run and count | `qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()` | a dictionary such as `{'01101': 3980, ...}`; the **rightmost** character is bit 0 |
| counts to a phase | `phi, freq = qff.bits.phase_from_counts(counts, m)` | the phase of the most frequent outcome and its frequency |
| a matrix as a gate | `UnitaryGate(matrix)` | the matrix is $2^k\times2^k$ for $k$ qubits (here $4\times4$: `np.eye(4)` has that size) |
| the matrix exponential | `import scipy.linalg as sla`, then `sla.expm(1j * M)` | the matrix $e^{iM}$ |
| the optional argument `unitary` of `qpe_run(h, m, unitary=None, shots=4096)` | `None` means "build the exact $U$ of equation (2)" | section 3.5 passes a Trotter gate here instead |
| what a `Ham` holds | `ham.op`, `ham.shift`, `ham.hf_x_qubits`, `ham.e_hf`, `ham.e_exact` | `ham.op.to_matrix()` is the Hamiltonian as a matrix; `ham.hf_x_qubits` lists the qubits that start in $\lvert1\rangle$ |
"""
_TB_CODE = r"""# Toolbox for 07, on toys (no hydrogen here).
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, phase_estimation

print("phase_from_counts:", qff.bits.phase_from_counts({"110": 4000, "010": 96}, 3), "<- (phase, frequency): the string is read reversed")
qc = QuantumCircuit(3, 2)                                  # 3 qubits, 2 classical bits named c
qc.x(0)                                                    # qubit 0 set
qc.measure([0, 1], [0, 1])                                 # qubit 0 -> bit 0, qubit 1 -> bit 1
counts = qff.quantum.sv_sampler(1000).run([qc]).result()[0].data.c.get_counts()
print("counts:", counts, "<- bit 0 is the rightmost character")
assert counts == {"01": 1000}
pe = phase_estimation(2, UnitaryGate(np.eye(2)))           # 2 evaluation qubits and a 1-qubit unitary: the circuit has 3 qubits
host = QuantumCircuit(3, 2)
host.compose(pe, inplace=True)
print("QPE circuit qubits:", pe.num_qubits, "| gates:", dict(host.count_ops()))
print("a matrix as a gate: np.eye(2 ** 2) has shape", np.eye(2 ** 2).shape)
print("H2 files in the data folder:", [f for f in sorted(os.listdir(DATA)) if f.startswith("h2_parity")])
import scipy.linalg as sla
print("e^{i pi diag(0, 1)}:", np.round(sla.expm(1j * np.pi * np.diag([0, 1])).real, 3).tolist())
assert np.allclose(sla.expm(1j * np.pi * np.diag([0, 1])), np.diag([1, -1]))
print("OK: toolbox 07 ran.")
"""
TB_MD, TB_CODE = TOOLBOX.get(7, (_TB_MD, _TB_CODE))

GLOSSARY_EXTRA = [
    ("quantum phase estimation (QPE)", "تقدير الطور الكمومي (QPE)", "G"), ("quantum Fourier transform", "تحويل فورييه الكمومي", "G"),
    ("inverse QFT", "تحويل فورييه الكمومي العكسي", "I"), ("eigenvalue", "القيمة الذاتية", "G"), ("eigenstate", "الحالة الذاتية", "G"),
    ("unitary matrix", "مصفوفة عكوسة/وحدوية", "G"), "Hamiltonian", "ground state",
    ("fault tolerance", "التسامح مع الخطأ", "G"), ("phase kickback", "ارتداد الطور", "I"),
    ("evaluation register", "سجل التقدير", "I"), ("Trotterisation", "تقريب تروتر", "I"),
]


def main():
    """Write the solution notebook."""
    nb = new(7, "Hydrogen electrolysis catalyst",
             use_case="Hydrogen: the molecule a hydrogen-evolution catalyst must release; find its energy",
             algorithm="Quantum phase estimation (QPE)", qubits=7, minutes=1)
    nb.section(1)
    nb.md(S1_MD_05)
    nb.section(2)
    nb.md(S2_MD_07)
    nb.code(S2_CODE_08)
    nb.md(S2_MD_09)
    nb.code(S2_CODE_10)
    nb.section(3)
    nb.md(S3_MD_12)
    nb.code(S3_CODE_13)
    nb.md(S3_MD_14)
    nb.md(S3_MD_15)
    nb.md(S3_MD_16)
    nb.code(S3_CODE_17)
    nb.md(S3_MD_18)
    nb.md(S3_MD_19)
    nb.md(S3_MD_20)
    nb.code(S3_CODE_21)
    nb.md(S3_MD_22)
    nb.code(S3_CODE_23)
    nb.md(TB_MD)
    nb.code(TB_CODE)
    nb.md(S3_MD_26)
    nb.task("T1", 'read the warm-up phase two ways and decide which is right', T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.md(S3_MD_29)
    nb.md(S3_MD_30)
    nb.task("T2", 'write qpe_run(h, m, unitary=None) and use it for H2 with m = 5; report the energy', T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_MD_33)
    nb.code(S3_CODE_34)
    nb.code(S3_CODE_35)
    nb.code(S3_CODE_36)
    nb.md(S3_MD_37)
    nb.md(S3_MD_38)
    nb.code(S3_CODE_39)
    nb.md(S3_MD_40)
    nb.md(S3_MD_41)
    nb.md(S3_MD_42)
    nb.code(S3_CODE_43)
    nb.code(S3_CODE_44)
    nb.md(S3_MD_45)
    nb.code(S3_CODE_46)
    nb.md(S3_MD_47)
    nb.md(S3_MD_48)
    nb.task("T3", 'how many evaluation qubits does chemical accuracy need? Then try another bond length', T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_MD_51)
    nb.md(S3_MD_52, tags=['solution-only'])
    nb.md(day.IBM_MD + "\n" + S3_IBM_NOTE)
    nb.code(S3_CODE_54)
    nb.md(S3_MD_55)
    nb.md(day.glossary_md(GLOSSARY_EXTRA), tags=["glossary"])
    nb.section(4)
    nb.code(S4_CODE_58)
    nb.md(S4_MD_59)
    nb.md(S4_MD_60)
    nb.md(S4_MD_61)
    nb.acceptance(ACC)
    nb.section5(use_case="Hydrogen stores surplus power, and a catalyst decides how cheaply it is made; to compare catalysts you need molecular energies to 1.6 mHa.",
                counting_fact="One more qubit, one more binary digit of the energy, twice the circuit.",
                single_idea="Let the molecule's energy turn a dial, then read the dial in binary with an inverse Fourier transform.",
                ideas=["A 'combination lock' interactive where each evaluation qubit reveals one bit of the energy",
                       "A zoom animation on the energy axis: each qubit halves the interval",
                       "A relay-race video where the runners are the evaluation qubits"],
                reuse="the precision table (m = 4 to 7), the histogram relabelled in Hartree, the CX count of the Trotter circuit")
    nb.section6(honest="With 7 evaluation qubits we got within 2 mHa of the exact H₂ energy in simulation; a real device cannot run a circuit this long yet.",
                question_a="Why does the answer jump when you add one qubit?",
                question_b="What if the starting state is bad?")
    path = nb.save(os.path.join(ROOT, "solutions", "07_store_hydrogen_qpe_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()