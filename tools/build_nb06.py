"""Build solutions/06_store_battery_vqe_solution.ipynb (notebook 6: battery cathode chemistry, VQE on LiH).

    python tools/build_nb06.py

Variational loop: Hartree-Fock start, an 8-knob chemistry ansatz built from the shipped excitation generators,
then a bond curve. Reusable pieces defined in the notebook: `ham` (loaded Hamiltonian), `hf`, `ansatz`, `curve`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1_MD_05 = r"""**The problem.** A battery stores energy in chemical bonds. How much voltage a cathode gives, and how long it lasts, depends on the energy of its electrons when lithium moves in and out. To predict that, you need the lowest energy (the *ground state*) of a molecule or crystal. Picture a lithium atom giving up an electron to its neighbour: the energy of that bond is the number we want.

**Who faces it.** A battery-materials research group, a cell manufacturer, or an energy-storage developer that wants to screen cathode candidates on a computer before making them in a lab.

<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

**What is real today.** WEF/Aramco Case 10 (p. 24): Ford and Quantinuum described a workflow to study a representative cathode material (LiCoO₂). The problem is split into smaller models, and the quantum step is used "where the hardest chemistry shows up". The paper does not give a qubit count, compares the result with classical methods, and calls the overall state "still early-stage R&D". The incumbent is classical quantum chemistry (density functional theory and coupled-cluster methods), which is excellent for most cathodes today.

</div>

**Why a quantum computer might help.** The state of *n* strongly interacting electrons needs a number of amplitudes that grows exponentially with *n*, and a quantum register stores such a state natively; what is still missing is hardware that is large and clean enough to measure energies to chemical accuracy on a real material.

| Paper's method | Our teaching route |
|---|---|
| A building-block workflow for LiCoO₂: classical methods for most of the material, a quantum step on the hardest part | The same algorithm family (VQE) on lithium hydride (LiH), 2 electrons in 3 orbitals, 4 qubits. LiH is a stand-in chosen for its size, not a cathode material |
"""

S2_MD_07 = r"""**The toy.** Lithium hydride (LiH) is the smallest molecule with a lithium bond. We keep **2 electrons in 3 orbitals** (an *active space*: only the orbitals where the interesting chemistry happens) and map them to **4 qubits**. The Hamiltonian is not built here: we generated it once with PySCF and qiskit-nature (`tools/generate_chemistry.py`) and shipped it as a small JSON file, so you only need Qiskit.

Energies are in **Hartree (Ha)**. *Chemical accuracy* is 1.6 mHa (0.0016 Ha): the error chemists regard as good enough to predict reactions. We work at a bond length R = 1.6 Å.

The classical baseline runs first. It is three numbers that need no quantum circuit: Hartree–Fock (every electron in its best single-orbital guess), exact diagonalisation of the 16×16 matrix, and the full-molecule FCI energy stored in the file.
"""

S2_CODE_08 = r"""QUBITS = 4                       # [KEEP] the test runner reads this
from qiskit import QuantumCircuit
ham = qff.chem.load_hamiltonian(f"{DATA}/lih_parity_1.6.json")
print(ham.description)
print("qubits:", ham.num_qubits, "| Pauli terms:", len(ham.op), "| shipped excitation generators:", len(ham.generators))
print("energy shift (nuclear repulsion + frozen core), Ha:", round(ham.shift, 6))
for p, c in list(zip(ham.op.paulis, ham.op.coeffs.real))[:5]:
    print(f"  {c:+.5f} * {p}")
"""

S2_CODE_09 = r"""# Classical baseline: always runs first. No quantum circuit anywhere in this cell.
H = ham.op.to_matrix()                                   # a 16 x 16 matrix
evals = np.linalg.eigvalsh(H) + ham.shift
e_exact = evals[0]                                       # exact diagonalisation of the active space
e_hf_file, e_fci_full = ham.e_hf, ham.e_fci_full         # Hartree-Fock and full FCI, read from the file
mha = 1000.0                                             # Ha -> mHa
print(f"Hartree-Fock                    {e_hf_file:.5f} Ha")
print(f"Exact (16x16 diagonalisation)   {e_exact:.5f} Ha   <- the answer VQE must reach")
print(f"Full FCI (whole molecule)       {e_fci_full:.5f} Ha")
print(f"correlation energy of the whole molecule : {(e_hf_file - e_fci_full) * mha:5.1f} mHa")
print(f"captured by our 4-qubit active space     : {(e_hf_file - e_exact) * mha:5.1f} mHa")
print(f"left out by the active space             : {(e_exact - e_fci_full) * mha:5.1f} mHa")
print("16 x 16 matrix: one linear-algebra call, no circuit")
"""

S2_MD_10 = r"""**A trap worth knowing.** A parity-mapped operator on 4 qubits also contains states with the *wrong number of electrons*, and taking the lowest eigenvalue of the whole matrix is only safe if that state has the right electron number. For LiH it does: the lowest eigenvalue lies between the full-molecule FCI energy and the Hartree–Fock energy at every bond length we ship, and the generator script asserts this. For other molecules, or charged ones, the lowest eigenvalue of the matrix can belong to the wrong sector, so always check it against a known reference.
"""

S2_CODE_11 = r"""print("lowest four eigenvalues of the whole 16 x 16 matrix (Ha):", np.round(evals[:4], 5))
print("right sector?  FCI <= exact <= HF:", e_fci_full <= e_exact <= e_hf_file)
"""

S3_MD_13 = r"""### 3.0 · The map of this section

<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**VQE in one sentence.** Nature's answer is the *lowest-energy state* of the molecule. We cannot read it off, so we prepare a guess with a quantum circuit that has knobs, **measure its average energy**, and let a classical optimiser turn the knobs downhill. A rule called the *variational principle* guarantees that a lower energy is always a better guess, and that no guess can go below the truth.

</div>

Everything below builds that sentence from things you already know: lists of numbers (vectors), matrices and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**, so you can watch the maths run and check it against Qiskit.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | A molecule becomes a sum of Pauli terms (the Hamiltonian) | (1) | `ham.op` | |
| 3.2 | Why the lowest average energy is the best guess: the variational principle | (2) | random-state cell | |
| 3.3 | A worked example by hand: hydrogen, 2 qubits, one knob | (3), (4) | `E_by_hand` | |
| 3.4 | Step 1: the Hartree-Fock starting state and its energy | (1) | `hf`, `estimator` | **T1** |
| 3.5 | Step 2: one knob per electron hop (the ansatz), and the loop | (5), (6) | `PauliEvolutionGate` | **T2** |
| 3.6 | Does the shape of the ansatz matter? | (2) again | `efficient_su2` | |
| 3.7 | Step 3: the bond curve | (7) | `curve` | **T3** |
| 3.8 | Why hardware makes this harder: measuring the energy | (8) | shots cell | |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it.
"""

S3_CODE_14 = r"""qff.plots.plot_flow(
    [("Start state", "Hartree-Fock", "source"), ("Hops", "ansatz, one knob each", "move"), ("Energy <H>", "measure", "store"), ("Optimiser", "SLSQP", "use")],
    loop=(3, 1, "turn the knobs downhill (equation 6)"), title="VQE for a molecule");
"""

S3_MD_15 = r"""**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).

| Symbol | Meaning | In the code |
|---|---|---|
| $H=\sum_kc_kP_k$ | the Hamiltonian as a sum of Pauli terms | `ham.op`, `ham.op.coeffs`, `ham.op.paulis` |
| `shift` | constant energy (nuclear repulsion, frozen core) | `ham.shift` |
| $\langle P_k\rangle$ | average of one Pauli term in the state | `estimator.run(...)` |
| $E_0,\ E_{HF}$ | exact ground energy; Hartree-Fock energy | `e_exact`, `ham.e_hf` |
| $\lvert HF\rangle$ | the Hartree-Fock state | `hf`, `ham.hf_x_qubits` |
| $G_k,\ \theta_k$ | generator of hop $k$, its knob | `ham.generators`, `theta[k]` |
| $\lvert\psi(\boldsymbol\theta)\rangle$ | the state of the 8-knob circuit (equation 5) | `ansatz.assign_parameters(...)` |
| $R,\ E(R)$ | bond length; energy at that length | `Rs`, `curve["vqe"]` |
| $\sigma_E,\ M$ | shot-noise error of the energy; shots per term | `variance_sum`, `M_needed` |
"""

S3_MD_16 = r"""<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**

**The one line to remember.** For *every* state, $\langle\psi|H|\psi\rangle\ge E_0$ (equation 2). VQE never needs the answer in advance: it lowers the average energy of a circuit with knobs, and the floor is the true ground-state energy.

</div>
"""

S3_MD_17 = r"""### 3.1 · From a molecule to a Hamiltonian

**What is being computed.** A molecule is nuclei plus electrons. The electrons spread over regions called **orbitals**, and each orbital has two "seats" (spin up and spin down). We keep the few orbitals where the interesting chemistry happens (an **active space**): for LiH, 3 orbitals, i.e. $3\times2=6$ seats (**spin orbitals**), holding 2 electrons. A state of the electrons says which seats are taken, one bit per seat: that is why qubits fit. Mapping the 6 seats to qubits with the *parity mapping*, and using two symmetries (the electron number and spin do not change), leaves **4 qubits**. All of this was done once with PySCF and qiskit-nature (`tools/generate_chemistry.py`); here you read the result from a file.

**The Hamiltonian** is the energy operator of the electrons. After the mapping it is a sum of **Pauli terms**: each $P_k$ is a string of $I$, $X$, $Y$, $Z$ gates, one per qubit, with a real number $c_k$ in front, plus a constant `shift` (nuclear repulsion and the frozen core electrons):

$$H=\sum_k c_k\,P_k,\qquad E=\langle\psi|H|\psi\rangle+\text{shift}=\sum_k c_k\,\langle P_k\rangle+\text{shift} \tag{1}$$

Here $|\psi\rangle$ is the state of the 4 qubits and $\langle P_k\rangle$ is the average value of $P_k$ in that state, a number between $-1$ and $1$. As a matrix, $H$ is $16\times16$ ($2^4$ states), **Hermitian** (equal to its own conjugate transpose, so its eigenvalues are real). Its eigenvalues are the possible energies, and the smallest, $E_0$, is the **ground state** energy: the number a battery chemist wants. It is measured in **Hartree** (Ha); a thousandth of it is a milli-Hartree (mHa), and getting within 1.6 mHa is called **chemical accuracy**.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Hamiltonian | مؤثر هاملتون (الهاملتوني) | G | Chemical accuracy | الدقة الكيميائية | I |
| Ground state | الحالة الأرضية | I | Pauli gates (X, Y, Z) | بوابات باولي | G |
| Eigenvalue | القيمة الذاتية | G | Expectation value | القيمة المتوقعة | I |
| Eigenstate | الحالة الذاتية | G | Molecule | الجزيء | I |
| Orbital | المدار | I | Electron | الإلكترون | I |
| Active space | الفضاء النشط | I | Qubit | كيوبت | G |
"""

S3_CODE_18 = r"""# Equation (1) in code: what the Hamiltonian of section 2 looks like as a matrix and as a sum of Pauli terms.
H_matrix = ham.op.to_matrix()
print("matrix shape:", H_matrix.shape, "| Hermitian:", np.allclose(H_matrix, H_matrix.conj().T), "| largest imaginary part:", np.abs(H_matrix.imag).max())
coeffs = ham.op.coeffs.real
print(f"Pauli terms: {len(coeffs)} | largest |c_k|: {np.abs(coeffs).max():.4f} | sum of c_k^2: {(coeffs ** 2).sum():.4f}")
print("terms per type of Pauli string (how many X or Y gates):",
      {k: sum(1 for p in ham.op.paulis if (p.x.sum()) == k) for k in range(5)})
fig, ax = plt.subplots(figsize=(6.5, 2.6))
order = np.argsort(-np.abs(coeffs))
ax.bar(range(len(coeffs)), coeffs[order], color="#0F62FE")
ax.set_xlabel("Pauli term (sorted by size)"); ax.set_ylabel("coefficient c_k (Ha)"); ax.set_title("the 100 terms of the LiH Hamiltonian", fontsize=10)
plt.tight_layout()
"""

S3_MD_19 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**Do not forget the shift.** The circuit and the Estimator only give $\sum_kc_k\langle P_k\rangle$. The constant `shift` of equation (1) (nuclear repulsion and the frozen core, here $-6.8040$ Ha) must be added back. The lowest eigenvalue of the $16\times16$ matrix alone is $-1.0771$ Ha; with the shift it is $-7.8811$ Ha. Compare a VQE value without the shift to the exact energy and you are off by thousands of mHa when the target is 1.6 mHa.

</div>
"""

S3_MD_20 = r"""### 3.2 · Why a lower average energy is a better guess: the variational principle

Any **Hermitian** matrix $H$ has eigenvectors $|E_n\rangle$ with real eigenvalues $E_0\le E_1\le\dots$, and any state can be written as a mix of them, $|\psi\rangle=\sum_nc_n|E_n\rangle$ with $\sum_n|c_n|^2=1$. The average energy is then a weighted average of the eigenvalues, with weights $|c_n|^2\ge0$ that add up to 1:

$$\langle\psi|H|\psi\rangle=\sum_n|c_n|^2E_n\ \ge\ \sum_n|c_n|^2E_0=E_0 \tag{2}$$

The inequality holds because every $E_n\ge E_0$. So **no state can have an average energy below the ground-state energy $E_0$**, and the average equals $E_0$ only if all the weight sits on the ground state. This is the **variational principle**, and it is the whole reason VQE works: we never need to know the answer in advance; *the lower the energy we can reach, the closer we are*, and the floor is the truth. (In notebook 2 the Hamiltonian was diagonal and the same fact was a one-line sum over layouts; here $H$ is a full matrix, so we need the eigenvectors.)

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Variational principle | المبدأ التغايري | I | Expectation value | القيمة المتوقعة | I |
| Eigenvalue | القيمة الذاتية | G | Hermitian matrix | مصفوفة هرميتية | I |
| Eigenstate | الحالة الذاتية | G | Ground state | الحالة الأرضية | I |
"""

S3_CODE_21 = r"""# Equation (2) in code: the average energy of 5000 random states never goes below the exact ground-state energy.
rng_states = np.random.default_rng(0)
random_states = rng_states.normal(size=(5000, 16)) + 1j * rng_states.normal(size=(5000, 16))
random_states /= np.linalg.norm(random_states, axis=1, keepdims=True)                  # 16 amplitudes, squares add up to 1
avg = np.einsum("ni,ij,nj->n", random_states.conj(), H_matrix, random_states).real + ham.shift   # <psi|H|psi> + shift for every state
fig, ax = plt.subplots(figsize=(7, 2.8))
ax.hist(avg, bins=40, color="#0F62FE")
ax.axvline(e_exact, color="k", ls="--", label=f"exact ground state {e_exact:.4f} Ha: nothing can go below")
ax.axvline(ham.e_hf, color="#8D8D8D", ls=":", label=f"Hartree-Fock {ham.e_hf:.4f} Ha")
ax.set_xlabel("average energy of a random state (Ha)"); ax.set_ylabel("number of states"); ax.legend(fontsize=8)
print(f"lowest average energy among 5000 random states: {avg.min():.4f} Ha | exact ground state: {e_exact:.4f} Ha")
assert avg.min() >= e_exact - 1e-9, "the variational principle should hold"
print("OK: no random state went below the ground-state energy. Random guessing is far from it; Hartree-Fock is already much better.")
"""

S3_MD_22 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

Replace the random states by *mixtures of the lowest two eigenstates* (use `np.linalg.eigh(H_matrix)` to get the eigenvectors). Check that the average energy is a weighted average of the two lowest eigenvalues, between the two, and that it equals the ground energy only when all the weight is on the ground state.

</div>
"""

S3_MD_23 = r"""### 3.3 · A worked example you can check on paper: hydrogen, two qubits, one knob

Hydrogen ($\text{H}_2$) at its usual bond length (0.735 Å) is the smallest molecule there is: after the mapping it needs only **2 qubits**, and the Hartree-Fock state is `01` (qubit 0 set). The $4\times4$ Hamiltonian (shift included) has only one pair of states that matters: Hartree-Fock $|01\rangle$ and the "both electrons hopped" state $|10\rangle$. In that two-state corner the matrix is

$$H_{2\times2}=\begin{pmatrix}a&c\\c&b\end{pmatrix}=\begin{pmatrix}-1.11700&0.18093\\0.18093&0.47475\end{pmatrix}\ \text{Ha} $$

($a$ = energy of Hartree-Fock, $b$ = energy of the other state, $c$ = how strongly the two are coupled). Prepare the one-knob state that rotates between them, $|\psi(\theta)\rangle=\cos\theta\,|01\rangle+\sin\theta\,|10\rangle$. Its average energy is

$$E(\theta)=a\cos^2\theta+b\sin^2\theta+2c\sin\theta\cos\theta=\bar a+d\cos2\theta+c\sin2\theta,\qquad \bar a=\tfrac{a+b}2,\ d=\tfrac{a-b}2 \tag{3}$$

This is a wave in $\theta$ with amplitude $R=\sqrt{d^2+c^2}$, so its lowest value is

$$E_{\min}=\bar a-\sqrt{d^2+c^2} \tag{4}$$

**Plug in the numbers.** $\bar a=\frac{-1.11700+0.47475}2=-0.32113$, $d=\frac{-1.11700-0.47475}2=-0.79588$, $c=0.18093$, so $\sqrt{d^2+c^2}=\sqrt{0.63341+0.03274}=0.81618$ and $E_{\min}=-0.32113-0.81618=-1.13731$ Ha. This is exactly the ground-state energy of $\text{H}_2$ in the file. At $\theta=0$ (Hartree-Fock) the energy is $a=-1.11700$, so one knob lowers the energy by $20.3$ mHa, far more than chemical accuracy (1.6 mHa). The best knob value satisfies $(\cos2\theta,\sin2\theta)=-(d,c)/R$, i.e. $\theta^*\approx-0.112$. Equation (2) is visible in the picture below: the curve never goes under $E_{\min}$.

<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**This is VQE in miniature.** One knob $\theta$ is the ansatz, $E(\theta)$ is the "average energy" the optimiser sees, and turning the knob downhill from $\theta=0$ (Hartree-Fock) reaches the exact answer. For LiH there are 8 knobs and a 16-dimensional space, but the idea is the same.

</div>
"""

S3_CODE_24 = r'''# The hydrogen example with numpy only: equations (3) and (4), checked against the exact eigenvalue stored in the file.
h2 = qff.chem.load_hamiltonian(f"{DATA}/h2_parity_0.735.json")
H2 = h2.op.to_matrix().real + h2.shift * np.eye(4)                       # 4 x 4 matrix, shift included
a, b, c = H2[1, 1], H2[2, 2], H2[1, 2]                                   # states |01> (Hartree-Fock) and |10>
abar, d = (a + b) / 2, (a - b) / 2
print(f"a = {a:.5f}, b = {b:.5f}, c = {c:.5f} Ha")


def E_by_hand(theta):
    """Equation (3): average energy of cos(theta)|01> + sin(theta)|10>."""
    return abar + d * np.cos(2 * theta) + c * np.sin(2 * theta)


psi = lambda t: np.array([0, np.cos(t), np.sin(t), 0])                   # the one-knob state in the basis |00>, |01>, |10>, |11>
for t in (0.0, 0.3, -0.5):
    assert abs(E_by_hand(t) - psi(t) @ H2 @ psi(t)) < 1e-12              # equation (3) = the matrix average <psi|H|psi>
E_min = abar - np.hypot(d, c)                                            # equation (4)
theta_star = 0.5 * np.arctan2(-c, -d)
print(f"E(0) = {E_by_hand(0):.5f} (Hartree-Fock) | E_min by equation (4) = {E_min:.5f} | exact eigenvalue in the file = {h2.e_exact:.5f}")
print(f"best knob value theta* = {theta_star:.4f} -> E(theta*) = {E_by_hand(theta_star):.5f} | one knob gains {(a - E_min) * 1000:.1f} mHa")
assert abs(E_min - h2.e_exact) < 1e-8
ts = np.linspace(-np.pi / 2, np.pi / 2, 300)
fig, ax = plt.subplots(figsize=(6.5, 3))
ax.plot(ts, [E_by_hand(t) for t in ts], color="#0F62FE")
ax.axhline(E_min, color="k", ls="--", label=f"floor {E_min:.5f} (equation 4)")
ax.plot([0], [a], "o", color="#8D8D8D", ms=9, label="Hartree-Fock (knob 0)")
ax.plot([theta_star], [E_min], "*", color="#D02670", ms=14, label="best knob value")
ax.set_xlabel("knob theta (radians)"); ax.set_ylabel("average energy (Ha)"); ax.legend(fontsize=8); ax.set_title("hydrogen: energy against one knob", fontsize=10)
plt.tight_layout()
'''

S3_MD_25 = r"""### 3.4 · Step 1: the Hartree-Fock starting state and its energy

VQE needs a sensible place to start. Hartree-Fock is the simplest picture of the molecule: each electron sits in the lowest free orbital, which on the qubits is a basis state with a few $X$ gates applied. The next cell is the toolbox for the calls this notebook uses; task T1 follows it.
"""

S3_MD_28 = r"""### T1 · Warm-up: the Hartree-Fock circuit and its energy · التمرين ١

**What you are about to build, and why.** The first thing every VQE needs is a **starting state**. For chemistry the natural one is **Hartree-Fock** (HF): each electron sits in its best single-orbital seat, and no electron has hopped. On our 4 qubits it is a basis state with 1s on the qubits listed in `ham.hf_x_qubits`, which a circuit prepares with **X gates** (an X gate flips $|0\rangle$ to $|1\rangle$). The other thing you build is the **measurement of the energy** with the Estimator, a Qiskit object that computes $\langle\psi|H|\psi\rangle$ for a circuit and an operator. A good check of your circuit: the energy you get must match the Hartree-Fock energy stored in the file.

**The formula you implement:** equation (1) for the HF state: $E_{HF}=\langle HF|H|HF\rangle+\text{shift}=\sum_kc_k\langle P_k\rangle+\text{shift}$.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Which qubits must be flipped? The file tells you. A circuit with 4 qubits is a `QuantumCircuit`; the gate that flips a qubit from 0 to 1 is the X gate. The Estimator is `qff.quantum.sv_estimator()`; its answer leaves out the constant `shift`.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps: create a 4-qubit circuit; apply X gates on the qubits listed in `ham.hf_x_qubits`; ask the estimator for the average energy of that circuit with the Hamiltonian `ham.op` (the result is a one-number array: turn it into a float); add the shift back. The answer must equal `ham.e_hf`.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
hf = ______                     # the Hartree-Fock circuit
e_hf_circuit = ______           # its energy from the estimator, shift included
print(f"HF energy from the circuit: {e_hf_circuit:.5f} Ha")
```
Self-check target: the energy equals the Hartree-Fock energy in the file (-7.86186 Ha).

</details>
"""

T1 = r"""hf = QuantumCircuit(4)
hf.x(ham.hf_x_qubits)                                      # X on the qubits listed in the file: [0, 1]
estimator = qff.quantum.sv_estimator()
e_hf_circuit = float(estimator.run([(hf, ham.op)]).result()[0].data.evs) + ham.shift
print(f"HF energy from the circuit: {e_hf_circuit:.5f} Ha")
"""

T1_STARTER = r"""# Goal: `hf` is a 4-qubit circuit that prepares the Hartree-Fock state; e_hf_circuit is its energy measured with the Estimator
# (shift included). It must equal the Hartree-Fock energy stored in the file.
estimator = qff.quantum.sv_estimator()
raise NotImplementedError("T1: build hf and e_hf_circuit, then delete this line")
"""

T1_CHECK = r"""assert hf.num_qubits == 4, "T1: the circuit needs 4 qubits"
assert hf.count_ops().get("x", 0) == 2, "T1: Hartree-Fock needs exactly two X gates"
assert abs(e_hf_circuit - ham.e_hf) < 1e-6, "T1: add ham.shift to the estimator value; it should equal the Hartree-Fock energy in the file"
print("T1 passed")
"""

S3_MD_31 = r"""### 3.5 · Step 2: one knob per electron hop (the ansatz)

Starting from Hartree-Fock we let electrons **hop** to empty orbitals, one electron (a **single**) or two together (a **double**) at a time. Each possible hop has a Hermitian operator $G_k$, a **generator** (shipped in the file as a sum of Pauli strings). Turning it into a gate with a knob is one formula: for a Hermitian $G$ and a number $\theta$, the matrix $e^{-i\theta G}$ is unitary (it rotates states without changing their length), it equals the identity at $\theta=0$, and as $\theta$ grows it moves the state smoothly along the hop. For a single Pauli $G=Y$ it is the familiar rotation $e^{-i\theta Y}=\cos\theta\,I-i\sin\theta\,Y$. The circuit applies all eight hops, one after the other, each with its own knob:

$$|\psi(\boldsymbol\theta)\rangle=e^{-i\theta_7G_7}\cdots e^{-i\theta_1G_1}\,e^{-i\theta_0G_0}\,|HF\rangle \tag{5}$$

In the hydrogen example of 3.3 the double-hop generator turned $|01\rangle$ into $\cos\theta\,|01\rangle+\sin\theta\,|10\rangle$: that *was* a hop. Because every $e^{-i\theta G}$ is the identity at $\theta=0$, **all knobs at zero gives exactly Hartree-Fock**, the right place to start. This ansatz is called **UCCSD** (unitary coupled-cluster with singles and doubles): the shape of the circuit comes from chemistry, which is why it needs only 8 knobs. In Qiskit each hop is a `PauliEvolutionGate(G_k, time=θ_k)`, which implements $e^{-i\theta_kG_k}$.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Ansatz | الافتراض المبدئي (ansatz) | G | Knob (parameter) | المعامل (المقبض) | I |
| Excitation (hop) | الإثارة (القفزة) | I | Variational | التغايري / المتغايرة | G |
| Generator | المولِّد | I | VQE | محلّل القيم الذاتية الكمومي التغايري (VQE) | G |
| Unitary matrix | مصفوفة عكوسة/وحدوية | G | Classical optimiser | المُحسِّن الكلاسيكي | I |
"""

S3_CODE_32 = r"""print("terms per generator:", [len(g) for g in ham.generators])
"""

S3_MD_33 = r"""### T2 · Core step: the ansatz and the VQE run · التمرين ٢

**What you are about to build, and why.** This is the heart of VQE. You build the 8-knob circuit of equation (5): the HF circuit from T1, followed by one `PauliEvolutionGate` per generator, each with its own parameter. Then you let Qiskit's `VQE` class run the loop of equation (6): it asks the Estimator for the average energy, hands the number to a classical optimiser (**SLSQP**, which uses the slope of the energy to descend), and repeats until the energy stops falling. A **callback** records the energy at every step so you can draw the descent.

$$\boldsymbol\theta^*=\arg\min_{\boldsymbol\theta}E(\boldsymbol\theta),\qquad E(\boldsymbol\theta)=\langle\psi(\boldsymbol\theta)|H|\psi(\boldsymbol\theta)\rangle+\text{shift} \tag{6}$$

("argmin" means "the values at which the function is smallest".)

**The formulas you implement:** (5) for the circuit and (6) for the loop; the energy you report is the optimiser's value plus the shift.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

You need eight knobs: a list of symbolic parameters. For each generator and its knob, one gate (a Pauli evolution gate) is appended to a copy of the HF circuit, on all four qubits. Which object runs the optimisation loop for you?

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps: make a `ParameterVector` with one entry per generator; copy the HF circuit and append one `PauliEvolutionGate` per generator, with its own parameter as the time; make a list for the history and a `VQE` object with your estimator, the circuit, the `SLSQP` optimiser limited to 200 steps, a start point of zeros (that is Hartree-Fock) and a callback storing each energy plus the shift; run `compute_minimum_eigenvalue` on `ham.op`; add the shift to the eigenvalue.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
theta = ______                  # one knob per generator
ansatz = ______                 # HF circuit plus one hop gate per generator (equation 5)
history = []
vqe = ______                    # estimator, circuit, optimiser, start at HF, a callback that stores energy + shift
result = ______                 # run it on the Hamiltonian
e_vqe = ______                  # the optimiser's energy, shift included
```
Self-check target: `e_vqe` within 1.6 mHa of the exact energy, and never below it.

</details>
"""

T2 = r"""from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

theta = ParameterVector("t", len(ham.generators))          # one knob per excitation
ansatz = hf.copy()
for k, gen in enumerate(ham.generators):
    ansatz.append(PauliEvolutionGate(gen, time=theta[k]), range(4))

history = []
vqe = VQE(estimator, ansatz, SLSQP(maxiter=200), initial_point=np.zeros(len(theta)),
          callback=lambda n, params, value, meta: history.append(value + ham.shift))
result = vqe.compute_minimum_eigenvalue(ham.op)
e_vqe = result.eigenvalue.real + ham.shift                 # the eigenvalue excludes the shift
print(f"VQE energy {e_vqe:.6f} Ha after {result.cost_function_evals} energy evaluations")
print(f"error versus exact: {(e_vqe - e_exact) * mha:.4f} mHa  (chemical accuracy: 1.6 mHa)")
"""

T2_STARTER = r"""# Goal: `ansatz` is the 8-knob circuit of equation (5) (HF state, then one hop per generator); run VQE on it from all-zero knobs,
# record the energy at every step in `history`, and keep e_vqe (shift included), within 1.6 mHa of the exact energy.
history = []
raise NotImplementedError("T2: build the ansatz and run VQE, then delete this line")
"""

T2_CHECK = r"""assert ansatz.num_parameters == 8, "T2: one parameter per generator (8)"
assert len(history) > 5, "T2: use the callback to record the energy at each evaluation"
assert abs(e_vqe - e_exact) < 1.6e-3, "T2: not within chemical accuracy yet; did you add ham.shift, and start from zeros?"
assert e_vqe >= e_exact - 1e-6, "T2: below the exact ground state is impossible; check the shift"
print("T2 passed")
"""

S3_MD_36 = r"""**Check equation (5) and the variational principle on your circuit.** The first part rebuilds the state $e^{-i\theta_7G_7}\cdots e^{-i\theta_0G_0}|HF\rangle$ with plain numpy matrices (`scipy.linalg.expm`) for random knob values and checks that it equals Qiskit's circuit state up to a global phase (one and the same turn applied to every amplitude, which changes no probability). The second part checks equation (2): the energy of *every* knob setting we try is at or above the exact energy.
"""

S3_CODE_37 = r"""# Check: equation (5) with numpy matrices equals the Qiskit circuit, and equation (2) holds for every setting.
from scipy.linalg import expm
from qiskit.quantum_info import Statevector
rng_k = np.random.default_rng(SEED)
theta_test = rng_k.uniform(-0.5, 0.5, 8)
state = np.zeros(16, dtype=complex); state[3] = 1                          # Hartree-Fock: qubits 0 and 1 set -> position 3
for k, gen in enumerate(ham.generators):
    state = expm(-1j * theta_test[k] * gen.to_matrix()) @ state            # equation (5): one hop after another
qiskit_state = Statevector(ansatz.assign_parameters(theta_test))
print("numpy product of hops == Qiskit circuit (up to a global phase):", qiskit_state.equiv(Statevector(state)))
assert qiskit_state.equiv(Statevector(state))
e_np = float(np.real(state.conj() @ H_matrix @ state)) + ham.shift
print(f"energy at these knobs: numpy {e_np:.6f} | Qiskit estimator {float(np.ravel(estimator.run([(ansatz, ham.op, [list(theta_test)])]).result()[0].data.evs)[0]) + ham.shift:.6f} Ha")
print(f"lowest energy VQE ever evaluated: {min(history):.6f} Ha | exact ground state: {e_exact:.6f} Ha")
assert min(history) >= e_exact - 1e-6, "no setting of the knobs can beat the ground-state energy"
print("OK: equation (5) matches Qiskit, and the variational principle holds.")
"""

S3_CODE_38 = r"""# The 8-knob circuit you built in T2: the Hartree-Fock X gates, then one Pauli-evolution box per electron hop.
try:
    display(ansatz.draw("mpl", fold=40))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(ansatz.draw("text", fold=120))
"""

S3_CODE_39 = r"""# Convergence: energy at every evaluation, with the three classical reference numbers.
fig, ax = plt.subplots(figsize=(6.5, 3.6))
qff.plots.plot_convergence({"VQE (8 parameters, SLSQP)": history}, exact=e_exact, band=(e_exact, e_exact + 1.6e-3), ax=ax)
ax.axhline(ham.e_hf, color="grey", ls=":", label="Hartree-Fock")
ax.axhline(e_fci_full, color="green", ls=":", label="full FCI")
ax.set_ylabel("energy (Ha)"); ax.set_xlabel("energy evaluation"); ax.legend(fontsize=8)
ax.set_title("LiH at 1.6 A: VQE walks from Hartree-Fock to exact"); plt.show()
"""

S3_MD_40 = r"""**Reading the picture.** The curve starts at the Hartree-Fock energy (all knobs zero), falls quickly as the optimiser turns the knobs, and stops inside the green band: within chemical accuracy of the exact energy (dashed). It never crosses the dashed line, as equation (2) guarantees. The dotted lines are the classical reference numbers from section 2.
"""

S3_MD_41 = r"""### 3.6 · Does the shape of the ansatz matter?

The UCCSD circuit knows chemistry: its gates are hops. A generic **hardware-efficient** circuit (rotations and CNOTs that any device does well, with no chemistry in it) also has 8 knobs. The variational principle (2) holds for *every* circuit, so any ansatz gives an energy at or above $E_0$; what changes is **how close the best state in the family can get**, and whether the optimiser finds it. We start both from Hartree-Fock plus small random angles, with three seeds, and compare the errors. The optimiser here is COBYLA, which needs no slopes. Lesson: the optimiser and the Hamiltonian are the same, the answer is not.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Hardware-efficient ansatz | دائرة فعّالة للعتاد | I | Local minimum | حدّ أدنى محلي | I |
| Entanglement | التشابك | G | Seed | البذرة العشوائية | I |
| Rotation gate | بوابة دوران | I |  |  |  |
"""

S3_CODE_42 = r"""from qiskit.circuit.library import efficient_su2
from qiskit_algorithms.optimizers import COBYLA
hea = hf.compose(efficient_su2(4, su2_gates=["ry"], reps=1, entanglement="linear"))
hea_errors, hea_evals = [], []
for seed in (1, 2, 3):
    x0 = np.random.default_rng(seed).normal(0, 0.05, hea.num_parameters)
    r = VQE(estimator, hea, COBYLA(maxiter=1000), initial_point=x0).compute_minimum_eigenvalue(ham.op)
    hea_evals.append(r.cost_function_evals)
    hea_errors.append((r.eigenvalue.real + ham.shift - e_exact) * mha)
    print(f"hardware-efficient, seed {seed}: {hea_errors[-1]:6.2f} mHa above exact  ({r.cost_function_evals} evaluations)")
print(f"chemistry-shaped ansatz        : {(e_vqe - e_exact) * mha:6.4f} mHa above exact")
"""

S3_MD_43 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

**More knobs, better answer?** In the cell above, change `reps=1` to `reps=2` in `efficient_su2` (that gives 12 knobs instead of 8) and rerun. Does the error against the exact energy fall? Does the number of energy evaluations grow? Compare with the chemistry-shaped ansatz, which reaches chemical accuracy with only 8.

</div>
"""

S3_MD_44 = r"""### 3.7 · Step 3: the bond curve

A single energy says little. Chemists want the **energy as a function of the bond length** $R$: the lowest point is the **equilibrium bond length**, where the molecule is most stable, and the slope tells how stiff the bond is. In T3 you run VQE at every bond length we ship (1.2 to 3.0 Å) and plot

$$E_{\text{VQE}}(R)=\min_{\boldsymbol\theta}\langle\psi(\boldsymbol\theta)|H(R)|\psi(\boldsymbol\theta)\rangle+\text{shift}(R) \tag{7}$$

The Hamiltonian $H(R)$ changes with $R$ (and so do the generators), but the circuit has the same shape. To save work you start each run from the previous optimum, a **warm start**: neighbouring bond lengths have similar best knobs, so the optimiser begins near the answer. As the bond stretches, Hartree-Fock gets *worse*: the electrons need to be **correlated** (they cannot be described by single-orbital guesses), and the gap between HF and the exact energy grows. That is where VQE earns its keep.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Bond length | طول الرابطة | I | Correlation energy | طاقة الترابط الإلكتروني | I |
| Equilibrium | الاتزان | I | Hartree-Fock | هارتري-فوك | I |
| Warm start | البدء الدافئ | G | Bond curve | منحنى الرابطة | I |
"""

S3_MD_45 = r"""### T3 · Stretch: the bond curve · التمرين ٣

**What you are about to build, and why.** In T3 you repeat the VQE run at seven bond lengths, warm-starting each run from the previous one, store the energies, and read off the equilibrium bond length among the shipped points. It tests whether you can turn the single run of T2 into a *loop over instances*, rebuilding the generators for every $R$ (they change with $R$) while keeping the same knob structure.

**The formula you implement:** equation (7) for $R=1.2,1.4,1.6,1.8,2.0,2.4,3.0$ Å.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

You already have everything for one bond length in T2. What changes with R? The file, and with it the Hamiltonian, the generators and the Hartree-Fock energy; what stays the same is the *shape* of the circuit. So you need a loop that loads a file, builds the same kind of circuit from *that file's* generators, runs VQE, and stores the numbers.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Inside the loop over `Rs`: load the file for this R; build the HF circuit and one hop gate per generator *of this file*; run VQE with SLSQP starting from the previous optimum (zeros for the first R); store the VQE energy (shift included) and the HF, FCI and exact energies from the file; keep the optimum as the next start. After the loop, `R_min` is the distance with the lowest VQE energy.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
x0 = ______                         # first start: Hartree-Fock
for R in Rs:
    h = ______                      # the Hamiltonian file for this distance
    a = ______                      # circuit: HF state plus one hop per generator of `h`
    r = ______                      # VQE from the start x0
    x0 = ______                     # warm start for the next distance
    ______                          # store the four energies in `curve`
R_min = ______                      # the distance with the lowest VQE energy
```

</details>
"""

T3 = r"""from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
Rs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]
curve = {"R": Rs, "hf": [], "vqe": [], "fci": [], "exact": []}
x0 = np.zeros(8)
th = ParameterVector("t", 8)
for R in Rs:
    h = qff.chem.load_hamiltonian(f"{DATA}/lih_parity_{R}.json")
    a = QuantumCircuit(4); a.x(h.hf_x_qubits)
    for k, gen in enumerate(h.generators):
        a.append(PauliEvolutionGate(gen, time=th[k]), range(4))
    r = VQE(estimator, a, SLSQP(maxiter=200), initial_point=x0).compute_minimum_eigenvalue(h.op)
    x0 = r.optimal_point                                   # warm start for the next distance
    curve["vqe"].append(r.eigenvalue.real + h.shift)
    curve["hf"].append(h.e_hf); curve["fci"].append(h.e_fci_full); curve["exact"].append(h.e_exact)
R_min = Rs[int(np.argmin(curve["vqe"]))]
print("VQE energies (Ha):", np.round(curve["vqe"], 5))
print("equilibrium bond length among the shipped points:", R_min, "A")
"""

T3_STARTER = r"""# Goal: run VQE at every bond length in Rs, warm-starting each run from the previous optimum, and fill `curve`
# (one value per R in each list: hf, vqe, fci, exact). R_min = the shipped distance with the lowest VQE energy.
Rs = [1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 3.0]
curve = {"R": Rs, "hf": [], "vqe": [], "fci": [], "exact": []}
raise NotImplementedError("T3: run the loop, then delete this line")
"""

T3_CHECK = r"""assert len(curve["vqe"]) == 7 and len(curve["hf"]) == 7, "T3: one VQE energy per bond length (7)"
assert max(abs(np.array(curve["vqe"]) - np.array(curve["exact"]))) < 1.6e-3, "T3: some VQE points are not within chemical accuracy"
assert R_min == 1.6, "T3: the lowest VQE energy among the shipped distances should be at 1.6 A"
print("T3 passed")
"""

S3_CODE_48 = r"""fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(curve["R"], curve["hf"], "o:", color="grey", label="Hartree-Fock")
ax.plot(curve["R"], curve["fci"], "s-", color="green", label="full FCI")
ax.plot(curve["R"], curve["vqe"], "x", color="#0F62FE", ms=9, mew=2, label="VQE (4 qubits)")
ax.axvline(R_min, color="#FA4D56", ls="--", lw=1, label=f"lowest shipped point R = {R_min} A")
ax.set_xlabel("Li-H distance (A)"); ax.set_ylabel("energy (Ha)"); ax.legend(fontsize=8)
ax.set_title("LiH bond curve"); plt.show()
# Hartree-Fock lies above the exact curve, and the gap grows as the bond stretches: electrons need to be correlated.
print("HF minus VQE (mHa) at each R:", np.round((np.array(curve["hf"]) - np.array(curve["vqe"])) * mha, 1))
"""

S3_MD_49 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**A warm start passes mistakes along.** If the run at one bond length ends in a poor valley, the next run starts from those poor knobs. The protection is the exact column: compare $E_{\text{VQE}}(R)$ with the full FCI value at *every* $R$, not only at the first one, and print the gaps before you trust the curve.

</div>
"""

S3_MD_50 = r"""### 3.8 · Measuring on hardware would be harder than it looks

The Estimator gave us $\langle H\rangle$ exactly, for free. A real device cannot. By equation (1), $\langle H\rangle=\sum_kc_k\langle P_k\rangle$ must be estimated **term by term** (or in groups of terms that can be measured together): each term is measured $M$ times, and each run returns $+1$ or $-1$. The average of $M$ runs estimates $\langle P_k\rangle$ with a random error of size $\sqrt{(1-\langle P_k\rangle^2)/M}$, so the energy has a random error

$$\sigma_E=\sqrt{\frac{\sum_kc_k^2\,\bigl(1-\langle P_k\rangle^2\bigr)}{M}}\qquad\Longrightarrow\qquad M\ \ge\ \frac{\sum_kc_k^2\,(1-\langle P_k\rangle^2)}{\varepsilon^2}\ \text{ to reach }\sigma_E\le\varepsilon \tag{8}$$

(for independent terms; grouping measurements makes the real count smaller). The cell below evaluates this for the Hartree-Fock state, with $\varepsilon=1.6$ mHa, checks equation (8) by simulating the measurements, and counts the total number of circuit runs. **Noise** adds a *systematic* error on top, which more shots do not remove. This is the main reason a noiseless 1.6 mHa result on a simulator says little about a real device.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Shots | عدد مرات التشغيل (shots) | I | Estimator | المُقدِّر | I |
| Noise | الضوضاء | G | Fault-tolerant | التسامح مع الخطأ | G |
| Measurement | القياس | G |  |  |  |
"""

S3_CODE_51 = r"""# Equation (8) in numbers: how many shots would a real device need for chemical accuracy on the Hartree-Fock state?
hf_vec = np.zeros(16); hf_vec[3] = 1                                              # the Hartree-Fock basis state
exp_pauli = np.array([hf_vec @ p.to_matrix() @ hf_vec for p in ham.op.paulis]).real  # <P_k> for every term
variance_sum = float((coeffs ** 2 * (1 - exp_pauli ** 2)).sum())
eps = 1.6e-3
M_needed = variance_sum / eps ** 2
print(f"sum c_k^2 (1 - <P_k>^2) = {variance_sum:.5f}")
print(f"shots per term for sigma_E <= 1.6 mHa: {M_needed:,.0f} | terms: {len(coeffs)} | total runs if each term is measured separately: {M_needed * len(coeffs):,.0f}")
M = 10_000
sim = np.random.default_rng(SEED)
estimates = [float(sum(ck * (2 * sim.binomial(M, (1 + ep) / 2) / M - 1) for ck, ep in zip(coeffs, exp_pauli))) for _ in range(300)]
print(f"{M:,} shots per term, 300 simulated experiments: spread of the energy estimate {np.std(estimates) * 1000:.2f} mHa | equation (8): {np.sqrt(variance_sum / M) * 1000:.2f} mHa")
assert abs(np.std(estimates) / np.sqrt(variance_sum / M) - 1) < 0.25, "equation (8) should predict the spread of simulated measurements"
"""

S3_CODE_52 = r"""circuit = ansatz.assign_parameters(result.optimal_point)     # the optimised circuit, for the preview cell below
"""

S3_IBM_NOTE = r"""The next cell does the offline part: it transpiles your circuit for a *pretend* device on this computer and reports its size.
"""

S3_CODE_54 = r"""preview = qff.quantum.device_preview(circuit)      # `circuit` = the main circuit of this notebook
print(f"depth {preview['depth']}, two-qubit gates {preview['two_qubit_gates']}")
"""

S3_MD_55 = r"""<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

Reading the preview: the circuit has eight Pauli-evolution gates acting on four qubits; once broken into gates a real device understands, its depth and two-qubit gate count are large for such a small molecule. A circuit this deep would not survive today's noise well enough to reach 1.6 mHa; that is why our result is a noiseless simulation, and why the claim stays modest.

</div>
"""

S4_CODE_58 = r"""# Results table: classical baseline rows first.
import time
t0 = time.time(); _ = np.linalg.eigvalsh(ham.op.to_matrix()); t_exact = time.time() - t0
rows = [
    ("Hartree-Fock (classical)", ham.e_hf, "1 energy"),
    ("Exact diagonalisation (classical)", e_exact, "16x16 matrix, %.4f s" % t_exact),
    ("Full FCI, whole molecule (classical)", e_fci_full, "from the file"),
    ("VQE, chemistry-shaped ansatz", e_vqe, f"{result.cost_function_evals} evaluations of <H>"),
] + [(f"VQE, hardware-efficient, seed {s}", e_exact + e / mha, f"{n} evaluations (COBYLA)") for s, e, n in zip((1, 2, 3), hea_errors, hea_evals)]
print(f"{'method':<40}{'energy (Ha)':>13}{'error vs exact (mHa)':>22}   cost")
for name, e, cost in rows:
    print(f"{name:<40}{e:>13.5f}{(e - e_exact) * mha:>22.3f}   {cost}")
err_uccsd = abs(e_vqe - e_exact) * mha
labels = ["Hartree-Fock", "VQE hardware-\nefficient (worst)", "VQE hardware-\nefficient (best)", "VQE chemistry-\nshaped"]
vals = [(ham.e_hf - e_exact) * mha, max(hea_errors), min(hea_errors), max(err_uccsd, 1e-4)]
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar(labels, vals, color=["grey", "#8D8D8D", "#8D8D8D", "#0F62FE"])
ax.axhline(1.6, color="#FA4D56", ls="--", label="chemical accuracy 1.6 mHa"); ax.set_yscale("log")
ax.set_ylabel("error vs exact (mHa, log)"); ax.legend(fontsize=8); ax.set_title("How close each method gets"); plt.show()
"""

S4_MD_59 = r"""**What this toy shows.** VQE with a chemistry-shaped ansatz recovers the exact energy of a 4-qubit model of one lithium bond to far better than chemical accuracy, and reproduces the bond-length curve; a generic ansatz does worse from most starts.
"""

S4_MD_60 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**What this toy does not show.** LiH in a minimal basis is solved exactly by a laptop in milliseconds. A cathode such as LiCoO₂ has transition-metal d electrons and a periodic crystal; the active spaces that matter are far beyond exact simulation and beyond what today's noisy devices can measure to 1.6 mHa. We ran noiseless, and LiH is not a cathode material.

</div>
"""

S4_MD_61 = r"""**Paper versus our route.** Same algorithm family as the paper. WEF/Aramco Case 10 (p. 24): Ford and Quantinuum model LiCoO₂ cathode chemistry with a building-block workflow, with the quantum step on the hardest part, compared against classical methods; the paper calls the overall state "still early-stage R&D". Our molecule (LiH) is a stand-in chosen for size, not a cathode material.
"""

ACC = r"""assert QUBITS == 4 and ham.op.num_qubits == 4 and len(ham.generators) == 8
assert abs(e_hf_circuit - (-7.86186)) < 1e-4
assert abs(e_exact - (-7.88107)) < 1e-4
assert abs(e_vqe - e_exact) < 1.6e-3
assert e_fci_full < e_vqe < e_hf_circuit
assert R_min == 1.6
assert sum(e > 1.6 for e in hea_errors) >= 2, "a generic ansatz should miss chemical accuracy from most starts"
print("acceptance passed")
"""

# Toolbox cells: taken from tools/toolbox.py when it has an entry for notebook 6, else the copy below (from the uploaded notebook).
_TB_MD = r"""### 🧰 Toolbox: how the calls you will need are written · صندوق الأدوات

No internet? No problem. The cell under this table runs each call once on a **small toy that has nothing to do with the task**, so you can see the exact spelling, the order of the arguments, and what comes back. Run it, read the printed lines, and come back to it whenever a hint says "a method of ...". *What* to build, and with *which numbers*, is your job.

| To do | How it is written | What you get |
|---|---|---|
| a circuit with 4 qubits; $X$ on listed qubits | `qc = QuantumCircuit(4)`, `qc.x([0, 1])` | `x` takes a qubit number or a list |
| average value of an operator in a circuit's state | `estimator = qff.quantum.sv_estimator()`, then `float(estimator.run([(qc, op)]).result()[0].data.evs)` | one number (without the constant `shift`) |
| a list of named knobs | `t = ParameterVector("t", 8)` | `t[0]`, `t[1]`, ... are knobs |
| a gate with a knob | `PauliEvolutionGate(G, time=t[0])` | the gate $e^{-i\,t_0G}$ |
| add a gate on chosen qubits | `qc.append(gate, range(4))` | added at the end of `qc` |
| one VQE run | `VQE(estimator, qc, SLSQP(maxiter=200), initial_point=x0, callback=fn).compute_minimum_eigenvalue(op)` | `res.eigenvalue.real`, `res.optimal_point`, `res.cost_function_evals` |
| the callback | `def fn(eval_count, params, value, meta): ...` | called after every energy evaluation |
| what a `Ham` object holds | `ham.op`, `ham.shift`, `ham.hf_x_qubits`, `ham.generators`, `ham.e_hf`, `ham.e_exact`, `ham.e_fci_full` | all the numbers the tasks need (`ham = qff.chem.load_hamiltonian(path)`) |
"""
_TB_CODE = r"""# Toolbox for 06, on a 1-qubit toy (find the lowest average of Z reachable by a rotation; the answer is -1).
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP

estimator = qff.quantum.sv_estimator()
flip = QuantumCircuit(2); flip.x(1)                        # X on qubit 1: the state |10> (qubit 1 set)
op2 = SparsePauliOp(["ZI", "IZ"], coeffs=[1.0, 0.5])       # Z on qubit 1 plus half of Z on qubit 0 (the LAST letter is qubit 0)
evs = float(estimator.run([(flip, op2)]).result()[0].data.evs)
print("average of the operator in the state:", evs, "(-1 + 0.5)")
assert abs(evs + 0.5) < 1e-12

t = ParameterVector("t", 1)                                # one named knob
rot = QuantumCircuit(1)
rot.append(PauliEvolutionGate(SparsePauliOp("Y"), time=t[0]), [0])      # the gate exp(-i t Y), on qubit 0
log = []
vqe = VQE(estimator, rot, SLSQP(maxiter=100), initial_point=[0.3], callback=lambda count, params, value, meta: log.append(value))
res = vqe.compute_minimum_eigenvalue(SparsePauliOp("Z"))
print(f"VQE: lowest value {res.eigenvalue.real:.4f} | knob {res.optimal_point[0]:.3f} | {res.cost_function_evals} evaluations | callback saw {len(log)} values")
assert abs(res.eigenvalue.real + 1) < 1e-4
files = [f for f in sorted(os.listdir(DATA)) if f.startswith("lih_parity")]
print("LiH files in the data folder:", files)
print("OK: toolbox 06 ran.")
"""
TB_MD, TB_CODE = TOOLBOX.get(6, (_TB_MD, _TB_CODE))

GLOSSARY_EXTRA = [
    "VQE", "ansatz", "Hamiltonian", "ground state", "expectation value",
    ("eigenvalue", "القيمة الذاتية", "G"), ("eigenstate", "الحالة الذاتية", "G"), ("active space", "الفضاء النشط", "I"),
    ("chemical accuracy", "الدقة الكيميائية", "I"), ("variational principle", "المبدأ التغايري", "I"),
    ("excitation (hop)", "الإثارة (القفزة)", "I"),
]


def main():
    """Write the solution notebook."""
    nb = new(6, "Battery cathode chemistry",
             use_case="Battery cathode chemistry: the energy of a lithium bond",
             algorithm="VQE on a small active space (LiH, shipped Hamiltonian files)", qubits=4, minutes=3)
    nb.section(1)
    nb.md(S1_MD_05)
    nb.section(2)
    nb.md(S2_MD_07)
    nb.code(S2_CODE_08)
    nb.code(S2_CODE_09)
    nb.md(S2_MD_10)
    nb.code(S2_CODE_11)
    nb.section(3)
    nb.md(S3_MD_13)
    nb.code(S3_CODE_14)
    nb.md(S3_MD_15)
    nb.md(S3_MD_16)
    nb.md(S3_MD_17)
    nb.code(S3_CODE_18)
    nb.md(S3_MD_19)
    nb.md(S3_MD_20)
    nb.code(S3_CODE_21)
    nb.md(S3_MD_22)
    nb.md(S3_MD_23)
    nb.code(S3_CODE_24)
    nb.md(S3_MD_25)
    nb.md(TB_MD)
    nb.code(TB_CODE)
    nb.md(S3_MD_28)
    nb.task("T1", 'prepare the Hartree-Fock state on 4 qubits and measure its energy with the Estimator', T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.md(S3_MD_31)
    nb.code(S3_CODE_32)
    nb.md(S3_MD_33)
    nb.task("T2", 'build the 8-parameter ansatz from the generators and run VQE to within 1.6 mHa of exact', T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_MD_36)
    nb.code(S3_CODE_37)
    nb.code(S3_CODE_38)
    nb.code(S3_CODE_39)
    nb.md(S3_MD_40)
    nb.md(S3_MD_41)
    nb.code(S3_CODE_42)
    nb.md(S3_MD_43)
    nb.md(S3_MD_44)
    nb.md(S3_MD_45)
    nb.task("T3", 'run VQE at all seven bond lengths with a warm start; find the equilibrium R', T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.code(S3_CODE_48)
    nb.md(S3_MD_49)
    nb.md(S3_MD_50)
    nb.code(S3_CODE_51)
    nb.code(S3_CODE_52)
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
    nb.section5(use_case="Battery makers need the energy of lithium bonds to choose good cathode materials.",
                counting_fact="4 qubits are enough here; describing the electrons of one cobalt oxide unit exactly needs more numbers than any classical memory holds.",
                single_idea="Guess a wavefunction with knobs, measure its energy, turn the knobs downhill.",
                ideas=["An interactive bond-stretch slider with the three curves (Hartree-Fock, VQE, full FCI)",
                       "A 'tuning a radio' metaphor video for the variational loop",
                       "A card game where each card is one electron excitation (a hop)"],
                reuse="the convergence plot with the chemical-accuracy band, the bond curve, the table of errors for the two ansätze")
    nb.section6(honest="VQE matched the exact answer for a 4-qubit toy; the exact answer was available classically.",
                question_a="What is chemical accuracy and why 1.6 mHa?",
                question_b="Why not just use more qubits for a real cathode?")
    path = nb.save(os.path.join(ROOT, "solutions", "06_store_battery_vqe_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()