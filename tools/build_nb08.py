"""Build solutions/08_store_carbon_capture_qpe_solution.ipynb (notebook 8: MOF carbon capture, QPE on an illustrative active-site model).

    python tools/build_nb08.py

No variational loop: QPE reads energies off the phase of exp(-iHt) for an invented two-orbital model (labelled illustrative
everywhere). The notebook finds the right electron-number sector, builds the QPE circuit, climbs the precision ladder and
subtracts two energies to get a binding curve. Reusable pieces defined in the notebook: `ham`, `qpe_energy` (T2) and `E_qpe`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1_MD_05 = r"""> **ILLUSTRATIVE MODEL · نموذج توضيحي.** Everything numeric in this notebook comes from a two-orbital model whose parameters we *invented* for teaching. It is not a calculation of any real material. No binding energy printed here may be quoted as a property of a real metal–organic framework.

**The problem.** A metal–organic framework (MOF) is a sponge made of metal sites joined by organic links. To capture carbon, a CO₂ molecule has to settle onto a metal site and stay there; later it has to be released again. The number that decides this is the **binding energy**: the energy released when CO₂ moves from far away to its place on the site. Too small and the CO₂ escapes; too large and releasing it costs too much energy.

**Who faces it.** A materials developer screening candidate sorbents, and an industrial emitter or carbon-capture operator that has to choose among them. Both want the binding energy before anyone makes the material in a lab.

<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

**What is real today.** The WEF/Aramco paper "Quantum for Energy and Utilities" (April 2026), Case 5 (p. 15): TotalEnergies and Quantinuum work on predicting CO₂ binding in an aluminium-fumarate MOF. The workflow is split into building blocks: a quantum step targets the active site, where the electrons interact most strongly, and classical methods handle the rest of the material. The results are checked against high-level classical calculations. The paper presents this as a workflow under development, not as a quantum advantage; check the paper for the sizes reached.

</div>

**Why a quantum computer might help.** The electrons at an active site interact strongly, and the memory a classical computer needs to describe them exactly doubles with every added orbital; QPE reads the exact energy of the active site without that blow-up. What is still missing is error-corrected hardware able to run the very long circuits that QPE needs, and a good starting state.

| Paper's method | Our teaching route |
|---|---|
| A building-block workflow: a quantum calculation on the active site of a real Al-fumarate MOF, classical methods for the rest, benchmarked against classical results. | We keep the split (quantum active site + classical remainder) but replace the chemistry with an **invented two-orbital model** and use **QPE**, because QPE is in the Part 2 lectures and is the method expected on future error-corrected machines. A teaching choice, not the paper's method. |
"""

S2_MD_07 = r"""> **ILLUSTRATIVE MODEL · نموذج توضيحي.** Invented parameters, invented geometry dependence. Energies are in Hartree (Ha); chemical accuracy = 1.6 mHa.

**The toy: CO₂ approaching an open metal site, in two orbitals.**
- Orbital **C**: a filled orbital on the CO₂ (the donor). Orbital **M**: an empty orbital on the framework's open metal site (the acceptor). Two electrons, both starting on CO₂.
- Parameters (Ha): ε_C = −0.9, ε_M = −0.2, U_C = 0.3, U_M = 0.5 (U is the extra cost of putting two electrons in one orbital).
- Distance R (Å) between CO₂ and the site changes two things: the **hopping** t(R) = 0.15·exp(−(R − 2.0)/0.5), which lets electrons move between C and M, and a **classical repulsion** V(R) = 0.05·exp(−(R − 2.0)/0.15) that stands for everything outside the two orbitals.
- Far away, t = 0 and the energy is exactly 2ε_C + U_C = −1.5 Ha.

**The quantity:** binding energy(R) = E₀(R) + V(R) − E₀(far), where E₀ is the lowest two-electron energy of the two-orbital part.

Each orbital holds a spin-up and a spin-down electron, so there are 4 spin orbitals and **4 system qubits** (Jordan–Wigner, order [C↑, M↑, C↓, M↓] = qubits 0 to 3). QPE adds *m* evaluation qubits; with the default m = 4 the circuit has **8 qubits**.
"""

S2_CODE_08 = r"""QUBITS = 8                       # 4 system qubits + m = 4 evaluation qubits
M_DEFAULT = 4
LABEL = "ILLUSTRATIVE MODEL (invented parameters)"
from qff.chem import load_hamiltonian

R_LIST = [1.8, 1.9, 2.0, 2.1, 2.2, 2.4, 2.8, 3.5]
ham = {R: load_hamiltonian(f"{DATA}/site_model_R{R}.json") for R in R_LIST}
ham["far"] = load_hamiltonian(f"{DATA}/site_model_far.json")
V = {key: h.meta["V"] for key, h in ham.items()}            # the classical repulsion, added by hand (not in the qubit operator)
print(LABEL)
print("qubits:", ham[2.0].num_qubits, "| Pauli terms at R = 2.0:", len(ham[2.0].op), "| at 'far':", len(ham["far"].op))
for label, coeff in zip(ham[2.0].op.paulis.to_labels(), ham[2.0].op.coeffs.real):
    print(f"  {coeff:+.3f}  {label}")
"""

S2_MD_09 = r"""**Classical baseline: runs first.** Two classical answers before any quantum circuit: a pencil-and-paper estimate and the exact answer stored in the shipped file. The pencil estimate is second-order perturbation theory: moving two electrons across a gap Δ = ε_M − ε_C − U_C = 0.4 Ha with hopping t lowers the energy by about 2t²/Δ. The exact answer comes from diagonalising the model (4 states in the right electron sector); in task T1 you will compute it yourself and meet a trap.
"""

S2_CODE_10 = r"""# Classical baseline (runs first)
eps_C, eps_M, U_C = -0.9, -0.2, 0.3
t = ham[2.0].meta["t"]
delta = eps_M - eps_C - U_C
second_order = -2 * t**2 / delta                                   # orbital energy lowering, pencil and paper
e_file = {key: h.e_exact for key, h in ham.items()}                # sector ground energy stored in the shipped files
binding_exact_mHa = (e_file[2.0] + V[2.0] - e_file["far"]) * 1000  # E0(R) + V(R) - E0(far)
print(f"[{LABEL}]")
print(f"pencil and paper : lowering {second_order*1000:7.1f} mHa  -> binding {(second_order + V[2.0])*1000:7.1f} mHa")
print(f"exact (from file): lowering {(e_file[2.0]-e_file['far'])*1000:7.1f} mHa  -> binding {binding_exact_mHa:7.1f} mHa  (E0 = {e_file[2.0]:.5f}, far = {e_file['far']:.5f})")
print("Forgetting V gives", round((e_file[2.0] - e_file["far"]) * 1000, 1), "mHa instead of", round(binding_exact_mHa, 1))
"""

S3_MD_12 = r"""### 3.0 · The map of this section

<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**The idea in one sentence.** The energy of CO₂ sitting on a site is the *lowest eigenvalue* of a small matrix (the model's Hamiltonian); quantum phase estimation (QPE) turns an eigenvalue into a *phase* and reads it as a binary number, one digit per extra qubit. The **binding energy** is the difference of two such energies, plus a classical correction. The model is **invented** for teaching: nothing here is a property of a real material.

</div>

Everything below builds from things you already know: lists of numbers (vectors), matrices, angles and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**, so you can watch the maths run and check it against Qiskit.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| 3.1 | The model's Hamiltonian, and why only 4 of its 16 states matter | (1), (2), (3) | `second_order`, `e_file` | **T1** |
| 3.2 | An energy becomes a phase; the window; aliasing | (4) | `phis_raw` | |
| 3.3 | Phase kickback, the inverse Fourier transform, the bit-order trap | (5), (6) | `register`, `phase_from_counts` | |
| 3.4 | The start state and what QPE returns | (7), (8) | `overlap`, `make_U` | **T2** |
| 3.5 | Precision ladder and the binding energy | (9), (10) | `binding_qpe_mHa` | **T3** |
| 3.6 | What hardware would need: Trotter steps | (11) | `cx_trotter` | |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it.
"""

S3_CODE_13 = r"""qff.plots.plot_flow(
    [("Start state", "both electrons on CO2", "source"), ("Hadamards", "on m evaluation qubits", "move"), ("Controlled U", "U, U^2, U^4, ...", "store"),
     ("Inverse QFT", "turns phase into bits", "use"), ("Measure", "read m bits", "source")],
    title="Quantum phase estimation: a fixed circuit, no tuning loop (ILLUSTRATIVE MODEL)");
"""

S3_MD_14 = r"""**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).

| Symbol | Meaning | In the code |
|---|---|---|
| $\varepsilon_C,\ \varepsilon_M,\ U_C,\ U_M$ | orbital energies and two-electron penalties (invented) | `eps_C`, `eps_M`, `U_C`, `U_M` |
| $t(R),\ V(R)$ | hopping and classical repulsion at distance $R$ (invented) | `ham[R].meta["t"]`, `V[R]` |
| $N_\uparrow,\ N_\downarrow$ | number of up and down electrons | the sector of a basis state |
| $E_a,\ E_c,\ E_b$ | energies of the three block states without hopping | `E_a`, `E_c`, `E_b` |
| $U,\ \varphi$ | unitary built from $H$; its phase | `make_U(model)`, `phi` |
| $E_{\min},\ W,\ m$ | window bottom, width, evaluation qubits | `E_MIN`, `W`, `m` |
| $c_n,\ \lvert c_n\rvert^2$ | eigenstate in the start state; overlap | `overlap` |
| $\text{binding}(R)$ | $E_0(R)+V(R)-E_0(\text{far})$ | `binding_qpe_mHa`, `binding_curve_mHa` |
"""

S3_MD_15 = r"""<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**

**The one line to remember.** Same as notebook 7: $E=E_{\min}+\varphi W$ with grid step $W/2^m$ (equations 4 and 9), and the binding energy is a *difference*, $E_0(R)+V(R)-E_0(\text{far})$ (equation 10), so both energies must be read precisely. (The model is invented; the maths is real.)

</div>
"""

S3_MD_16 = r"""### 3.1 · The model's Hamiltonian, and which states matter

> **ILLUSTRATIVE MODEL · نموذج توضيحي.** The numbers $\varepsilon_C,\varepsilon_M,U_C,U_M,t(R),V(R)$ are invented for teaching.

**Electrons and orbitals.** An electron sits in an **orbital**, and each orbital has two "seats" (spin up and spin down). Our model has two orbitals, C (on the CO₂) and M (on the metal site), so $2\times2=4$ seats (**spin orbitals**) $[C\!\uparrow,M\!\uparrow,C\!\downarrow,M\!\downarrow]$, one qubit each: a state says which seats are taken, one bit per seat. The **Hamiltonian** $H$ is the energy operator: it adds up the energy of each electron in its orbital, a penalty $U$ when two electrons share an orbital, and a **hopping** term $-t$ that lets an electron jump between C and M:

$$H=\sum_{\sigma=\uparrow,\downarrow}\Bigl[\varepsilon_C\,n_{C\sigma}+\varepsilon_M\,n_{M\sigma}-t\,\bigl(c^\dagger_{C\sigma}c_{M\sigma}+c^\dagger_{M\sigma}c_{C\sigma}\bigr)\Bigr]+U_C\,n_{C\uparrow}n_{C\downarrow}+U_M\,n_{M\uparrow}n_{M\downarrow} \tag{1}$$

Here $n=c^\dagger c$ is 1 if its seat is taken and 0 otherwise, and $c^\dagger_{M\sigma}c_{C\sigma}$ moves an electron from C to M. As a $16\times16$ matrix this is what `ham[R].op` holds (after a mapping called Jordan-Wigner). The parameters are $\varepsilon_C=-0.9$, $\varepsilon_M=-0.2$, $U_C=0.3$, $U_M=0.5$ Ha.

**Hopping never changes the number of electrons.** Every term in (1) either counts electrons or moves one, so $H$ conserves the number of spin-up electrons $N_\uparrow$ and of spin-down electrons $N_\downarrow$. Sorting the 16 states by $(N_\uparrow,N_\downarrow)$ makes the matrix **block diagonal**: states with different electron numbers never mix. With two seats per spin, the block with $N_\uparrow$ up and $N_\downarrow$ down electrons has size

$$\dim=\binom{2}{N_\uparrow}\binom{2}{N_\downarrow},\qquad \text{e.g. } N_\uparrow=N_\downarrow=1:\ 2\cdot2=4\ \text{states} \tag{2}$$

(the nine blocks have sizes $1,2,1,2,4,2,1,2,1$, adding up to 16). **Our molecule has one electron of each spin**, so only the $(1,1)$ block, 4 states, is the physical one.

**A worked example you can check on paper: that block by hand.** Its four states are: **a** = both electrons on CO₂ ($C\!\uparrow C\!\downarrow$); **b** = both on the metal ($M\!\uparrow M\!\downarrow$); **c** and **d** = one electron on each ($C\!\uparrow M\!\downarrow$ and $M\!\uparrow C\!\downarrow$). Their energies without hopping are $E_a=2\varepsilon_C+U_C=-1.5$, $E_{c}=E_d=\varepsilon_C+\varepsilon_M=-1.1$, $E_b=2\varepsilon_M+U_M=+0.1$ Ha. Hopping connects **a** to $\mathbf c$ and $\mathbf d$ and **b** to $\mathbf c$ and $\mathbf d$, each with strength $t$ (up to a sign). The combination $\mathbf s=(\mathbf c+\mathbf d)/\sqrt2$ couples to both with strength $\sqrt2\,t$, while $(\mathbf c-\mathbf d)/\sqrt2$ is left alone at $-1.1$. So only a $3\times3$ matrix in the basis $(\mathbf a,\mathbf s,\mathbf b)$ matters:

$$H_{3\times3}=\begin{pmatrix}E_a&\sqrt2\,t&0\\ \sqrt2\,t&E_c&\sqrt2\,t\\ 0&\sqrt2\,t&E_b\end{pmatrix}\ \xrightarrow{\ t=0.15\ }\ \begin{pmatrix}-1.5&0.2121&0\\0.2121&-1.1&0.2121\\0&0.2121&0.1\end{pmatrix} \tag{3}$$

*Pencil-and-paper estimate* (second-order perturbation theory, valid for small $t$): the energy gap is $\Delta=E_c-E_a=0.4$, and state **a** is pushed down by $\frac{(\sqrt2t)^2}{\Delta}=\frac{2t^2}\Delta=\frac{2\cdot0.0225}{0.4}=0.1125$ Ha, so $E_0\approx-1.6125$. Keeping only **a** and **s** (a $2\times2$ matrix) gives $\bar E-\sqrt{(\Delta/2)^2+2t^2}=-1.3-\sqrt{0.04+0.045}=-1.5915$. The full $3\times3$ answer, found by diagonalising (3), is $-1.59588$: the estimate is too low by 16.6 mHa because $t=0.15$ is not small compared with $\Delta=0.4$, the two-state answer is 4.3 mHa too high, and the exact value, stored in the file, lies between. This is why we need the exact diagonalisation, and QPE has to reproduce it.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Hamiltonian | مؤثر هاملتون (الهاملتوني) | G | Block-diagonal matrix | مصفوفة قطرية كتلية | I |
| Orbital | المدار | I | Binding energy | طاقة الارتباط | I |
| Spin orbital | المدار البرمي | I | Eigenvalue | القيمة الذاتية | G |
| Hopping | القفز (الانتقال بين المدارات) | I | Active site | الموقع النشط | I |
| Electron number sector | قطاع عدد الإلكترونات | I | Illustrative model | نموذج توضيحي | I |
"""

S3_CODE_17 = r"""# Equations (1)-(3) with numbers: the 3x3 block at R = 2.0 (ILLUSTRATIVE MODEL, invented parameters).
t_20 = ham[2.0].meta["t"]
U_M = 0.5
E_a, E_c, E_b = 2 * eps_C + U_C, eps_C + eps_M, 2 * eps_M + U_M      # energies without hopping: both on CO2, one each, both on the metal
H3 = np.array([[E_a, np.sqrt(2) * t_20, 0], [np.sqrt(2) * t_20, E_c, np.sqrt(2) * t_20], [0, np.sqrt(2) * t_20, E_b]])
w3 = np.linalg.eigvalsh(H3)
two_level = (E_a + E_c) / 2 - np.hypot((E_c - E_a) / 2, np.sqrt(2) * t_20)
print("3x3 block eigenvalues (Ha):", np.round(w3, 5))
print(f"pencil estimate (second order) {second_order + E_a:.5f} | two-state formula {two_level:.5f} | exact 3x3 {w3[0]:.5f} | value in the file {e_file[2.0]:.5f}")
assert abs(w3[0] - e_file[2.0]) < 1e-8, "the 3x3 block should reproduce the ground energy stored in the file"
print("OK: the 3x3 block of equation (3) gives the exact ground energy of the model.")
print("the 4th state of the block, (c - d)/sqrt2, is untouched by hopping at", E_c, "Ha")
"""

S3_MD_20 = r"""### T1 · The lowest eigenvalue of the matrix is the wrong answer · التمرين ١

**What you are about to build, and why.** The $16\times16$ matrix contains states with 0, 1, 2, 3 and 4 electrons. If you ask a computer for its lowest eigenvalue, it returns the lowest energy over *all* of them, and that is **not** our molecule: it is a state with an extra electron. In T1 you write `sector_ground(op)`, which keeps only the basis states with exactly one electron among the up seats (qubits 0 and 1) and exactly one among the down seats (qubits 2 and 3), builds the small block, and returns *its* lowest eigenvalue. Then you compare the two answers at R = 2.0 and far away. This is the main trap of the notebook.

**The formula you implement:** equation (2): keep the $(N_\uparrow,N_\downarrow)=(1,1)$ block. Basis state number $k$ (a number from 0 to 15) has seat $i$ taken when bit $i$ of $k$ is 1.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Basis state number k has qubit i set when bit i of k is 1. How many up seats are taken in k? Count the set bits among bits 0 and 1; the down seats are bits 2 and 3. Python can count the set bits of a number in a few ways (for example by turning it into a binary string).

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps inside `sector_ground(op)`: get the matrix of the operator; make the list of basis states that have exactly one up and one down electron; pick that block out of the matrix (NumPy has a helper to select rows and columns together); return the smallest eigenvalue of the block. Outside, compute the lowest eigenvalue of the whole matrix for comparison.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
def sector_ground(op):
    M = ______          # the 16 x 16 matrix
    keep = ______       # the positions k with one up electron and one down electron
    return ______       # the smallest eigenvalue of the block M[keep, keep]
```
Test yourself: at R = 2.0 the sector ground energy is -1.59588 Ha, far away it is -1.5 Ha, and the full-matrix minimum is far lower (a 3-electron state).

</details>
"""

T1_CELL = r'''# TODO (team) T1: sector ground energy versus the full-matrix minimum
def sector_ground(op):
    """Lowest eigenvalue among basis states with one electron in qubits {0,1} and one in qubits {2,3}."""
    ### STARTER: raise NotImplementedError("T1: write sector_ground(op)")
    ### BEGIN SOLUTION
    M = op.to_matrix()
    keep = [k for k in range(16) if bin(k & 0b0011).count("1") == 1 and bin(k & 0b1100).count("1") == 1]
    return np.linalg.eigvalsh(M[np.ix_(keep, keep)])[0]
    ### END SOLUTION

e_sector = {key: sector_ground(ham[key].op) for key in (2.0, "far")}
full_min = {key: np.linalg.eigvalsh(ham[key].op.to_matrix())[0] for key in (2.0, "far")}
for key in (2.0, "far"):
    print(f"{key!s:>4}: sector ground {e_sector[key]:.5f} | lowest eigenvalue of the full 16x16 matrix {full_min[key]:.5f}")
'''

S3_CODE_22 = r"""# Self-check
assert abs(e_sector[2.0] - (-1.59588)) < 1e-5 and abs(e_sector["far"] - (-1.5)) < 1e-9, "T1: sector ground should be -1.59588 at R = 2.0 and -1.5 far away"
assert abs(e_sector[2.0] - e_file[2.0]) < 1e-8, "T1: your sector ground should match the value stored in the file"
assert full_min[2.0] < e_sector[2.0] - 0.1, "T1: the full-matrix minimum should sit well below the sector ground (it is a 3-electron state)"
"""

S3_MD_23 = r"""The full-matrix minimum (about −1.72 at R = 2.0 and −1.70 far away) belongs to **three-electron states**: a molecule with an extra electron, which is not the molecule we want. A binding energy built from those two numbers would be +25.7 mHa, the wrong sign. We will **not** repair this by editing the operator. QPE never sees those states, because the Hamiltonian conserves electron number and our start state has two electrons.

**The start state.** How much of the right answer does "both electrons on CO₂" contain?
"""

S3_CODE_24 = r"""# After T1: the block structure of equation (2), found from the matrix itself. Left: the matrix as stored. Right: states grouped by block.
from scipy.sparse.csgraph import connected_components
from math import comb

M_full = ham[2.0].op.to_matrix()
n_blocks, block_of = connected_components(np.abs(M_full) > 1e-12, directed=False)         # states joined by a non-zero entry share a block
order = np.argsort(block_of, kind="stable")
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
axes[0].imshow(np.abs(M_full) > 1e-12, cmap="Blues"); axes[0].set_title("as stored: nonzero entries", fontsize=9)
axes[1].imshow(np.abs(M_full[np.ix_(order, order)]) > 1e-12, cmap="Blues"); axes[1].set_title(f"states grouped by block: {n_blocks} blocks", fontsize=9)
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout()
sizes = sorted(np.bincount(block_of).tolist())
expected = sorted(comb(2, nu) * comb(2, nd) for nu in range(3) for nd in range(3))          # equation (2) for every (N_up, N_down)
print("block sizes found in the matrix:", sizes)
print("equation (2) predicts          :", expected)
assert sizes == expected, "equation (2) should predict the block sizes"
print("OK: hopping never mixes states with different electron numbers; the largest block (4 states) is the physical one.")
"""

S3_MD_25 = r"""<div style="border:1px solid rgba(0,125,121,0.35);border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">💡 Idea · فكرة</span>

**A conserved quantity cuts a big problem into small ones.** Because hopping never changes the number of electrons of each spin, the $16\times16$ matrix is really nine independent blocks, and only the $(1,1)$ block matters for our molecule (and inside it only a $3\times3$ matrix). Always look for such a symmetry *before* you spend qubits: it is free, and it tells you which states QPE can ever return.

</div>
"""

S3_MD_26 = r"""### 3.2 · From an energy to a phase

**Eigenstates and phases.** A **unitary** $U$ is a reversible operation (a matrix that rotates states without changing their length). Some special states, its **eigenstates** $|u\rangle$, come back unchanged except for a **phase**, an angle written $e^{2\pi i\varphi}$ ($e^{i\theta}=\cos\theta+i\sin\theta$ is a point on a circle of radius 1 at angle $\theta$; multiplying an amplitude by it *turns it* and changes no probability). The number $\varphi$, a fraction of a full turn with $0\le\varphi<1$, is what **quantum phase estimation (QPE)** measures. To use it for chemistry we build $U$ *from the Hamiltonian* so that **the energy sets the phase**. If every energy we care about lies in a window $[E_{\min},E_{\min}+W)$:

$$U=e^{\,i\,(H-E_{\min})\,2\pi/W}\quad\Longrightarrow\quad U|E\rangle=e^{2\pi i\varphi}|E\rangle,\qquad \varphi=\frac{E-E_{\min}}{W},\quad E=E_{\min}+\varphi\,W \tag{4}$$

A phase is an *angle*: $\varphi$ and $\varphi+1$ are the same turn. An energy **outside** the window **wraps around** (aliases) and is read as a different energy inside it. We use $E_{\min}=-1.75$ and $W=0.5$, i.e. the window $[-1.75,-1.25)$, which holds the ground state ($-1.5959$) but **not** the excited state at $-1.0413$ Ha: its phase is $\frac{-1.0413+1.75}{0.5}=1.417$, which wraps to $0.417$ and is read as $-1.5413$ Ha (a small wrong peak that section 3.4 shows). The ground state gives $\varphi=\frac{-1.5959+1.75}{0.5}=0.3082$.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Quantum phase estimation (QPE) | تقدير الطور الكمومي (QPE) | G | Unitary matrix | مصفوفة عكوسة/وحدوية | G |
| Phase | الطور | U | Energy window | نافذة الطاقة | I |
| Eigenvalue | القيمة الذاتية | G | Aliasing (wrap-around) | الالتفاف الطوري | I |
| Eigenstate | الحالة الذاتية | G |  |  |  |
"""

S3_CODE_27 = r"""# Equation (4) in code: the four energies of the (1 up, 1 down) block as phases, with the wrap-around of the excited state.
E_MIN_WIN, W_WIN = -1.75, 0.5
sector_vals = np.sort(np.concatenate([np.linalg.eigvalsh(H3), [E_c]]))             # the 3x3 block plus the state (c - d)/sqrt2 that hopping leaves alone
phis_raw = (sector_vals - E_MIN_WIN) / W_WIN                                    # equation (4), without wrapping
phis_wrapped = phis_raw % 1                                                      # a phase is an angle: only the fractional part is measured
print(f"{'energy (Ha)':>12}{'phase':>9}{'wrapped':>9}{'read as (Ha)':>14}")
for e, p0, p1 in zip(sector_vals, phis_raw, phis_wrapped):
    print(f"{e:>12.5f}{p0:>9.4f}{p1:>9.4f}{E_MIN_WIN + p1 * W_WIN:>14.5f}" + ("   <- outside the window" if not (0 <= p0 < 1) else ""))
fig = plt.figure(figsize=(4.4, 4.4))
ax = fig.add_subplot(111, projection="polar")
for k in range(16):
    ax.plot([2 * np.pi * k / 16] * 2, [0.97, 1.0], color="#C6C6C6", lw=1)             # the grid of a 4-qubit register
for e, p1 in zip(sector_vals, phis_wrapped):
    inside = E_MIN_WIN <= e < E_MIN_WIN + W_WIN
    ax.annotate("", xy=(2 * np.pi * p1, 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#0F62FE" if inside else "#D02670", lw=2))
    ax.text(2 * np.pi * p1, 1.22, f"{e:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.4); ax.set_yticks([]); ax.set_title("energy = angle (blue: inside the window, magenta: wrapped)", fontsize=8, pad=12)
plt.tight_layout()
"""

S3_MD_28 = r"""### 3.3 · Phase kickback and the inverse Fourier transform

**Kickback.** Take one extra *control* qubit in the equal mixture $\frac1{\sqrt2}(|0\rangle+|1\rangle)$ (a Hadamard gate makes it from $|0\rangle$) and apply $U$ to the system **only when the control is $|1\rangle$**. If the system is in an eigenstate, it is unchanged, but the control picks up the phase, **kicked back** onto it: $\frac1{\sqrt2}(|0\rangle+e^{2\pi i\varphi}|1\rangle)$. With $m$ control qubits (the **evaluation register**), control $k$ applying $U^{2^k}$ (that is $U,U^2,U^4,\dots$), and $y=\sum_ky_k2^k$ written in binary, the $2^m$ amplitudes of the register are

$$\frac1{\sqrt{2^m}}\sum_{y=0}^{2^m-1}e^{2\pi i\,\varphi\,y}\,|y\rangle \tag{5}$$

a *spiral* of unit arrows turning by $2\pi\varphi$ per step. The **inverse quantum Fourier transform** turns the spiral of speed $\varphi$ into a single outcome $|k\rangle$ when $\varphi=k/2^m$ exactly. The amplitude of outcome $k$ is

$$A_k=\frac1{2^m}\sum_{y=0}^{2^m-1}e^{2\pi i\,(\varphi-k/2^m)\,y} \tag{6}$$

If $\varphi=k_0/2^m$: for $k=k_0$ the sum is $\frac1{2^m}\cdot2^m=1$ (all arrows agree), for any other $k$ it is 0 (the arrows cancel in a full circle). **The register reads $k_0$ with certainty, and $\varphi=k_0/2^m$ is a binary fraction with $m$ digits.** Check with $\varphi=\frac38=0.011_2$ and $m=3$: outcome $k=3$ with probability 1 (cell below).

<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**Bit order: the trap.** Qiskit's `phase_estimation` leaves out the final qubit swaps of the Fourier transform. If evaluation qubit $i$ is measured into classical bit $i$, the printed count string has qubit 0 on the **right** and the bits of $k$ come out **reversed**. Always read the string reversed, or use `qff.bits.phase_from_counts`: it does exactly that. (Notebook 7 has a one-qubit warm-up for this trap.)

</div>

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Phase kickback | ارتداد الطور | I | Control qubit | كيوبت التحكم | I |
| Evaluation register | سجل التقدير | I | Superposition | التراكب | G |
| Quantum Fourier transform | تحويل فورييه الكمومي | G | Hadamard gate | بوابة هادامارد | G |
| Inverse QFT | تحويل فورييه الكمومي العكسي | I |  |  |  |
"""

S3_CODE_29 = r"""# Equations (5)-(6) with plain numpy, no quantum library: phi = 3/8 with m = 3 evaluation qubits reads k = 3 with certainty.
m_demo, phi_demo = 3, 3 / 8
M_demo = 2 ** m_demo
y = np.arange(M_demo)
register = np.exp(2j * np.pi * phi_demo * y) / np.sqrt(M_demo)                                   # equation (5): a spiral of unit arrows
A = np.array([np.exp(2j * np.pi * (phi_demo - k / M_demo) * y).sum() / M_demo for k in range(M_demo)])   # equation (6)
print("outcome probabilities |A_k|^2 for k = 0..7:", np.round(np.abs(A) ** 2, 6))
assert np.isclose(abs(A[3]) ** 2, 1.0), "phi = 3/8 = 0.011 in binary: the register must read k = 3 with certainty"
fig = plt.figure(figsize=(8, 3.2))
ax1 = fig.add_subplot(1, 2, 1, projection="polar")
for yy in y:
    ax1.annotate("", xy=(np.angle(register[yy]), 1), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#0F62FE", lw=1.5))
    ax1.text(np.angle(register[yy]), 1.2, f"y={yy}", ha="center", fontsize=7)
ax1.set_ylim(0, 1.4); ax1.set_yticks([]); ax1.set_title("equation (5): the spiral for phi = 3/8", fontsize=9, pad=12)
ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(range(M_demo), np.abs(A) ** 2, color="#0F62FE"); ax2.set_xlabel("outcome k"); ax2.set_ylabel("probability")
ax2.set_title("equation (6): all weight on k = 3", fontsize=9)
plt.tight_layout()
"""

S3_MD_30 = r"""### 3.4 · The start state, and what QPE returns

When $\varphi$ is *not* a fraction $k/2^m$ (the usual case), equation (6) is a geometric sum and the probability of outcome $k$ is

$$P(k)=F_M\!\left(\varphi-\frac kM\right),\qquad F_M(\delta)=\frac{\sin^2(\pi M\delta)}{M^2\sin^2(\pi\delta)},\qquad M=2^m \tag{7}$$

$F_M(0)=1$ and $F_M$ falls quickly away from 0. If the phase sits almost half way between two grid points, the nearest outcome gets only about half of the shots (never less than $4/\pi^2\approx0.4$): QPE's answer can *jump* when you add a qubit. And the state we prepare matters: write it as a mix of eigenstates, $\sum_nc_n|E_n\rangle$. The register is a mixture of the answers for each eigenstate, weighted by the **overlap** $|c_n|^2$, the probability of finding the start state in eigenstate $n$:

$$P(k)=\sum_n|c_n|^2\,F_M\!\left(\varphi_n-\frac kM\right) \tag{8}$$

Our start state, both electrons on CO₂ (qubits 0 and 2), is *state* $\mathbf a$ of 3.1. The cell below shows how much of each eigenstate it contains: 82.8% in the ground state, 17.1% in the excited state at $-1.0413$ Ha. So QPE can return the ground energy with probability at most about **0.83**: it is the overlap, not a flaw in QPE. The excited state is *outside* the window, so (equation 4) it wraps and shows up as a wrong energy near $-1.54$: **read the tallest peak**.

<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Overlap | التراكب (التداخل الجزئي) بين حالتين | I | Evaluation qubits | كيوبتات التقدير | I |
| Start state | الحالة الابتدائية | I | Ground state | الحالة الأرضية | I |
| Grid step | خطوة الشبكة | I | Excited state | الحالة المثارة | I |
"""

S3_CODE_31 = r"""# Overlap of the start state with the eigenstates of the 2-electron sector (guided cell)
M = ham[2.0].op.to_matrix()
keep = np.flatnonzero(block_of == block_of[5])      # the block that contains the start state (position 5: qubits 0 and 2 set)
vals, vecs = np.linalg.eigh(M[np.ix_(keep, keep)])
start = np.zeros(16); start[(1 << 0) | (1 << 2)] = 1                      # X on qubits 0 and 2
overlap = np.abs(vecs.conj().T @ start[keep]) ** 2
print(f"start-state energy {(start @ M @ start).real:.4f} Ha (this is the 'far away' energy)")
for e, o in zip(vals, overlap):
    print(f"  eigenstate E = {e:+.4f} Ha   overlap {o:.3f}")
print("QPE can return the ground energy with probability at most about", round(overlap[0], 2), "(it is the overlap, not a flaw in QPE).")
"""

S3_MD_32 = r"""**Energy to phase to circuit.** The cell below builds U for any geometry with an exact matrix exponential. That is a classical shortcut (it is as much work as diagonalising the matrix); on a real machine U would have to be built from gates, which we price at the end of this section.
"""

S3_CODE_33 = r'''import scipy.linalg as sl
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, phase_estimation

E_MIN, W = -1.75, 0.5                         # the energy window [E_MIN, E_MIN + W)

def make_U(model, e_min=E_MIN, w=W):
    """U = exp(+i (H - e_min) 2 pi / w) as a gate: an eigenstate with energy E gets phase (E - e_min) / w."""
    H = model.op.to_matrix()
    return UnitaryGate(sl.expm(1j * (H - e_min * np.eye(H.shape[0])) * 2 * np.pi / w), label="U")

phi_true = (e_sector[2.0] - E_MIN) / W
print(f"ground state at R = 2.0: phase = ({e_sector[2.0]:.5f} - ({E_MIN})) / {W} = {phi_true:.5f}")
print("with 4 evaluation qubits the phase grid has 16 points:", phi_true * 16, "-> nearest grid point 5/16 = 0.3125 -> E =", E_MIN + 0.3125 * W)
'''

S3_MD_34 = r"""### T2 · Core step: assemble the QPE circuit and read the energy · التمرين ٢

**What you are about to build, and why.** The heart of the notebook: a function `qpe_energy(model, m)` that returns the energy QPE reads for a given geometry (it also accepts the energy window `e_min`, `w` and an optional ready-made gate `unitary`, because later cells reuse it with another window and with a Trotterised $U$). Four pieces: (1) a circuit with $m$ evaluation qubits **first** and then the 4 system qubits; (2) X gates on the system qubits that prepare the start state (`model.hf_x_qubits`, shifted by $m$ because the evaluation register comes first); (3) Qiskit's `phase_estimation(m, make_U(model))`, where `make_U` (given above) builds $U=e^{i(H-E_{\min})2\pi/W}$ as a gate (equation 4); (4) a measurement of the evaluation qubits into $m$ classical bits. Then you convert the most frequent outcome to an energy with `qff.bits.phase_from_counts` and $E=E_{\min}+\varphi W$. Run it at R = 2.0 and far away with $m=4$ and compute the **binding energy including $V$**.

**The formulas you implement:** (4) for $U$ and for $E$; the binding energy (10) below, $E_0(R)+V(R)-E_0(\text{far})$.

$$\text{binding}(R)=E_0(R)+V(R)-E_0(\text{far}) \tag{10}$$

$V(R)$ is a plain number from the file (`V[R]`), the classical correction for everything outside the two orbitals; it is **not** in the qubit operator, so you must add it by hand.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Think about which qubit is which. The evaluation register comes first: circuit qubits 0 to m-1. The system qubits come after, so system qubit q is circuit qubit m + q. The start state is prepared on the system qubits listed in `model.hf_x_qubits`. `make_U(model)` gives you the gate of equation (4).

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps inside `qpe_energy`: a circuit with m + 4 qubits and m classical bits; X gates on the shifted start-state qubits; compose the phase estimation circuit built from the gate (`make_U(model, e_min, w)` unless a gate was passed in); measure the evaluation qubits; sample `shots` runs and read the counts of register `c`; convert the phase with the `qff.bits` helper and then to an energy with the *given* window; return the energy, the frequency, the counts and the circuit. Afterwards fill `E_qpe`, `top_freq` and the binding energy (equation 10, including V).

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
def qpe_energy(model, m, shots=4096, e_min=E_MIN, w=W, unitary=None):
    qc = ______             # m evaluation qubits first, then the 4 system qubits
    ______                  # start state on the system qubits (mind the shift by m)
    ______                  # the phase estimation circuit for the gate (make_U(...) unless `unitary` is given)
    ______                  # measure the evaluation qubits
    counts = ______         # run the sampler
    phi, freq = ______      # the phase and its frequency
    return ______, freq, counts, qc    # the energy from equation (4), with this window
```
Self-check targets: far away the energy is exactly -1.5 Ha; at R = 2.0 with m = 4 it is the grid point -1.59375.

</details>
"""

T2_CELL = r'''# TODO (team) T2: QPE circuit, energies and binding energy at m = 4
def qpe_energy(model, m, shots=4096, e_min=E_MIN, w=W, unitary=None):
    """QPE for `model` with m evaluation qubits and the window [e_min, e_min + w).
    Returns (energy, frequency of the top outcome, counts, circuit). A ready-made gate `unitary` replaces make_U(model, e_min, w)."""
    ### STARTER: raise NotImplementedError("T2: write qpe_energy")
    ### BEGIN SOLUTION
    qc = QuantumCircuit(m + 4, m)
    qc.x([m + q for q in model.hf_x_qubits])                       # start state on the system register (after the m eval qubits)
    qc.compose(phase_estimation(m, make_U(model, e_min, w) if unitary is None else unitary), inplace=True)
    qc.measure(range(m), range(m))
    counts = qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()
    phi, freq = qff.bits.phase_from_counts(counts, m)               # reads the string reversed
    return e_min + phi * w, freq, counts, qc
    ### END SOLUTION

E_qpe = {2.0: {}, "far": {}}; top_freq = {2.0: {}, "far": {}}
for key in (2.0, "far"):
    E_qpe[key][4], top_freq[key][4], counts_m4, qc_m4 = qpe_energy(ham[key], 4)
    print(f"{key!s:>4}: QPE energy {E_qpe[key][4]:.5f} Ha, top outcome frequency {top_freq[key][4]:.2f}")
binding_qpe_mHa = {4: (E_qpe[2.0][4] + V[2.0] - E_qpe["far"][4]) * 1000}      # equation (10): binding(R) = E_0(R) + V(R) - E_0(far)
print(f"binding energy with m = 4: {binding_qpe_mHa[4]:.1f} mHa   (exact {binding_exact_mHa:.1f})")
'''

S3_CODE_36 = r"""# Self-check
assert abs(E_qpe["far"][4] - (-1.5)) < 1e-9, "T2: far away the start state is an eigenstate at -1.5 Ha; check the bit order (use phase_from_counts)"
assert abs(E_qpe[2.0][4] - (-1.59375)) < 1e-9, "T2: with 4 evaluation qubits the grid point nearest -1.59588 is -1.59375"
assert abs(binding_qpe_mHa[4] - (-43.75)) < 0.01, "T2: the binding energy includes the classical term V(R)"
"""

S3_MD_37 = r"""**Check equation (8) against the circuit.** The cell below takes the *exact* outcome probabilities of the circuit you built (no shots) and compares them with $P(k)=\sum_n|c_n|^2F_M(\varphi_n-k/M)$ for all 16 outcomes of $m=4$. The excited state's phase, outside the window, is folded back automatically because $F_M$ repeats with period 1 in $\delta$.
"""

S3_CODE_38 = r'''# Check: the exact outcome probabilities of the QPE circuit equal equation (8), including the aliased excited state.
from qiskit.quantum_info import Statevector


def F(delta, M):
    """Equation (7): sin^2(pi M delta) / (M^2 sin^2(pi delta)), equal to 1 at delta = 0."""
    delta = np.asarray(delta, dtype=float)
    out = np.ones_like(delta)
    nz = np.abs(np.sin(np.pi * delta)) > 1e-12
    out[nz] = np.sin(np.pi * M * delta[nz]) ** 2 / (M ** 2 * np.sin(np.pi * delta[nz]) ** 2)
    return out


def bit_reversed(r, bits):
    """The integer whose binary digits are those of r, written backwards (the QPE bit-order trap)."""
    return int(format(r, f"0{bits}b")[::-1], 2)


m_chk = 4
qc_chk = qpe_energy(ham[2.0], m_chk)[3].remove_final_measurements(inplace=False)      # the circuit you built, without the measurements
raw = Statevector(qc_chk).probabilities(list(range(m_chk)))                      # exact probabilities of the evaluation register
p_circuit = np.zeros(2 ** m_chk)
for r, pr in enumerate(raw):
    p_circuit[bit_reversed(r, m_chk)] = pr                                       # undo the bit order
Mg = 2 ** m_chk
p_formula = np.array([(overlap * F(phis_raw - k / Mg, Mg)).sum() for k in range(Mg)])      # equation (8): overlaps from 3.4, phases not yet wrapped
assert np.allclose(p_circuit, p_formula, atol=1e-9), "the circuit should follow equation (8)"
k_top = int(np.argmax(p_circuit))
print(f"top outcome k = {k_top} -> energy {E_MIN + k_top / Mg * W:.5f} Ha with probability {p_circuit[k_top]:.3f} (circuit) = {p_formula[k_top]:.3f} (formula)")
print("OK: equation (8) reproduces the circuit exactly, including the wrapped peak of the excited state.")
fig, ax = plt.subplots(figsize=(7, 3))
ax.bar(range(Mg), p_circuit, color="#0F62FE", label="circuit (exact probabilities)")
ax.plot(range(Mg), p_formula, "o", color="#D02670", ms=4, label="equation (8)")
ax.set_xlabel("outcome k (energy = -1.75 + 0.5 k / 16)"); ax.set_ylabel("probability"); ax.legend(fontsize=8)
ax.set_title("R = 2.0, m = 4: ground-state peak and the wrapped excited-state peak (ILLUSTRATIVE MODEL)", fontsize=8)
plt.tight_layout()
'''

S3_CODE_39 = r"""# The QPE circuit your function builds, drawn for m = 3 so that it stays readable (4 system qubits below the evaluation register).
try:
    display(qpe_energy(ham[2.0], 3)[3].draw("mpl", fold=40))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(qpe_energy(ham[2.0], 3)[3].draw("text", fold=120))
"""

S3_MD_40 = r"""**Reading the histogram.** The x axis is relabelled in Hartree. The tall bar is the ground-state energy rounded to the 16-point grid; the small bars are rounding neighbours and the leftover probability of the excited state.
"""

S3_CODE_41 = r"""_, _, counts_m4, qc_m4 = qpe_energy(ham[2.0], 4)    # the R = 2.0, m = 4 run (the loop above ended on 'far')
circuit = qc_m4                                  # the main circuit of this notebook (used by the IBM preview below)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7, 3.2))
energies = {}
for s, n in counts_m4.items():
    e = E_MIN + qff.bits.phase_from_counts({s: 1}, 4)[0] * W
    energies[e] = energies.get(e, 0) + n / 4096
ax.bar([f"{e:.3f}" for e in sorted(energies)], [energies[e] for e in sorted(energies)], color="#0F62FE")
ax.axvline(f"{E_MIN + 0.3125 * W:.3f}", color="#FA4D56", ls="--", lw=1)
ax.set_xlabel("QPE energy (Ha), 16-point grid"); ax.set_ylabel("probability")
ax.set_title(f"R = 2.0, m = 4. {LABEL}", fontsize=9); plt.xticks(rotation=90, fontsize=7); plt.tight_layout(); plt.show()
print("exact ground energy:", round(e_sector[2.0], 5), "| tallest bar:", max(energies, key=energies.get))
"""

S3_MD_42 = r"""### 3.5 · The precision ladder and the binding energy

Each extra evaluation qubit halves the energy step $W/2^m$. T3 climbs that ladder at one geometry and then computes the binding energy at two geometries.
"""

S3_MD_43 = r"""### T3 · Stretch: the precision ladder and the binding curve · التمرين ٣

**What you are about to build, and why.** The grid step of QPE is $W/2^m$, so precision costs qubits. In T3 you climb the ladder $m=5,\dots,8$, store how the energy and the binding energy improve, then run QPE at *every* shipped distance for $m=6$ and $m=8$ to draw the binding curve and find the distance with the strongest binding. You also work out the number of evaluation qubits that would give chemical accuracy (1.6 mHa), and explain why the top outcome never exceeds about 0.83.

**The formulas you implement:** grid step $W/2^m$; the design rule

$$\frac W{2^m}\le\varepsilon\quad\Longleftrightarrow\quad m\ \ge\ \log_2\frac W\varepsilon \tag{9}$$

and the binding energy (10) at each $R$. Notice what (9) says for our window: $W=0.5$ Ha and $\varepsilon=1.6$ mHa need $m\ge\log_2(0.5/0.0016)=8.3$, so $m=9$.

**Stuck? Open the hints one at a time, and try again after each. · تلميحات**

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 1 · a nudge · تلميح ١</summary>

Part (a) is a loop over m = 5..8 that calls `qpe_energy` for both geometries and fills the dictionaries, as in T2. Part (b) is a double loop over m in (6, 8) and over the distances: the binding energy is equation (10) times 1000; compute the far energy once per m. Part (c) is equation (9). The top frequency is bounded by the overlap of the start state with the ground state (3.4).

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 2 · the structure · تلميح ٢</summary>

Steps: (a) for each m and each geometry store the QPE energy and top frequency, then the binding energy for that m; (b) for m = 6 and 8, `binding_curve_mHa[m]` maps each R in `R_LIST` to its binding energy; `R_best` is the key with the lowest value at m = 8; (c) `m_needed` is the first whole m for which the grid step is at most 0.0016 Ha.

</details>

<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">
<summary style="cursor:pointer;font-weight:600">Hint 3 · nearly there · تلميح ٣</summary>

```python
for m in range(5, 9):
    ______                       # E_qpe and top_freq for both geometries, then binding_qpe_mHa[m]

binding_curve_mHa = {}
for m in (6, 8):
    ______                       # {R: binding energy in mHa} for every R in R_LIST
R_best = ______                  # the distance with the lowest binding energy at m = 8
m_needed = ______                # equation (9)
```

</details>
"""

T3 = r"""import time
for m in range(5, 9):
    for key in (2.0, "far"):
        E_qpe[key][m], top_freq[key][m], _, _ = qpe_energy(ham[key], m)
    binding_qpe_mHa[m] = (E_qpe[2.0][m] + V[2.0] - E_qpe["far"][m]) * 1000    # equation (10)
print(f"{'m':>2} {'circuit qubits':>15} {'top freq':>9} {'E (Ha)':>10} {'error (mHa)':>12} {'resolution':>11} {'binding (mHa)':>14}")
for m in range(4, 9):
    print(f"{m:>2} {m + 4:>15} {top_freq[2.0][m]:>9.2f} {E_qpe[2.0][m]:>10.5f} {(E_qpe[2.0][m] - e_sector[2.0]) * 1000:>+12.1f} "
          f"{W / 2**m * 1000:>8.1f} mHa {binding_qpe_mHa[m]:>14.1f}")

t0 = time.perf_counter()
binding_curve_mHa = {}
E_far = {m: qpe_energy(ham["far"], m)[0] for m in (6, 8)}
for m in (6, 8):
    binding_curve_mHa[m] = {R: (qpe_energy(ham[R], m)[0] + V[R] - E_far[m]) * 1000 for R in R_LIST}
print(f"curves took {time.perf_counter() - t0:.1f} s")
R_best = min(binding_curve_mHa[8], key=binding_curve_mHa[8].get)
R_best_m6 = min(binding_curve_mHa[6], key=binding_curve_mHa[6].get)
m_needed = next(m for m in range(1, 30) if W / 2**m <= 0.0016)    # equation (9): W / 2^m <= eps  <=>  m >= log2(W / eps)
print("best distance: m = 8 ->", R_best, "A | m = 6 ->", R_best_m6, "A (m = 6 steps are 7.8 mHa, bigger than the gap between R = 2.0 and 2.1)")
print("evaluation qubits for 1.6 mHa resolution:", m_needed, "(circuit qubits:", m_needed + 4, ")")
"""

T3_STARTER = r"""# Goal: (a) fill E_qpe[key][m], top_freq[key][m] for key in (2.0, "far") and m = 5..8, and binding_qpe_mHa[m];
# (b) binding_curve_mHa = {6: {R: ...}, 8: {R: ...}} (binding energy in mHa, including V) for every R in R_LIST, and R_best at m = 8;
# (c) m_needed = the smallest m with W / 2**m <= 0.0016 (equation 9).
raise NotImplementedError("T3: ladder, curve and m_needed")
"""

T3_CHECK = r"""assert abs(E_qpe[2.0][8] - e_sector[2.0]) < 0.002, "T3: m = 8 should land within one grid step (2 mHa) of the exact energy"
assert all(abs(E_qpe["far"][m] - (-1.5)) < 1e-9 for m in range(4, 9)), "T3: far away every m should return -1.5 exactly"
assert abs(binding_qpe_mHa[8] - (-45.7)) < 0.3, "T3: m = 8 binding energy should be about -45.7 mHa"
assert R_best == 2.0, "T3: the binding curve at m = 8 should have its minimum at R = 2.0"
assert m_needed == 9, "T3: W / 2**m <= 1.6 mHa first holds at m = 9"
"""

S3_MD_46 = r"""**Reading the ladder.** More evaluation qubits do not always help *at that step*: at m = 7 the true phase sits almost halfway between two grid points, so the top outcome has only about 40% probability and lands 1.8 mHa on the other side. The top frequency never exceeds about 0.83 for a different reason: that is the overlap of the start state with the ground state; the other 17% belongs to an excited state.

**Aliasing: where did the other 17% go?** The two-electron excited state at −1.041 Ha lies **outside** our window [−1.75, −1.25), so its phase wraps around the circle and QPE reports it as a different, wrong energy (a small peak near −1.54). That is aliasing. Read the **tallest** peak. Widening the window to [−2.0, −1.0) moves the second peak to its true place, near −1.04. Compare the three tallest peaks.
"""

S3_MD_47 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**Aliasing is silent.** Nothing in the counts marks an energy as wrapped: a state outside the window simply shows up as a different, wrong energy, and it can look just like a real result. The only protection is to choose a window that holds every energy you care about (use equation (4) and a classical estimate), to read the tallest peak, and to repeat the run with a second window and see which peaks move.

</div>
"""

S3_MD_48 = r"""<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Aliasing | الالتفاف الطوري | I | Window | النافذة | I |
| Peak | ذروة | I |  |  |  |
"""

S3_CODE_49 = r"""# Guided: same circuit, two windows (m = 6). Peaks are (energy, probability).
def top_peaks(model, m, e_min, w, k=3, shots=8192):
    counts = qpe_energy(model, m, shots=shots, e_min=e_min, w=w)[2]            # your QPE function with another window
    return [(round(e_min + qff.bits.phase_from_counts({s: 1}, m)[0] * w, 4), round(n / shots, 3))
            for s, n in sorted(counts.items(), key=lambda kv: -kv[1])[:k]]

print("narrow window [-1.75, -1.25):", top_peaks(ham[2.0], 6, -1.75, 0.5))
print("wide window   [-2.00, -1.00):", top_peaks(ham[2.0], 6, -2.00, 1.0))
print("true energies: ground -1.5959, excited -1.0413; in the narrow window the excited state wraps around and shows up near -1.54")
"""

S3_MD_50 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

**Watch the sampling wobble.** Call `top_peaks(ham[2.0], 6, -1.75, 0.5, k=1, shots=s)` for `s` = 256, 1024, 8192 and 65536. The frequency of the top peak jumps around for few shots and settles for many. A rough size of the wobble is $\sqrt{p(1-p)/s}$, where $p$ is the frequency you see at 65536 shots: does it match how much the lower values differ from it?

</div>
"""

S3_MD_51 = r"""<div style="border:1px solid rgba(208,38,112,0.35);border-left:6px solid #D02670;background:rgba(208,38,112,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#D02670;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🌍 Real world · في الواقع</span>

In a building-block workflow like the one in the paper (Case 5), the quantum step is reserved for the **active site**, the few orbitals where the electrons interact most strongly, and classical methods handle the rest of the material. That is why our model is only two orbitals, and why the classical term $V(R)$ is added by hand to the quantum energies.

</div>
"""

S3_MD_52 = r"""<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>

Predict first, then run: call `top_peaks(ham[2.0], 6, -2.0, 0.5)`, a window $[-2.0,-1.5)$ that holds the ground state but not the excited state. Where should the ground-state peak sit, and where does the excited state wrap to? Use equation (4): $\varphi=(E-E_{\min})/W$, keep only the fractional part.

</div>
"""

S3_MD_53 = r"""### 3.6 · What would hardware need?

Our exact $U$ was a classical matrix exponential: as much work as diagonalising the matrix, so it proves nothing about speed. The gate-level version splits $H$ into its Pauli terms $H=\sum_kA_k$ and uses a **Trotter** approximation:

$$e^{-i(A_1+A_2+\dots)t}\ \approx\ \Bigl(e^{-iA_1t/r}\,e^{-iA_2t/r}\cdots\Bigr)^{r} \tag{11}$$

The approximation improves as the number of steps $r$ grows (and with a symmetric ordering, "order 2", which we use) but costs more gates. For QPE, evaluation qubit $k$ needs $U^{2^k}$, i.e. a total of $2^m-1$ uses of $U$ in the textbook construction. (Notebook 7 shows that Qiskit's `PauliEvolutionGate` builds the power by lengthening the evolution time instead, which keeps the gate count low but lets the Trotter error grow.) Count the two-qubit gates for $m=6$ with one Trotter step ($r=1$) per use of $U$ in the cell below.
"""

S3_MD_54 = r"""<sub>Arabic terms for this step. Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>

| English | العربية | Tier | English | العربية | Tier |
|---|---|---|---|---|---|
| Trotterisation | تقريب تروتر (Trotterisation) | I | CNOT gate | بوابة النفي المتحكّم بها | G |
| Gate count | عدد البوابات | I | Circuit depth | عمق الدائرة | G |
"""

S3_CODE_55 = r"""from qiskit import transpile
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import SuzukiTrotter

m = 6
H_shift = (ham[2.0].op - E_MIN * SparsePauliOp("IIII")).simplify()
trotter_gate = PauliEvolutionGate(H_shift, time=-2 * np.pi / W, synthesis=SuzukiTrotter(order=2, reps=1))   # Equation (11): Trotter formula; exp(-i t op) with t = -2 pi / W
E_tr, f_tr, counts_tr, qc_tr = qpe_energy(ham[2.0], m, unitary=trotter_gate)     # your QPE function with the Trotterised U
cx_trotter = transpile(qc_tr, basis_gates=["cx", "rz", "sx", "x"]).count_ops().get("cx", 0)
print(f"Trotter QPE, m = 6: energy {E_tr:.5f} Ha (frequency {f_tr:.2f}) | CX gates after transpiling: {cx_trotter}")
print("Exact-unitary QPE at m = 6 gave", round(E_qpe[2.0][6], 5), "- same answer, but now with a gate count a real device must survive.")
"""

S3_IBM_NOTE = r"""The next cell does the offline part: it transpiles your circuit for a *pretend* device on this computer and reports its size.
"""

S3_CODE_57 = r"""preview = qff.quantum.device_preview(circuit)      # `circuit` = the main circuit of this notebook
print(f"depth {preview['depth']}, two-qubit gates {preview['two_qubit_gates']}")
"""

S4_MD_60 = r"""> **ILLUSTRATIVE MODEL · نموذج توضيحي.** All rows and plots below describe the invented two-orbital model.
"""

S4_CODE_61 = r"""# Results table: classical baselines first, then QPE.
import time
t0 = time.perf_counter(); e_check = {k: sector_ground(ham[k].op) for k in (2.0, "far")}; t_exact = time.perf_counter() - t0
t0 = time.perf_counter(); e8 = {k: qpe_energy(ham[k], 8)[0] for k in (2.0, "far")}; t_qpe = time.perf_counter() - t0
rows = [
    ("pencil and paper, -2t^2/D + V", (second_order + V[2.0]) * 1000, "-", "a formula"),
    ("exact diagonalisation, 1e+1e sector", (e_check[2.0] + V[2.0] - e_check["far"]) * 1000, f"{t_exact:.3f} s", "4 states (of 16)"),
    ("naive: lowest eigenvalue of 16x16", (full_min[2.0] + V[2.0] - full_min["far"]) * 1000, "-", "WRONG sector (3 electrons)"),
    ("QPE, m = 4 (8 qubits)", binding_qpe_mHa[4], "-", "U applied 15 times per shot"),
    ("QPE, m = 8 (12 qubits)", (e8[2.0] + V[2.0] - e8["far"]) * 1000, f"{t_qpe:.1f} s", "U applied 255 times per shot"),
]
print(f"[{LABEL}]   binding energy of CO2 at R = 2.0 A")
print(f"{'method':<38}{'binding (mHa)':>14}  {'time':<9}{'cost'}")
for name, b, tm, cost in rows:
    print(f"{name:<38}{b:>14.1f}  {tm:<9}{cost}")
"""

S4_CODE_62 = r"""# One plot: the binding curve, exact against QPE.
R_axis = R_LIST
exact_curve = [(e_file[R] + V[R] - e_file["far"]) * 1000 for R in R_axis]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(R_axis, exact_curve, "k-o", label="exact (sector diagonalisation)")
ax.plot(R_axis, [binding_curve_mHa[6][R] for R in R_axis], "s--", color="#8A3FFC", label="QPE, m = 6 (step 7.8 mHa)")
ax.plot(R_axis, [binding_curve_mHa[8][R] for R in R_axis], "^--", color="#0F62FE", label="QPE, m = 8 (step 2.0 mHa)")
ax.axhline(0, color="grey", lw=0.5); ax.set_xlabel("CO2 to site distance R (Angstrom)"); ax.set_ylabel("binding energy (mHa)")
ax.set_title(f"{LABEL}: binding curve", fontsize=10); ax.legend(fontsize=8); plt.tight_layout(); plt.show()
"""

S4_MD_63 = r"""**What this toy shows.** A binding energy is a difference of two energies, each read by QPE; the quantum part handles the two orbitals where bonding happens and a classical term handles the rest; the state you prepare decides which eigenvalue you read and how often. With 8 evaluation qubits, in a noiseless simulation, QPE reproduces the exact binding energy of the invented model to about 0.2 mHa.
"""

S4_MD_64 = r"""<div style="border:1px solid rgba(138,63,252,0.35);border-left:6px solid #8A3FFC;background:rgba(138,63,252,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">

<span style="background:#8A3FFC;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">⚠️ Watch out · انتبه</span>

**What this toy does not show.** The parameters are made up: nothing here says anything about any real material. A real metal–organic framework has hundreds of atoms, dispersion forces and an active site with many more orbitals. Binding energies of a few tens of mHa need both energies to about 1 mHa, so 9 or more evaluation qubits and very long circuits, which is fault-tolerant territory; our exact U also cheated by using a classical matrix exponential, and the Trotter cell shows the real price. A laptop solves this model in microseconds, so QPE is not beating anything here. We ran noiseless.

</div>
"""

S4_MD_65 = r"""**Paper versus our route.** WEF/Aramco Case 5 (p. 15): TotalEnergies and Quantinuum predict CO₂ binding in an Al-fumarate MOF with a building-block workflow in which the quantum step targets the active site and classical methods handle the rest, benchmarked against high-level classical results. We keep the split (quantum site + classical remainder) and replace the chemistry with an invented two-orbital model and QPE, a teaching choice.
"""

ACC = r"""assert QUBITS == 8
assert abs(e_sector[2.0] - (-1.59588)) < 1e-5 and abs(e_sector["far"] - (-1.5)) < 1e-9
assert full_min[2.0] < e_sector[2.0] - 0.1                     # the sector trap exists
assert abs(E_qpe["far"][4] - (-1.5)) < 1e-9
assert abs(E_qpe[2.0][4] - (-1.59375)) < 1e-9 and abs(E_qpe[2.0][8] - e_sector[2.0]) < 0.002
assert abs(binding_exact_mHa - (-45.9)) < 0.1 and abs(binding_qpe_mHa[8] - (-45.7)) < 0.3
assert R_best == 2.0
assert cx_trotter > 100
"""

# Toolbox cells: taken from tools/toolbox.py when it has an entry for notebook 8, else the copy below (from the uploaded notebook).
_TB_MD = r"""### 🧰 Toolbox: how the calls you will need are written · صندوق الأدوات

No internet? No problem. The cell under this table runs each call once on a **small toy that has nothing to do with the task**, so you can see the exact spelling, the order of the arguments, and what comes back. Run it, read the printed lines, and come back to it whenever a hint says "a method of ...". *What* to build, and with *which numbers*, is your job.

| To do | How it is written | What you get |
|---|---|---|
| a circuit with $m+4$ qubits and $m$ classical bits | `qc = QuantumCircuit(m + 4, m)` | the classical bits are named `c` |
| $X$ on a list of qubits | `qc.x([m + q for q in model.hf_x_qubits])` | the system register starts after the $m$ evaluation qubits |
| put the QPE circuit in | `qc.compose(phase_estimation(m, gate), inplace=True)` | added at the end; evaluation qubits first, then the system qubits |
| measure chosen qubits | `qc.measure(range(m), range(m))` | qubit $i$ into classical bit $i$ |
| run and count | `qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()` | a dictionary; the **rightmost** character is bit 0 |
| counts to a phase | `phi, freq = qff.bits.phase_from_counts(counts, m)` | the phase of the most frequent outcome and its frequency |
| a sub-matrix of rows and columns | `M[np.ix_(rows, cols)]` | the entries of `M` at those rows and those columns |
| timing | `t0 = time.perf_counter()` ... `time.perf_counter() - t0` | seconds |
"""
_TB_CODE = r"""# Toolbox for 08, on toys (no CO2 here).
import time
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate, phase_estimation

m_toy = 2
qc = QuantumCircuit(m_toy + 1, m_toy)                      # 2 evaluation qubits + 1 system qubit; 2 classical bits named c
qc.x(m_toy)                                                # X on the system qubit (it comes AFTER the evaluation qubits)
qc.compose(phase_estimation(m_toy, UnitaryGate(np.diag([1, -1]))), inplace=True)    # U = diag(1, -1) acts on the system qubit
qc.measure(range(m_toy), range(m_toy))
counts = qff.quantum.sv_sampler(1000).run([qc]).result()[0].data.c.get_counts()
phi, freq = qff.bits.phase_from_counts(counts, m_toy)
print("counts:", counts, "| phase:", phi, "| frequency:", freq)
assert phi == 0.5 and freq == 1.0                          # the system qubit is an eigenstate with U|1> = -|1>, phase 1/2
M = np.arange(16).reshape(4, 4)
print("sub-matrix on rows/cols [1, 3]:", M[np.ix_([1, 3], [1, 3])].tolist())
t0 = time.perf_counter(); _ = sum(range(10 ** 5)); print("seconds:", round(time.perf_counter() - t0, 4))
print("OK: toolbox 08 ran.")
"""
TB_MD, TB_CODE = TOOLBOX.get(8, (_TB_MD, _TB_CODE))

GLOSSARY_EXTRA = [
    ("quantum phase estimation (QPE)", "تقدير الطور الكمومي (QPE)", "G"), ("quantum Fourier transform", "تحويل فورييه الكمومي", "G"),
    ("inverse QFT", "تحويل فورييه الكمومي العكسي", "I"), ("eigenvalue", "القيمة الذاتية", "G"), ("eigenstate", "الحالة الذاتية", "G"),
    "Hamiltonian", "ground state", ("fault tolerance", "التسامح مع الخطأ", "G"),
    ("binding energy", "طاقة الارتباط", "I"), ("carbon capture", "احتجاز الكربون", "I"),
    ("metal-organic framework (MOF)", "الإطار المعدني العضوي (MOF)", "I"), ("active site", "الموقع النشط", "I"),
    ("illustrative model", "نموذج توضيحي", "I"), ("phase kickback", "ارتداد الطور", "I"),
    ("electron number sector", "قطاع عدد الإلكترونات", "I"),
]


def main():
    """Write the solution notebook."""
    nb = new(8, "MOF carbon capture (illustrative model)",
             use_case="Does a porous material hold CO₂ firmly enough to capture it? (an invented two-orbital model, labelled illustrative everywhere)",
             algorithm="Quantum phase estimation (QPE) on an active-site model", qubits=8, minutes=3)
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
    nb.md(TB_MD)
    nb.code(TB_CODE)
    nb.md(S3_MD_20)
    # T1 has markers inside its code (partly a shared function body), so it is kept as written in the notebook
    nb.code(T1_CELL)
    nb.code(S3_CODE_22)
    nb.md(S3_MD_23)
    nb.code(S3_CODE_24)
    nb.md(S3_MD_25)
    nb.md(S3_MD_26)
    nb.code(S3_CODE_27)
    nb.md(S3_MD_28)
    nb.code(S3_CODE_29)
    nb.md(S3_MD_30)
    nb.code(S3_CODE_31)
    nb.md(S3_MD_32)
    nb.code(S3_CODE_33)
    nb.md(S3_MD_34)
    # T2 has markers inside its code (partly a shared function body), so it is kept as written in the notebook
    nb.code(T2_CELL)
    nb.code(S3_CODE_36)
    nb.md(S3_MD_37)
    nb.code(S3_CODE_38)
    nb.code(S3_CODE_39)
    nb.md(S3_MD_40)
    nb.code(S3_CODE_41)
    nb.md(S3_MD_42)
    nb.md(S3_MD_43)
    nb.task("T3", 'precision ladder to m = 8, binding curve, evaluation qubits needed', T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_MD_46)
    nb.md(S3_MD_47)
    nb.md(S3_MD_48)
    nb.code(S3_CODE_49)
    nb.md(S3_MD_50)
    nb.md(S3_MD_51)
    nb.md(S3_MD_52)
    nb.md(S3_MD_53)
    nb.md(S3_MD_54)
    nb.code(S3_CODE_55)
    nb.md(day.IBM_MD + "\n" + S3_IBM_NOTE)
    nb.code(S3_CODE_57)
    nb.md(day.glossary_md(GLOSSARY_EXTRA), tags=["glossary"])
    nb.section(4)
    nb.md(S4_MD_60)
    nb.code(S4_CODE_61)
    nb.code(S4_CODE_62)
    nb.md(S4_MD_63)
    nb.md(S4_MD_64)
    nb.md(S4_MD_65)
    nb.acceptance(ACC)
    nb.section5(use_case="Does a porous material hold CO₂ firmly enough to capture it, yet loosely enough to release it? (on an illustrative model; say so in your product)",
                counting_fact="2 orbitals → 4 qubits; a realistic active site → hundreds.",
                single_idea="read two energies in binary and subtract; the difference tells you how firmly the material holds CO₂.",
                ideas=["A 'sponge' interactive where the visitor slides CO₂ toward the site and watches the energy curve",
                       "A two-scales animation (near and far) with binary readouts",
                       "A board-game tile for the site"],
                reuse="the binding curve and the precision table (always labelled illustrative)")
    nb.section6(honest="On an invented two-orbital model, QPE with 8 evaluation qubits reproduced the binding energy to 0.2 mHa in simulation.",
                question_a="Why is the lowest eigenvalue of your matrix not the answer?",
                question_b="Where do your parameters come from? (Answer: we chose them; say so.)")
    path = nb.save(os.path.join(ROOT, "solutions", "08_store_carbon_capture_qpe_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()