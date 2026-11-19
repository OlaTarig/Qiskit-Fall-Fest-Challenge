"""The 'Toolbox' cell pair (a short table plus one runnable cell) that every notebook shows before its first task.

A participant with no internet cannot look up how a Qiskit call is written, so each toolbox shows the *call shapes* the three tasks
need (imports, argument order, what comes back) on a small toy that has nothing to do with the task's own numbers. What to build and
with which numbers stays the participant's job. Every code cell here is run by tools/run_tests.py as part of its notebook.

    from toolbox import TOOLBOX            # TOOLBOX[n] = (markdown, code) for notebook n (05 and 09 share one)
"""

HEAD = ("### 🧰 Toolbox: how the calls you will need are written · صندوق الأدوات\n\n"
        "No internet? No problem. The cell under this table runs each call once on a **small toy that has nothing to do with the task**, "
        "so you can see the exact spelling, the order of the arguments, and what comes back. Run it, read the printed lines, and come back "
        "to it whenever a hint says \"a method of ...\". *What* to build, and with *which numbers*, is your job.\n\n")

# ---------------------------------------------------------------------------------------------------------------- 01
MD01 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with 3 qubits | `qc = QuantumCircuit(3)` | all qubits start in $\\lvert0\\rangle$ |
| Hadamard on qubit 0 | `qc.h(0)` | one gate; `qc.h(range(3))` puts it on every qubit in the list |
| load amplitudes into *some* qubits | `qc.initialize(amplitudes, [1, 2])` | `amplitudes` is a normalised list of $2^k$ numbers for $k$ listed qubits |
| apply a matrix as one gate | `qc.unitary(matrix, [0, 1, 2])` | `matrix` is $2^k\\times2^k$ for $k$ listed qubits |
| read all amplitudes | `Statevector(qc).data` | a complex numpy array of length $2^n$ (position $k$: qubit $i$ is bit $i$ of $k$) |
| count the gates | `qc.count_ops()` | a dictionary such as `{'h': 1, 'unitary': 1}` |
| seeded random numbers | `rng = np.random.default_rng(SEED)`, then `rng.normal(0, sigma, size)` | the same numbers on every run |
| keep numbers inside $[0,1]$ | `np.clip(a, 0, 1)` | smaller values become 0, larger become 1 |
| the helper `classical_edges(image, t)` | `gh, gv, mask = classical_edges(image, t)` | **three** things: horizontal differences, vertical differences, and the yes/no edge mask (the last one) |"""
CODE01 = '''# Toolbox for 01, on a 2-qubit toy that has nothing to do with the image.
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

toy = QuantumCircuit(2)                                   # 2 qubits, both start in |0>
toy.initialize([0.6, 0.8], [1])                           # load a normalised amplitude list into qubit 1 only
toy.h(0)                                                  # Hadamard on qubit 0
toy.unitary(np.array([[0, 1], [1, 0]]), [0])              # a 2 x 2 matrix applied as one gate to qubit 0 (this matrix is the X gate)
amp = Statevector(toy).data                               # the 4 amplitudes (position k = 2 * qubit1 + qubit0)
print("amplitudes:", np.round(amp.real, 3), "| gates:", dict(toy.count_ops()))
assert np.allclose(amp.real, np.array([0.6, 0.6, 0.8, 0.8]) / np.sqrt(2)), "the toolbox toy should give these amplitudes"
rng = np.random.default_rng(SEED)                         # the day's seeded generator
print("four noisy numbers kept inside [0, 1]:", np.round(np.clip(rng.normal(0.5, 0.3, 4), 0, 1), 3))
print("OK: toolbox 01 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 02
MD02 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with knobs | `ansatz = real_amplitudes(n, reps=1, entanglement="linear")` | `ansatz.num_parameters` tells how many knobs |
| the optimiser | `COBYLA(maxiter=400)` | an object that turns the knobs |
| one VQE run | `VQE(estimator, ansatz, optimizer, initial_point=x0, callback=fn)` | an object; nothing runs yet |
| run it on an operator | `res = vqe.compute_minimum_eigenvalue(op)` | `res.eigenvalue.real` (lowest energy found, **without** any constant offset), `res.optimal_point` (best knobs) |
| the callback | `def fn(eval_count, params, energy, metadata): ...` | called by VQE after every energy evaluation; keep the energy |
| probabilities of a state | `Statevector(ansatz.assign_parameters(knobs)).probabilities_dict()` | `{'010110': 0.52, ...}`; the **last** character is qubit 0 |
| the most likely key | `max(probs, key=probs.get)` | the string with the largest probability |
| string to layout | `qff.bits.to_x('110')` | `[0, 1, 1]`: the list $x$ with $x_0$ first |
| seeded random start | `np.random.default_rng(seed).uniform(-np.pi, np.pi, n)` | $n$ numbers between $-\\pi$ and $\\pi$, the same on every run with the same seed |"""
CODE02 = '''# Toolbox for 02: one complete VQE call on a 1-qubit toy (find the lowest energy of the operator Z; the answer is -1).
from qiskit.circuit.library import real_amplitudes
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

toy_op = SparsePauliOp("Z")                               # the energy operator
toy_ansatz = real_amplitudes(1, reps=1)                   # a circuit with knobs
trace = []


def record(eval_count, params, energy, metadata):          # VQE calls this after every energy evaluation, with these four arguments
    trace.append(energy)


vqe_toy = VQE(StatevectorEstimator(), toy_ansatz, COBYLA(maxiter=100), initial_point=[0.3] * toy_ansatz.num_parameters, callback=record)
res = vqe_toy.compute_minimum_eigenvalue(toy_op)
print("lowest energy found:", round(res.eigenvalue.real, 4), "| best knobs:", np.round(res.optimal_point, 3), "| energies recorded:", len(trace))
probs = Statevector(toy_ansatz.assign_parameters(res.optimal_point)).probabilities_dict()
print("probabilities of the final state:", {str(k): round(float(v), 3) for k, v in probs.items()}, "| most likely:", max(probs, key=probs.get))
print("string to list:", qff.bits.to_x("110"), "(x_0 first)")
print("seeded random start (3 numbers, seed 1):", np.round(np.random.default_rng(1).uniform(-np.pi, np.pi, 3), 3))
assert abs(res.eigenvalue.real + 1) < 1e-3 and qff.bits.to_x("110") == [0, 1, 1]
print("OK: toolbox 02 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 03
MD03 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| try every $x$ of a problem | `x, value, table = qff.classical.brute_force(qp)` | best **legal** $x$, its value, and `table` = a list of `(x, value, legal?)` for every $x$, cheapest first |
| minimise with scipy | `res = minimize(f, x0=[0.1, 0.1], method="COBYLA", options={"maxiter": 300})` | `res.x` (best inputs), `res.fun` (value there); `f` takes one list |
| size of a circuit on a pretend device | `qff.quantum.device_preview(circuit)` | a dictionary with `depth`, `two_qubit_gates`, `size`, `num_qubits` (nothing is sent anywhere); a circuit with knobs must have them bound first: `qff.quantum.device_preview(circuit.assign_parameters(theta))` |
| a small problem | `qp = QuadraticProgram(); qp.binary_var("a"); qp.minimize(linear={"a": 1})` | the same object type the notebook calls `qp` |"""
CODE03 = '''# Toolbox for 03: the three helpers and the scipy optimiser, on toys.
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit_optimization import QuadraticProgram

toy_qp = QuadraticProgram()                               # choose a, b in {0, 1} to minimise  -a - b + 3 a b
toy_qp.binary_var("a"); toy_qp.binary_var("b")
toy_qp.minimize(linear={"a": -1, "b": -1}, quadratic={("a", "b"): 3})
x_toy, v_toy, table_toy = qff.classical.brute_force(toy_qp)
print("best x:", x_toy, "| value:", v_toy, "| table rows:", len(table_toy), "| first row (x, value, legal?):", table_toy[0])

res = minimize(lambda t: (t[0] - 1) ** 2 + (t[1] + 0.5) ** 2, x0=[0.1, 0.1], method="COBYLA", options={"maxiter": 200})
print("scipy found knobs", np.round(res.x, 2), "with value", round(float(res.fun), 4))

bell = QuantumCircuit(2); bell.h(0); bell.cx(0, 1)
print("pretend device:", qff.quantum.device_preview(bell))
assert v_toy == -1 and abs(res.x[0] - 1) < 0.05 and abs(res.x[1] + 0.5) < 0.05
print("OK: toolbox 03 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 04
MD04 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| the QAOA circuit of a cost operator | `circuit = qaoa_ansatz(op, reps=1)` | a circuit whose knobs are, in this order, **all betas first, then all gammas** |
| diagonal of the cost operator | `diag = qff.classical.diag_of(op, offset)` | entry $k$ is the cost of basis state $k$ (qubit $i$ is bit $i$ of $k$) |
| the average cost as a function of the knobs | `f = qff.quantum.qaoa_energy(circuit, diag)`, then `f([beta, gamma])` | one number $\\langle C\\rangle$, computed exactly |
| minimise it with scipy | `res = minimize(f, x0=[0.1, 0.1], method="COBYLA", options={"maxiter": 300})` | `res.x` (best knobs), `res.fun` (lowest average cost found) |
| a QUBO and its operator | `op, offset = qubo.to_ising()` | the same `op` and `offset` used in section 3 |
| try every $x$ of a problem | `x, value, table = qff.classical.brute_force(qp)` | best **legal** $x$, its value, and `table` = a list of `(x, value, legal?)` for every $x$, cheapest first |
| the cost of one list $x$ | `qp.objective.evaluate(x)` | the cost of that $x$ for the problem `qp` |"""
CODE04 = '''# Toolbox for 04, on a 2-qubit toy cost operator (not the LNG problem).
from scipy.optimize import minimize
from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp

toy_cost = SparsePauliOp("ZZ")                            # cost +1 when the two bits agree, -1 when they differ
toy_circuit = qaoa_ansatz(toy_cost, reps=1)
toy_diag = qff.classical.diag_of(toy_cost, 0.0)
print("knob names, in order:", [str(p) for p in toy_circuit.parameters], "| diagonal:", toy_diag)
f = qff.quantum.qaoa_energy(toy_circuit, toy_diag)
print("average cost at the knobs (0.1, 0.1):", round(f([0.1, 0.1]), 4))
res = minimize(f, x0=[0.1, 0.1], method="COBYLA", options={"maxiter": 300})
print("after minimising: knobs", np.round(res.x, 3), "average cost", round(float(res.fun), 4), "(the lowest possible is -1)")
assert res.fun < f([0.1, 0.1]) and np.allclose(toy_diag, [1, -1, -1, 1])
from qiskit_optimization import QuadraticProgram
toy_qp = QuadraticProgram()                               # choose a, b in {0, 1} to minimise  -a - b + 3 a b
toy_qp.binary_var("a"); toy_qp.binary_var("b")
toy_qp.minimize(linear={"a": -1, "b": -1}, quadratic={("a", "b"): 3})
x_toy, v_toy, table_toy = qff.classical.brute_force(toy_qp)
print("brute force: best x", x_toy, "value", v_toy, "| evaluate(best x):", toy_qp.objective.evaluate(x_toy))
assert abs(toy_qp.objective.evaluate(x_toy) - v_toy) < 1e-9
print("OK: toolbox 04 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 05 and 09
MD05 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with $n$ qubits | `qc = QuantumCircuit(n)` | all qubits start in $\\lvert0\\rangle$ |
| Hadamard on every qubit | `qc.h(range(n))` | one call puts the gate on each listed qubit |
| a list of $\\pm1$ as a gate | `DiagonalGate([1, 1, 1, -1])` | multiplies the amplitude of position $k$ by the $k$-th number |
| put a gate on chosen qubits | `qc.append(gate, range(n))` | the gate is added at the end of `qc` |
| one full Grover step | `G = grover_operator(oracle_circuit)` | a **circuit**: the sign flip you give it, followed by the reflection about the average |
| put a circuit after another | `qc.compose(G, inplace=True)` | `G` is added at the end of `qc` |
| probabilities of all positions | `Statevector(qc).probabilities()` | a numpy array; **position $k$ means qubit $i$ equals bit $i$ of $k$** (so $k=2$ is qubit 1 set, qubit 0 clear) |"""
CODE05 = '''# Toolbox for 05 and 09, on a 2-qubit toy with one marked position (not the notebook's cost table).
from qiskit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, grover_operator
from qiskit.quantum_info import Statevector

n_toy = 2
sign_flip = QuantumCircuit(n_toy)
sign_flip.append(DiagonalGate([1, 1, 1, -1]), range(n_toy))      # flips the sign of position 3 only
toy_step = grover_operator(sign_flip)                            # one full Grover step, as a circuit
toy = QuantumCircuit(n_toy)
toy.h(range(n_toy))                                              # the equal start state
print("before any step:", np.round(Statevector(toy).probabilities(), 3))
toy.compose(toy_step, inplace=True)                              # one Grover step after it
probs_toy = Statevector(toy).probabilities()
print("after one step :", np.round(probs_toy, 3), "<- position", int(np.argmax(probs_toy)), "now has almost all the probability")
assert abs(probs_toy[3] - 1) < 1e-9
print("OK: toolbox ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 06
MD06 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with 4 qubits; $X$ on listed qubits | `qc = QuantumCircuit(4)`, `qc.x([0, 1])` | `x` takes a qubit number or a list |
| average value of an operator in a circuit's state | `estimator = qff.quantum.sv_estimator()`, then `float(estimator.run([(qc, op)]).result()[0].data.evs)` | one number (without the constant `shift`) |
| a list of named knobs | `t = ParameterVector("t", 8)` | `t[0]`, `t[1]`, ... are knobs |
| a gate with a knob | `PauliEvolutionGate(G, time=t[0])` | the gate $e^{-i\\,t_0G}$ |
| add a gate on chosen qubits | `qc.append(gate, range(4))` | added at the end of `qc` |
| one VQE run | `VQE(estimator, qc, SLSQP(maxiter=200), initial_point=x0, callback=fn).compute_minimum_eigenvalue(op)` | `res.eigenvalue.real`, `res.optimal_point`, `res.cost_function_evals` |
| the callback | `def fn(eval_count, params, value, meta): ...` | called after every energy evaluation |
| what a `Ham` object holds | `ham.op`, `ham.shift`, `ham.hf_x_qubits`, `ham.generators`, `ham.e_hf`, `ham.e_exact`, `ham.e_fci_full` | all the numbers the tasks need (`ham = qff.chem.load_hamiltonian(path)`) |"""
CODE06 = '''# Toolbox for 06, on a 1-qubit toy (find the lowest average of Z reachable by a rotation; the answer is -1).
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
print("OK: toolbox 06 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 07
MD07 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with 7 qubits and 5 classical bits | `qc = QuantumCircuit(7, 5)` | the bits are named `c` |
| $X$ on one qubit | `qc.x(5)` | the qubit is set to $\\lvert1\\rangle$ |
| put the QPE circuit in | `qc.compose(phase_estimation(m, gate), inplace=True)` | added at the end; it uses qubits $0..m-1$ for evaluation, the rest for the system |
| measure chosen qubits | `qc.measure(range(m), range(m))` | qubit $i$ into classical bit $i$ |
| run and count | `qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()` | a dictionary such as `{'01101': 3980, ...}`; the **rightmost** character is bit 0 |
| counts to a phase | `phi, freq = qff.bits.phase_from_counts(counts, m)` | the phase of the most frequent outcome and its frequency |
| a matrix as a gate | `UnitaryGate(matrix)` | the matrix is $2^k\\times2^k$ for $k$ qubits (here $4\\times4$: `np.eye(4)` has that size) |
| the matrix exponential | `import scipy.linalg as sla`, then `sla.expm(1j * M)` | the matrix $e^{iM}$ |
| the optional argument `unitary` of `qpe_run(h, m, unitary=None, shots=4096)` | `None` means "build the exact $U$ of equation (2)" | section 3.5 passes a Trotter gate here instead |
| what a `Ham` holds | `ham.op`, `ham.shift`, `ham.hf_x_qubits`, `ham.e_hf`, `ham.e_exact` | `ham.op.to_matrix()` is the Hamiltonian as a matrix; `ham.hf_x_qubits` lists the qubits that start in $\\lvert1\\rangle$ |"""
CODE07 = '''# Toolbox for 07, on toys (no hydrogen here).
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
print("OK: toolbox 07 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 08
MD08 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| a circuit with $m+4$ qubits and $m$ classical bits | `qc = QuantumCircuit(m + 4, m)` | the classical bits are named `c` |
| $X$ on a list of qubits | `qc.x([m + q for q in model.hf_x_qubits])` | the system register starts after the $m$ evaluation qubits |
| put the QPE circuit in | `qc.compose(phase_estimation(m, gate), inplace=True)` | added at the end; evaluation qubits first, then the system qubits |
| measure chosen qubits | `qc.measure(range(m), range(m))` | qubit $i$ into classical bit $i$ |
| run and count | `qff.quantum.sv_sampler(shots).run([qc]).result()[0].data.c.get_counts()` | a dictionary; the **rightmost** character is bit 0 |
| counts to a phase | `phi, freq = qff.bits.phase_from_counts(counts, m)` | the phase of the most frequent outcome and its frequency |
| a sub-matrix of rows and columns | `M[np.ix_(rows, cols)]` | the entries of `M` at those rows and those columns |
| timing | `t0 = time.perf_counter()` ... `time.perf_counter() - t0` | seconds |"""
CODE08 = '''# Toolbox for 08, on toys (no CO2 here).
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
print("OK: toolbox 08 ran.")'''

# ---------------------------------------------------------------------------------------------------------------- 10
MD10 = HEAD + """| To do | How it is written | What you get |
|---|---|---|
| the exact-simulation sampler | `from qiskit.primitives import StatevectorSampler` | `StatevectorSampler(seed=1, default_shots=4096)` |
| run a circuit | `job = sampler.run([qc])` | then `job.result()[0].data.meas.get_counts()` gives a dictionary like `{'0000': 3900, ...}` |
| measure every qubit | `qc.measure_all()` | adds a classical register named `meas` |
| the undo of a circuit | `qc.inverse()` | a new circuit that reverses `qc` |
| one circuit after another | `a.compose(b)` | a **new** circuit: `a` followed by `b` |
| bind a car's numbers | `fm.assign_parameters(x)` | a **new** circuit with one car's numbers filled in (the original `fm` is unchanged) |
| the quantum classifier | `from qiskit_machine_learning.algorithms import QSVC` | `QSVC(quantum_kernel=qk, C=10).fit(X, y)`, then `.score(X_test, y_test)` |
| share of a count | `counts.get("0000", 0) / shots` | 0 if the key never appeared |"""
CODE10 = '''# Toolbox for 10, on toys (not the EV fleet).
from qiskit import QuantumCircuit
from qiskit.circuit.library import zz_feature_map
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

bell = QuantumCircuit(2); bell.h(0); bell.cx(0, 1)
undo = bell.compose(bell.inverse())                        # a circuit followed by its own undo gives back |00>
undo.measure_all()                                         # adds a classical register named "meas"
counts_toy = StatevectorSampler(seed=1, default_shots=2000).run([undo]).result()[0].data.meas.get_counts()
print("counts:", counts_toy, "| share of '00':", counts_toy.get("00", 0) / 2000)
assert counts_toy == {"00": 2000}

toy_X = np.array([[0.1, 0.2], [0.2, 0.1], [0.8, 0.9], [0.9, 0.8]])
toy_y = np.array([0, 0, 1, 1])
toy_clf = QSVC(quantum_kernel=FidelityStatevectorKernel(feature_map=zz_feature_map(2, reps=1)), C=10).fit(toy_X, toy_y)
print("QSVC accuracy on its own 4 training points:", toy_clf.score(toy_X, toy_y))
print("OK: toolbox 10 ran.")'''

TOOLBOX = {1: (MD01, CODE01), 2: (MD02, CODE02), 3: (MD03, CODE03), 4: (MD04, CODE04), 5: (MD05, CODE05), 6: (MD06, CODE06),
           7: (MD07, CODE07), 8: (MD08, CODE08), 9: (MD05, CODE05), 10: (MD10, CODE10)}
