"""Build solutions/10_use_ev_fleet_kernel_solution.ipynb (notebook 10: EV fleet scheduling, V2G, quantum kernel classifier).

    python tools/build_nb10.py

Data: data/v2g_fleet.csv (made by tools/generate_v2g_data.py). The quantum kernel is taught here, in section 3 (there is no
lecture on it), so section 3 opens with a short classical primer. Result to remember: the best quantum kernel ties the classical
RBF kernel; nothing here is tuned to make the quantum side win.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, arabic_table, callout, details, hints, new  # noqa: E402
from toolbox import TOOLBOX  # noqa: E402
from qff import day  # noqa: E402

S1 = """**The problem.** Thousands of electric cars sit plugged in for hours. Each one can charge now, wait, or even **sell power back to the grid** (vehicle-to-grid, V2G) when the grid is stretched and the price is high, as long as it still leaves with enough charge. Deciding, car by car and hour by hour, who should do what is a big scheduling problem. A good optimiser can solve it, but slowly; a **learned model** can imitate the optimiser's decisions in a fraction of a second. Our question: can a model that uses a *quantum kernel* learn the rule "should this car discharge to the grid now?"

**Who faces it.** A utility or fleet operator with many plug-in vehicles (a depot, a car park, a charging network) and a grid operator who wants the flexibility.

""" + callout("real", """**What is real today.** The WEF/Aramco paper (Case 14, p. 28) describes E.ON and IBM scheduling a vehicle-to-grid fleet with a kernel-based learned model trained on approximate dynamic programming, plus quantum-kernel experiments with error mitigation, reporting performance "close to classical baselines". That is the honest state: quantum kernels have been tried on this kind of problem and come out about level with classical ones, not ahead.""") + """

**Why a quantum computer might help.** A quantum kernel measures how similar two situations are by comparing the quantum states a circuit makes from them. The hope is that some circuits make states whose similarity is hard to compute classically yet useful for classifying. What is still missing: no one has shown such a circuit that is useful *and* beats a classical kernel on real data.

| Paper's method | Our teaching route |
|---|---|
| Kernel-based learned model trained on approximate dynamic programming, plus quantum-kernel experiments with error mitigation (Case 14, p. 28) | A quantum-kernel support vector machine (Qiskit `FidelityStatevectorKernel` and `QSVC`) on 80 synthetic cars, 4 qubits |"""

S2_MD = """The data is a small synthetic fleet: **80 cars**, one row each, made by `tools/generate_v2g_data.py` (read it, it is short). Four numbers describe a car's situation, each scaled to run between 0 and 1 or close to it:

| column | meaning |
|---|---|
| `soc` | state of charge now (how full the battery is) |
| `hours_norm` | hours until the car leaves, divided by 12 |
| `price` | electricity price now, normalised |
| `load` | local grid load now, normalised |

and the label `discharge` is 1 if the car should sell power to the grid now, 0 if not. **This label comes from a simple rule written by us** (the car has spare charging time, and price and load are both on the high side). It stands in for the decisions of a real optimiser; it is *not* real fleet data. Four features means **4 qubits**: one qubit per feature.

We keep 60 cars for training and hide 20 for testing, so the test measures how well the model handles cars it has not seen."""

S2_DATA = """QUBITS = 4                       # one qubit per feature; the test runner reads this
import pandas as pd
from sklearn.model_selection import train_test_split

fleet = pd.read_csv(os.path.join(DATA, "v2g_fleet.csv"))
FEATURES = ["soc", "hours_norm", "price", "load"]
X = fleet[FEATURES].to_numpy()
y = fleet["discharge"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)
print(f"{len(fleet)} cars | discharge rate {y.mean():.0%} | train {X_train.shape}, test {X_test.shape}")
fleet.head()"""

S2_CL_MD = """**Classical baseline first.** Before any quantum circuit, see what ordinary machine learning does. A **support vector machine** (SVM) draws the best dividing line between "discharge" and "do not discharge" cars. The first guess to beat is the laziest one: always say "do not discharge". The next is a straight-line SVM. Your task T1 adds the stronger classical model, an SVM with an **RBF kernel** (it can draw curved boundaries; the next section explains what a kernel is)."""

S2_MAJORITY = """# Classical baseline 1: the laziest guess, and a straight-line SVM.
from sklearn.svm import SVC

acc_majority = float(np.mean(y_test == 0))                    # always answer "do not discharge"
linear = SVC(kernel="linear", C=10).fit(X_train, y_train)
acc_linear = linear.score(X_test, y_test)
print(f"always say no: {acc_majority:.2f} | straight-line SVM: train {linear.score(X_train, y_train):.2f}, test {acc_linear:.2f}")"""

T1 = """rbf = SVC(kernel="rbf", C=10).fit(X_train, y_train)
acc_rbf_train = rbf.score(X_train, y_train)
acc_rbf = rbf.score(X_test, y_test)
print(f"RBF-kernel SVM: train {acc_rbf_train:.2f}, test {acc_rbf:.2f}")"""
T1_STARTER = """# Goal: `rbf` is the trained classical RBF-kernel SVM (C = 10, unscaled features); acc_rbf_train and acc_rbf are its accuracy
# on the training set and on the test set.
raise NotImplementedError("T1: train the RBF-kernel SVM baseline and report train and test accuracy")"""
T1_CHECK = """assert abs(acc_rbf - 0.80) < 0.051, "T1: the RBF test accuracy should be about 0.80 (16 of 20 cars)"
assert acc_rbf_train >= 0.9, "T1: the RBF model should fit the 60 training cars almost perfectly" """

S2_NOTE = """So the bar is **0.80**: the RBF SVM gets 16 of the 20 test cars right, the straight line 15 of 20, and just saying "no" 13 of 20. With only 20 test cars, **one car is 5 points of accuracy**, so treat differences of 5 points as noise. Everything below has to be read against that."""

S3_PRIMER = """# Primer: a straight-line SVM on two features, price and load.
from matplotlib.colors import ListedColormap

two = [FEATURES.index("price"), FEATURES.index("load")]
svm2 = SVC(kernel="linear", C=10).fit(X_train[:, two], y_train)
gx, gy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
zone = svm2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
ax.contourf(gx, gy, zone, alpha=0.15, cmap=ListedColormap(["#8D8D8D", "#FA4D56"]))
ax.scatter(*X_train[:, two].T, c=np.where(y_train == 1, "#FA4D56", "#525252"), edgecolors="k", s=28)
ax.set_xlabel("price"); ax.set_ylabel("load")
ax.set_title("red = should discharge; the line is the SVM's guess", fontsize=9)
print(f"two features only: straight-line SVM test accuracy {svm2.score(X_test[:, two], y_test):.2f} (all four features: {acc_linear:.2f})")"""

S3_SCALE = """from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler((0, 1)).fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)
print("training features now run from", X_train_s.min().round(2), "to", X_train_s.max().round(2))

# Equation (5) by hand: every car, test cars included, is scaled with the TRAINING minimum and maximum.
lo, hi = X_train.min(axis=0), X_train.max(axis=0)
assert np.allclose((X_test - lo) / (hi - lo), X_test_s), "equation (5) should match MinMaxScaler"
assert np.allclose(X_train_s.min(axis=0), 0) and np.allclose(X_train_s.max(axis=0), 1), "the training cars fill [0, 1] exactly"
outside = int(((X_test_s < 0) | (X_test_s > 1)).sum())
print(f"test values outside [0, 1]: {outside} of {X_test_s.size} (expected: they are scaled with the training range)")
fig, ax = plt.subplots(figsize=(6, 2.8))
ax.axhspan(0, 1, color="#24A148", alpha=0.15, label="range of the 60 training cars")
for j, name in enumerate(FEATURES):
    ax.scatter(np.full(len(X_test_s), j) + np.linspace(-0.15, 0.15, len(X_test_s)), X_test_s[:, j], s=14, color="#0F62FE", label="20 test cars" if j == 0 else None)
ax.set_xticks(range(len(FEATURES))); ax.set_xticklabels(FEATURES); ax.set_ylabel("scaled value"); ax.legend(fontsize=8, loc="lower right")
plt.tight_layout()
print("OK: equation (5) reproduces the scaler; the training cars fill [0, 1]; the test cars may sit slightly outside.")"""

S3_FM = """from qiskit.circuit.library import zz_feature_map, z_feature_map
from qiskit.quantum_info import Statevector

fm = zz_feature_map(QUBITS, reps=1)
print("feature-map parameters (one per feature):", list(fm.parameters))
try:
    display(fm.decompose().draw("mpl", fold=30))
except Exception:                                      # drawing with matplotlib needs pylatexenc; fall back to text
    print(fm.decompose().draw("text", fold=120))

car_a, car_b = X_train_s[0], X_train_s[1]
state_a = Statevector(fm.assign_parameters(car_a))      # equation (6): |phi(x)> = U(x)|0...0> for car A
state_b = Statevector(fm.assign_parameters(car_b))      # equation (6) again, for car B
print(f"car A (label {y_train[0]}): {np.round(car_a, 2)}   car B (label {y_train[1]}): {np.round(car_b, 2)}")
print("the two cars' states differ; their overlap |<a|b>|^2 =", round(abs(state_a.inner(state_b)) ** 2, 3))"""

S3_KERNEL_MD = """**Step 3: the whole similarity table, exactly.** `FidelityStatevectorKernel` computes every K(a, b) from the exact statevectors of the simulator, so it takes well under a second. We then sort the 60 cars by label and draw K as a heat map: a kernel that separates the classes well would show brighter blocks along the diagonal. Look at how faint any such blocks are, and read the two averages printed under the picture: most pairs of cars score low (around 0.1), and the gap between same-class and different-class pairs is small. The SVM can still use small differences, but this is not a sharp picture. One more property worth checking: a table of squared overlaps is a valid kernel matrix, which means it is symmetric and has no negative eigenvalue (the SVM's mathematics needs that); the cell prints the smallest eigenvalue."""

S3_KERNEL = """from qiskit_machine_learning.kernels import FidelityStatevectorKernel

qk = FidelityStatevectorKernel(feature_map=fm)
K_train = qk.evaluate(X_train_s)                      # 60 x 60 table of similarities
order = np.argsort(y_train, kind="stable")             # no-discharge cars first, then discharge cars
n_no = int((y_train == 0).sum())
print("K is symmetric:", np.allclose(K_train, K_train.T), "| every car is identical to itself (diagonal = 1):", np.allclose(np.diag(K_train), 1))
smallest = np.linalg.eigvalsh(K_train).min()                  # a similarity table made of overlaps has no negative eigenvalue
print(f"smallest eigenvalue of K: {smallest:.2e} (a valid kernel matrix never goes below 0, up to rounding)")
assert smallest > -1e-9, "a table of squared overlaps should be positive semi-definite"
fig, ax = plt.subplots(figsize=(4.6, 4))
im = ax.imshow(K_train[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
ax.axhline(n_no - 0.5, color="w", lw=1); ax.axvline(n_no - 0.5, color="w", lw=1)
ax.set_title("kernel matrix, 60 training cars sorted by label"); fig.colorbar(im, ax=ax, label="similarity K")
same = (y_train[:, None] == y_train[None, :]) & ~np.eye(len(y_train), dtype=bool)
print(f"mean similarity: same-class pairs {K_train[same].mean():.3f}, different-class pairs {K_train[y_train[:, None] != y_train[None, :]].mean():.3f}")"""

S3_ZOOM_MD = """**A closer look: 12 cars.** Sixty rows is too many to read. Here are 12 cars, the first 6 "no" and the first 6 "discharge" in the training set, with their kernel matrix and where they sit on the price-load map."""

S3_ZOOM = """no_idx = [i for i in range(len(y_train)) if y_train[i] == 0][:6]
yes_idx = [i for i in range(len(y_train)) if y_train[i] == 1][:6]
twelve = no_idx + yes_idx                              # indices into the training set
K12 = K_train[np.ix_(twelve, twelve)]
labels12 = [f"{k}{'+' if y_train[i] else '-'}" for k, i in enumerate(twelve)]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
im = axes[0].imshow(K12, cmap="viridis", vmin=0, vmax=1)
axes[0].set_xticks(range(12)); axes[0].set_xticklabels(labels12, fontsize=7); axes[0].set_yticks(range(12)); axes[0].set_yticklabels(labels12, fontsize=7)
axes[0].set_title("12-car kernel matrix (- no, + discharge)", fontsize=9); fig.colorbar(im, ax=axes[0])
axes[1].scatter(*X_train[:, two].T, c="#C6C6C6", s=18, label="other training cars")
axes[1].scatter(*X_train[twelve][:, two].T, c=np.where(y_train[twelve] == 1, "#FA4D56", "#0F62FE"), edgecolors="k", s=60)
for k, i in enumerate(twelve):
    axes[1].annotate(str(k), X_train[i, two], textcoords="offset points", xytext=(4, 4), fontsize=8)
axes[1].set_xlabel("price"); axes[1].set_ylabel("load"); axes[1].set_title("the same 12 cars (red = discharge)", fontsize=9)
plt.tight_layout()
inside_yes = K12[6:, 6:][~np.eye(6, dtype=bool)].mean()
inside_no = K12[:6, :6][~np.eye(6, dtype=bool)].mean()
across = K12[:6, 6:].mean()
print(f"12 cars: similarity inside 'no' {inside_no:.3f} | inside 'discharge' {inside_yes:.3f} | across the two groups {across:.3f}")
print("car   soc  hours price  load  label")
for k, i in enumerate(twelve):
    print(f"{k:>3}  " + "  ".join(f"{v:.2f}" for v in X_train[i]) + f"  {y_train[i]}")"""

T2 = """from qiskit.primitives import StatevectorSampler


def kernel_entry(x_a, x_b, shots=4096):
    # circuit: feature map of car a, then the undo (inverse) of the feature map of car b
    qc = fm.assign_parameters(x_a).compose(fm.assign_parameters(x_b).inverse())
    qc.measure_all()
    counts = StatevectorSampler(seed=SEED, default_shots=shots).run([qc]).result()[0].data.meas.get_counts()
    return counts.get("0" * QUBITS, 0) / shots


k_manual_same = kernel_entry(X_train_s[0], X_train_s[0])
k_manual_pair = kernel_entry(X_train_s[0], X_train_s[1])
print(f"K(car 0, car 0) by circuit: {k_manual_same:.3f} (must be 1)")
print(f"K(car 0, car 1) by circuit: {k_manual_pair:.3f} | from the exact table: {K_train[0, 1]:.3f}")

from qiskit_machine_learning.algorithms import QSVC
qsvc = QSVC(quantum_kernel=qk, C=10).fit(X_train_s, y_train)
acc_qsvc_train = qsvc.score(X_train_s, y_train)
acc_qsvc = qsvc.score(X_test_s, y_test)
print(f"QSVC (zz_feature_map, reps=1, scaled to 0-1): train {acc_qsvc_train:.2f}, test {acc_qsvc:.2f}")"""
T2_STARTER = """# Goal: kernel_entry(x_a, x_b, shots=4096) estimates one kernel entry the way a device would (equation 9, second form);
# keep k_manual_same (car 0 with itself) and k_manual_pair (cars 0 and 1). Then train the quantum-kernel classifier `qsvc`
# (kernel object `qk`, C = 10) on the scaled training cars, and keep acc_qsvc_train and acc_qsvc.
raise NotImplementedError("T2: compute one kernel entry by hand, check it against the table, train QSVC")"""
T2_CHECK = """assert abs(k_manual_same - 1.0) < 1e-9, "T2: a car must be identical to itself, K = 1"
assert abs(k_manual_pair - K_train[0, 1]) < 0.03, "T2: the circuit estimate should match the exact table within 0.03"
assert 0.6 <= acc_qsvc <= 0.85, "T2: QSVC test accuracy should be about 0.70" """

T3 = """maps = {"zz1": zz_feature_map(QUBITS, reps=1),
        "zz2": zz_feature_map(QUBITS, reps=2, entanglement="linear"),
        "z2": z_feature_map(QUBITS, reps=2)}
scalings = {"unit": (0, 1), "pi": (0, np.pi)}
acc_q, acc_q_train = {}, {}
for mname, fmap in maps.items():
    for sname, rng in scalings.items():
        sc = MinMaxScaler(rng).fit(X_train)
        a, b = sc.transform(X_train), sc.transform(X_test)
        model = QSVC(quantum_kernel=FidelityStatevectorKernel(feature_map=fmap), C=10).fit(a, y_train)
        acc_q[(mname, sname)] = model.score(b, y_test)
        acc_q_train[(mname, sname)] = model.score(a, y_train)
print(f"{'feature map':<26}{'scaled to 0-1':<22}{'scaled to 0-pi'}")
for mname in maps:
    print(f"{mname:<26}" + "".join(f"train {acc_q_train[(mname, s)]:.2f} / test {acc_q[(mname, s)]:.2f}   " for s in scalings))
best_key = max(acc_q, key=acc_q.get)
explanation = (f"The best row is {best_key} at {acc_q[best_key]:.2f}. Scaling to 0-pi makes the rotation angles wrap around, so cars that are "
               "different end up with similar states and the kernel stops telling them apart (the ZZ map falls to a coin flip). "
               "The best map has no entanglement, which a classical computer can simulate easily, so it is not a quantum advantage.")
print(explanation)"""
T3_STARTER = """# Goal: for the three feature maps zz1 = zz_feature_map(4, reps=1), zz2 = zz_feature_map(4, reps=2, entanglement="linear"),
# z2 = z_feature_map(4, reps=2) and the two scalings unit = (0, 1), pi = (0, np.pi) (scaler fitted on X_train only), train a quantum-kernel
# QSVC (C = 10) and store test and training accuracy in acc_q[(map_name, scaling_name)] and acc_q_train[...] (six entries each).
# Then write explanation = "...": which row is best, which is worst, and why scaling matters.
raise NotImplementedError("T3: fill the variations table and explain the best and worst rows")"""
T3_CHECK = """assert len(acc_q) == 6 and len(acc_q_train) == 6, "T3: six rows (3 feature maps x 2 scalings) for test and for train"
assert acc_q[("zz1", "pi")] <= acc_q[("zz1", "unit")], "T3: scaling the ZZ map to 0-pi should be no better than 0-1"
assert isinstance(explanation, str) and len(explanation) > 60, "T3: write your explanation in a sentence or two" """

S3_SAMPLED = """import time
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

t0 = time.perf_counter()
qk_sampled = FidelityQuantumKernel(feature_map=fm, fidelity=ComputeUncompute(sampler=StatevectorSampler(seed=SEED)))
qsvc_sampled = QSVC(quantum_kernel=qk_sampled, C=10).fit(X_train_s, y_train)
acc_sampled = qsvc_sampled.score(X_test_s, y_test)
sampled_seconds = time.perf_counter() - t0
n_tr, n_te = len(X_train_s), len(X_test_s)
circuits = n_tr * (n_tr - 1) // 2 + n_te * n_tr
print(f"sampled kernel: test accuracy {acc_sampled:.2f} in {sampled_seconds:.0f} s, {circuits} circuits (exact kernel: {acc_qsvc:.2f} in a fraction of a second)")"""

S3_IBM_AFTER = callout("real", """The circuit above is the feature map for a single car on a pretend device. On real hardware every kernel entry is a separate job estimated from shots, noise shifts each entry, and for 6 000 cars the number of circuits would be about 18 million. That is why error mitigation and small datasets appear in the paper's experiments.""")

S4_TABLE = """# Results table: classical baselines first.
rows = [("always say no", None, acc_majority),
        ("SVM, straight line", acc_linear, acc_linear),
        ("SVM, RBF kernel", acc_rbf_train, acc_rbf)]
rows += [("QSVC zz_feature_map(reps=1), 0-1", acc_q_train[("zz1", "unit")], acc_q[("zz1", "unit")]),
         ("QSVC zz_feature_map(reps=2, linear), 0-1", acc_q_train[("zz2", "unit")], acc_q[("zz2", "unit")]),
         ("QSVC z_feature_map(reps=2), 0-1", acc_q_train[("z2", "unit")], acc_q[("z2", "unit")]),
         ("QSVC zz_feature_map(reps=1), 0-pi", acc_q_train[("zz1", "pi")], acc_q[("zz1", "pi")]),
         ("QSVC z_feature_map(reps=2), 0-pi", acc_q_train[("z2", "pi")], acc_q[("z2", "pi")]),
         ("QSVC zz_feature_map(reps=1), sampled kernel", None, acc_sampled)]
rows[1] = ("SVM, straight line", linear.score(X_train, y_train), acc_linear)
print(f"{'model':<46}{'train':<8}{'test':<8}")
for name, tr, te in rows:
    print(f"{name:<46}{'' if tr is None else format(tr, '.2f'):<8}{te:<8.2f}")
best_quantum = max(acc_q.values())
print(f"\\nbest quantum test accuracy {best_quantum:.2f} | RBF baseline {acc_rbf:.2f} | one test car = {1 / len(y_test):.2f}")"""

S4_PLOT = """fig, ax = plt.subplots(figsize=(8, 3.4))
names = [r[0].replace("QSVC ", "Q: ").replace("SVM, ", "C: ") for r in rows]
vals = [r[2] for r in rows]
colors = ["#8D8D8D"] * 3 + ["#0F62FE"] * (len(rows) - 3)
ax.barh(range(len(rows)), vals, color=colors)
ax.axvline(acc_rbf, ls="--", color="k", label="classical RBF")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(names, fontsize=7); ax.invert_yaxis()
ax.set_xlabel("test accuracy on 20 cars (grey = classical, blue = quantum kernel)"); ax.legend(loc="lower right")
plt.tight_layout()"""

S4_SHOWS = """**What this toy shows.** A circuit can serve as a similarity measure between two cars' situations, and a standard SVM can learn the charge-or-discharge rule from it. With sensible scaling, the best quantum kernel reaches 80% on 20 test cars, the same as the classical RBF kernel. The same circuit family scaled badly (0 to π) falls to 40-65%, which shows how much the encoding matters."""

S4_NOTSHOW = callout("watch", """**What this toy does not show.** Any advantage. The classical RBF is as good or better, trains instantly, and the best quantum row is the one **without entanglement** (`z_feature_map`), which is easy to simulate classically. Every quantum model has 100% training accuracy and lower test accuracy: it memorises the 60 training cars (**overfitting**). With 20 test cars one car is 5 points, so 0.75 and 0.80 are not meaningfully different. The labels come from a simple rule we wrote, not a real fleet optimiser. And we did not tune anything to help the quantum side: the three feature maps and two scalings were chosen before looking at results, and all rows are shown. On hardware every kernel entry is estimated from shots, and the number of circuits grows with the square of the data size.""")

S4_PAPER = """**Paper versus our route.** Same idea as the paper. WEF/Aramco Case 14 (p. 28): E.ON and IBM schedule a vehicle-to-grid fleet with a kernel-based learned model trained on approximate dynamic programming, plus quantum-kernel experiments with error mitigation, and report performance "close to classical baselines". Our data is synthetic and our model is a plain quantum-kernel SVM."""

T1_INTRO = r"""### T1 · Warm-up: the classical RBF baseline · التمرين ١

**What you are about to build, and why.** Before any quantum circuit we need the classical model to beat. In T1 you train a support vector machine with an **RBF kernel**, the standard classical choice, on the 60 training cars and measure its accuracy on the 20 hidden test cars. Everything quantum later is judged against this number. (Section 3 explains what a kernel is and why the RBF one works; for now think of it as "a way of saying how alike two cars are".)

**The formula behind it.** The RBF kernel scores the similarity of two cars $a$ and $b$ by their distance in feature space, with a width $\gamma$ ("gamma"):

$$K_{\text{RBF}}(a,b)=e^{-\gamma\,\lVert a-b\rVert^2}\ \in(0,1] \tag{R}$$

Identical cars score 1; the further apart, the closer to 0. By hand: two cars at squared distance $0.25$ with $\gamma=1$ score $e^{-0.25}=0.78$. The SVM learns from the table of all such scores (section 3.1).

""" + hints(
    r"""You are using scikit-learn, which you have seen in the straight-line SVM cell just above. What do you change to get the RBF kernel instead of the linear one? What does `.score` return? If the training accuracy comes out at 1.00 and the test accuracy is lower, what does that tell you about the model?""",
    r"""Steps: create and train the model (the support-vector classifier `SVC`, with the RBF kernel and C = 10) on the unscaled training features; compute its accuracy on the training set and on the test set; print both. Two details: `C` is a setting you pass when you create the model (a larger C makes it try harder to get every training car right), and in T1 the features stay *unscaled*, so do not use the scaler from section 3. The model's `.score(X, y)` method returns the share of cars it labels correctly, so you call it once with the training cars and once with the test cars.""",
    r"""```python
rbf = ______               # the trained RBF-kernel SVM
acc_rbf_train = ______     # accuracy on the 60 training cars
acc_rbf = ______           # accuracy on the 20 test cars
```
Self-check target: test accuracy about 0.80 (16 of 20 cars).""")

S3_ROADMAP = r"""### 3.0 · The map of this section

""" + callout("idea", r"""**A quantum kernel in one sentence.** A support vector machine needs only one thing from the data: a table saying how *similar* every pair of cars is. A quantum circuit can fill in that table: turn each car into a quantum state, and score two cars by how much their states overlap. Everything else (the learning) stays classical, and nothing here is tuned to make the quantum side win.""") + r"""

There is **no knob-tuning of a circuit** here: the circuit is fixed, and what matters is the similarity it produces. We build the idea from things you already know: vectors, angles and probability. Every step has a **numbered equation** and, right after it, the **code cell that implements it**.

| Step | What happens | Equation | Where in the code | Your task |
|---|---|---|---|---|
| (section 2) | The classical RBF baseline | (R) | `rbf` | **T1** |
| 3.1 | A line, a kernel, and the SVM's decision rule | (1) to (4) | `svm2`, `K_rbf_demo` | |
| 3.2 | Features become angles: scaling | (5) | `scaler` | |
| 3.3 | The quantum feature map: a car becomes a state | (6), (7), (8) | `fm`, `zz_by_hand` | |
| 3.4 | The quantum kernel: how much two states overlap | (9), (10) | `K_train`, `kernel_entry` | **T2** |
| 3.5 | Same classifier, different circuits and scalings | (10) again | `maps`, `scalings` | **T3** |
| 3.6 | The kernel the way a device computes it: shots and cost | (11) | `qk_sampled` | |

Words in **bold** are new ideas; each is explained where it first appears, and the Arabic term sits in a small table next to it."""

S3_FLOW = """qff.plots.plot_flow(
    [("Car", "4 numbers", "source"), ("Scale", "angles 0 to 1", "move"), ("Feature map", "circuit makes a state", "store"),
     ("Kernel", "overlap of two states", "use"), ("SVM", "classical learning", "source")],
    title="Quantum kernel classifier: only the similarity table is quantum");"""

S3_KERNEL_THEORY_MD = r"""### 3.1 · A line, a similarity table, and the SVM

**Classification.** We want a rule that says "discharge" or "do not discharge" from a car's four numbers $x=(\text{soc},\text{hours},\text{price},\text{load})$. We learn it from labelled examples (the 60 *training* cars) and judge it on cars it has not seen (the 20 *test* cars): the share it gets right is its **accuracy**.

**A straight line.** A **support vector machine (SVM)** draws the dividing line $w\cdot x+b=0$ that leaves the widest empty gap (the **margin**) between the two groups; a car is classified by the sign of

$$f(x)=w\cdot x+b \tag{1}$$

If no straight line separates the groups, we can first *re-describe* each car by a richer list of numbers $\phi(x)$ and draw the line there. The trick that makes this cheap is that the SVM never needs $\phi(x)$ itself, only the **similarity** of every pair of cars, the **kernel**

$$K(a,b)=\langle\phi(a),\phi(b)\rangle\qquad(\text{an inner product: sum of products of matching entries}) \tag{2}$$

so the learned rule becomes a weighted vote of the training cars, each voting by how similar it is to the new car:

$$f(x)=\sum_{i\in\text{training}}\alpha_i\,y_i\,K(x_i,x)+b \tag{3}$$

($y_i=\pm1$ is the label of training car $i$; the weights $\alpha_i\ge0$ are found by the SVM, and most are 0.) **All the SVM needs is the table $K$ of similarities, the *kernel matrix*.** The classical **RBF** kernel is the similarity (R) of T1:

$$K_{\text{RBF}}(a,b)=e^{-\gamma\lVert a-b\rVert^2} \tag{4}$$

A **quantum kernel** replaces (4) by a similarity computed by a quantum circuit. Everything else, equations (1) to (3), stays classical.

""" + arabic_table([
    ("Support vector machine (SVM)", "آلة المتجهات الداعمة", "I"), ("Kernel", "النواة", "I"), ("Quantum kernel", "النواة الكمية", "G"), ("Feature map", "خريطة السمات", "I"),
    ("Classifier", "المصنِّف", "I"), ("Accuracy", "الدقة", "I"), ("Training data / test data", "بيانات التدريب / بيانات الاختبار", "I"), ("Margin", "الهامش", "I"),
    ("Similarity", "التشابه", "I"), ("Quantum machine learning", "تعلم الآلة الكمومي", "I")])

S3_RBF_CODE = '''# Equation (R) of T1, which is the same formula as equation (4): K_RBF(a, b) = exp(-gamma |a - b|^2), by hand and against scikit-learn: two cars at squared distance 0.25 with gamma = 1 score exp(-0.25) = 0.78.
from sklearn.metrics.pairwise import rbf_kernel

a_demo, b_demo = np.array([0.2, 0.5, 0.5, 0.5]), np.array([0.7, 0.5, 0.5, 0.5])          # they differ in one feature by 0.5
K_rbf_demo = np.exp(-1.0 * np.sum((a_demo - b_demo) ** 2))
print(f"squared distance {np.sum((a_demo - b_demo) ** 2):.2f} -> K_RBF = exp(-0.25) = {K_rbf_demo:.4f}")
assert abs(K_rbf_demo - rbf_kernel([a_demo], [b_demo], gamma=1.0)[0, 0]) < 1e-12, "equation (4) should match scikit-learn"
print("OK: equation (4) matches scikit-learn's rbf_kernel.")

# Equation (3): f(x) = sum_i alpha_i y_i K(x_i, x) + b, rebuilt from the trained RBF model `rbf` of T1.
# `dual_coef_` holds alpha_i * y_i (one number per support vector), `support_vectors_` are the x_i, `intercept_` is b.
g = 1 / (X_train.shape[1] * X_train.var())                                    # scikit-learn's default "scale" width gamma
f_manual = rbf.dual_coef_ @ rbf_kernel(rbf.support_vectors_, X_test, gamma=g) + rbf.intercept_
assert np.allclose(f_manual.ravel(), rbf.decision_function(X_test)), "equation (3) should reproduce the decision function"
print("OK: equation (3) reproduces scikit-learn's decision function for all", len(X_test), "test cars.")'''

S3_SCALE_MD = r"""### 3.2 · Features become angles: scaling

A feature map uses each feature as a rotation **angle**, so the *range* of the numbers matters a great deal (we will see how much). We rescale every feature linearly to the interval $[a,b]$ using the training cars only (the test cars must not leak into the scaling):

$$x'=a+(b-a)\,\frac{x-x_{\min}}{x_{\max}-x_{\min}} \tag{5}$$

with $x_{\min},x_{\max}$ the smallest and largest value of that feature *in the training set*. We start with $[a,b]=[0,1]$ (angles up to about 57°).

""" + callout("idea", r"""**A number becomes a turn.** Think of each feature as a dial. The circuit will turn qubit $i$ by an angle that grows with $x_i'$, so two cars whose feature values are close end up with nearly the same state, and cars far apart end up with very different states. The scaling decides *how far round the dial the biggest difference can reach*: a short turn (range 0 to 1) keeps all cars within one quarter of a turn of each other; a long turn (0 to $\pi$) lets the two most different cars land on the *same* spot again (section 3.5 shows why that hurts).""") + "\n\n" + callout("watch", r"""**Fit the scaler on the training cars only.** The minimum and maximum in equation (5) come from the 60 training cars; the 20 test cars are scaled with those same numbers, so a test value can land a little *outside* $[0,1]$. That is correct and expected. Refitting the scaler on the test cars (or on all 80) would let information about the hidden cars leak into the model, and the test accuracy would no longer be an honest number. The cell below checks equation (5) by hand and counts how many test values fall outside.""") + "\n\n" + arabic_table([
    ("Scaling", "تحجيم المقياس", "I"), ("Feature", "السمة", "I"), ("Data leakage", "تسرّب البيانات", "I"), ("Angle", "الزاوية", "I")])

S3_MAP_MD = r"""### 3.3 · The quantum feature map: a car becomes a state

**Amplitudes and states.** $n$ qubits hold a list of $2^n$ numbers called **amplitudes**; the squared size of each is the **probability** of measuring that outcome. A **feature map** is a circuit $U(x)$ that starts from all qubits in $|0\rangle$ and, using the car's numbers as rotation angles, prepares a state that depends on the car:

$$|\phi(x)\rangle=U(x)\,|0\ldots0\rangle \tag{6}$$

**The simplest map (one qubit per feature, no entanglement).** A Hadamard gate $H$ turns $|0\rangle$ into the equal mixture $\frac1{\sqrt2}(|0\rangle+|1\rangle)$; a phase gate $P(2x)$ then multiplies the $|1\rangle$ part by $e^{2ix}$ (a **phase**: a turn by the angle $2x$; $e^{i\theta}=\cos\theta+i\sin\theta$ is a point on a circle of radius 1). For one feature:

$$|\phi(x)\rangle=\tfrac1{\sqrt2}\bigl(|0\rangle+e^{2ix}|1\rangle\bigr) \tag{7}$$

This is Qiskit's `z_feature_map` (one repetition). **The standard map with entanglement**, `zz_feature_map`, adds for every *pair* of features $i<j$ a coupling between their qubits (a CX gate, a phase gate, and the same CX gate again) with angle $2(\pi-x_i)(\pi-x_j)$:

$$|\phi(x)\rangle=\prod_{i<j}\Bigl[\text{CX}_{ij}\,P_j\bigl(2(\pi-x_i)(\pi-x_j)\bigr)\,\text{CX}_{ij}\Bigr]\ \prod_iP_i(2x_i)\ H^{\otimes4}|0000\rangle \tag{8}$$

(with `reps=2` the whole block is repeated). **Entanglement** here means the qubits' states are tied together by the pair terms, so the state cannot be written as one independent state per qubit.

""" + arabic_table([
    ("Feature map", "خريطة السمات", "I"), ("Amplitude", "السعة", "G"), ("Phase", "الطور", "U"), ("Entanglement", "التشابك", "G"),
    ("Hadamard gate", "بوابة هادامارد", "G"), ("Rotation angle", "زاوية الدوران", "I"), ("Quantum circuit", "الدائرة الكمومية", "U"), ("Quantum state", "الحالة الكمومية", "I")])

S3_ZZ_CODE = '''# Equation (8) with plain gates, checked against Qiskit's zz_feature_map: the same state, up to a global phase.
from qiskit import QuantumCircuit


def zz_by_hand(x):
    """H on every qubit, P(2 x_i) on qubit i, then for every pair i < j: CX, P(2 (pi - x_i)(pi - x_j)) on j, CX."""
    n = len(x)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for i in range(n):
        qc.p(2 * x[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j)
            qc.p(2 * (np.pi - x[i]) * (np.pi - x[j]), j)
            qc.cx(i, j)
    return qc


car0 = X_train_s[0]
same_state = Statevector(zz_by_hand(car0)).equiv(Statevector(fm.assign_parameters(car0)))
assert same_state, "equation (8) should reproduce Qiskit's zz_feature_map"
print("hand-built equation (8) == zz_feature_map(4, reps=1) (same state up to a global phase):", same_state)
print("gates in the hand-built map:", dict(zz_by_hand(car0).count_ops()), "(4 H, 4 P for the features, 6 pairs x (2 CX + 1 P))")'''

S3_KERNEL_THEORY2_MD = r"""### 3.4 · The quantum kernel: how much two states overlap

Two cars $a$ and $b$ give two states $|\phi(a)\rangle$ and $|\phi(b)\rangle$. Their **overlap** $\langle\phi(a)|\phi(b)\rangle$ is an inner product of the two amplitude lists (with complex conjugation on the first); its squared size is a number between 0 (states completely different) and 1 (identical states): the **quantum kernel**

$$K(a,b)=\bigl|\langle\phi(a)|\phi(b)\rangle\bigr|^2=P\bigl(\text{measuring }0000\ \text{after }U(a)\text{ then }U(b)^\dagger\bigr) \tag{9}$$

The second form is how a device does it: run the circuit for car $a$, then the *undo* ($U^\dagger$, the inverse) of the circuit for car $b$, and measure. If the two states are the same, the undo works perfectly and every run gives $0000$, so $K=1$; otherwise $K$ is the *fraction of runs* that give $0000$. A simulator can also read $K$ exactly from the statevectors.

**A worked example you can check on paper (one qubit).** With the simplest map (7), the overlap of cars with feature values $a$ and $b$ is $\frac12\bigl(1+e^{2i(b-a)}\bigr)$ and its squared size is

$$K(a,b)=\cos^2(a-b)\ \text{ for one feature},\qquad K(a,b)=\prod_{i=1}^4\cos^2(a_i-b_i)\ \text{ for the four-qubit Z map} \tag{10}$$

Because there is no entanglement, the four qubits are independent and the kernel is a product. (With two repetitions, as in the best row of T3, each factor is a different function of the two feature values, but it is still a product of four single-qubit factors, so the conclusions below still hold.) For $a=0.2$ and $b=0.9$: $K=\cos^2(0.7)=0.585$. **Two things follow.** (i) $\cos^2(\Delta)\approx e^{-\Delta^2}$ for small differences $\Delta$: the Z-map kernel behaves like an RBF kernel with $\gamma=1$, which is why it can tie the classical one. (ii) A kernel with this product form costs a classical computer 4 cosines per pair of cars: **it is trivially easy to simulate**, so a good result with it is no quantum advantage. With the ZZ map (8) there is no such formula, but the cost of computing it exactly on a classical computer grows quickly with the number of qubits.

""" + callout("watch", r"""**Kernel concentration.** If the off-diagonal similarities are all small, the kernel matrix is close to the *identity matrix* (every car is similar only to itself): the SVM can then fit any labels on the training cars (memorise them) but has little to say about new cars. This is **overfitting**, and the plots below show how strongly it affects the ZZ map.""") + "\n\n" + arabic_table([
    ("Overlap", "التراكب (التداخل الجزئي) بين حالتين", "I"), ("Inner product", "الجداء الداخلي", "I"), ("Kernel matrix", "مصفوفة النواة", "I"),
    ("Overfitting", "الإفراط في الملاءمة", "I"), ("Inverse circuit", "الدائرة العكسية", "I"), ("Probability", "الاحتمال", "I"), ("Measurement", "القياس", "G")])

S3_Z_CODE = '''# Equation (10) in code: the Z-map kernel is a product of cos^2 of the feature differences, checked against Qiskit.
from qiskit.circuit.library import z_feature_map
from qiskit_machine_learning.kernels import FidelityStatevectorKernel as _Fidelity

K_one = _Fidelity(feature_map=z_feature_map(1, reps=1)).evaluate(np.array([[0.2], [0.9]]))
print(f"one qubit, a = 0.2, b = 0.9: Qiskit {K_one[0, 1]:.4f} | cos^2(0.7) = {np.cos(0.7) ** 2:.4f}")
assert abs(K_one[0, 1] - np.cos(0.7) ** 2) < 1e-12
K_z = _Fidelity(feature_map=z_feature_map(QUBITS, reps=1)).evaluate(X_train_s)
K_prod = np.prod(np.cos(X_train_s[:, None, :] - X_train_s[None, :, :]) ** 2, axis=2)
print("four-qubit Z map: largest difference between Qiskit's 60x60 table and the product of cos^2:", np.abs(K_z - K_prod).max())
assert np.allclose(K_z, K_prod, atol=1e-10), "equation (10) should reproduce the Z-map kernel"
K_gauss = np.exp(-((X_train_s[:, None, :] - X_train_s[None, :, :]) ** 2).sum(axis=2))
print(f"cos^2 product against exp(-distance^2): average difference {np.abs(K_prod - K_gauss).mean():.3f}, largest {np.abs(K_prod - K_gauss).max():.3f}")
# Equation (7): for one qubit the Z map gives (|0> + e^{2ix}|1>)/sqrt(2); here x = 0.3, equal up to a global phase.
state_one = Statevector(z_feature_map(1, reps=1).assign_parameters([0.3]))
assert state_one.equiv(Statevector(np.array([1, np.exp(2j * 0.3)]) / np.sqrt(2))), "equation (7) should describe the one-qubit Z map"
print("OK: equation (7), the one-qubit Z-map state, matches Qiskit up to a global phase.")
print("OK: this quantum kernel is a product of cosines: a classical computer gets it from 4 cosines per pair.")'''

S3_Z_TRY = callout("try", r"""**Do it by hand.** Pick two cars, for example `i, j = 0, 1`, and compute `np.prod(np.cos(X_train_s[i] - X_train_s[j]) ** 2)`. Compare it with `K_z[i, j]` from the cell above. Then find the pair of cars with the *smallest* `K_z` (hint: `np.argmin` on the table, or look at `np.unravel_index`) and print their four scaled features. Which feature differs most, and does that match the cos² picture?""")

T2_INTRO = r"""### T2 · Core step: one kernel entry by hand, and the QSVC · التمرين ٢

**What you are about to build, and why.** The kernel table in the previous cell was computed exactly by the simulator from the statevectors. A real device cannot do that: it estimates *each* entry from repeated runs, as equation (9) says. In T2 you compute **one** entry the way a device would, for cars 0 and 1: build the circuit "feature map of car a, then the *inverse* of the feature map of car b", measure all four qubits many times, and count the fraction of runs that give `0000`. You compare it with the exact table, and then you train the full quantum-kernel classifier, Qiskit's `QSVC`, on the 60 training cars and read its test accuracy.

**The formula you implement:** equation (9), second form: $K(a,b)=\text{(number of runs giving 0000)}/\text{(number of runs)}$. A car compared with itself must give exactly 1.

""" + hints(
    r"""Look at equation (9): the circuit has two halves. The first half is the feature map for car a. The second half is the *undo* of the feature map for car b. Two circuits are joined with a method of the first one. To count outcomes you must measure all qubits. Ask yourself two things first: why does a car compared with itself have to give exactly 1, and which objects do you already have from section 3 that you can reuse (the feature map `fm` and the kernel object `qk`)?""",
    r"""Steps inside `kernel_entry`: assign car a's numbers to the feature map; assign car b's numbers to a copy and take its inverse; compose them; measure all qubits; run the circuit with a `StatevectorSampler` (seeded, `shots` runs) and read the counts of the register `meas`; the answer is the share of runs that gave all zeros. Then compute the two entries and train `QSVC` with the kernel object `qk` on the scaled training data.""",
    r"""```python
def kernel_entry(x_a, x_b, shots=4096):
    qc = ______               # feature map of car a, then the undo of the feature map of car b
    ______                    # measure all qubits
    counts = ______           # run it with the seeded sampler and read the counts
    return ______             # the share of runs that gave 0000
k_manual_same = ______        # a car against itself
k_manual_pair = ______        # cars 0 and 1
qsvc = ______                 # the quantum-kernel classifier, trained on the scaled training cars
acc_qsvc_train, acc_qsvc = ______, ______
```
Self-check targets: the same-car entry is exactly 1, the pair entry matches the exact table within about 0.03, and the QSVC test accuracy is about 0.70.""")

S3_CONC_MD = """**Kernel concentration, in one picture.** The histograms show the off-diagonal similarities (pairs of *different* training cars) for four kernels. The ZZ map with angles up to $\\pi$ squeezes almost every pair towards 0; the plain ZZ map at 0-1 is better but still low; the Z map and the classical RBF kernel spread their values out. A kernel matrix with values squeezed near 0 is close to the identity matrix: the SVM memorises the training cars."""

S3_CONC = '''# Off-diagonal similarities of four kernels on the 60 training cars (pairs of different cars).
sc_pi = MinMaxScaler((0, np.pi)).fit(X_train)
K_zz_pi = FidelityStatevectorKernel(feature_map=fm).evaluate(sc_pi.transform(X_train))
K_rbf_train = rbf_kernel(X_train, gamma=1 / (X_train.shape[1] * X_train.var()))                      # scikit-learn's default width
offdiag = ~np.eye(len(X_train), dtype=bool)
kernels = {"ZZ map, 0-pi": K_zz_pi, "ZZ map, 0-1": K_train, "Z map, 0-1": K_z, "classical RBF": K_rbf_train}
fig, axes = plt.subplots(1, 4, figsize=(12, 2.6), sharey=True)
for ax, (name, Kmat) in zip(axes, kernels.items()):
    ax.hist(Kmat[offdiag], bins=20, range=(0, 1), color="#0F62FE")
    ax.set_title(f"{name}\\nmean {Kmat[offdiag].mean():.2f}", fontsize=9); ax.set_xlabel("similarity K")
axes[0].set_ylabel("number of car pairs")
plt.tight_layout()
print({name: round(float(Kmat[offdiag].mean()), 3) for name, Kmat in kernels.items()})'''

S3_VAR_MD = r"""### 3.5 · The same classifier with different circuits and scalings

Two choices change the kernel: the **feature map** (how the car's numbers become a state) and the **scaling** (how large the angles get). By equation (10) the Z-map kernel depends on the feature differences through $\cos^2(\Delta)$, which is **periodic**: it has period $\pi$, so two values that differ by exactly $\pi$ look *identical*. If we scale the features to $[0,\pi]$, the two most different cars (a difference of $\pi$ in a feature) get $K=1$ in that feature, as if they were the same car. Scaling to $[0,1]$ keeps every difference inside $[0,1]$ where $\cos^2$ falls steadily from 1 to $0.29$. The ZZ map has the same wrap-around problem, only worse. The cell below draws the curve and the two ranges, and then T3 measures what it does to accuracy. `z_feature_map` has no entanglement (each qubit is treated alone), which makes it easy to simulate on an ordinary computer. We report every row, good or bad.

""" + arabic_table([("Periodic function", "دالة دورية", "I"), ("Period", "الدور (الدورة)", "I"), ("Repetitions (reps)", "التكرارات", "I")])

S3_PERIOD = '''# Why the scaling matters: cos^2 of a feature difference, over the range of differences each scaling allows.
delta = np.linspace(0, np.pi, 300)
fig, ax = plt.subplots(figsize=(6.5, 2.8))
ax.plot(delta, np.cos(delta) ** 2, color="#0F62FE")
ax.axvspan(0, 1, color="#24A148", alpha=0.25, label="differences when scaled to 0-1")
ax.axvspan(1, np.pi, color="#FA4D56", alpha=0.12, label="extra differences when scaled to 0-pi")
ax.set_xlabel("difference of one feature between two cars"); ax.set_ylabel("K = cos^2(difference)"); ax.legend(fontsize=8)
plt.tight_layout()
print(f"scaled to 0-1: cos^2 stays between {np.cos(1) ** 2:.2f} and 1 | scaled to 0-pi: a difference of pi/2 gives K = {np.cos(np.pi / 2) ** 2:.2f} and a difference of pi gives K = {np.cos(np.pi) ** 2:.2f}")
# The same thing in Qiskit itself: move one car's feature by exactly pi and the one-qubit Z-map kernel does not change.
k1 = _Fidelity(feature_map=z_feature_map(1, reps=1))
K_before = k1.evaluate(np.array([[0.2], [0.9]]))[0, 1]
K_shifted = k1.evaluate(np.array([[0.2], [0.9 + np.pi]]))[0, 1]
print(f"a = 0.2, b = 0.9: K = {K_before:.4f} | b moved by pi: K = {K_shifted:.4f}   (the kernel cannot see a shift of pi)")
assert abs(K_before - K_shifted) < 1e-9, "cos^2 has period pi, so the Z-map kernel cannot tell b from b + pi"
print("OK: the kernel is periodic with period pi; scaling to 0-pi makes the most different cars look identical in that feature.")'''

T3_INTRO = r"""### T3 · Stretch: three circuits, two scalings, and what the table says · التمرين ٣

**What you are about to build, and why.** The honest test of a quantum kernel is not one number but a table: change the feature map and the scaling, and report *every* row, good or bad, next to the classical baseline. In T3 you loop over three feature maps (`zz_feature_map` with one repetition, `zz_feature_map` with two repetitions and linear entanglement, and `z_feature_map` with two repetitions) and two scalings (0 to 1 and 0 to $\pi$), train a `QSVC` for each of the six combinations, and store test and training accuracy. Then you write one or two sentences saying which row is best, which is worst, and why the scaling matters, using what you learned in 3.4 and 3.5.

**The formulas you use:** equation (5) for the two scalings (`MinMaxScaler((0, 1))` and `MinMaxScaler((0, np.pi))`, fitted on the training cars only) and the kernel (9) inside `FidelityStatevectorKernel`.

""" + hints(
    r"""You are doing the same three things six times: scale the features, build the quantum kernel from a feature map, train `QSVC`, and read two accuracies. What changes between the six runs? Which parts can you put in a dictionary and loop over? Before you code anything, write down which of the six runs you expect to be best and which worst; sections 3.4 and 3.5 contain what you need, and the table will tell you whether you were right.""",
    r"""Define the three feature maps and the two scalings as dictionaries; two nested loops over them. Inside: fit a `MinMaxScaler` with the chosen range on the training cars only, scale both sets, train a `QSVC` whose kernel is a `FidelityStatevectorKernel` built from the feature map, and store its test and training accuracy under the key (map name, scaling name). After the loops write the explanation. Two details: the only thing that differs between the two scalings is the range you give the scaler, and the only thing that differs between the three maps is the feature map you hand to the kernel; and the Self-check reads results by tuple keys such as `("zz1", "unit")`, so your dictionaries need keys of exactly that kind.""",
    r"""```python
maps = ______                 # {"zz1": ..., "zz2": ..., "z2": ...} as described in the task
scalings = ______             # {"unit": the range 0 to 1, "pi": the range 0 to pi}
acc_q, acc_q_train = {}, {}
for mname, fmap in maps.items():
    for sname, rng in scalings.items():
        ______                # scale (fit on training only), train QSVC with this map, store both accuracies
explanation = "..."           # which row is best, which is worst, and why scaling matters
```""")

S3_SAMPLED_MD = r"""### 3.6 · The kernel the way a device computes it: shots and cost

A real device estimates every entry from $M$ runs (**shots**): $\hat K$ is the fraction of runs that give $0000$, so it carries a random error. If the true value is $K$, the typical size of that error is

$$\sigma_{\hat K}=\sqrt{\frac{K(1-K)}{M}}\ \approx\ 0.009\ \text{ for }K=0.1,\ M=1024,\qquad \text{number of circuits}=\frac{N_{\text{train}}(N_{\text{train}}-1)}2+N_{\text{test}}N_{\text{train}} \tag{11}$$

For our data $60\cdot59/2=1770$ circuits for the training table plus $20\cdot60=1200$ for the test cars: **2 970 circuits** for 80 cars. For 6 000 training cars the table alone has $6000\cdot5999/2\approx18$ million entries: the cost grows with the **square** of the data size. `FidelityQuantumKernel` with a sampler does what a device would (here a simulated sampler, 1 024 shots per circuit by default). The cell below first checks the error formula by simulation, then runs the sampled kernel; it takes about half a minute.

""" + arabic_table([("Shots", "عدد مرات التشغيل (shots)", "I"), ("Sampling error", "خطأ أخذ العيّنات", "I"), ("Noise", "الضوضاء", "G"),
                    ("Error mitigation", "تخفيف الأخطاء", "I"), ("Quantum advantage", "الأفضلية الكمومية", "G")])

S3_SHOT_CHECK = '''# Equation (11): the error of a shot-based kernel entry, and the number of circuits.
sim = np.random.default_rng(SEED)
K_true, shots_demo = 0.1, 1024
estimates = sim.binomial(shots_demo, K_true, 20000) / shots_demo
print(f"K = {K_true}, {shots_demo} shots: spread of the estimate {estimates.std():.4f} | equation (11): {np.sqrt(K_true * (1 - K_true) / shots_demo):.4f}")
assert abs(estimates.std() / np.sqrt(K_true * (1 - K_true) / shots_demo) - 1) < 0.05
n_tr_, n_te_ = len(X_train_s), len(X_test_s)
print(f"circuits for 60 training + 20 test cars: {n_tr_ * (n_tr_ - 1) // 2} + {n_te_ * n_tr_} = {n_tr_ * (n_tr_ - 1) // 2 + n_te_ * n_tr_}")
print(f"circuits for the training table of 6000 cars: {6000 * 5999 // 2:,}")'''


ACC = """assert QUBITS == 4 and X_train.shape == (60, 4) and X_test.shape == (20, 4)
assert abs(acc_rbf - 0.80) < 0.051 and abs(acc_linear - 0.75) < 0.051
assert abs(k_manual_same - 1.0) < 1e-9 and abs(k_manual_pair - K_train[0, 1]) < 0.03
assert np.allclose(np.diag(K_train), 1.0) and np.allclose(K_train, K_train.T)
assert 0.6 <= acc_q[("zz1", "unit")] <= 0.85 and acc_q[("zz1", "pi")] <= acc_q[("zz1", "unit")]
assert max(acc_q.values()) <= acc_rbf + 0.051          # the honest result: no quantum win
print("quantum test accuracies:", {f"{k[0]}/{k[1]}": v for k, v in acc_q.items()}, "| classical RBF:", acc_rbf)"""

INSTRUCTOR = """**Instructor note (never reaches the starter).** Expected numbers (seed 2026, split `random_state=7`): majority 0.65, linear SVM train 1.00 / test 0.75, RBF train 1.00 / test 0.80. Quantum test accuracy with C = 10: zz1 0.70 (0-1) and 0.50 (0-π); zz2 0.75 and 0.65; z2 **0.80** and 0.40; train accuracy is 1.00 in all six. Sampled kernel (1 024 shots) about 0.75 in 30 s (the plan expected 0.70; sampling noise moves it by a car). Pitfalls: `FidelityStatevectorKernel` is the exact path; `QSVC` passes C to scikit-learn; do not drop `load` or otherwise change the features to make quantum look better (a 3-feature variant gave RBF 1.00 and quantum 0.95, which is a different, easier problem)."""


S3_SYMBOLS = '**Symbol table** (keep it open while you read; the right column is where the symbol lives in the code).\n\n| Symbol | Meaning | In the code |\n|---|---|---|\n| $x=(\\text{soc},\\text{hours},\\text{price},\\text{load})$ | one car\'s four numbers | `X`, `X_train[i]` |\n| $y\\in\\{0,1\\}$ | label: should this car discharge now? | `y`, `y_train` |\n| $f(x)$ | the classifier\'s decision (sign of the score) | `svm.predict`, `qsvc.predict` |\n| $K(a,b)$ | similarity of two cars (the kernel) | `K_train[i, j]`, `kernel_entry(a, b)` |\n| $\\gamma$ | width of the RBF kernel | `SVC(kernel="rbf")` (default `gamma="scale"`) |\n| $\\lvert\\phi(x)\\rangle=U(x)\\lvert0\\ldots0\\rangle$ | the quantum state of a car | `fm.assign_parameters(x)` |\n| $\\alpha_i,\\ b$ | the SVM\'s weights and offset | `qsvc`, `svm.dual_coef_` |\n| $M,\\ N$ | shots per circuit; number of cars | `shots`, `len(X_train)` |'

S3_MATH_ONE = '<div style="border:1px solid rgba(15,98,254,0.35);border-left:6px solid #0F62FE;background:rgba(15,98,254,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#0F62FE;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">∑ Math · رياضيات</span>&nbsp; **The line to remember**\n\n**The one line to remember.** $K(a,b)=\\bigl|\\langle\\phi(a)|\\phi(b)\\rangle\\bigr|^2$ is the fraction of runs that return $0000$ when you run the circuit for car $a$ and then the undo of the circuit for car $b$ (equation 9). Everything else in the classifier stays classical.\n\n</div>'

S3_TRY = '<div style="border:2px dashed #007D79;border-left:6px solid #007D79;background:rgba(0,125,121,0.1);border-radius:8px;padding:6px 16px 8px 16px;margin:10px 0">\n\n<span style="background:#007D79;color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;white-space:nowrap">🧪 Try it · جرّب</span>\n\nGo back to section 2, change `random_state=7` in the train/test split to another number (for example 3), and rerun the notebook. Which accuracies move, and by how much? Why is one test car worth 5 points, and what does that tell you about trusting a difference of 0.05?\n\n</div>'


def main():
    """Write the solution notebook."""
    nb = new(10, "EV fleet scheduling (V2G) with a quantum kernel",
             use_case="Decide which plugged-in electric cars should sell power back to the grid now (vehicle-to-grid)",
             algorithm="Quantum kernel classifier (FidelityStatevectorKernel + QSVC)", qubits=4, minutes=3)
    nb.section(1)
    nb.md(S1)
    nb.section(2)
    nb.md(S2_MD)
    nb.code(S2_DATA)
    nb.md(S2_CL_MD)
    nb.code(S2_MAJORITY)
    nb.md(TOOLBOX[10][0])
    nb.code(TOOLBOX[10][1])
    nb.md(T1_INTRO)
    nb.task("T1", "train the RBF-kernel SVM baseline and report train and test accuracy", T1, starter=T1_STARTER, selfcheck=T1_CHECK)
    nb.md(S2_NOTE)
    nb.section(3)
    nb.md(S3_ROADMAP)
    nb.code(S3_FLOW)
    nb.md(S3_SYMBOLS)
    nb.md(S3_MATH_ONE)
    nb.md(S3_KERNEL_THEORY_MD)
    nb.code(S3_PRIMER)
    nb.code(S3_RBF_CODE)
    nb.md(S3_SCALE_MD)
    nb.code(S3_SCALE)
    nb.md(S3_MAP_MD)
    nb.code(S3_FM)
    nb.code(S3_ZZ_CODE)
    nb.md(S3_KERNEL_THEORY2_MD)
    nb.code(S3_Z_CODE)
    nb.md(S3_Z_TRY)
    nb.md(S3_KERNEL_MD)
    nb.code(S3_KERNEL)
    nb.md(S3_ZOOM_MD)
    nb.code(S3_ZOOM)
    nb.md(S3_CONC_MD)
    nb.code(S3_CONC)
    nb.md(T2_INTRO)
    nb.task("T2", "compute one kernel entry by hand, check it against the table, train QSVC", T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.md(S3_VAR_MD)
    nb.code(S3_PERIOD)
    nb.md(T3_INTRO)
    nb.task("T3", "stretch: fill the variations table and explain the best and worst rows", T3, starter=T3_STARTER, selfcheck=T3_CHECK)
    nb.md(S3_SAMPLED_MD)
    nb.code(S3_SHOT_CHECK)
    nb.code(S3_SAMPLED)
    nb.md(S3_TRY)
    nb.md(INSTRUCTOR, tags=["solution-only"])
    nb.code("circuit = fm.assign_parameters(X_train_s[0])        # the main circuit of this notebook: the feature map for one car", tags=[])
    nb.md(day.IBM_MD)
    nb.code(day.IBM_PREVIEW_CODE)
    nb.md(S3_IBM_AFTER)
    nb.md(day.glossary_md(["quantum kernel", "feature map", "support vector machine", ("kernel matrix", "مصفوفة النواة", "I"), ("overlap", "التراكب (التداخل الجزئي) بين حالتين", "I"),
                           ("quantum machine learning", "تعلم الآلة الكمومي", "I"), ("classifier", "المصنِّف", "I"),
                           ("training data", "بيانات التدريب", "I"), ("overfitting", "الإفراط في الملاءمة", "I")]), tags=["glossary"])
    nb.section(4)
    nb.code(S4_TABLE)
    nb.code(S4_PLOT)
    nb.md(S4_SHOWS)
    nb.md(S4_NOTSHOW)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="Decide which plugged-in electric cars should sell power back to the grid now.",
                counting_fact="60 training cars need 1 770 kernel circuits; 6 000 cars would need about 18 million, because the count grows with the square of the data.",
                single_idea="Two situations are \"similar\" if the circuit for one undoes the circuit for the other.",
                ideas=["A \"should this car sell power now?\" quiz where the visitor competes with both classifiers",
                       "A similarity heat-map explorer: pick a car, see which others the circuit thinks are alike",
                       "A short film of one car's evening plugged in"],
                reuse="the kernel heat map (60 cars or the 12-car zoom), and the variations table")
    nb.section6(honest="our best quantum kernel tied the classical one at 80% on 20 cars; it did not beat it.",
                question_a="Why did changing the scaling drop accuracy to a coin flip?",
                question_b="Why is training accuracy 100% but test accuracy lower?")
    path = nb.save(os.path.join(ROOT, "solutions", "10_use_ev_fleet_kernel_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
