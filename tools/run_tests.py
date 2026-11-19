"""CI-free test runner.

    python tools/run_tests.py            # selftest + data + template + every solution notebook that exists
    python tools/run_tests.py 3 5        # only notebooks 3 and 5
    python tools/run_tests.py data       # only the data-file checks (also: selftest, template)

For each notebook (plan 1.7):
  1. regenerate the starter with make_starters and fail if notebooks/ differs from what is on disk;
  2. execute the solution (fail on any exception or more than 180 s);
  3. execute the starter and check it stops with NotImplementedError whose message starts with T1;
  4. structure: seven H2 headings in order, exactly three `# TODO (team) T` markers (T1, T2, T3), a glossary cell
     (tag `glossary`), an `acceptance` cell, no credentials;
  5. print a table: notebook, result, seconds, QUBITS.
Exit code 0 only if everything passed.
"""
import copy
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
import make_starters  # noqa: E402
from qff import day  # noqa: E402

TIME_LIMIT = 180
SECRET_RES = [re.compile(r"token\s*="), re.compile(r"QiskitRuntimeService\.save_account"),
              re.compile(r"(?i)api[_-]?key\s*[=:]\s*['\"][^'\"]{8,}"),
              re.compile(r"(?=[A-Za-z0-9_-]*\d)(?=[A-Za-z0-9_-]*[A-Za-z])[A-Za-z0-9_-]{40,}")]
TODO_RE = re.compile(r"^\s*#\s*TODO \(team\) (T\d)\b", re.M)


def check_structure(nb):
    """Return a list of problems with the notebook's structure (empty list = fine)."""
    bad, heads, todos = [], [], []
    for c in nb.cells:
        first = c.source.split("\n", 1)[0]
        if c.cell_type == "markdown" and re.match(r"^## [0-6] ·", first):
            heads.append(first)
        if c.cell_type == "code":
            todos += TODO_RE.findall(c.source)
        for rx in SECRET_RES:
            if rx.search(c.source):
                bad.append(f"possible credential in a cell (pattern {rx.pattern[:30]})")
    if heads != day.HEADINGS:
        bad.append(f"section headings wrong or out of order: {heads}")
    if sorted(todos) != ["T1", "T2", "T3"]:
        bad.append(f"expected exactly one TODO marker each for T1, T2, T3; found {todos}")
    tags = [t for c in nb.cells for t in c.get("metadata", {}).get("tags", [])]
    for need in ("glossary", "acceptance"):
        if need not in tags:
            bad.append(f"no cell tagged '{need}'")
    return bad


def run_notebook(nb, workdir):
    """Execute a copy of nb in workdir with a final cell that prints QUBITS. Returns (seconds, qubits, error or None)."""
    nb = copy.deepcopy(nb)
    nb.cells.append(nbformat.v4.new_code_cell("print('QFF_QUBITS', QUBITS)"))
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": workdir}})
    t0 = time.time()
    try:
        client.execute()
    except CellExecutionError as e:
        return time.time() - t0, None, e
    out = "".join(o.get("text", "") for o in nb.cells[-1].outputs)
    m = re.search(r"QFF_QUBITS (\d+)", out)
    return time.time() - t0, int(m.group(1)) if m else None, None


def test_notebook(sol_path, workdir, starter_file=None):
    """Run all checks for one solution notebook. Returns (ok, seconds, qubits, [messages])."""
    msgs = []
    sol = nbformat.read(sol_path, as_version=4)
    starter = make_starters.make_starter(sol)
    if starter_file is not None:
        want = nbformat.writes(starter)
        want += "" if want.endswith("\n") else "\n"          # nbformat.write adds a final newline
        have = open(starter_file).read() if os.path.exists(starter_file) else None
        if have != want:
            msgs.append(f"{os.path.relpath(starter_file, ROOT)} is missing or differs from the generated starter; run tools/make_starters.py")
    msgs += [f"solution: {m}" for m in check_structure(sol)]
    msgs += [f"starter: {m}" for m in check_structure(starter)]
    if any("BEGIN SOLUTION" in c.source for c in starter.cells):
        msgs.append("starter still contains a solution block")
    secs, qubits, err = run_notebook(sol, workdir)
    if err is not None:
        msgs.append(f"solution raised {err.ename}: {err.evalue[:200]}")
    elif secs > TIME_LIMIT:
        msgs.append(f"solution took {secs:.0f} s (limit {TIME_LIMIT} s)")
    _, _, serr = run_notebook(starter, workdir)
    if serr is None:
        msgs.append("starter ran to the end; it must stop at T1 with NotImplementedError")
    elif serr.ename != "NotImplementedError" or not serr.evalue.startswith("T1"):
        msgs.append(f"starter stopped with {serr.ename}: {serr.evalue[:120]} (expected NotImplementedError starting 'T1')")
    return not msgs, secs, qubits, msgs


def test_selftest():
    """python -m qff.selftest in a fresh interpreter."""
    r = subprocess.run([sys.executable, "-m", "qff.selftest"], cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, [r.stdout[-300:] + r.stderr[-800:]] if r.returncode else []


def regen_matches(script, args, files):
    """Run a generator into a temp dir/file and compare with the shipped files byte for byte."""
    msgs = []
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, os.path.join(HERE, script), "--out", tmp if len(files) > 1 else os.path.join(tmp, "x.csv")],
                       check=True, capture_output=True)
        for f in files:
            new = os.path.join(tmp, os.path.basename(f)) if len(files) > 1 else os.path.join(tmp, "x.csv")
            if open(new).read() != open(f).read():
                msgs.append(f"{os.path.basename(f)} is not reproduced by {script}")
    return msgs


def test_data():
    """Load every JSON with qff.chem, re-check energies, and check the generators reproduce the files exactly."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from qff.chem import load_hamiltonian
    msgs = []
    jsons = sorted(glob.glob(os.path.join(ROOT, "data", "*.json")))
    for f in jsons:
        name = os.path.basename(f)
        h = load_hamiltonian(f)
        qc = QuantumCircuit(h.num_qubits)
        qc.x(h.hf_x_qubits)
        e0 = Statevector(qc).expectation_value(h.op).real + h.shift
        w = np.linalg.eigvalsh(h.op.to_matrix())[0] + h.shift
        if abs(e0 - h.e_hf) > 1e-8:
            msgs.append(f"{name}: e_hf does not match the start state")
        if name.startswith(("lih", "h2")):
            if abs(w - h.e_exact) > 1e-8 or not (h.e_fci_full - 1e-8 <= h.e_exact <= h.e_hf + 1e-8):
                msgs.append(f"{name}: exact/FCI/HF energies inconsistent")
        elif abs(h.e_exact - h.meta["e_exact_sector"]) > 1e-12 or h.meta["e_min_full_matrix"] > h.e_exact + 1e-12:
            msgs.append(f"{name}: site-model energies inconsistent")
    want = {"lih": 7, "h2": 6, "site": 9}
    got = {"lih": sum("lih_" in f for f in jsons), "h2": sum("h2_" in f for f in jsons), "site": sum("site_model" in f for f in jsons)}
    if got != want:
        msgs.append(f"unexpected number of Hamiltonian files {got}, expected {want}")
    rows = open(os.path.join(ROOT, "data", "v2g_fleet.csv")).read().strip().split("\n")
    if len(rows) != 81 or rows[0] != "soc,hours_norm,price,load,discharge":
        msgs.append("v2g_fleet.csv should have a header and 80 rows")
    msgs += regen_matches("generate_chemistry.py", [], [f for f in jsons if "lih_" in f or "h2_" in f])
    msgs += regen_matches("generate_site_model.py", [], [f for f in jsons if "site_model" in f])
    msgs += regen_matches("generate_v2g_data.py", [], [os.path.join(ROOT, "data", "v2g_fleet.csv")])
    return not msgs, msgs


def main(argv):
    """Select what to run from the command line, print the table, return the exit code."""
    words = [a.lower() for a in argv]
    nums = [int(a) for a in words if a.isdigit()]
    names = [a for a in words if not a.isdigit()]
    run_all = not argv
    rows, ok_all = [], True

    def record(label, ok, secs=None, qubits=None, msgs=()):
        nonlocal ok_all
        ok_all &= ok
        rows.append((label, "PASS" if ok else "FAIL", "" if secs is None else f"{secs:.1f}", "" if qubits is None else qubits))
        for m in msgs:
            print(f"  [{label}] {m}")

    for key, fn in (("selftest", test_selftest), ("data", test_data)):
        if run_all or key in names:
            t0 = time.time()
            ok, msgs = fn()
            record(key, ok, time.time() - t0, None, msgs)
    if run_all or "template" in names:
        ok, secs, q, msgs = test_notebook(os.path.join(ROOT, "templates", "template_solution.ipynb"),
                                          os.path.join(ROOT, "templates"))
        record("template", ok, secs, q, msgs)
    sols = make_starters.solution_files(nums or None)
    for n in nums:
        if not any(int(os.path.basename(s)[:2]) == n for s in sols):
            record(f"notebook {n:02d}", False, None, None, ["no solution notebook found in solutions/"])
    if run_all and len(sols) < 10 and not nums:
        print(f"note: {len(sols)} of 10 solution notebooks exist so far")
    for s in sols if (run_all or nums) else []:
        ok, secs, q, msgs = test_notebook(s, os.path.join(ROOT, "solutions"), make_starters.starter_path(s))
        record(os.path.basename(s).replace("_solution.ipynb", ""), ok, secs, q, msgs)
    print(f"\n{'check':<44}{'result':<8}{'seconds':<9}qubits")
    for r in rows:
        print(f"{r[0]:<44}{r[1]:<8}{r[2]:<9}{r[3]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
