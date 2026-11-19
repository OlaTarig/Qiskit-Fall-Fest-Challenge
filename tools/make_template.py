"""Build templates/template_solution.ipynb: the empty frame every notebook builder starts from.

    python tools/make_template.py

The template is a real, runnable solution notebook for a tiny dummy problem (2 qubits), so it also tests make_starters and
run_tests (python tools/run_tests.py template).  Builders copy it, keep every cell marked [KEEP], and replace the rest.
Sections 0, 5, 6, the glossary and the IBM cell come from qff.day; the section 1 and section 4 shapes are in the plan (2.4, 2.5).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nbkit import ROOT, new  # noqa: E402

S1 = """**The problem.** [what decision or quantity, as one concrete picture]

**Who faces it.** [sector classes only: "a national grid operator", "a desalination utility"; no Saudi organisation names]

**What is real today.** [the paper's case study with page number, its organisations as the paper names them, the size reached, the classical incumbent]

**Why a quantum computer might help.** [one mechanism, one sentence on what is still missing]

| Paper's method | Our teaching route |
|---|---|
| [paper's method] | [our method and why we chose it] |"""

S2_INSTANCE = """QUBITS = 2                       # [KEEP] the test runner reads this
from qiskit import QuantumCircuit
# Dummy problem for the template: two switches, pick the cheapest setting. Replace with your toy instance.
def cost(x):
    return 3 - 2 * x[0] - x[1] + 2 * x[0] * x[1]"""

S2_CLASSICAL = """# Classical baseline: always runs first.
from qff.bits import index_to_x
table = sorted(((cost(index_to_x(k, QUBITS)), index_to_x(k, QUBITS)) for k in range(2 ** QUBITS)))
classical_value, classical_x = table[0]
print("best x:", classical_x, "cost:", classical_value, "| states checked:", 2 ** QUBITS)"""

T1 = """cost_table = {tuple(index_to_x(k, QUBITS)): cost(index_to_x(k, QUBITS)) for k in range(2 ** QUBITS)}"""
T1_CHECK = """assert len(cost_table) == 4, "T1: the table needs one entry per state (4)"
assert min(cost_table.values()) == classical_value, "T1: the smallest cost should match the classical baseline" """

T2 = """circuit = QuantumCircuit(2)
circuit.ry(2 * np.arcsin(np.sqrt(0.9)), 0)     # puts 90% of the probability on x_0 = 1
circuit.measure_all()
counts = qff.quantum.sv_sampler(4096).run([circuit]).result()[0].data.meas.get_counts()"""
T2_STARTER = """circuit = QuantumCircuit(2)      # build your circuit here, then measure_all()
# run it with qff.quantum.sv_sampler() and keep the counts
raise NotImplementedError("T2: build the circuit and get counts")"""
T2_CHECK = """assert sum(counts.values()) == 4096, "T2: counts should add up to the number of shots (4096)" """

T3 = """more = qff.quantum.sv_sampler(4096).run([circuit.copy()]).result()[0].data.meas.get_counts()
print("T3: what changed when the probability knob moved? [explain in one sentence]")"""
T3_CHECK = """assert isinstance(more, dict), "T3: run the circuit again with your change" """

S4_RESULTS = """# Results table: classical baseline row first.
probs = {qff.bits.from_x(qff.bits.to_x(k)): v / 4096 for k, v in counts.items()}
print(f"{'method':<22}{'answer':<12}{'value':<8}{'states / evaluations'}")
print(f"{'brute force':<22}{str(classical_x):<12}{classical_value:<8}{2 ** QUBITS}")
best = max(counts, key=counts.get)
print(f"{'quantum circuit':<22}{str(qff.bits.to_x(best)):<12}{cost(qff.bits.to_x(best)):<8}{4096} shots")
qff.plots.plot_counts({qff.bits.from_x(qff.bits.to_x(k)): v / 4096 for k, v in counts.items()}, title="one plot");"""

S4_TEXT = """**What this toy shows.** [from the plan's block for this notebook]

**What this toy does not show.** [from the plan's block for this notebook]"""

S4_PAPER = """**Paper versus our route.** [from the plan's block for this notebook]"""

ACC = """assert QUBITS == 2
assert classical_x == [1, 0] and classical_value == 1 + 1 - 1, "classical baseline changed"   # [replace with the plan's asserts]
assert max(counts, key=counts.get) in ("01", "11"), "the circuit should favour x_0 = 1" """


def main():
    """Write templates/template_solution.ipynb."""
    nb = new(5, "[English title]", use_case="[one line]", algorithm="[algorithm name]", qubits=2, minutes=1)
    nb.section(1)
    nb.md(S1)
    nb.section(2)
    nb.code(S2_INSTANCE)
    nb.code(S2_CLASSICAL)
    nb.section(3)
    nb.md("[Concept first, in plain words, then guided Qiskit cells. Three team tasks per notebook: T1 warm-up, T2 core step, T3 stretch.]")
    nb.task("T1", "[warm-up: fill the cost table]", T1, selfcheck=T1_CHECK)
    nb.task("T2", "[core step: build the circuit and get counts]", T2, starter=T2_STARTER, selfcheck=T2_CHECK)
    nb.task("T3", "[stretch: change a knob and explain what moves]", T3, selfcheck=T3_CHECK)
    nb.md("[solution-only] Instructor note: delete this cell or keep it; it never reaches the starter.", tags=["solution-only"])
    nb.end_of_section3(["Hamiltonian", "optimization"])
    nb.section(4)
    nb.code(S4_RESULTS)
    nb.md(S4_TEXT)
    nb.md(S4_PAPER)
    nb.acceptance(ACC)
    nb.section5(use_case="[use-case one-liner]", counting_fact="[the counting fact for this notebook]",
                single_idea="[the single idea for this notebook]",
                ideas=["[idea 1]", "[idea 2]", "[idea 3]"], reuse="[the 2-3 figures or numbers worth reusing]")
    nb.section6(honest="[notebook-specific one-liner]", question_a="[notebook-specific question A]",
                question_b="[notebook-specific question B]")
    path = nb.save(os.path.join(ROOT, "templates", "template_solution.ipynb"))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
