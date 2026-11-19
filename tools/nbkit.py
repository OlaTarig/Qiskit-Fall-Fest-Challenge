"""Small notebook-building kit for the ten solution notebooks (mentors and builders only).

Usage (see tools/make_template.py for a complete working example):

    import sys; sys.path.insert(0, "tools")
    from nbkit import new
    nb = new(5, "Unit commitment", use_case="...", algorithm="Grover adaptive search", qubits=11, minutes=3)
    nb.section(1)                                  # adds the exact heading "## 1 · This stop — هذه المحطة"
    nb.md("**The problem.** ...")
    nb.task("T1", "fill the cost table", solution="cost = ...", selfcheck="assert len(cost) == 16, 'expected 16 entries'")
    nb.end_of_section3(["Grover's algorithm", "oracle"])   # optional IBM cell + glossary
    nb.section(4); nb.acceptance("assert QUBITS == 11")
    nb.section5(...); nb.section6(...)
    nb.save("solutions/05_move_unit_commitment_grover_solution.ipynb")

Sections 0, 5 and 6 and the glossary and IBM cells come from qff.day, so they stay identical across notebooks.
Section 5 and 6 text is exactly the plan's template; only the bracketed parts change.
"""
import os
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from qff import day  # noqa: E402

STOP_OF = {c["n"]: c["stop"] for c in day.CHALLENGES}


# ---------------------------------------------------------------------------------------------------------------------
# House style for explanations: coloured callout boxes, hint ladders, Arabic term tables.
# Everything is plain HTML with inline styles inside Markdown cells, so it renders in Colab (light and dark), JupyterLab
# and VS Code.  Backgrounds are translucent and text colour is inherited, so the same cell reads well on a dark theme.
# ---------------------------------------------------------------------------------------------------------------------
STOP_HEX = {"source": "#007D79", "move": "#0F62FE", "store": "#8A3FFC", "use": "#D02670"}      # teal, blue, purple, magenta
CALLOUTS = {                       # kind: (stop colour, icon, English label, Arabic label, dashed border?)
    "idea": ("source", "💡", "Idea", "فكرة", False),
    "math": ("move", "∑", "Math", "رياضيات", False),
    "watch": ("store", "⚠️", "Watch out", "انتبه", False),
    "try": ("source", "🧪", "Try it", "جرّب", True),
    "real": ("use", "🌍", "Real world", "في الواقع", False),
}


def _rgba(hex_colour, alpha):
    """'#RRGGBB' -> 'rgba(r,g,b,alpha)'."""
    h = hex_colour.lstrip("#")
    return "rgba({},{},{},{})".format(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def callout(kind, body, title=""):
    """A coloured box: kind is one of idea (teal), math (blue), watch (purple), try (teal, dashed), real (magenta).

    The body is Markdown (LaTeX allowed).  The blank lines around it are what make Markdown render inside the HTML box.
    """
    stop, icon, en, ar, dashed = CALLOUTS[kind]
    c = STOP_HEX[stop]
    border = f"2px dashed {c}" if dashed else f"1px solid {_rgba(c, 0.35)}"
    head = (f'<span style="background:{c};color:#fff;border-radius:12px;padding:2px 10px;font-size:0.85em;font-weight:600;'
            f'white-space:nowrap">{icon} {en} · {ar}</span>')
    if title:
        head += f"&nbsp; **{title}**"
    return (f'<div style="border:{border};border-left:6px solid {c};background:{_rgba(c, 0.10)};border-radius:8px;'
            f'padding:6px 16px 8px 16px;margin:10px 0">\n\n{head}\n\n{body.strip()}\n\n</div>')


HINT_LABELS = [("Hint 1 · a nudge", "تلميح ١"), ("Hint 2 · the structure", "تلميح ٢"), ("Hint 3 · nearly there", "تلميح ٣")]


def hints(*texts):
    """A hint ladder: one collapsed <details> block per text (nudge, structure, near-complete). Never put the answer in Hint 3."""
    out = ["**Stuck? Open the hints one at a time, and try again after each. · تلميحات**"]
    for (en, ar), text in zip(HINT_LABELS, texts):
        out.append('<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">\n'
                   f'<summary style="cursor:pointer;font-weight:600">{en} · {ar}</summary>\n\n{text.strip()}\n\n</details>')
    return "\n\n".join(out)


def details(summary, body):
    """A collapsed optional block (for the algebra a curious reader may want)."""
    return ('<details style="border:1px solid rgba(128,128,128,0.45);border-radius:8px;padding:4px 12px;margin:6px 0">\n'
            f'<summary style="cursor:pointer;font-weight:600">{summary}</summary>\n\n{body.strip()}\n\n</details>')


def arabic_table(rows, caption="Arabic terms for this step."):
    """Small English / Arabic / tier table, two terms per row. rows = [(english, arabic, tier)].

    Tier G = a headword of Quantabic's glossary; tier I = proposed here, still to be checked against quantabic.com.
    """
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:] + [("", "", "")] * (half - len(rows[half:]))
    body = "\n".join(f"| {a} | {b} | {c} | {d} | {e} | {f} |" for (a, b, c), (d, e, f) in zip(left, right))
    return (f"<sub>{caption} Tier G: in Quantabic's glossary. Tier I: proposed, to be checked against quantabic.com.</sub>\n\n"
            f"| English | العربية | Tier | English | العربية | Tier |\n|---|---|---|---|---|---|\n{body}")


class NB:
    """A notebook under construction."""

    def __init__(self, n):
        """Start an empty notebook for challenge number n."""
        self.n, self.stop = n, STOP_OF[n]
        self.nb = new_notebook()
        self.nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                            "language_info": {"name": "python"}}

    def md(self, text, tags=()):
        """Add a markdown cell."""
        c = new_markdown_cell(text.strip("\n"))
        if tags:
            c.metadata["tags"] = list(tags)
        self.nb.cells.append(c)

    def code(self, text, tags=()):
        """Add a code cell (text is dedented by the caller)."""
        c = new_code_cell(text.strip("\n"))
        if tags:
            c.metadata["tags"] = list(tags)
        self.nb.cells.append(c)

    def section(self, k):
        """Add the exact heading of section k (0-6)."""
        self.md(day.HEADINGS[k])

    def task(self, tid, title, solution, starter=None, selfcheck=None):
        """A team task cell plus its self-check cell.

        The cell shows `# TODO (team) Tn: title`; `solution` goes between BEGIN/END markers; `starter` (optional, may be
        multi-line) becomes the STARTER stub, else the starter raises NotImplementedError("Tn: fill this in").
        """
        lines = [f"# TODO (team) {tid}: {title}"]
        for s in (starter or "").splitlines():
            lines.append(f"### STARTER: {s}")
        lines += ["### BEGIN SOLUTION", solution.strip("\n"), "### END SOLUTION"]
        self.code("\n".join(lines))
        if selfcheck:
            self.code("# Self-check\n" + selfcheck.strip())

    def end_of_section3(self, glossary_extra=()):
        """The optional IBM cell (markdown + executed preview) and the glossary: the last cells of section 3."""
        self.md(day.IBM_MD)
        self.code(day.IBM_PREVIEW_CODE)
        self.md(day.glossary_md(glossary_extra), tags=["glossary"])

    def acceptance(self, asserts):
        """The acceptance cell: last code cell of section 4, tagged `acceptance`, present in starter and solution."""
        self.code("# Acceptance: the numbers this notebook promises\n" + asserts.strip(), tags=["acceptance"])

    def section5(self, use_case, counting_fact, single_idea, ideas, reuse):
        """Section 5 from the shared template."""
        self.md(day.section5_md(use_case, counting_fact, single_idea, ideas, reuse))

    def section6(self, honest, question_a, question_b):
        """Section 6 from the shared template."""
        self.md(day.section6_md(honest, question_a, question_b))

    def save(self, path):
        """Validate and write the notebook (cell ids are normalised)."""
        for k, c in enumerate(self.nb.cells):
            c["id"] = f"s{k:03d}"
        nbformat.validate(self.nb)
        nbformat.write(self.nb, path)
        return path


def new(n, title, use_case, algorithm, qubits, minutes):
    """Start a notebook: header, setup cell, section 0 (text + stop map). Returns the NB to add sections 1-6 to."""
    stop = day.STOPS[STOP_OF[n] - 1]
    nb = NB(n)
    nb.md(day.header_md(n, title, stop["ar"], stop["en"], use_case, algorithm, qubits, minutes))
    nb.code(day.SETUP_CODE)
    nb.md(day.SECTION0_MD)
    nb.code(day.section0_code(STOP_OF[n], n))
    return nb
