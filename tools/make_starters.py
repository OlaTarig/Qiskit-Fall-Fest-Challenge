"""Turn solution notebooks into starter notebooks.   solutions/NN_..._solution.ipynb  ->  notebooks/NN_....ipynb

    python tools/make_starters.py                  # all solutions -> notebooks/
    python tools/make_starters.py 3 5              # only notebooks 3 and 5
    python tools/make_starters.py --release DIR    # export the tree teams clone: no solutions/, templates/ or mentor tools

Rules (plan 1.2):
  * every  ### BEGIN SOLUTION ... ### END SOLUTION  block is replaced by the stub given in the `### STARTER: <code>` line(s)
    placed immediately above BEGIN; with no STARTER line the stub is  raise NotImplementedError("T2: fill this in")
    (the task id comes from the `# TODO (team) Tn` line in the same cell);
  * all outputs are cleared; cells tagged `solution-only` are dropped;
  * cell ids are made deterministic so the output is byte-identical from run to run.
"""
import argparse
import copy
import glob
import os
import re
import shutil
import sys

import nbformat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEGIN, END, STARTER = "### BEGIN SOLUTION", "### END SOLUTION", "### STARTER:"
TODO_RE = re.compile(r"#\s*TODO \(team\) (T\d)")


def task_id(lines):
    """First `# TODO (team) Tn` in the cell, or 'TODO'."""
    for ln in lines:
        m = TODO_RE.search(ln)
        if m:
            return m.group(1)
    return "TODO"


def strip_cell_source(src):
    """Replace solution blocks in one code cell's source by their starter stubs."""
    lines = src.split("\n")
    tid, out, i = task_id(lines), [], 0
    while i < len(lines):
        ln = lines[i]
        if ln.strip() == BEGIN:
            indent = ln[:len(ln) - len(ln.lstrip())]
            stub = []
            while out and out[-1].strip().startswith(STARTER):          # STARTER lines sit right above BEGIN
                text = out.pop().strip()[len(STARTER):].rstrip()      # keep the indentation written after the marker
                stub.insert(0, text[1:] if text.startswith(" ") else text)
            j = i + 1
            while j < len(lines) and lines[j].strip() != END:
                j += 1
            if j == len(lines):
                raise ValueError("### BEGIN SOLUTION without ### END SOLUTION")
            out.extend(indent + s for s in (stub or [f'raise NotImplementedError("{tid}: fill this in")']))
            i = j + 1
            continue
        if ln.strip() == END:
            raise ValueError("### END SOLUTION without BEGIN")
        out.append(ln)
        i += 1
    return "\n".join(out)


def make_starter(nb):
    """Return a new starter notebook made from solution notebook `nb` (the input is not modified)."""
    nb = copy.deepcopy(nb)
    cells = []
    for c in nb.cells:
        if "solution-only" in c.get("metadata", {}).get("tags", []):
            continue
        if c.cell_type == "code":
            c.source = strip_cell_source(c.source)
            c.outputs, c.execution_count = [], None
        cells.append(c)
    for k, c in enumerate(cells):
        c["id"] = f"c{k:03d}"
    nb.cells = cells
    nb.metadata.pop("widgets", None)
    return nb


def solution_files(numbers=None):
    """solutions/*_solution.ipynb, optionally only the given challenge numbers."""
    files = sorted(glob.glob(os.path.join(ROOT, "solutions", "[0-9][0-9]_*_solution.ipynb")))
    if numbers:
        files = [f for f in files if int(os.path.basename(f)[:2]) in numbers]
    return files


def starter_path(solution_path):
    """solutions/NN_x_solution.ipynb -> notebooks/NN_x.ipynb"""
    name = os.path.basename(solution_path).replace("_solution.ipynb", ".ipynb")
    return os.path.join(ROOT, "notebooks", name)


def write_nb(nb, path):
    """Write a notebook the same way every time."""
    nbformat.validate(nb)
    nbformat.write(nb, path)


def release(dest):
    """Copy the public tree (no solutions/, templates/, mentor tools, caches) into dest."""
    skip_dirs = {".git", "solutions", "templates", "__pycache__", ".ipynb_checkpoints", "_review"}      # _review: local HTML previews of solutions
    skip_files = {"tools/run_tests.py", "tools/make_starters.py", "tools/nbkit.py", "tools/make_template.py", "tools/toolbox.py", ".gitattributes"}
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in skip_dirs]
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), ROOT).replace(os.sep, "/")
            if rel in skip_files or f.endswith(".pyc") or rel.startswith("tools/build_nb"):
                continue
            os.makedirs(os.path.join(dest, os.path.dirname(rel)), exist_ok=True)
            shutil.copy2(os.path.join(ROOT, rel), os.path.join(dest, rel))
    print(f"released public tree to {dest}")


def main(argv=None):
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("numbers", nargs="*", type=int)
    ap.add_argument("--release", metavar="DIR")
    a = ap.parse_args(argv)
    for f in solution_files(a.numbers):
        out = starter_path(f)
        write_nb(make_starter(nbformat.read(f, as_version=4)), out)
        print("wrote", os.path.relpath(out, ROOT))
    if a.release:
        release(a.release)


if __name__ == "__main__":
    sys.exit(main())
