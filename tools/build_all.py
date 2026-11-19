"""Run every notebook build script, then the template and starter steps, with one command.

    python tools/build_all.py            # all of tools/build_nb*.py, then make_template.py, then make_starters.py
    python tools/build_all.py 1 6 8      # only builds 1, 6 and 8 first (the two final steps still run)
    python tools/build_all.py --force    # run the final steps even if a build failed

Each script runs in its own process, so one failure does not stop the other builds. If any build fails, the final steps are
skipped (they would work from stale solution notebooks) unless --force is given. The exit code is 1 if anything failed.
"""
import glob
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
FINAL_STEPS = ["make_template.py", "make_starters.py"]
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")


def run(script):
    """Run one script from the project root. Returns True if it succeeded."""
    name = os.path.basename(script)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, script], cwd=ROOT, env=ENV, capture_output=True, text=True, encoding="utf-8")
    secs = time.perf_counter() - t0
    if r.returncode == 0:
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        print(f"ok      {name:<22}{secs:5.1f}s  {last}")
        return True
    print(f"FAILED  {name:<22}{secs:5.1f}s")
    print("\n".join("        " + line for line in (r.stderr or r.stdout).strip().splitlines()[-8:]))
    return False


def main():
    args = sys.argv[1:]
    force = "--force" in args
    wanted = {int(a) for a in args if a.isdigit()}
    scripts = sorted(glob.glob(os.path.join(HERE, "build_nb*.py")))
    if wanted:
        scripts = [s for s in scripts if int(re.search(r"build_nb(\d+)", os.path.basename(s)).group(1)) in wanted]
    if not scripts:
        print("no build_nb*.py scripts found in", HERE)
        return 1
    failed = [os.path.basename(s) for s in scripts if not run(s)]
    print(f"\n{len(scripts) - len(failed)} of {len(scripts)} notebooks built" + (f"; failed: {', '.join(failed)}" if failed else ""))

    if failed and not force:
        print("\nSkipping", " and ".join(FINAL_STEPS), "because a build failed (use --force to run them anyway).")
        return 1
    print()
    for step in FINAL_STEPS:
        path = os.path.join(HERE, step)
        if not os.path.exists(path):
            print(f"skipped {step} (not found in tools/)")
            failed.append(step)
        elif not run(path):
            failed.append(step)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())