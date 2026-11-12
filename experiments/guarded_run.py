"""Run an analysis script with the output guard enforced process-wide.

    python3 experiments/guarded_run.py experiments/analyze_v7.py [args...]
        installs safe_motion.paths.install_guard() and runs the script as __main__; a write-mode
        open of a path inside a protected tree (historical results/, the frozen v14 and v15
        trees, SM_PROTECTED_EXTRA) raises ProtectedPathError.
    python3 experiments/guarded_run.py --check PATH [PATH...]
        exits 1 if writing a file inside any PATH would be refused.
"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from safe_motion import paths as P                     # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--check":
        bad = []
        for d in sys.argv[2:]:
            try:
                P.guard_output(os.path.join(d, "__probe__"))
            except P.ProtectedPathError as e:
                bad.append(str(e))
        for b in bad:
            print("REFUSED", b)
        print(f"guard check: {len(sys.argv) - 2} paths, {len(bad)} refused")
        sys.exit(1 if bad else 0)
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
    P.install_guard()
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
