"""Output guard as the release driver uses it, on scratch fixtures only.

A scratch folder is declared protected through SM_PROTECTED_EXTRA and a fake tree links its
results/ into it; writes through the installed guard must be refused before any file changes."""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER = os.path.join(REPO, "experiments", "guarded_run.py")


def _env(protected):
    env = dict(os.environ)
    env["SM_PROTECTED_EXTRA"] = str(protected)
    return env


def test_check_mode_refuses_tree_linking_into_protected_fixture(tmp_path):
    fixture = tmp_path / "historical"; fixture.mkdir()
    (fixture / "v7").mkdir()
    tree = tmp_path / "tree"; tree.mkdir()
    os.symlink(fixture, tree / "results")
    (tree / "paper").mkdir()
    r = subprocess.run([sys.executable, RUNNER, "--check", str(tree / "results" / "v7"), str(tree / "paper")],
                       capture_output=True, text=True, env=_env(fixture))
    assert r.returncode == 1 and "REFUSED" in r.stdout
    assert not any(fixture.rglob("*__probe__*"))


def test_check_mode_accepts_fresh_tree(tmp_path):
    tree = tmp_path / "tree"; (tree / "results" / "v7").mkdir(parents=True)
    r = subprocess.run([sys.executable, RUNNER, "--check", str(tree / "results" / "v7")],
                       capture_output=True, text=True, env=_env(tmp_path / "elsewhere"))
    assert r.returncode == 0


def test_installed_guard_blocks_every_writer(tmp_path):
    fixture = tmp_path / "historical"; fixture.mkdir()
    tree = tmp_path / "tree"; tree.mkdir()
    os.symlink(fixture, tree / "results")
    script = tmp_path / "writer.py"
    script.write_text(
        "import csv, json, sys, numpy as np, shutil\n"
        "p = sys.argv[1]\n"
        "ok = sys.argv[2]\n"
        "open(ok + '/fine.txt', 'w').write('x')\n"
        "try:\n"
        "    json.dump({}, open(p + '/a.json', 'w')); print('JSON WROTE')\n"
        "except Exception as e: print('json refused', type(e).__name__)\n"
        "try:\n"
        "    np.savez(p + '/a.npz', a=np.zeros(2)); print('NPZ WROTE')\n"
        "except Exception as e: print('npz refused', type(e).__name__)\n"
        "try:\n"
        "    shutil.copy(ok + '/fine.txt', p + '/c.txt'); print('COPY WROTE')\n"
        "except Exception as e: print('copy refused', type(e).__name__)\n"
        "try:\n"
        "    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt\n"
        "    plt.figure(); plt.savefig(p + '/f.pdf'); print('FIG WROTE')\n"
        "except Exception as e: print('fig refused', type(e).__name__)\n")
    ok = tmp_path / "ok"; ok.mkdir()
    r = subprocess.run([sys.executable, RUNNER, str(script), str(tree / "results"), str(ok)],
                       capture_output=True, text=True, env=_env(fixture))
    assert "WROTE" not in r.stdout, r.stdout + r.stderr
    assert r.stdout.count("refused") == 4, r.stdout + r.stderr
    assert os.listdir(fixture) == []
    assert (ok / "fine.txt").read_text() == "x"


def test_guard_reads_are_unaffected(tmp_path):
    fixture = tmp_path / "historical"; fixture.mkdir()
    (fixture / "data.txt").write_text("hello")
    script = tmp_path / "reader.py"
    script.write_text("import sys; print(open(sys.argv[1]).read())")
    r = subprocess.run([sys.executable, RUNNER, str(script), str(fixture / "data.txt")],
                       capture_output=True, text=True, env=_env(fixture))
    assert r.returncode == 0 and "hello" in r.stdout


def test_real_protected_roots_present():
    from safe_motion import paths as P
    names = {os.path.basename(p) for p in P.PROTECTED}
    assert {"results", "release_v14_orig", "release_v14_clean", "release_v15_orig", "release_v15_clean"} <= names
