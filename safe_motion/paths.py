"""Output-path guard for analyses that run inside a fresh release tree.

Archived results (results/, including the fitted models under results/fits, and the frozen v14/v15 build trees) are never
written by analysis code. guard_output(path) and guarded_open() are explicit checks that resolve
symlinks; install_guard() wraps builtins.open and io.open so every write-mode open in the process
is checked (used by experiments/guarded_run.py). SM_PROTECTED_EXTRA (colon-separated paths) adds
protected roots, e.g. for test fixtures.
"""
import builtins
import io
import os

REPO = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))   # resolved, so a tree's symlink maps back to the repo
PROTECTED = tuple(os.path.realpath(os.path.join(REPO, *p)) for p in (
    ("results",),
    ("build", "release_v14_orig"), ("build", "release_v14_clean"),
    ("build", "repair_v14_orig"), ("build", "repair_v14_clean"),
    ("build", "release_v15_orig"), ("build", "release_v15_clean"),
    ("build", "paper_pre_repair_2026-11-07"),
)) + tuple(os.path.realpath(p) for p in os.environ.get("SM_PROTECTED_EXTRA", "").split(":") if p)


class ProtectedPathError(RuntimeError):
    pass


def guard_output(path):
    """Return `path` if it may be written; raise ProtectedPathError otherwise."""
    path = os.fspath(path)
    target = os.path.realpath(path)
    parent = os.path.realpath(os.path.dirname(path) or ".")
    for p in PROTECTED:
        for cand in (target, parent):
            if cand == p or cand.startswith(p + os.sep):
                raise ProtectedPathError(f"refusing to write {path}: resolves into protected tree {p}")
    return path


def guarded_open(path, mode="w", **kw):
    if any(c in mode for c in "wax+"):
        guard_output(path)
    return open(path, mode, **kw)


_ORIG_OPEN = io.open
_INSTALLED = False


def _guarded_builtin_open(file, mode="r", *args, **kw):
    if isinstance(file, (str, bytes, os.PathLike)) and any(c in str(mode) for c in "wax+"):
        guard_output(os.fsdecode(file) if isinstance(file, bytes) else file)
    return _ORIG_OPEN(file, mode, *args, **kw)


def install_guard():
    """Enforce the guard on every write-mode open in this process (idempotent)."""
    global _INSTALLED
    if not _INSTALLED:
        builtins.open = _guarded_builtin_open
        io.open = _guarded_builtin_open
        _INSTALLED = True


def uninstall_guard():
    global _INSTALLED
    builtins.open = _ORIG_OPEN
    io.open = _ORIG_OPEN
    _INSTALLED = False
