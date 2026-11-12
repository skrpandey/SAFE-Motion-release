import os

import pytest

from safe_motion import paths as P


def test_guard_accepts_build_tree(tmp_path):
    assert P.guard_output(str(tmp_path / "x.csv")) == str(tmp_path / "x.csv")
    assert P.guard_output(os.path.join(P.REPO, "build", "release_v16_clean", "results", "v7", "analysis", "x.csv"))


@pytest.mark.parametrize("rel", ["results/v7/analysis/x.csv", "results/fits/v7/results/v7/analysis/x.csv", "build/release_v15_clean/results/v7/analysis/x.csv",
                                 "build/release_v14_clean/paper/tables/x.tex", "build/repair_v14_orig/results/v11/x.csv"])
def test_guard_rejects_protected(rel):
    with pytest.raises(P.ProtectedPathError):
        P.guard_output(os.path.join(P.REPO, rel))


def test_guard_follows_symlinks(tmp_path):
    link = tmp_path / "results_link"
    os.symlink(os.path.join(P.REPO, "results"), link)
    with pytest.raises(P.ProtectedPathError):
        P.guard_output(str(link / "v7" / "analysis" / "x.csv"))
    with pytest.raises(P.ProtectedPathError):
        P.guarded_open(str(link / "v7" / "analysis" / "x.csv"), "w")
