"""Shadow working tree per selection rule for repair_selection_v14.

build/repair_v14_<rule>/ gets symlinks to the code, a copy of paper/, and a results/ folder in
which every input that depends on an ASDPose model is replaced by the repaired output of that
rule (scores, features and registry of v7, v10, v12, v13 and v15conf; repaired checkpoints for
score_v11.py). Model-independent inputs are copied or linked from the historical results, so the
unchanged analysis, table and figure scripts regenerate every retained result.
    python experiments/build_repair_view.py --rule clean --repaired results/fits/repair_v14
"""
import argparse, glob, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--rule", choices=["orig", "clean"], required=True)
ap.add_argument("--repaired", default="results/fits/repair_v14")
ap.add_argument("--out", default=None, help="tree path (default build/repair_v14_<rule>)")
ap.add_argument("--rescore-all-ours", action="store_true",
                help="drop every real-split model of ours from results/v11 so all 30 are rescored")
ap.add_argument("--v15conf", default="results/fits/v15conf/results",
                help="protocol v15conf result set; its <rule>/ scores and registry are copied to results/v15conf (2026-11-09)")
ap.add_argument("--v16", nargs="+", default=["results/fits/v16a/results", "results/fits/v16bc/results"],
                help="protocol v16 result sets (ST-GCN; CompactCTRGCN); their <rule>/ scores and registries are merged into results/v16 (2026-11-11)")
ap.add_argument("--v17", nargs="*", default=["results/fits/v17/results"],
                help="protocol v17 result sets, merged into results/v17 when present (2026-11-11)")
a = ap.parse_args()
REPO = os.getcwd()
B = a.out or f"build/repair_v14_{a.rule}"
R = os.path.abspath(a.repaired)
if os.path.exists(B):
    shutil.rmtree(B)
os.makedirs(f"{B}/results/fits/v7", exist_ok=True); os.makedirs(f"{B}/results/fits/v10", exist_ok=True)
for d in ("experiments", "safe_motion", "configs", "tests", "cache"):   # cache: published checkpoints (read only)
    os.symlink(f"{REPO}/{d}", f"{B}/{d}")
shutil.copytree("paper", f"{B}/paper", ignore=shutil.ignore_patterns("*.aux", "*.log", "*.fls", "*.fdb_latexmk", "*.pdf", "*.bbl", "*.blg"))
for f in glob.glob("paper/figures/*.pdf"):          # figures not regenerated stay as they are
    shutil.copy(f, f"{B}/paper/figures/")
REPLACED = {"v7", "v9", "v10", "v11", "v12", "v13", "v15conf", "v16", "v17"}
for e in os.listdir("results"):
    if e not in REPLACED and e != "fits":
        os.symlink(os.path.realpath(f"results/{e}"), f"{B}/results/{e}")
for e in os.listdir("results/fits"):
    if e not in ("v7", "v10"):
        os.symlink(os.path.realpath(f"results/fits/{e}"), f"{B}/results/fits/{e}")


def from_rerun(proto, extra_copy=(), extra_link=()):
    dst = f"{B}/results/{proto}"; src_h = os.path.realpath(f"results/{proto}"); src_r = f"{R}/results/{a.rule}/{proto}"
    os.makedirs(f"{dst}/analysis", exist_ok=True)
    for sub in ("scores", "features"):
        if os.path.isdir(f"{src_r}/{sub}"):
            shutil.copytree(f"{src_r}/{sub}", f"{dst}/{sub}")
    if os.path.exists(f"{src_r}/registry.json"):
        shutil.copy(f"{src_r}/registry.json", f"{dst}/registry.json")
    for f in extra_copy:
        p = f"{src_h}/{f}"
        if os.path.isdir(p):
            shutil.copytree(p, f"{dst}/{f}", dirs_exist_ok=True)
        elif os.path.exists(p):
            shutil.copy(p, f"{dst}/{f}")
    for f in extra_link:
        if os.path.exists(f"{src_h}/{f}"):
            os.symlink(f"{src_h}/{f}", f"{dst}/{f}")


from_rerun("v7", extra_copy=("oracles.json",))
from_rerun("v10", extra_link=("masks",))
from_rerun("v12")
from_rerun("v13")
# v15conf was trained with both selection rules from the start; use the rule's own score set
src15 = f"{os.path.abspath(a.v15conf)}/{a.rule}"
os.makedirs(f"{B}/results/v15conf/analysis", exist_ok=True)
shutil.copytree(f"{src15}/scores", f"{B}/results/v15conf/scores")
shutil.copy(f"{src15}/registry.json", f"{B}/results/v15conf/registry.json")
# v16 and v17 were trained with both selection rules from the start as well; one or more result
# sets per protocol (v16: one instance per architecture) are merged into results/<proto>
import json as _json
for proto, sets in (("v16", a.v16), ("v17", a.v17)):
    sets = [s for s in sets if os.path.isdir(f"{os.path.abspath(s)}/{a.rule}/scores")]
    if not sets:
        print(f"{proto}: no result set with {a.rule}/scores; results/{proto} not built")
        continue
    os.makedirs(f"{B}/results/{proto}/analysis", exist_ok=True); os.makedirs(f"{B}/results/{proto}/scores", exist_ok=True)
    reg = []
    for s in sets:
        src = f"{os.path.abspath(s)}/{a.rule}"
        for f in sorted(glob.glob(f"{src}/scores/*.csv")):
            shutil.copy(f, f"{B}/results/{proto}/scores/")
        reg += _json.load(open(f"{src}/registry.json"))
    assert len({m["name"] for m in reg}) == len(reg), f"{proto}: duplicate model names across result sets"
    _json.dump(reg, open(f"{B}/results/{proto}/registry.json", "w"), indent=1)
    if proto == "v16":                       # permutation-reliance scores of the v16 models (clean selection only)
        cr = os.path.realpath("results/v16/conf_reliance")
        if os.path.isdir(cr) and a.rule == "clean":
            os.symlink(cr, f"{B}/results/v16/conf_reliance")
# inputs that do not depend on any ASDPose fit (probe of interpolated coordinates; v9 hosts)
for f in ("interp_probe.csv", "interp_probe.json"):
    p = os.path.realpath("results/v10") + f"/analysis/{f}"
    if os.path.exists(p):
        shutil.copy(p, f"{B}/results/v10/analysis/{f}")
# model-independent inputs (masks, data flow, invariance checks) copied from the historical analysis
for f in ("v7/analysis/dataflow.csv", "v7/analysis/host_reaudit.csv", "v7/analysis/severities_realized.csv",
          "v7/analysis/operator_clip_counts.csv", "v10/analysis/conf_invariance.csv",
          "v10/analysis/conf_invariance_leak.csv", "v13/analysis/conf_invariance_v13.csv"):
    src = os.path.realpath(f"results/{f.split('/')[0]}") + "/" + "/".join(f.split("/")[1:])
    if os.path.exists(src):
        shutil.copy(src, f"{B}/results/{f}")
# v9: analysis/ copied (analyze_v9_variance.py rewrites its ASDPose-host rows), the rest linked
v9h = os.path.realpath("results/v9"); os.makedirs(f"{B}/results/v9", exist_ok=True)
for e in os.listdir(v9h):
    if e == "analysis":
        shutil.copytree(f"{v9h}/analysis", f"{B}/results/v9/analysis")
    else:
        os.symlink(f"{v9h}/{e}", f"{B}/results/v9/{e}")
# v11: drop the files of real-split models whose selected epoch changes under this rule, so
# score_v11.py rescores exactly those from the repaired checkpoints
import json as _json
changed = set()
for lp in glob.glob(f"{R}/results/epochs/*/train_log.json"):
    j = _json.load(open(lp))
    if a.rule == "clean" and j["selected_epoch_orig"] != j["selected_epoch_clean"]:
        changed.add(j["run"])
if a.rescore_all_ours:
    changed = {os.path.basename(os.path.dirname(lp)) for lp in glob.glob(f"{R}/results/epochs/*/train_log.json")}
v11_drop = {f"ours_v7_{r[len('v7_real_'):]}" for r in changed if r.startswith("v7_real_")} | \
           {f"ours_v10_{r[len('v10_real_I_'):]}" for r in changed if r.startswith("v10_real_I_")}
print("v11 models to rescore:", sorted(v11_drop))
shutil.copytree(os.path.realpath("results/v11"), f"{B}/results/v11", symlinks=True,
                ignore=lambda d, names: [n for n in names if n.split(".")[0] in v11_drop])
for proto, fam in (("v7", "pilot_v7_real_"), ("v10", "pilot_v10_real_I_")):
    os.makedirs(f"{B}/results/fits/{proto}/cache", exist_ok=True)
    for p in sorted(glob.glob(f"{R}/cache/{a.rule}/{fam}*.pt")):
        os.symlink(p, f"{B}/results/fits/{proto}/cache/{os.path.basename(p)}")
n = {p: len(glob.glob(f"{B}/results/{p}/scores/*.csv")) for p in ("v7", "v10", "v12", "v13", "v15conf", "v16", "v17")}
print(f"built {B}: score files {n}; v11 checkpoints v7 {len(os.listdir(f'{B}/results/fits/v7/cache'))}, "
      f"v10 {len(os.listdir(f'{B}/results/fits/v10/cache'))}")
