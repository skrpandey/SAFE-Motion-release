"""One authoritative provenance map of every final ASDPose model (review B, item M5).

Joins, for a release tree, each protocol's registry.json (protocol, architecture, world, training
assignment = world seed, optimizer seed, cell, validity rule) with every fit's train_log.json
(validity rule and architecture of the cell as trained, loader recipe, selected epochs under both
selection rules, code commit) and, when given, MANIFEST.json (checkpoint and log hashes). Writes
  <tree>/results/provenance/models.csv      one row per final model (all fields)
  <tree>/paper/tables/provenance.tex        one row per protocol block and analysis role
and refuses to write when the counts disagree with the run lists (configs/runs/queue_*.txt) or with the
manifest. Run from the repository root with --tree pointing at a release tree (or ".").
    python experiments/model_provenance.py --tree build/release_v17_clean [--manifest build/release_v17_clean/MANIFEST.json]
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROTOS = ("v7", "v10", "v12", "v13", "v15conf", "v16", "v17")
EPOCH_DIRS = {"v7": "results/fits/repair_v14/results/epochs", "v8": "results/fits/repair_v14/results/epochs",
              "v10": "results/fits/repair_v14/results/epochs", "v12": "results/fits/repair_v14/results/epochs",
              "v13": "results/fits/repair_v14/results/epochs", "v15conf": "results/fits/v15conf/results/epochs",
              "v16": ("results/fits/v16a/results/epochs", "results/fits/v16bc/results/epochs"),
              "v17": "results/fits/v17/results/epochs"}
# run lists: protocol -> queue files whose line count is the number of final models of that protocol
QUEUES = {"v7": ("configs/runs/queue_v7.txt", "configs/runs/queue_v8.txt"), "v10": ("configs/runs/queue_v10.txt",), "v12": ("configs/runs/queue_v12.txt",),
          "v13": ("configs/runs/queue_v13.txt",), "v15conf": ("configs/runs/queue_v15conf.txt",),
          "v16": ("configs/runs/queue_v16a.txt", "configs/runs/queue_v16bc.txt"), "v17": ("configs/runs/queue_v17.txt",)}
BLOCK = {"v7": "v7, v8, v10, v12", "v8": "v7, v8, v10, v12", "v10": "v7, v8, v10, v12", "v12": "v7, v8, v10, v12",
         "v13": "v13", "v15conf": "v15conf", "v16": "v16", "v17": "v17"}
RECEIVING = {"v7": "Tables 4, S2, S3, S14, S18; Fig. 2", "v8": "Tables 4, S2, S3, S18; Fig. 2", "v10": "Tables S5, S20; S6",
             "v12": "Tables 3, S9, S10", "v13": "Tables 2, 4, S4, S18", "v15conf": "Tables 2, 3, S11",
             "v16": "Tables 2, 3, S16, S17", "v17": "Table S-v17"}
TEST_MASKS = {None: "clean only", "planted": "own operator, masks 2000/3000/4000; four operators, mask 2000"}


def role(m):
    p, cell, w, ws = m["proto"], m["cell"], m.get("world"), int(m.get("world_seed") or 0)
    if w is None:
        return "real-split models" if p == "v7" else "real-split models retrained with label-independent missing runs"
    if p == "v10":
        return "interpolated (confidence-removed) control"
    if cell.startswith("C_"):
        return {"C_groupdro": "Group DRO, planted-severity groups", "C_groupdro_obs": "Group DRO, input-derived groups",
                "C_adversarial": "adversarial head", "C_augment": "operator augmentation"}.get(cell, cell)
    if cell == "A_none_stgcn":
        return "ST-GCN, no remedy"
    if cell != "A_none":
        return "input configuration (" + cell[2:].replace("_", " ") + ")"
    if p == "v17":
        return f"no remedy, regenerated training assignment {ws}"
    if ws != 0:
        return f"no remedy, extra World S training-mask realization {ws}"
    if p == "v16":
        return "no remedy, seeds 3-8 (pooled with v15conf seeds 0-2)"
    return "no remedy (planted worlds)"


def load_registry(tree, proto):
    path = f"{tree}/results/{proto}/registry.json"
    if not os.path.exists(path):
        return []
    reg = json.load(open(path))
    reg = reg["models"] if isinstance(reg, dict) else reg
    for m in reg:
        m.setdefault("proto", proto)
    return reg


def train_log(tree, m):
    dirs = EPOCH_DIRS[m["proto"]]
    for d in ([dirs] if isinstance(dirs, str) else dirs):
        p = f"{tree}/{d}/{m['name']}/train_log.json"
        if os.path.exists(p):
            return json.load(open(p)), p
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=".")
    ap.add_argument("--manifest", default=None, help="MANIFEST.json of the tree (optional cross-check and hashes)")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-tex", default=None)
    ap.add_argument("--no-strict", action="store_true", help="report count mismatches instead of refusing")
    a = ap.parse_args(argv)
    T = a.tree.rstrip("/")
    out_csv = a.out_csv or f"{T}/results/provenance/models.csv"
    out_tex = a.out_tex or f"{T}/paper/tables/provenance.tex"
    man = json.load(open(a.manifest)) if a.manifest else None
    man_fits = {}
    if man:
        for sec in ("repaired_fits", "v15conf_fits", "v16_fits", "v17_fits"):
            man_fits.update(man.get(sec, {}))
    rows, problems = [], []
    for proto in PROTOS:
        for m in load_registry(T, proto):
            log, lp = train_log(T, m)
            cell = (log or {}).get("cell", {})
            rec = {"name": m["name"], "block": BLOCK[m["proto"]], "proto": m["proto"], "cell": m["cell"],
                   "arch": cell.get("arch", m.get("arch", "compact")), "world": m.get("world") or "",
                   "world_seed": int(m.get("world_seed") or 0), "seed": m.get("seed"),
                   "mechanism": m.get("mechanism") or "", "strength": m.get("strength") or "",
                   "valid_rule": cell.get("valid_rule", m.get("valid_rule", "conf")),
                   # the trainer records the recipe since the v15 switch; earlier fits ran the only
                   # recipe that existed, the historical ("legacy") one (as extend_manifest.py records it)
                   "train_recipe": (log or {}).get("train_recipe", "legacy"),
                   "selected_epoch_orig": (log or {}).get("selected_epoch_orig", ""),
                   "selected_epoch_clean": (log or {}).get("selected_epoch_clean", ""),
                   "code_commit": ((log or {}).get("code") or {}).get("commit", (log or {}).get("code_commit", "")),
                   "role": role(m), "test_masks": TEST_MASKS[None if m.get("world") is None else "planted"],
                   "receiving": RECEIVING[m["proto"]], "train_log": lp or "",
                   "checkpoint_clean_sha256": man_fits.get(m["name"], {}).get("checkpoint_clean_sha256", ""),
                   "train_log_sha256": man_fits.get(m["name"], {}).get("train_log_sha256", "")}
            if m.get("valid_rule") and m["valid_rule"] != rec["valid_rule"]:
                problems.append(f"{m['name']}: registry rule {m['valid_rule']} != trained rule {rec['valid_rule']}")
            if log is None:
                problems.append(f"{m['name']}: no train_log.json under {EPOCH_DIRS[m['proto']]}")
            rows.append(rec)
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        problems.append("duplicate model names across registries")
    # counts against the run lists
    by_proto = {}
    for r in rows:
        by_proto.setdefault(r["proto"], 0)
        by_proto[r["proto"]] += 1
    counts = {}
    for proto, qs in QUEUES.items():
        want = sum(sum(1 for l in open(q) if l.strip()) for q in qs if os.path.exists(q))
        got = sum(by_proto.get(p, 0) for p in (("v7", "v8") if proto == "v7" else (proto,)))
        counts[proto] = (got, want)
        # a protocol whose results folder is absent from the tree is not yet part of the model set
        # (its queue may be written before its fits exist); a present protocol must match its run list
        if want and got != want and os.path.exists(f"{T}/results/{proto}/registry.json"):
            problems.append(f"{proto}: {got} registry models, {want} queue lines")
    if man_fits:
        extra = sorted(set(names) - set(man_fits)); missing = sorted(set(man_fits) - set(names))
        if extra or missing:
            problems.append(f"manifest mismatch: {len(extra)} models not in manifest, {len(missing)} manifest fits not in registries")
    blocks = {}
    for r in rows:
        key = (r["block"], r["role"], r["arch"], r["valid_rule"], r["train_recipe"] or "legacy")
        b = blocks.setdefault(key, {"n": 0, "seeds": set(), "assignments": set(), "receiving": set(), "masks": set()})
        b["n"] += 1; b["seeds"].add(r["seed"]); b["assignments"].add(r["world_seed"]); b["receiving"].add(r["receiving"]); b["masks"].add(r["test_masks"])
    total = len(rows)
    print(f"models per protocol (registry/queue): {counts}; total {total}")
    print("blocks:", {k[0]: sum(v["n"] for kk, v in blocks.items() if kk[0] == k[0]) for k in blocks})
    if problems:
        print("PROBLEMS:\n  " + "\n  ".join(problems))
        if not a.no_strict:
            raise SystemExit("provenance map not written (count or provenance mismatch)")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: (PROTOS.index(r["proto"]) if r["proto"] in PROTOS else 1, r["name"])))
    order = ["v7, v8, v10, v12", "v13", "v15conf", "v16", "v17"]
    NET = lambda a: "CompactCTRGCN" if a == "compact" else a.upper().replace("STGCN", "ST-GCN")
    RULE = lambda r: "conf.\\ $>0$" if r == "conf" else "coords $\\neq 0$"
    RECIPE = lambda r: "historical" if r == "legacy" else "corrected"
    SEL = lambda blk: "replayed" if blk in ("v7, v8, v10, v12", "v13") else "at training"

    def seeds_txt(ss):
        ss = sorted(ss)
        return f"{ss[0]}--{ss[-1]}" if len(ss) > 1 and ss == list(range(ss[0], ss[-1] + 1)) else ", ".join(map(str, ss))

    # main-text table: one row per protocol block, named by experiment (release protocol identifiers in S1)
    NAME = {"v7, v8, v10, v12": "main experiments", "v13": "corrected-rule cells",
            "v15conf": "corrected-recipe Group DRO study", "v16": "extension (seeds, ST-GCN, input groups)",
            "v17": "assignment replication"}
    L = [r"\begin{tabular}{@{}p{1.55in}lcr@{}}", r"\toprule",
         r"Experiment & Validity rule & Assignments; seeds & Models \\", r"\midrule"]
    for blk in order:
        rs = [r for r in rows if r["block"] == blk]
        if not rs:
            continue
        rules = sorted({r["valid_rule"] for r in rs}); recs = sorted({r["train_recipe"] for r in rs})
        ws = sorted({r["world_seed"] for r in rs if r["world"]}); seeds = {r["seed"] for r in rs}
        assign = "w" + seeds_txt(ws) if ws else "real split"
        if blk == "v7, v8, v10, v12":
            assign = "w0 (+w1, w2)"
        L.append(f"{NAME[blk]} & {', '.join(RULE(x) for x in rules)} & "
                 f"{assign}; {seeds_txt(seeds)} & {len(rs)} \\\\")
    L += [r"\midrule", f"Total final models & & & {total} \\\\", r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.dirname(out_tex), exist_ok=True)
    with open(out_tex, "w") as f:
        f.write("% generated by experiments/model_provenance.py from the registries, training logs and run lists; do not edit\n" + "\n".join(L) + "\n")
    # supplement table: one row per block and analysis role
    K = [r"\begin{tabular}{lllllc}", r"\toprule",
         r"Protocols & Analysis role & Validity rule & Recipe & Assignments; seeds & Models \\", r"\midrule"]
    for blk in order:
        ks = sorted((k for k in blocks if k[0] == blk), key=lambda k: (k[1], k[2]))
        for i, k in enumerate(ks):
            b = blocks[k]
            K.append(f"{blk if i == 0 else ''} & {k[1]}{'' if k[2] == 'compact' or 'ST-GCN' in k[1] else ' (' + NET(k[2]) + ')'} & {RULE(k[3])} & {RECIPE(k[4])} & "
                     f"w{seeds_txt(b['assignments'])}; {seeds_txt(b['seeds'])} & {b['n']} \\\\")
        if ks:
            K.append(r"\addlinespace")
    K[-1:] = [r"\midrule", f"\\multicolumn{{5}}{{l}}{{Total final models}} & {total} \\\\", r"\bottomrule", r"\end{tabular}"]
    roles_tex = out_tex.replace("provenance.tex", "provenance_roles.tex")
    with open(roles_tex, "w") as f:
        f.write("% generated by experiments/model_provenance.py from the registries, training logs and run lists; do not edit\n" + "\n".join(K) + "\n")
    print(f"wrote {out_csv} ({total} models), {out_tex} and {roles_tex} ({len(blocks)} role rows)")
    return rows


if __name__ == "__main__":
    main()
