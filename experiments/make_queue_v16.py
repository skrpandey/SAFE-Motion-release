"""Write the training lines of protocol v16 (configs/protocols/confonly_extension_v16.yaml).

configs/runs/queue_v16a.txt: part A, ST-GCN no remedy, worlds S and I, seeds 0-2 (6 fits).
configs/runs/queue_v16bc.txt: part C, CompactCTRGCN Group DRO with observed groups, seeds 0-2 (6 fits),
then part B, CompactCTRGCN no remedy, seeds 3-8 (12 fits).
Each line holds train_asdpose.py arguments only; the trainer skips a finished fit. Run with
SM_TRAIN_RECIPE=v15 bash experiments/train_runs.sh configs/runs/queue_v16a.txt (and queue_v16bc.txt).
    python experiments/make_queue_v16.py
"""
COMMON = ("--proto v16 --amp --batch-size 128 --epochs 25 --workers 6 --mechanism armruns "
          "--strength mid --repair-protocol configs/protocols/confonly_extension_v16.yaml")


def lines_a():
    return [f"{COMMON} --arch stgcn --cell A_none_stgcn --world {w} --seed {k}"
            for k in (0, 1, 2) for w in ("S", "I")]


def lines_bc():
    c = [f"{COMMON} --arch compact --cell C_groupdro_obs --world {w} --seed {k}"
         for k in (0, 1, 2) for w in ("S", "I")]
    b = [f"{COMMON} --arch compact --cell A_none --world {w} --seed {k}"
         for k in range(3, 9) for w in ("S", "I")]
    return c + b


def main():
    for name, ls in (("configs/runs/queue_v16a.txt", lines_a()), ("configs/runs/queue_v16bc.txt", lines_bc())):
        with open(name, "w") as f:
            f.write("\n".join(ls) + "\n")
        print(f"{len(ls)} training lines -> {name}")


if __name__ == "__main__":
    main()
