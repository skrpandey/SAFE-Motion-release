"""Write configs/runs/queue_v15conf.txt: the 12 training lines of protocol v15conf
(configs/protocols/remedies_confonly_v15.yaml).

Each line holds train_asdpose.py arguments only; the trainer itself skips a finished fit. Run with
SM_TRAIN_RECIPE=v15 bash experiments/train_runs.sh configs/runs/queue_v15conf.txt.
    python experiments/make_queue_v15conf.py
"""
SEEDS = (0, 1, 2)
CELLS = ("C_groupdro", "A_none")
COMMON = ("--proto v15conf --arch compact --amp --batch-size 128 --epochs 25 --workers 6 "
          "--mechanism armruns --strength mid --repair-protocol configs/protocols/remedies_confonly_v15.yaml")


def lines():
    return [f"{COMMON} --cell {c} --world {w} --seed {k}" for c in CELLS for w in ("S", "I") for k in SEEDS]


def main():
    ls = lines()
    with open("configs/runs/queue_v15conf.txt", "w") as f:
        f.write("\n".join(ls) + "\n")
    print(f"{len(ls)} training lines -> configs/runs/queue_v15conf.txt")


if __name__ == "__main__":
    main()
