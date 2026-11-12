"""Write the training lines of protocol v17 (configs/protocols/confonly_assignment_v17.yaml).

configs/runs/queue_v17.txt: CompactCTRGCN no remedy under confidence-only planting, regenerated training
assignments (world seeds 1, 2, 3), worlds S and I, optimizer seeds 0-2 (18 fits). Each line holds
train_asdpose.py arguments only; the trainer skips a finished fit.
    python experiments/make_queue_v17.py
"""
COMMON = ("--proto v17 --amp --batch-size 128 --epochs 25 --workers 6 --mechanism armruns "
          "--strength mid --repair-protocol configs/protocols/confonly_assignment_v17.yaml")
WORLD_SEEDS = (1, 2, 3)
SEEDS = (0, 1, 2)


def lines():
    return [f"{COMMON} --arch compact --cell A_none --world-seed {ws} --world {w} --seed {k}"
            for ws in WORLD_SEEDS for k in SEEDS for w in ("S", "I")]


def main():
    ls = lines()
    with open("configs/runs/queue_v17.txt", "w") as f:
        f.write("\n".join(ls) + "\n")
    print(f"{len(ls)} training lines -> configs/runs/queue_v17.txt")


if __name__ == "__main__":
    main()
