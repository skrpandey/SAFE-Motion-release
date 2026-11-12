"""Write configs/runs/queue_v12.txt: the 36 training lines of protocol v12 (configs/protocols/remedies_v12.yaml).

Each line is "<out_dir> <train_asdpose.py arguments>" (run with experiments/train_runs.sh); scoring
is score_v12.py.
    python experiments/make_queue_v12.py
"""
SEEDS = (0, 1, 2)
COMMON = "--proto v12 --arch compact --amp --batch-size 128 --epochs 25 --workers 6"
CELLS = ("C_groupdro", "C_adversarial", "C_augment")


def lines():
    return [f"results/v12/{w}_w0_{mech}_mid/{cell}/seed{k} {COMMON} --cell {cell} --world {w} "
            f"--mechanism {mech} --strength mid --seed {k}"
            for cell in CELLS for mech in ("runs", "frozen") for w in ("S", "I") for k in SEEDS]


def main():
    ls = lines()
    with open("configs/runs/queue_v12.txt", "w") as f:
        f.write("\n".join(ls) + "\n")
    print(f"{len(ls)} training lines -> configs/runs/queue_v12.txt")


if __name__ == "__main__":
    main()
