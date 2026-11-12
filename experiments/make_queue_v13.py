"""Write configs/runs/queue_v13.txt: the 6 training lines of protocol v13 (configs/protocols/armconf_v13.yaml).

Each line is "<out_dir> <train_asdpose.py arguments>" (run with experiments/train_runs.sh); scoring
is score_v13.py.
    python experiments/make_queue_v13.py
"""
SEEDS = (0, 1, 2)
COMMON = "--proto v13 --arch compact --amp --batch-size 128 --epochs 25 --workers 6"


def lines():
    return [f"results/v13/{w}_w0_armruns_mid/A_none/seed{k} {COMMON} --cell A_none --world {w} "
            f"--mechanism armruns --strength mid --seed {k}"
            for w in ("S", "I") for k in SEEDS]


def main():
    ls = lines()
    with open("configs/runs/queue_v13.txt", "w") as f:
        f.write("\n".join(ls) + "\n")
    print(f"{len(ls)} training lines -> configs/runs/queue_v13.txt")


if __name__ == "__main__":
    main()
