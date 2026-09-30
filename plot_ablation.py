import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.ticker import FuncFormatter

from plot_readability import plain_number

RULES = ("separation", "alignment", "cohesion")
#one colour per round, the same as its draft in plot_readability
ROUND_COLORS = ["#eb6834", "#1baf7a"]


def plot_ablation(data, readability=None, *, linear=False):
    #one panel per rule: how much of that push the model does, how much of it each
    #round carries (lost when that round hears only itself), and, if given, how well
    #the draft that round writes shows the push
    curve = data["curve"]
    if linear:
        rows = curve
    else:
        rows = [row for row in curve if row["batch"] > 0]
    batches = [row["batch"] for row in rows]
    last = curve[-1]
    batches_per_pass = last["batch"] / last["pass"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, rule in zip(axes, RULES):
        done = [row["done"][rule] for row in rows]
        ax.plot(batches, done, color="black", linewidth=2, label="the model does")
        n_blocks = len(rows[0]["blocks"])
        for k in range(n_blocks):
            lost = [row["blocks"][k]["lost"][rule] for row in rows]
            ax.plot(batches, lost, color=ROUND_COLORS[k], linewidth=2,
                    label=f"lost when round {k + 1} hears only itself")

        if readability is not None:
            read_rows = []
            for row in readability["curve"]:
                if linear or row["batch"] > 0:
                    read_rows.append(row)
            read_batches = [row["batch"] for row in read_rows]
            for k in range(n_blocks):
                #draft k + 1 is what round k + 1 wrote
                scores = [row["scores"][rule][k + 1] for row in read_rows]
                ax.plot(read_batches, scores, color=ROUND_COLORS[k], linestyle="--", linewidth=1.2,
                        label=f"readable in draft {k + 1}")

        ax.axvline(batches_per_pass, color="#555555", linestyle=":", linewidth=1.2)
        ax.text(batches_per_pass, 1.0, " end of pass 1", color="#555555", fontsize=9,
                va="top", transform=ax.get_xaxis_transform())
        ax.set_title(rule, loc="left")
        if linear:
            ax.set_xscale("linear")
            ax.set_xlabel("training steps (batches)")
        else:
            ax.set_xscale("log")
            ax.set_xlabel("training steps (batches), log scale")
        ax.xaxis.set_major_formatter(FuncFormatter(plain_number))
        ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("share of the push (1 = all of it)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.suptitle(f"readable vs used: {data['check_clips']} held-out clips, "
                 f"birds away from the speed limits", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("curve_file")          #what ablate.py --sweep --out wrote
    parser.add_argument("--readability", default=None)   #what probe.py --sweep --out wrote, same seed
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.curve_file).read_text())
    readability = None
    if args.readability is not None:
        readability = json.loads(Path(args.readability).read_text())
    fig = plot_ablation(data, readability, linear=args.linear)
    fig.savefig(args.out, dpi=150)
    return fig


if __name__ == "__main__":
    main()
