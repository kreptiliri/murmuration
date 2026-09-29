import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

RULES = ("separation", "alignment", "cohesion")
#one colour per draft, the same in every panel
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
DRAFT_NAMES = ["draft 0 (own move only)", "draft 1 (listened once)", "draft 2 (listened twice)"]


def plain_number(value, position):
    #1000 -> "1,000" instead of 10 to the 3
    return f"{value:,.0f}"


def plot_readability(data, *, linear=False):
    #one panel per rule: how well each draft shows that rule's push, over training
    curve = data["curve"]
    untrained = curve[0]
    if untrained["batch"] != 0:
        raise ValueError("the first row has to be batch 0, the untrained control")
    #a log axis has no room for batch 0, so there it is only the dashed line
    if linear:
        rows = curve
    else:
        rows = [row for row in curve if row["batch"] > 0]
    batches = [row["batch"] for row in rows]
    #how many batches make one pass, read off the curve itself
    last = curve[-1]
    batches_per_pass = last["batch"] / last["pass"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    for ax, rule in zip(axes, RULES):
        n_drafts = len(untrained["scores"][rule])
        for k in range(n_drafts):
            scores = [row["scores"][rule][k] for row in rows]
            ax.plot(batches, scores, color=COLORS[k], linestyle="-", linewidth=2)
            #what a random network already shows: the free part
            ax.axhline(untrained["scores"][rule][k], color=COLORS[k], linestyle="--", linewidth=1.2)
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
    axes[0].set_ylabel("probe score on held-out clips (1 = perfect)")

    handles = []
    for k in range(len(DRAFT_NAMES)):
        handles.append(Line2D([], [], color=COLORS[k], linewidth=2, label=DRAFT_NAMES[k]))
    handles.append(Line2D([], [], color="#555555", linestyle="--", linewidth=1.2, label="same draft, untrained model"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"can the model's drafts show each push? probe fitted on {data['learn_clips']} clips, "
                 f"scored on {data['check_clips']} held-out clips", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("curve_file")          #what probe.py --sweep --out wrote
    parser.add_argument("--out", required=True)
    parser.add_argument("--linear", action="store_true")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.curve_file).read_text())
    fig = plot_readability(data, linear=args.linear)
    fig.savefig(args.out, dpi=150)
    return fig


if __name__ == "__main__":
    main()
