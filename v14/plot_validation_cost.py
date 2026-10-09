"""
Cost (1.1 - J) vs CMA-ES generation for the trained v14 variants, on validation and/or train.

J = success + 0.1 * progress, read from each variant's cmaes_state.json:
    validation  the search mean m, scored on 100 val scenarios x 3 futures every val_every generations
                (gen 0 = the starting set). Square: the selected checkpoint (highest validation J).
    train       the 15 candidates of each generation on 100 train scenarios x 2 futures, summarised per
                generation by --train-stat (best and/or median); gen 0 = the starting set.

    python plot_validation_cost.py                                    # all variants, validation only
    python plot_validation_cost.py --variants base adaptive --data validation train
    python plot_validation_cost.py --data train --train-stat best median --out train_cost.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
J_MAX = 1.1

# folder -> (label, color); categorical slots 1..3 in fixed order, so a variant keeps its color
# whichever subset is plotted
VARIANTS = {
    "base": ("base ", "#2a78d6"),
    "base_delta_v": ("|Δv| numerator", "#eb6834"),
    "adaptive": ("adaptive", "#1baf7a"),
}
TRAIN_STYLE = {"best": "--", "median": ":"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def load(folder):
    st = json.loads((HERE / folder / "cmaes_state.json").read_text())
    cps = st.get("checkpoints", [])
    val = sorted((c["gen"], J_MAX - c["val"]["J"]) for c in cps if c.get("kind") != "best train")
    sel = max(cps, key=lambda c: (round(c["val"]["J"], 9), -c["gen"])) if cps else None
    start = st.get("start_train") or st.get("base_train")     # base / adaptive name for the gen-0 set
    train = {}
    for stat in TRAIN_STYLE:
        pts = [(h["gen"], J_MAX - h[f"J_{stat}"]) for h in st.get("history", [])]
        train[stat] = ([(0, J_MAX - start["J"])] if start else []) + pts
    return val, sel, train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                    help="variants to plot (default: all)")
    ap.add_argument("--data", nargs="+", choices=["validation", "train"], default=["validation"],
                    help="which cost to plot (default: validation)")
    ap.add_argument("--train-stat", nargs="+", choices=list(TRAIN_STYLE), default=["best"],
                    help="per-generation summary of the 15 train candidates (default: best)")
    ap.add_argument("--out", default=str(HERE.parent / "reports" / "sep25-oct09" / "validation_cost.png"))
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    plotted = []
    for folder in args.variants:
        if not (HERE / folder / "cmaes_state.json").exists():
            print(f"skip {folder}: no cmaes_state.json")
            continue
        label, color = VARIANTS[folder]
        val, sel, train = load(folder)
        end = None                                  # where the direct label goes
        if "train" in args.data:
            for stat in args.train_stat:
                if len(train[stat]) < 2:
                    continue
                gx, gy = zip(*train[stat])
                ax.plot(gx, gy, color=color, lw=1.5, ls=TRAIN_STYLE[stat], zorder=2)
                end = (gx[-1], gy[-1])
        if "validation" in args.data and len(val) >= 2:
            gx, gy = zip(*val)
            ax.plot(gx, gy, color=color, lw=2, marker="o", ms=5, mec=SURFACE, mew=1.5, zorder=3)
            end = (gx[-1], gy[-1])
            if sel is not None:
                sy = J_MAX - sel["val"]["J"]
                ax.plot(sel["gen"], sy, marker="s", ms=9, color=color, mec=SURFACE, mew=1.5, zorder=4)
                label =f"{label}"# f"{label}: selected gen {sel['gen']}, val cost {sy:.3f}"
        if end is None:
            continue
        ax.annotate(folder, end, xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
        plotted.append(Line2D([], [], color=color, lw=2, label=label))

    # legend: one entry per variant (color), then one per line style (data)
    keys = []
    if "validation" in args.data:
        keys.append(Line2D([], [], color=MUTED, lw=2, marker="o", ms=5, mec=SURFACE, label="validation (search mean)"))
        keys.append(Line2D([], [], ls="none", marker="s", ms=8, color=MUTED, label="selected"))
    if "train" in args.data:
        for stat in args.train_stat:
            keys.append(Line2D([], [], color=MUTED, lw=1.5, ls=TRAIN_STYLE[stat],
                               label=f"train ({stat} of 15 candidates)"))
    ax.legend(handles=plotted + keys, frameon=False, fontsize=8, loc="upper right", labelcolor=INK2)

    what = " and ".join("validation" if d == "validation" else "train" for d in args.data)
    ax.set_title(f"Cost vs generation", loc="left", color=INK, fontsize=12)
    ax.set_xlabel("generation", color=INK2)
    ax.set_ylabel("cost", color=INK2)
    ax.set_ylim(bottom=0)
    ax.set_xlim(-1, None)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    # fig.text(0.01, 0.01, "cost = 1.1 − J,  J = success + 0.1·progress;  validation: 100 scenarios × 3 futures, "
    #          "train: 100 × 2;  lower is better", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=SURFACE)
    print(args.out)


if __name__ == "__main__":
    main()
