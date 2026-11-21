"""Runs of one policy-sweep model split by attack (es / random / stepwise), no new runs.

Reads the ``runs`` of the corridor-on cells at the model's own tau from policy-sweep.json files (Task 19 and 19b)
and draws, per policy, the 9 per-run upper bounds of the uplift (3 attacks x seeds 1-3). The cell's U is the largest
of them. md1 is the extreme point (more than 75 % of cards blocked at least once) and is drawn in grey.

    python scripts/plot_runs_by_attack.py out/t19/m2-hard/policy-sweep.json out/t19b/m2-hard/policy-sweep.json \
        --out out/t19b/summary --name runs-by-attack-m2-hard
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ATTACKS = ("es", "random", "stepwise")
COLORS = {"es": "#2a78d6", "random": "#eb6834", "stepwise": "#1baf7a"}  # categorical slots 1-3
MARKERS = {"es": "o", "random": "s", "stepwise": "^"}
GREY = "#8a8985"
TEXT = {
    "en": {"x": "uplift, upper bound of one run, % of the episode amount", "md1": "impractical: > 75 % of cards blocked at least once",
           "u": "U (worst run)", "title": "{label}: runs by attack, corridor on, own τ (FPR {fpr:g})",
           "es": "ES", "random": "random search", "stepwise": "stepwise"},
    "uk": {"x": "приріст, верхня межа одного прогону, % від суми епізоду", "md1": "непрактично: > 75 % карток блокувались хоч раз",
           "u": "U (найгірший прогін)", "title": "{label}: прогони за атаками, коридор увімкнено, власний τ (FPR {fpr:g})",
           "es": "ES", "random": "випадковий пошук", "stepwise": "покрокова"},
}


def load_cells(paths, label=None):
    """Corridor-on cells at the own tau, one per distinct (reaction, kappa, q), later files do not override."""
    cells, own = {}, None
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        label = label or d["label"]
        own = d["own_target_fpr"]
        for c in d["cells"]:
            if c["corridor"] != "on" or abs(c["target_fpr"] - own) > 1e-12 or not c.get("runs"):
                continue
            kappa = c.get("corridor_kappa") if c.get("corridor_kappa") is not None else d["corridor_params"]["daily"]["kappa"]
            q = c.get("corridor_q") if c.get("corridor_q") is not None else d["corridor_params"]["q"]
            cells.setdefault((c["reaction"], float(kappa), float(q)), c)
    return label, own, cells


def row_label(key):
    reaction, kappa, q = key
    return f"{reaction}, κ {kappa:g}, q {q:g}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sweeps", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/t19b/summary"))
    ap.add_argument("--name", default="runs-by-attack")
    a = ap.parse_args(argv)
    label, own, cells = load_cells(a.sweeps)
    if not cells:
        raise SystemExit("no corridor-on cells at the own tau in the given files")
    keys = sorted(cells, key=lambda k: (k[0] != "md1", k[0], -k[1], -k[2]))  # md1 first (top), then md3, md3+esc
    a.out.mkdir(parents=True, exist_ok=True)
    for lang, t in TEXT.items():
        fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(keys) + 1.4), dpi=200)
        for i, k in enumerate(keys):
            grey = k[0] == "md1"
            runs = cells[k]["runs"]
            for j, atk in enumerate(ATTACKS):
                ys = i + (j - 1) * 0.2
                for s in (1, 2, 3):
                    v = runs[f"{atk}/{s}"]["uplift_upper"] * 100
                    ax.scatter(v, ys, s=34, marker=MARKERS[atk], color=GREY if grey else COLORS[atk],
                               edgecolor="white", linewidth=0.8, alpha=0.55 if grey else 0.95, zorder=3)
            u = cells[k]["U"] * 100
            ax.text(61.5, i, f"{u:.1f}", va="center", ha="right", fontsize=8, color="#52514e")
            if grey:
                ax.text(1, i - 0.52, t["md1"], fontsize=7, color=GREY, va="center")
        ax.set_yticks(range(len(keys)), [row_label(k) for k in keys], fontsize=8)
        for lbl, k in zip(ax.get_yticklabels(), keys):
            lbl.set_color(GREY if k[0] == "md1" else "#0b0b0b")
        ax.invert_yaxis()
        ax.set_xlim(0, 62)
        ax.set_xlabel(t["x"], fontsize=8)
        ax.text(61.5, -0.95, t["u"], fontsize=7, ha="right", color="#52514e")
        ax.grid(axis="x", color="#e6e5e1", linewidth=0.6, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(t["title"].format(label=label, fpr=own), fontsize=9, loc="left")
        handles = [plt.Line2D([], [], marker=MARKERS[x], color=COLORS[x], linestyle="", markersize=6, label=t[x]) for x in ATTACKS]
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, ncol=1, bbox_to_anchor=(0.9, 0.0))
        fig.tight_layout()
        fig.savefig(a.out / f"{a.name}-{lang}.png")
        plt.close(fig)
        print(a.out / f"{a.name}-{lang}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
