"""Figure dU of the budget sensitivity on the 12 cells of three families (R45 (6), R52 (7)).

One panel: dU = U(B) - U(2000) in pp for each cell, the cells grouped by family (MLP, XGB, GRU), the seed spread 9.6 pp
as a line. Filled markers – the gate seeds 1-3 at budgets 8 000 / 20 000 / 50 000 (the data of H and H50; 8 000 is
marked "other commit"). Small hollow markers beside the GRU cells – the maximum of the cell on fresh seeds 4-9 at
20 000 and 50 000 (Task 27e, descriptive), not per seed.

Inputs are the ``budget-sensitivity.json`` files already written by ``fraudaudit budget-sensitivity``; nothing is
recomputed except the check ``dU == U(B) - U(2000)``. The budget-2000 reference of a cell must be bitwise equal in every
gate file that holds the cell (the reference was recomputed at each commit and matched, register 5.41 / 5.49).

    fraudaudit budget-figure --gate out/t19c/summary/budget-sensitivity.json \
        --gate out/t19c-t24/summary-xgb/budget-sensitivity.json --gate out/t19c-t24/summary-gru/budget-sensitivity.json \
        --gate out/t27-chain1/t27f/summary-mlp/budget-sensitivity.json ... \
        --fresh out/t27-chain1/t27e/summary/budget-sensitivity.json --out out/t27-chain1/summary-du
"""
from __future__ import annotations

import json
from pathlib import Path

from fraudaudit.budget_sensitivity import REFERENCE_BUDGET, SEED_SPREAD

GATE_SEEDS = (1, 2, 3)
OTHER_COMMIT_BUDGETS = (8000,)
FAMILY_OF_PREFIX = {"m3": "MLP", "m2": "MLP", "xgb": "XGB", "gru": "GRU"}
FAMILY_ORDER = ("MLP", "XGB", "GRU")
BUDGET_STYLE = {8000: ("^", "#8a8a8a"), 20000: ("s", "#2a78d6"), 50000: ("o", "#eb6834")}
DU_TOL = 1e-12


def family(label: str) -> str:
    prefix = label.split("-", 1)[0]
    if prefix not in FAMILY_OF_PREFIX:
        raise ValueError(f"label {label!r}: unknown family prefix {prefix!r} (known: {sorted(FAMILY_OF_PREFIX)})")
    return FAMILY_OF_PREFIX[prefix]


def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _du(row: dict, b: int, where: str) -> float:
    if str(b) not in row:
        raise ValueError(f"{where}: no row for budget {b}")
    ref, cur = row[str(REFERENCE_BUDGET)]["U"], row[str(b)]["U"]
    du = row[str(b)]["dU"]
    if abs(du - (cur - ref)) > DU_TOL:
        raise ValueError(f"{where}: stored dU {du!r} != U({b}) - U({REFERENCE_BUDGET}) = {cur - ref!r}")
    return du


def _check_file(p, d: dict) -> None:
    if d.get("reference_budget", REFERENCE_BUDGET) != REFERENCE_BUDGET:
        raise ValueError(f"{p}: reference budget {d.get('reference_budget')} != {REFERENCE_BUDGET}")
    if d.get("seed_spread", SEED_SPREAD) != SEED_SPREAD:
        raise ValueError(f"{p}: seed spread {d.get('seed_spread')} != {SEED_SPREAD}")


def collect(gate_paths, fresh_paths=()) -> dict:
    """Cells, their family and dU per budget from the gate files (seeds 1-3) and the fresh files (other seeds).

    Returns ``{"cells": [{label, key, family, ref_U, dU: {b: x}, fresh_dU: {b: x}}], "budgets", "fresh_budgets",
    "fresh_seeds", "sources"}``, cells ordered by family (MLP, XGB, GRU, then others) and by first appearance."""
    cells: dict = {}
    sources = {"gate": [str(p) for p in gate_paths], "fresh": [str(p) for p in fresh_paths]}
    budgets: set = set()
    for p in gate_paths:
        d = _load(p)
        # Task 19c files predate the key (gate seeds 1-3); a present key must name exactly the gate seeds (null = the
        # labels of the file ran on different seeds, refused).
        if "attack_seeds" in d and d["attack_seeds"] != list(GATE_SEEDS):
            raise ValueError(f"{p}: gate file has seeds {d['attack_seeds']}, expected {list(GATE_SEEDS)}")
        _check_file(p, d)
        for label, rows in d["labels"].items():
            for key, row in rows.items():
                ref = row[str(REFERENCE_BUDGET)]
                c = cells.setdefault((label, key), {"label": label, "key": key, "family": family(label),
                                                    "ref_U": ref["U"], "ref_runs": ref.get("runs"), "dU": {},
                                                    "fresh_dU": {}})
                if c["ref_U"] != ref["U"] or c["ref_runs"] != ref.get("runs"):
                    raise ValueError(f"{p}: {label} {key}: budget-{REFERENCE_BUDGET} reference (U {ref['U']!r} and its "
                                     f"runs) differs from another gate file (U {c['ref_U']!r})")
                for b in d["budgets"]:
                    if b in c["dU"]:
                        raise ValueError(f"{p}: {label} {key}: budget {b} given twice")
                    c["dU"][b] = _du(row, b, f"{p}: {label} {key}")
                    budgets.add(b)
    if not cells:
        raise ValueError("no cells in the gate files")
    for c in cells.values():
        if set(c["dU"]) != budgets:
            raise ValueError(f"{c['label']} {c['key']}: budgets {sorted(c['dU'])}, expected every gate budget "
                             f"{sorted(budgets)} (a gate file missing?)")
    fresh_budgets: set = set()
    fresh_seeds: set = set()
    for p in fresh_paths:
        d = _load(p)
        seeds = d.get("attack_seeds")
        if not seeds or set(seeds) & set(GATE_SEEDS):
            raise ValueError(f"{p}: fresh file must name its seeds, disjoint from the gate seeds {list(GATE_SEEDS)}; "
                             f"got {seeds}")
        if d.get("verdict") is not None:
            raise ValueError(f"{p}: a fresh file is descriptive (no verdict); got {d['verdict']}")
        if fresh_seeds and set(seeds) != fresh_seeds:
            raise ValueError(f"{p}: fresh files must share their seeds; got {seeds} and {sorted(fresh_seeds)}")
        _check_file(p, d)
        fresh_seeds.update(seeds)
        for label, rows in d["labels"].items():
            for key, row in rows.items():
                if (label, key) not in cells:
                    raise ValueError(f"{p}: {label} {key} is not among the gate cells")
                c = cells[(label, key)]
                for b in d["budgets"]:
                    if b in c["fresh_dU"]:
                        raise ValueError(f"{p}: {label} {key}: fresh budget {b} given twice")
                    c["fresh_dU"][b] = _du(row, b, f"{p}: {label} {key}")
                    fresh_budgets.add(b)
    for c in cells.values():
        del c["ref_runs"]
    fam_rank = {f: i for i, f in enumerate(FAMILY_ORDER)}
    order = sorted(enumerate(cells.values()), key=lambda ic: (fam_rank[ic[1]["family"]], ic[0]))
    return {"cells": [c for _, c in order], "budgets": sorted(budgets), "fresh_budgets": sorted(fresh_budgets),
            "fresh_seeds": sorted(fresh_seeds), "gate_seeds": list(GATE_SEEDS), "seed_spread": SEED_SPREAD,
            "reference_budget": REFERENCE_BUDGET, "other_commit_budgets": list(OTHER_COMMIT_BUDGETS),
            "sources": sources}


def _key_parts(key: str) -> dict:
    out = {}
    for i, part in enumerate(key.split("|")):
        name, sep, val = part.partition("=")
        out[name if sep else f"_{i}"] = val if sep else part
    return out


def cell_name(res: dict, c: dict) -> str:
    """Model label and reaction, plus any key part that differs between the cells (none for the 12 cells of 19c)."""
    parts = [_key_parts(x["key"]) for x in res["cells"]]
    mine = _key_parts(c["key"])
    common = {k for k in mine if all(p.get(k) == mine[k] for p in parts)}
    reaction = next((v for k, v in mine.items() if k.startswith("_")), "")
    extra = [f"{k}={v}" for k, v in mine.items() if k not in common and not k.startswith("_")]
    return " ".join([c["label"], reaction, *extra]).strip()


def _common_params(res: dict) -> dict:
    parts = [_key_parts(x["key"]) for x in res["cells"]]
    return {k: v for k, v in parts[0].items() if not k.startswith("_") and all(p.get(k) == v for p in parts)}


def markdown(res: dict) -> str:
    pp = lambda x: "–" if x is None else f"{100 * x:+.1f}"  # noqa: E731
    bs, fb = res["budgets"], res["fresh_budgets"]
    oc = set(res["other_commit_budgets"])
    seeds = res["fresh_seeds"]
    fs = f"seeds {seeds[0]}-{seeds[-1]}" if seeds else "fresh seeds"
    lines = [f"# dU of the budget sensitivity, {len(res['cells'])} cells (R45 (6), R52 (7))", "",
             f"dU = U(B) - U({res['reference_budget']}), pp; U = worst one-sided upper bound over attacks x seeds. "
             f"Seeds 1-3 = the data of H / H50; {fs} = maximum of the cell against its own 2000 on those seeds "
             f"(descriptive, no verdict). "
             f"Seed spread {100 * res['seed_spread']:.1f} pp. "
             + (f"Budget(s) {', '.join(map(str, sorted(oc)))}: other commit. " if oc else ""), "",
             "| family | cell | " + " | ".join(f"dU {b}" + (" (other commit)" if b in oc else "") for b in bs)
             + "".join(f" | dU {b}, {fs}" for b in fb) + " |",
             "|---|---|" + "---:|" * (len(bs) + len(fb))]
    for c in res["cells"]:
        lines.append(f"| {c['family']} | `{c['label']} {c['key']}` | " + " | ".join(pp(c["dU"].get(b)) for b in bs)
                     + "".join(f" | {pp(c['fresh_dU'].get(b))}" for b in fb) + " |")
    lines += ["", "Sources: " + ", ".join(f"`{s}`" for s in res["sources"]["gate"] + res["sources"]["fresh"])]
    return "\n".join(lines) + "\n"


TEXT = {
    "en": {"y": "dU = U(B) − U(2,000), pp", "spread": "seed spread {s} pp", "budget": "budget {b}, seeds 1–3",
           "other": "budget {b}, seeds 1–3 (other commit)",
           "fresh": "budget {b}, max of the cell on seeds {s} against its 2,000 (Task 27e)",
           "note": "FPR {fpr}, corridor κ {kappa}, q {q}; filled – seeds 1–3 (H, H50); hollow – Task 27e, descriptive"},
    "uk": {"y": "dU = U(B) − U(2 000), п. п.", "spread": "розкид seed {s} п. п.", "budget": "бюджет {b}, seed 1–3",
           "other": "бюджет {b}, seed 1–3 (інший коміт)",
           "fresh": "бюджет {b}, максимум комірки на seed {s} відносно їхніх 2 000 (27e)",
           "note": "FPR {fpr}, коридор κ {kappa}, q {q}; заповнені – seed 1–3 (H, H50); порожні – 27e, описово"},
}


def _num(x: float, lang: str, d: int = 1) -> str:
    s = f"{x:.{d}f}".replace("-", "\u2212")
    return s.replace(".", ",") if lang == "uk" else s


def _budget(b: int, lang: str) -> str:
    return f"{b:,}".replace(",", " " if lang == "uk" else ",")


def positions(res: dict) -> tuple[list[float], list[tuple[str, float, float]]]:
    """x of each cell (a gap of 0.8 between families) and ``(family, x_start, x_end)`` spans."""
    xs, spans, x, prev = [], [], 0.0, None
    for c in res["cells"]:
        if prev is not None and c["family"] != prev:
            x += 0.8
        if prev != c["family"]:
            spans.append([c["family"], x, x])
        spans[-1][2] = x
        xs.append(x)
        prev = c["family"]
        x += 1.0
    return xs, [tuple(s) for s in spans]


def budget_du_plot(res: dict, path: Path, lang: str = "en") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    t = TEXT[lang]
    xs, spans = positions(res)
    bs, fb = res["budgets"], res["fresh_budgets"]
    oc = set(res["other_commit_budgets"])
    seeds = res["fresh_seeds"]
    fs = f"{seeds[0]}–{seeds[-1]}" if seeds else ""
    gate_off = {b: (i - (len(bs) - 1) / 2) * 0.17 - 0.08 for i, b in enumerate(bs)}
    fresh_off = {b: 0.24 + i * 0.12 for i, b in enumerate(fb)}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=1)
    ax.axhline(100 * res["seed_spread"], color="#333333", lw=1.0, ls="--", zorder=1,
               label=t["spread"].format(s=_num(100 * res["seed_spread"], lang)))
    for b in bs:
        mk, col = BUDGET_STYLE.get(b, ("D", "#555555"))
        pts = [(x + gate_off[b], 100 * c["dU"][b]) for x, c in zip(xs, res["cells"]) if b in c["dU"]]
        if pts:
            lab = (t["other"] if b in oc else t["budget"]).format(b=_budget(b, lang))
            ax.scatter(*zip(*pts), s=46, marker=mk, color=col, edgecolors=col, lw=1.0, zorder=3, label=lab)
    for b in fb:
        mk, col = BUDGET_STYLE.get(b, ("D", "#555555"))
        pts = [(x + fresh_off[b], 100 * c["fresh_dU"][b]) for x, c in zip(xs, res["cells"]) if b in c["fresh_dU"]]
        if pts:
            ax.scatter(*zip(*pts), s=22, marker=mk, facecolors="white", edgecolors=col, lw=1.1, zorder=3,
                       label=t["fresh"].format(b=_budget(b, lang), s=fs))
    for i, (fam, a, z) in enumerate(spans):
        if i:
            ax.axvline(a - 0.9, color="#dddddd", lw=0.8, zorder=0)
        ax.text((a + z) / 2, 1.01, fam, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels([cell_name(res, c).replace(" ", "\n", 1) for c in res["cells"]], fontsize=7.5)
    ax.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    ax.set_ylabel(t["y"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: _num(v, lang, 0)))
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.12), ncol=3)
    cp = _common_params(res)
    if {"fpr", "kappa", "q"} <= cp.keys():
        leg.set_title(t["note"].format(fpr=_num(float(cp["fpr"]), lang, 2), kappa=_num(float(cp["kappa"]), lang, 2),
                                       q=_num(float(cp["q"]), lang, 2)), prop={"size": 7.5})
        leg._legend_box.align = "left"
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def figure(gate_paths, fresh_paths, out_dir, meta: dict | None = None) -> dict:
    res = collect(gate_paths, fresh_paths)
    if meta is not None:
        res["meta"] = meta
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "budget-du.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (out / "budget-du.md").write_text(markdown(res), encoding="utf-8")
    for lang in ("en", "uk"):
        budget_du_plot(res, out / f"budget-du-{lang}.png", lang)
    return res
