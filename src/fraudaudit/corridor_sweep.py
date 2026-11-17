"""``fraudaudit corridor-sweep``: the Pareto front «guaranteed loss bound vs owner-confirmation burden» of the
soft spending corridor (Task 22) over a pre-registered grid, without retraining.

Grid: the corridor quantile q × the model threshold τ, τ chosen on clean val for each target FPR
(``threshold_at_fpr``; the same weights, another operating point – a derived model directory with a fresh
manifest, ``artifacts.derive_with_tau``). For every τ a reference cell without the corridor is added. Per cell:

* guaranteed bound (``corridor.episode_bound``) over the gate's target episodes: mean / median / max in $ and as a
  share of A, the tighter headroom bound, the share of targets whose bound is below ρ·A;
* empirical: the gate's ensemble (attacks × attack seeds of ``gate.yaml``, same targets/spares/seed) under the
  corridor – worst one-sided upper bound of the uplift, per-run uplifts, the max VER_adv and the bound check;
* cost on the legitimate test stream (``metrics.confirmation_burden``, p cached once per model): share of
  legitimate operations needing confirmation and share of card-days with at least one confirmation.

Pareto: among the corridor cells, a cell is on the front when no other cell is at least as good on every objective
and better on one; objectives (all minimised): mean bound share of A, worst empirical uplift upper bound, share of
card-days with a confirmation (3-D front); the 2-D front (bound share, card-day burden) is marked as well.
"""
from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import corridor as corridor_mod

Q_GRID = (0.95, 0.98, 0.99, 0.995, 0.999)
FPR_GRID = (0.003, 0.01)


def val_tau(df: pd.DataFrame, model, target_fpr: float) -> float:
    """τ at ``target_fpr`` on clean val (``threshold_at_fpr``, as in training): features on the WHOLE stream (the
    windowed aggregates need the history before val), scores of the val rows."""
    from .data import period
    from .metrics import threshold_at_fpr
    from .model.pipeline import features
    m = (period(df) == "val").to_numpy()
    p = np.asarray(model.predict_proba(features(df, model.encoder)[m]), dtype=np.float64)
    return float(threshold_at_fpr(df.loc[m, "is_fraud"].to_numpy(dtype=np.int64), p, target_fpr))


def _model_target_fpr(model_dir):
    meta = json.loads((Path(model_dir) / "model.json").read_text(encoding="utf-8")).get("meta", {})
    v = meta.get("target_fpr")
    return None if v is None else float(v)


def pareto(points: list[tuple]) -> list[bool]:
    """Non-dominated flags (every objective minimised)."""
    out = []
    for i, a in enumerate(points):
        dom = any(all(b[k] <= a[k] for k in range(len(a))) and any(b[k] < a[k] for k in range(len(a)))
                  for j, b in enumerate(points) if j != i)
        out.append(not dom)
    return out


def corridor_sweep(df, model, model_dir, tm, cfg, *, qs=Q_GRID, fprs=FPR_GRID, seed: int = 42,
                   n_jobs: int | None = None, out_dir, threat_path=None, log=None, reference: bool = True) -> dict:
    from .audit import select_episodes
    from .gate import run_ensemble
    from .metrics import confirmation_burden, legit_test_scores
    from .model.artifacts import derive_with_tau, load_artifacts
    from .model.pipeline import FraudModel
    from .threat import contexts_for
    say = log or (lambda m: None)
    if tm.corridor is None:
        raise ValueError("corridor-sweep needs a threat model with a corridor section (the base parameters)")
    t0 = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tm = corridor_mod.prepare(tm, df)
    say(f"multipliers: {json.dumps(tm.corridor.fitted.summary() if tm.corridor.fitted else None)}")
    targets = contexts_for(df, select_episodes(df, ("test",), cfg.spares_period, seed)[0])
    scores = legit_test_scores(df, model, n_jobs=n_jobs, model_dir=model_dir)
    say(f"legit test scores: {sum(len(v) for v in scores.values())} ops, {time.perf_counter() - t0:.0f} s")
    cells = []
    model_fpr = _model_target_fpr(model_dir)
    for fpr in fprs:
        tau = val_tau(df, model, fpr)
        if model_fpr is not None and abs(fpr - model_fpr) < 1e-12 and not math.isclose(tau, float(model.tau),
                                                                                        rel_tol=1e-6):
            raise ValueError(f"tau recomputed at the model's own target_fpr {fpr} is {tau}, model.json has "
                             f"{model.tau}: the val scores differ from training")
        if math.isclose(tau, float(model.tau), rel_tol=1e-6):
            mdir, mdl = Path(model_dir), model
        else:
            mdir = out / "models" / f"fpr-{fpr:g}"
            derive_with_tau(model_dir, mdir, tau, {"target_fpr": float(fpr),
                                                   "note": "corridor-sweep: same weights, tau at target_fpr on val"})
            mdl = FraudModel.from_artifacts(load_artifacts(mdir))
        say(f"target_fpr {fpr}: tau {tau:.6g} (model dir {mdir})")
        for q in ([None] if reference else []) + list(qs):
            c = None if q is None else dataclasses.replace(tm.corridor, q=float(q))
            tm_c = dataclasses.replace(tm, corridor=c)
            t1 = time.perf_counter()
            runs = run_ensemble(df, mdl, tm_c, cfg, seed, n_jobs=n_jobs, model_dir=mdir, threat_path=threat_path)
            cell = {"target_fpr": float(fpr), "tau": tau, "q": None if q is None else float(q),
                    "runs": {k: {"uplift_mean": r.uplift[0], "uplift_upper": r.uplift[2], "ver_adv_mean": r.ver_adv[0],
                                 "ver_clean_mean": r.ver_clean[0], "asr": r.asr} for k, r in runs.items()}}
            worst = max(cell["runs"], key=lambda k: cell["runs"][k]["uplift_upper"])
            first = next(iter(runs.values()))
            cell.update(worst_run=worst, worst_uplift_upper=cell["runs"][worst]["uplift_upper"],
                        mean_uplift=float(np.mean([v["uplift_mean"] for v in cell["runs"].values()])),
                        max_ver_adv=float(max(t["ver_adv"] for r in runs.values() for t in r.per_target)),
                        test_recall=first.clean_quality["test"]["recall"], test_fpr=first.clean_quality["test"]["fpr"])
            if c is not None:
                b = [corridor_mod.episode_bound(c, ctx.state, ctx.episode.start, tm.window_days) for ctx in targets]
                pt = [{"amount": float(ctx.episode.amount), "corridor_bound": x["bound_cents"] / 100,
                       "corridor_headroom": x["headroom_cents"] / 100} for ctx, x in zip(targets, b)]
                cell["bound"] = corridor_mod.bound_stats(pt, tm.rho)
                cell["n_bound_violations"] = int(sum(r.meta["corridor"]["n_bound_violations"] for r in runs.values()))
                bu = confirmation_burden(df, scores, tau, c, n_jobs=n_jobs)
                cell["burden"] = dataclasses.asdict(bu)
            else:
                cell["bound"], cell["n_bound_violations"] = None, None
                cell["burden"] = {"share_confirm_ops": 0.0, "share_confirm_card_days": 0.0}
            cell["wall_seconds"] = round(time.perf_counter() - t1, 1)
            cells.append(cell)
            say(_cell_line(cell))
            (out / "corridor-sweep.partial.json").write_text(json.dumps(cells, indent=2) + "\n", encoding="utf-8")
    grid = [c for c in cells if c["q"] is not None]
    p3 = pareto([(c["bound"]["bound_share"]["mean"], c["worst_uplift_upper"], c["burden"]["share_confirm_card_days"])
                 for c in grid])
    p2 = pareto([(c["bound"]["bound_share"]["mean"], c["burden"]["share_confirm_card_days"]) for c in grid])
    for c, a, b in zip(grid, p3, p2):
        c["pareto_3d"], c["pareto_2d"] = bool(a), bool(b)
    data = {"cells": cells, "q_grid": list(qs), "fpr_grid": list(fprs), "seed": int(seed),
            "corridor_base": tm.corridor.public(),
            "multipliers": None if tm.corridor.fitted is None else tm.corridor.fitted.summary(),
            "multiplier_estimates": [] if tm.corridor.fitted is None else list(tm.corridor.fitted.estimates),
            "gate": {"attacks": cfg.attacks, "attack_seeds": cfg.attack_seeds, "spares_period": cfg.spares_period},
            "n_targets": len(targets), "wall_seconds": round(time.perf_counter() - t0, 1)}
    return data


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _cell_line(c: dict) -> str:
    q = "none" if c["q"] is None else f"{c['q']:g}"
    b = c["bound"]
    bs = "–" if b is None else (f"bound mean {b['bound']['mean']:.0f} $ ({_pct(b['bound_share']['mean'])} of A)")
    return (f"fpr {c['target_fpr']:g} q {q}: worst uplift {_pct(c['worst_uplift_upper'])} ({c['worst_run']}), {bs}, "
            f"confirm ops {_pct(c['burden']['share_confirm_ops'])}, card-days "
            f"{_pct(c['burden']['share_confirm_card_days'])}, {c['wall_seconds']} s")


def summary_markdown(data: dict) -> str:
    lines = ["# Spending corridor sweep", "",
             f"Targets: {data['n_targets']} test episodes; attacks {', '.join(data['gate']['attacks'])} × seeds "
             f"{data['gate']['attack_seeds']}; seed {data['seed']}. Base corridor: `{json.dumps(data['corridor_base'])}`.",
             "", "| target FPR | τ | q | bound mean $ | bound median $ | bound max $ | bound/A mean | bound/A median | "
             "bound/A max | headroom/A mean | bound < ρA | worst uplift (run) | mean uplift | confirm ops | "
             "confirm card-days | test recall | Pareto 3-D / 2-D |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for c in data["cells"]:
        b = c["bound"]
        if b is None:
            bcols = ["–"] * 8
        else:
            bcols = [f"{b['bound']['mean']:.0f}", f"{b['bound']['median']:.0f}", f"{b['bound']['max']:.0f}",
                     _pct(b["bound_share"]["mean"]), _pct(b["bound_share"]["median"]), _pct(b["bound_share"]["max"]),
                     _pct(b["headroom_share"]["mean"]), _pct(b["share_targets_bound_below_rho_a"])]
        par = "–" if c["q"] is None else f"{'yes' if c['pareto_3d'] else 'no'} / {'yes' if c['pareto_2d'] else 'no'}"
        lines.append("| " + " | ".join([f"{c['target_fpr']:g}", f"{c['tau']:.4g}",
                                        "none" if c["q"] is None else f"{c['q']:g}", *bcols,
                                        f"{_pct(c['worst_uplift_upper'])} ({c['worst_run']})", _pct(c["mean_uplift"]),
                                        _pct(c["burden"]["share_confirm_ops"]),
                                        _pct(c["burden"]["share_confirm_card_days"]),
                                        f"{c['test_recall']:.4f}", par]) + " |")
    lines += ["", "Bound = max over the episode window of the 7-day corridor (Proposition, D ≤ W); headroom = the "
              "tighter bound net of the owner's own auto-approved spend still in the window. Uplift = one-sided upper "
              "bound of VER_adv − VER_clean, both legs under the corridor. Burden on the legitimate test stream "
              "(owner confirms; no blocking). Bound violations over all runs: "
              f"{sum(c['n_bound_violations'] or 0 for c in data['cells'])}.", ""]
    sig = [e for e in data["multiplier_estimates"] if e["significant"]]
    lines += ["## Significant multipliers (fit period, population)", ""]
    lines += [f"- {e['kind']} {e['group']}: {e['ratio']:.3f} [{e['lower']:.3f}; {e['upper']:.3f}], {e['n_days']} days"
              for e in sig] or ["- none"]
    return "\n".join(lines) + "\n"


def plot(data: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = [c for c in data["cells"] if c["q"] is not None]
    colors = {0: "#2a78d6", 1: "#eb6834"}
    markers = {0: "o", 1: "s"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    for k, fpr in enumerate(sorted({c["target_fpr"] for c in grid})):
        cs = sorted([c for c in grid if c["target_fpr"] == fpr], key=lambda c: c["q"])
        x = [100 * c["burden"]["share_confirm_card_days"] for c in cs]
        for ax, key in ((axes[0], "bound"), (axes[1], "uplift")):
            y = [100 * (c["bound"]["bound_share"]["mean"] if key == "bound" else c["worst_uplift_upper"]) for c in cs]
            ax.plot(x, y, color=colors[k % 2], lw=2, marker=markers[k % 2], ms=8,
                    label=f"target FPR {fpr:g} (τ {cs[0]['tau']:.3g})")
            for c, xi, yi in zip(cs, x, y):
                on = c["pareto_3d"] if key == "uplift" else c["pareto_2d"]
                ax.annotate(f"q {c['q']:g}" + (" *" if on else ""), (xi, yi), textcoords="offset points",
                            xytext=(5, 5), fontsize=8, color="#444444")
    for c in data["cells"]:
        if c["q"] is None:
            axes[1].axhline(100 * c["worst_uplift_upper"], color="#888888", lw=1, ls="--")
            axes[1].annotate(f"no corridor, FPR {c['target_fpr']:g}", (0, 100 * c["worst_uplift_upper"]),
                             textcoords="offset points", xytext=(4, 3), fontsize=8, color="#666666")
    axes[0].set_ylabel("guaranteed bound, mean share of A (%)")
    axes[1].set_ylabel("worst empirical uplift, one-sided upper (%)")
    for ax in axes:
        ax.set_xlabel("card-days with an owner confirmation (%)")
        ax.grid(color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Bound vs burden (* = 2-D Pareto front)", fontsize=10)
    axes[1].set_title("Empirical uplift vs burden (* = 3-D Pareto front)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_outputs(data: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corridor-sweep.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "corridor-sweep.md").write_text(summary_markdown(data), encoding="utf-8")
    plot(data, out / "corridor-sweep.png")
