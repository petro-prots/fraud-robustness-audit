"""``fraudaudit corridor-sweep``: the Pareto front «guaranteed loss bound vs owner-confirmation burden» of the
soft spending corridor (Tasks 22, 22c) over a pre-registered grid, without retraining.

Grid: corridor variant (ruling R27: ``base``; ``v1:κ`` = daily corridor as a weekly split, C_24h = κ · s(j) · C_7d;
``v2`` = owner-confirmed amounts extend the corridor; ``v1:κ+v2``) × quantile q × the model threshold τ (τ chosen on
clean val for each target FPR, ``threshold_at_fpr``; a derived model directory when it differs from the model's own).
The grid must contain the model's own target FPR (its τ is checked to reproduce the trained τ). For every τ a
reference cell without the corridor is added. The population floor (R28) is refitted for every q. Per cell:

* guaranteed bound M(c) per target in $ (ruling R29): P90 / median / mean, the share of targets with M(c) < ρ·A,
  bound/A as an auxiliary column; with V2 the post-hoc (attack-independent) bound; V1 cells also the curve
  B(h), h = 1..D (Proposition 2) and the empirical cumulative auto-approved amount per 24-hour block of every run;
  how many targets the floor touched;
* empirical: the gate's ensemble (attacks × attack seeds of ``gate.yaml``) under the corridor – worst one-sided
  upper bound of the uplift, per-run uplifts, violations of the bound, the headroom bound and B(h);
* cost on the legitimate test stream (``metrics.confirmation_burden``, p cached once per model): share of
  legitimate operations needing confirmation and share of card-days with at least one confirmation.

Pareto (per τ, every objective minimised): 2-D (P90 of M(c) in $, card-day burden); 3-D adds the worst uplift.
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

Q_GRID = (0.98, 0.99, 0.995)
FPR_GRID = (0.01,)
VARIANTS = ("base", "v1:1.5", "v1:2", "v1:3", "v2", "v1:2+v2")


def parse_variant(name: str) -> dict:
    """``base`` | ``v1:κ`` | ``v2`` | ``v1:κ+v2`` → the corridor fields that set the variant (ruling R27)."""
    d = {"daily_mode": "quantile", "kappa": None, "confirmed_extends": False}
    for part in name.strip().lower().split("+"):
        if part == "base" and name.strip().lower() == "base":
            continue
        if part == "v2":
            d["confirmed_extends"] = True
        elif part.startswith("v1:"):
            try:
                k = float(part[3:])
            except ValueError:
                raise ValueError(f"bad variant {name!r}: kappa must be a number") from None
            d["daily_mode"], d["kappa"] = "weekly_split", k
        else:
            raise ValueError(f"bad variant {name!r}: expected base, v1:κ, v2 or v1:κ+v2")
    return d


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


def corridor_sweep(df, model, model_dir, tm, cfg, *, qs=Q_GRID, fprs=FPR_GRID, variants=VARIANTS, seed: int = 42,
                   n_jobs: int | None = None, out_dir, threat_path=None, log=None, reference: bool = True) -> dict:
    from .audit import select_episodes
    from .gate import run_ensemble
    from .metrics import confirmation_burden, legit_test_scores
    from .model.artifacts import derive_with_tau, load_artifacts
    from .model.pipeline import FraudModel
    from .threat import contexts_for
    say = log or (lambda m: None)
    specs = [(v, parse_variant(v)) for v in variants]  # fail fast on a bad variant name
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
    if model_fpr is None:
        raise ValueError(f"{model_dir}/model.json has no meta.target_fpr: the check that tau at the model's own "
                         "target FPR reproduces the trained tau cannot run")
    if not any(abs(f - model_fpr) < 1e-12 for f in fprs):
        raise ValueError(f"the model's own target_fpr {model_fpr} is not in the FPR grid {list(fprs)}: the check that "
                         "tau at it reproduces the trained tau cannot run")
    for fpr in fprs:
        tau = val_tau(df, model, fpr)
        if abs(fpr - model_fpr) < 1e-12 and not math.isclose(tau, float(model.tau), rel_tol=1e-6):
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
        todo = ([(None, None, None)] if reference else []) + [(vn, vd, q) for vn, vd in specs for q in qs]
        for vname, vfields, q in todo:
            c = None
            if q is not None:
                c = dataclasses.replace(tm.corridor, q=float(q), **vfields)
                c = corridor_mod.prepare(dataclasses.replace(tm, corridor=c), df).corridor  # floor per q, split
            tm_c = dataclasses.replace(tm, corridor=c)
            t1 = time.perf_counter()
            runs = run_ensemble(df, mdl, tm_c, cfg, seed, n_jobs=n_jobs, model_dir=mdir, threat_path=threat_path)
            cell = {"target_fpr": float(fpr), "tau": tau, "q": None if q is None else float(q),
                    "variant": None if q is None else vname,
                    "variant_label": None if c is None else c.variant(),
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
                       "corridor_headroom": x["headroom_cents"] / 100, "b_h": [y / 100 for y in x["b_h_cents"]],
                       "floor_touched": x["floor_touched"]} for ctx, x in zip(targets, b)]
                # M(c): per target the max over the runs of the bound each run reports (post-hoc with V2; the
                # pre-episode bound otherwise, which every run must reproduce)
                by_t = {}
                for r in runs.values():
                    for t in r.per_target:
                        k = (t["card"], t["start"])
                        prev = by_t.get(k)
                        by_t[k] = t if prev is None or t["corridor_bound"] > prev["corridor_bound"] else prev
                for ctx, x, e in zip(targets, b, pt):
                    t = by_t[(int(ctx.episode.card), pd.Timestamp(ctx.episode.start).isoformat())]
                    if not c.confirmed_extends and round(t["corridor_bound"] * 100) != x["bound_cents"]:
                        raise RuntimeError("a run reported a bound other than the pre-episode bound")
                    e.update(corridor_bound=t["corridor_bound"], corridor_headroom=t["corridor_headroom"],
                             b_h=t["b_h"])
                cell["bound"] = corridor_mod.bound_stats(pt, tm.rho)
                cell["n_bound_violations"] = int(sum(r.meta["corridor"]["n_bound_violations"] for r in runs.values()))
                cell["n_headroom_violations"] = int(sum(r.meta["corridor"]["n_headroom_violations"]
                                                        for r in runs.values()))
                cell["n_block_violations"] = int(sum(r.meta["corridor"].get("n_block_violations", 0)
                                                     for r in runs.values()))
                cell["n_floor_touched"] = int(sum(e["floor_touched"] for e in pt))
                cell["floor_fit"] = None if c.floor_fit is None else dataclasses.asdict(c.floor_fit)
                cell["weekly_split"] = None if c.split is None else dataclasses.asdict(c.split)
                cell["empirical_blocks"] = _empirical_blocks(runs)
                bu = confirmation_burden(df, scores, tau, c, n_jobs=n_jobs)
                cell["burden"] = dataclasses.asdict(bu)
            else:
                cell["bound"], cell["n_bound_violations"], cell["n_headroom_violations"] = None, None, None
                cell["n_block_violations"] = None
                cell["burden"] = {"share_confirm_ops": 0.0, "share_confirm_card_days": 0.0}
            cell["wall_seconds"] = round(time.perf_counter() - t1, 1)
            cells.append(cell)
            say(_cell_line(cell))
            (out / "corridor-sweep.partial.json").write_text(json.dumps(cells, indent=2) + "\n", encoding="utf-8")
    grid = [c for c in cells if c["q"] is not None]
    for fpr in fprs:  # fronts per τ (R29 axes: P90 of M(c) in $ vs card-day burden; 3-D adds the worst uplift)
        g = [c for c in grid if abs(c["target_fpr"] - fpr) < 1e-12]
        p3 = pareto([(c["bound"]["bound"]["q90"], c["worst_uplift_upper"], c["burden"]["share_confirm_card_days"])
                     for c in g])
        p2 = pareto([(c["bound"]["bound"]["q90"], c["burden"]["share_confirm_card_days"]) for c in g])
        for c, a, b in zip(g, p3, p2):
            c["pareto_3d"], c["pareto_2d"] = bool(a), bool(b)
    data = {"cells": cells, "q_grid": list(qs), "fpr_grid": list(fprs), "variants": list(variants), "seed": int(seed),
            "corridor_base": tm.corridor.public(),
            "multipliers": None if tm.corridor.fitted is None else tm.corridor.fitted.summary(),
            "multiplier_estimates": [] if tm.corridor.fitted is None else list(tm.corridor.fitted.estimates),
            "gate": {"attacks": cfg.attacks, "attack_seeds": cfg.attack_seeds, "spares_period": cfg.spares_period},
            "n_targets": len(targets), "wall_seconds": round(time.perf_counter() - t0, 1)}
    return data


def _empirical_blocks(runs: dict) -> dict:
    """Proposition 2 in the runs: per h the cumulative auto-approved attacker amount after h blocks over every
    run × target (mean / max in $, max share of B(h)) and the number of (run, target, h) above B(h)."""
    cum = np.array([t["auto_by_block"] for r in runs.values() for t in r.per_target], dtype=np.float64) / 100
    bh = np.array([t["b_h"] for r in runs.values() for t in r.per_target], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(bh > 0, cum / bh, 0.0)
    return {"mean": cum.mean(axis=0).tolist(), "max": cum.max(axis=0).tolist(),
            "max_share_of_b_h": ratio.max(axis=0).tolist(),
            "n_over": int((np.round(cum * 100) > np.round(bh * 100)).sum()), "n": int(cum.shape[0])}


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f} %"


def _cell_line(c: dict) -> str:
    q = "none" if c["q"] is None else f"{c.get('variant_label') or 'base'} q {c['q']:g}"
    b = c["bound"]
    bs = "–" if b is None else (f"M(c) P90 {b['bound']['q90']:.0f} $ / median {b['bound']['median']:.0f} $ / "
                                f"mean {b['bound']['mean']:.0f} $, < ρA {_pct(b['share_targets_bound_below_rho_a'])}")
    return (f"fpr {c['target_fpr']:g} {q}: worst uplift {_pct(c['worst_uplift_upper'])} ({c['worst_run']}), {bs}, "
            f"confirm ops {_pct(c['burden']['share_confirm_ops'])}, card-days "
            f"{_pct(c['burden']['share_confirm_card_days'])}, {c['wall_seconds']} s")


def summary_markdown(data: dict) -> str:
    lines = ["# Spending corridor sweep", "",
             f"Targets: {data['n_targets']} test episodes; attacks {', '.join(data['gate']['attacks'])} × seeds "
             f"{data['gate']['attack_seeds']}; seed {data['seed']}. Base corridor: `{json.dumps(data['corridor_base'])}`.",
             "", "M(c) = the guaranteed bound of target c in $ (max over the episode window of C_7d; with V2 the "
             "post-hoc, attack-independent max of C'_7d). Bound share of A is auxiliary.", "",
             "| target FPR | τ | variant | q | M(c) P90 $ | M(c) median $ | M(c) mean $ | M < ρA | bound/A median "
             "(aux) | worst uplift (run) | mean uplift | confirm ops | confirm card-days | floor touched | "
             "violations bound / headroom / blocks | Pareto 2-D / 3-D |",
             "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|"]
    for c in data["cells"]:
        b = c["bound"]
        if b is None:
            bcols = ["–"] * 5
        else:
            bcols = [f"{b['bound']['q90']:.0f}", f"{b['bound']['median']:.0f}", f"{b['bound']['mean']:.0f}",
                     _pct(b["share_targets_bound_below_rho_a"]), _pct(b["bound_share"]["median"])]
        par = "–" if c["q"] is None else f"{'yes' if c['pareto_2d'] else 'no'} / {'yes' if c['pareto_3d'] else 'no'}"
        viol = "–" if c["q"] is None else (f"{c['n_bound_violations']} / {c['n_headroom_violations']} / "
                                           f"{c['n_block_violations']}")
        lines.append("| " + " | ".join([f"{c['target_fpr']:g}", f"{c['tau']:.4g}",
                                        "none" if c["q"] is None else c["variant_label"],
                                        "–" if c["q"] is None else f"{c['q']:g}", *bcols,
                                        f"{_pct(c['worst_uplift_upper'])} ({c['worst_run']})", _pct(c["mean_uplift"]),
                                        _pct(c["burden"]["share_confirm_ops"]),
                                        _pct(c["burden"]["share_confirm_card_days"]),
                                        "–" if c["q"] is None else str(c["n_floor_touched"]), viol, par]) + " |")
    curves = [c for c in data["cells"] if c["q"] is not None and c["bound"].get("b_h")
              and str(c.get("variant", "")).startswith("v1")]
    if curves:
        n = len(curves[0]["bound"]["b_h"]["median"])
        lines += ["", "## B(h): guaranteed loss after h days (Proposition 2), V1 cells", "",
                  "B(h) = min(max C_7d, Σ_{i<=h} max_{B_i} C_24h) per target; median $ and median / q90 of B(h)/A. "
                  "Empirical: max over all runs × targets of the cumulative auto-approved attacker amount after h "
                  "blocks, $ (and its max share of B(h)).", "",
                  "| variant | q | " + " | ".join(f"h={h}" for h in range(1, n + 1)) + " |",
                  "|---|---:|" + "---|" * n]
        for c in curves:
            bh, bs, e = c["bound"]["b_h"], c["bound"]["b_h_share"], c["empirical_blocks"]
            lines.append(f"| {c['variant_label']} | {c['q']:g} | " + " | ".join(
                f"{bh['median'][h]:.0f} $ ({_pct(bs['median'][h])} / {_pct(bs['q90'][h])}); emp. max "
                f"{e['max'][h]:.0f} $ ({_pct(e['max_share_of_b_h'][h])})" for h in range(n)) + " |")
    ff = sorted({(c["q"], c["floor_fit"]["floor_24h"], c["floor_fit"]["floor_7d"], c["floor_fit"]["n_cards"])
                 for c in data["cells"] if c.get("floor_fit")})
    if ff:
        lines += ["", "## Population floor (R28, new cards only)", ""]
        lines += [f"- q {q:g}: {f24:.2f} $ (24 h) / {f7:.2f} $ (7 d), P10 over {n} established cards"
                  for q, f24, f7, n in ff]
    ws = next((c["weekly_split"] for c in data["cells"] if c.get("weekly_split")), None)
    if ws:
        lines += ["", "Weekly split s(j), Monday..Sunday: " + ", ".join(f"{x:.4f}" for x in ws["shares"])]
    lines += ["", "Uplift = one-sided upper bound of VER_adv − VER_clean, both legs under the corridor. Burden on the "
              "legitimate test stream (owner confirms; no blocking). Bound violations over all runs: "
              f"{sum(c['n_bound_violations'] or 0 for c in data['cells'])}; headroom-bound violations: "
              f"{sum(c.get('n_headroom_violations') or 0 for c in data['cells'])}; block (Proposition 2) violations: "
              f"{sum(c.get('n_block_violations') or 0 for c in data['cells'])}. The corridor profile is lagged by "
              "7 days and label-free (ruling R26).", ""]
    sig = [e for e in data["multiplier_estimates"] if e["significant"]]
    lines += ["## Significant multipliers (fit period, population)", ""]
    lines += [f"- {e['kind']} {e['group']}: {e['ratio']:.3f} [{e['lower']:.3f}; {e['upper']:.3f}], {e['n_days']} days"
              for e in sig] or ["- none"]
    return "\n".join(lines) + "\n"


def plot(data: dict, path: Path) -> None:
    """Pareto front (R29): P90 of M(c) in $ vs card-day burden; right panel: worst empirical uplift vs burden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    grid = [c for c in data["cells"] if c["q"] is not None]
    names = list(dict.fromkeys(c["variant_label"] for c in grid))
    palette = ["#2a78d6", "#eb6834", "#2e9e5b", "#8a4fd1", "#c23b6b", "#7a7a7a"]
    markers = ["o", "s", "^", "D", "v", "P"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for k, name in enumerate(names):
        cs = sorted([c for c in grid if c["variant_label"] == name], key=lambda c: c["q"])
        x = [100 * c["burden"]["share_confirm_card_days"] for c in cs]
        for ax, key in ((axes[0], "bound"), (axes[1], "uplift")):
            y = [c["bound"]["bound"]["q90"] if key == "bound" else 100 * c["worst_uplift_upper"] for c in cs]
            ax.plot(x, y, color=palette[k % 6], lw=1.6, marker=markers[k % 6], ms=7, label=name)
            for c, xi, yi in zip(cs, x, y):
                on = c["pareto_2d"] if key == "bound" else c["pareto_3d"]
                ax.annotate(f"{c['q']:g}" + (" *" if on else ""), (xi, yi), textcoords="offset points",
                            xytext=(5, 3), fontsize=7, color="#444444")
    for c in data["cells"]:
        if c["q"] is None:
            axes[1].axhline(100 * c["worst_uplift_upper"], color="#888888", lw=1, ls="--")
            axes[1].annotate(f"no corridor: {100 * c['worst_uplift_upper']:.1f} %",
                             (axes[1].get_xlim()[0], 100 * c["worst_uplift_upper"]), textcoords="offset points",
                             xytext=(4, 3), fontsize=8, color="#666666")
    axes[0].set_ylabel("guaranteed bound M(c), P90 over targets ($)")
    axes[1].set_ylabel("worst empirical uplift, one-sided upper (%)")
    for ax in axes:
        ax.set_xlabel("card-days with an owner confirmation (%)")
        ax.grid(color="#e5e5e5", lw=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("P90 bound vs burden (labels q; * = 2-D Pareto front)", fontsize=10)
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
