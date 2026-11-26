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
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import corridor as corridor_mod
from .sweep_cells import (check_own_tau, model_at_tau, model_target_fpr, run_cell, tau_at,  # noqa: F401
                          val_scores, val_tau)

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
    from .metrics import confirmation_burden, legit_test_scores
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
    model_fpr = model_target_fpr(model_dir)
    if model_fpr is None:
        raise ValueError(f"{model_dir}/model.json has no meta.target_fpr: the check that tau at the model's own "
                         "target FPR reproduces the trained tau cannot run")
    if not any(abs(f - model_fpr) < 1e-12 for f in fprs):
        raise ValueError(f"the model's own target_fpr {model_fpr} is not in the FPR grid {list(fprs)}: the check that "
                         "tau at it reproduces the trained tau cannot run")
    y_val, p_val = val_scores(df, model)
    for fpr in fprs:
        tau = tau_at(y_val, p_val, fpr)
        if abs(fpr - model_fpr) < 1e-12:
            check_own_tau(model_dir, model, tau, fpr)
        mdir, mdl = model_at_tau(model_dir, model, tau, fpr, out / "models" / f"fpr-{fpr:g}",
                                 "corridor-sweep: same weights, tau at target_fpr on val")
        say(f"target_fpr {fpr}: tau {tau:.6g} (model dir {mdir})")
        todo = ([(None, None, None)] if reference else []) + [(vn, vd, q) for vn, vd in specs for q in qs]
        for vname, vfields, q in todo:
            c = None
            if q is not None:
                c = dataclasses.replace(tm.corridor, q=float(q), **vfields)
                c = corridor_mod.prepare(dataclasses.replace(tm, corridor=c), df).corridor  # floor per q, split
            tm_c = dataclasses.replace(tm, corridor=c)
            t1 = time.perf_counter()
            rec, runs = run_cell(df, mdl, mdir, tm_c, cfg, seed, n_jobs=n_jobs, threat_path=threat_path)
            cell = {"target_fpr": float(fpr), "tau": tau, "q": None if q is None else float(q),
                    "variant": None if q is None else vname,
                    "variant_label": None if c is None else c.variant(), **rec}
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


# --- review I1: the owner-replay post-hoc bound M'(c) of V2 cells ------------------------------------------------


def owner_window_scores(df: pd.DataFrame, model_or_serving, contexts, window_days: int) -> list[dict]:
    """Per target context: the owner's stream – the card's legitimate operations (``is_fraud == 0``; the label only
    selects the stream, as in the burden replay) in ``[t0 − 7 d, t0 + D)`` without the episode rows – the state at
    ``t0 − 7 d`` (``from_history``, history counted as auto-approved) and the model's p of every operation (every
    operation is pushed, so p does not depend on the corridor; computed once per model)."""
    from .serving.core import Serving
    from .serving.state import CardState, Tx
    serving = model_or_serving if hasattr(model_or_serving, "decide") else Serving(model_or_serving)
    W = pd.Timedelta(days=7)
    groups = dict(tuple(df[df["cc_num"].isin({c.episode.card for c in contexts})].groupby("cc_num", sort=False)))
    out = []
    for ctx in contexts:
        e = ctx.episode
        g = groups[e.card]
        g = g[~g["row"].isin(e.rows)]
        t0 = pd.Timestamp(e.start)
        lo, hi = t0 - W, t0 + pd.Timedelta(days=int(window_days))
        own = g[(g["ts"] >= lo) & (g["ts"] < hi) & (g["is_fraud"] == 0)].sort_values(["ts", "row"], kind="stable")
        state = CardState.from_history(ctx.profile, g[g["ts"] < lo], at=lo)
        txs = [Tx(ts=pd.Timestamp(r.ts), amt=float(r.amt), merchant=str(r.merchant), category=str(r.category),
                  merch_lat=float(r.merch_lat), merch_long=float(r.merch_long)) for r in own.itertuples(index=False)]
        st = state.copy()
        p = np.empty(len(txs), dtype=np.float64)
        for i, x in enumerate(txs):
            p[i] = serving.decide(st, x).p
            st.push(x)
        out.append({"state": state, "txs": txs, "p": p})
    return out


def owner_posthoc_bounds(contexts, data: list[dict], corridor, tau: float, window_days: int) -> list[dict]:
    """M'(c) per target for a V2 corridor (``confirmed_extends``): the owner's stream of ``owner_window_scores``
    replayed through the bank (``owner=True``: the owner confirms when asked; no blocking; threshold ``tau``), then
    ``posthoc_bound`` of the audit's pre-episode bound with the owner's confirmations. Owner only: no attacker
    operation is in this replay (with an attack, the attacker's auto-approved spend can push more owner operations
    into confirmation, so this is not an upper bound on the post-hoc bound under attack)."""
    from types import SimpleNamespace

    from .episode import _Bank
    from .metrics import _SeqServing, _TauServing
    if not corridor.confirmed_extends:
        raise ValueError("owner_posthoc_bounds needs a V2 corridor (confirmed_extends: true)")
    out = []
    for ctx, d in zip(contexts, data):
        b = corridor_mod.episode_bound(corridor, ctx.state, ctx.episode.start, window_days)
        bank = _Bank(_TauServing(_SeqServing(d["p"]), tau), SimpleNamespace(episode=SimpleNamespace(amount=1.0),
                                                                           state=d["state"]),
                     None, 1.0, None, corridor, owner=True)
        for x in d["txs"]:
            bank.submit(x)
        conf = d["state"].confirmed_ops() + tuple(bank.confirmed_ops)
        p = corridor_mod.posthoc_bound(b, conf, ctx.episode.start, window_days)
        t0 = pd.Timestamp(ctx.episode.start).value
        out.append({"card": int(ctx.episode.card), "start": pd.Timestamp(ctx.episode.start).isoformat(),
                    "amount": float(ctx.episode.amount), "bound_pre": b["bound_cents"] / 100,
                    "bound_post": p["bound_cents"] / 100, "b_h_post": [x / 100 for x in p["b_h_cents"]],
                    "n_owner_ops": len(d["txs"]), "n_owner_ops_in_window": sum(x.ts.value >= t0 for x in d["txs"]),
                    "owner_in_window_amount": float(sum(x.amt for x in d["txs"] if x.ts.value >= t0)),
                    "n_owner_confirms": len(bank.confirmed_ops),
                    "owner_confirmed_amount": sum(c for _, c in bank.confirmed_ops) / 100})
    return out


def _stats(x) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"q90": float(np.quantile(x, 0.9)), "median": float(np.median(x)), "mean": float(x.mean()),
            "max": float(x.max())}


def owner_posthoc_sweep(df, model, tm, cfg, *, qs=Q_GRID, variants=("v2", "v1:2+v2"), seed: int = 42,
                        log=None) -> dict:
    """``owner_posthoc_bounds`` over the V2 cells of the grid (same targets as ``corridor_sweep``; the model at its
    own τ). Per cell: M(c) pre-episode and M'(c) post-hoc (P90 / median / mean / max $, share < ρA), owner
    confirmations, targets with M' > M."""
    from .audit import select_episodes
    from .threat import contexts_for
    say = log or (lambda m: None)
    specs = [(v, parse_variant(v)) for v in variants]
    if not all(f["confirmed_extends"] for _, f in specs):
        raise ValueError("corridor-posthoc measures V2 cells only (variants with +v2 or v2)")
    targets = contexts_for(df, select_episodes(df, ("test",), cfg.spares_period, seed)[0])
    t1 = time.perf_counter()
    data = owner_window_scores(df, model, targets, tm.window_days)
    say(f"owner streams of {len(targets)} targets: {sum(len(d['txs']) for d in data)} ops, "
        f"{time.perf_counter() - t1:.0f} s")
    cells = []
    for vname, vfields in specs:
        for q in qs:
            c = dataclasses.replace(tm.corridor, q=float(q), **vfields)
            c = corridor_mod.prepare(dataclasses.replace(tm, corridor=c), df).corridor
            per = owner_posthoc_bounds(targets, data, c, float(model.tau), tm.window_days)
            pre = np.array([r["bound_pre"] for r in per])
            post = np.array([r["bound_post"] for r in per])
            a = np.array([r["amount"] for r in per])
            cell = {"variant": vname, "variant_label": c.variant(), "q": float(q), "tau": float(model.tau),
                    "m_pre": _stats(pre), "m_post": _stats(post),
                    "share_pre_below_rho_a": float(np.mean(pre < tm.rho * a)),
                    "share_post_below_rho_a": float(np.mean(post < tm.rho * a)),
                    "n_targets_lifted": int((post > pre).sum()),
                    "lift": _stats(post - pre),
                    "n_owner_confirms": int(sum(r["n_owner_confirms"] for r in per)),
                    "n_targets_with_owner_ops_in_window": int(sum(r["n_owner_ops_in_window"] > 0 for r in per)),
                    "owner_in_window_amount": _stats([r["owner_in_window_amount"] for r in per]),
                    "per_target": per}
            cells.append(cell)
            say(f"{c.variant()} q {q:g}: M P90 {cell['m_pre']['q90']:.0f} → M' {cell['m_post']['q90']:.0f} $, "
                f"median {cell['m_pre']['median']:.0f} → {cell['m_post']['median']:.0f} $, lifted "
                f"{cell['n_targets_lifted']} targets, {cell['n_owner_confirms']} owner confirmations")
    return {"cells": cells, "n_targets": len(targets), "seed": int(seed), "variants": list(variants),
            "q_grid": list(qs), "corridor_base": tm.corridor.public()}


def merge_posthoc(sweep: dict, post: dict) -> dict:
    """A copy of the sweep cells where every V2 cell measured by ``owner_posthoc_sweep`` uses M'(c) P90 for the 2-D
    and 3-D fronts (``m_p90_front``; other cells keep M(c) P90), with the fronts recomputed per τ
    (``pareto_2d_posthoc``, ``pareto_3d_posthoc``)."""
    import copy
    pm = {(c["variant"], c["q"]): c for c in post["cells"]}
    cells = copy.deepcopy(sweep["cells"])
    for c in cells:
        if c["q"] is None:
            continue
        m = pm.get((c["variant"], c["q"]))
        c["m_p90_front"] = m["m_post"]["q90"] if m else c["bound"]["bound"]["q90"]
        if m:
            c["m_post"], c["share_post_below_rho_a"] = m["m_post"], m["share_post_below_rho_a"]
    for fpr in sorted({c["target_fpr"] for c in cells if c["q"] is not None}):
        g = [c for c in cells if c["q"] is not None and abs(c["target_fpr"] - fpr) < 1e-12]
        p2 = pareto([(c["m_p90_front"], c["burden"]["share_confirm_card_days"]) for c in g])
        p3 = pareto([(c["m_p90_front"], c["worst_uplift_upper"], c["burden"]["share_confirm_card_days"]) for c in g])
        for c, a, b in zip(g, p2, p3):
            c["pareto_2d_posthoc"], c["pareto_3d_posthoc"] = bool(a), bool(b)
    return {"cells": cells}


def posthoc_markdown(post: dict, merged: dict | None = None) -> str:
    lines = ["# V2 post-hoc bound with the owner's real operations (review I1)", "",
             f"{post['n_targets']} test targets; owner stream = the card's legitimate operations in [t0 − 7 d, t0 + 7 d) "
             "(episode rows excluded) replayed with owner confirmations at the model's own τ, no attacker operations. "
             "M = pre-episode bound, M' = post-hoc max_t C'_7d over the episode window. $ values.", "",
             "| variant | q | M P90 / median / mean | M' P90 / median / mean | M < ρA | M' < ρA | targets lifted | "
             "lift median / max | owner confirmations | targets with owner ops in window |",
             "|---|---:|---|---|---:|---:|---:|---|---:|---:|"]
    for c in post["cells"]:
        a, b = c["m_pre"], c["m_post"]
        lines.append(f"| {c['variant_label']} | {c['q']:g} | {a['q90']:.0f} / {a['median']:.0f} / {a['mean']:.0f} | "
                     f"{b['q90']:.0f} / {b['median']:.0f} / {b['mean']:.0f} | {_pct(c['share_pre_below_rho_a'])} | "
                     f"{_pct(c['share_post_below_rho_a'])} | {c['n_targets_lifted']} | {c['lift']['median']:.0f} / "
                     f"{c['lift']['max']:.0f} | {c['n_owner_confirms']} | {c['n_targets_with_owner_ops_in_window']} |")
    if merged:
        lines += ["", "## Pareto with M'(c) for the V2 cells (FPR fronts per τ)", "",
                  "| variant | q | P90 $ used | confirm card-days | worst uplift | 2-D (sweep → post-hoc) | "
                  "3-D (sweep → post-hoc) |", "|---|---:|---:|---:|---:|---|---|"]
        yn = lambda x: "yes" if x else "no"  # noqa: E731
        for c in merged["cells"]:
            if c["q"] is None:
                continue
            lines.append(f"| {c['variant_label']} | {c['q']:g} | {c['m_p90_front']:.0f} | "
                         f"{_pct(c['burden']['share_confirm_card_days'])} | {_pct(c['worst_uplift_upper'])} | "
                         f"{yn(c['pareto_2d'])} → {yn(c['pareto_2d_posthoc'])} | "
                         f"{yn(c['pareto_3d'])} → {yn(c['pareto_3d_posthoc'])} |")
    return "\n".join(lines) + "\n"


def write_outputs(data: dict, out_dir) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "corridor-sweep.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "corridor-sweep.md").write_text(summary_markdown(data), encoding="utf-8")
    plot(data, out / "corridor-sweep.png")
