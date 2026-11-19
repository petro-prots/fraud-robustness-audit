import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from fraudaudit import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudaudit",
        description="Episode-level adversarial robustness audit of a fraud-detection model.",
    )
    parser.add_argument("--version", action="version", version=f"fraudaudit {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    train = sub.add_parser("train", help="train the baseline model (bachelor architecture, bachelor+hour-calendar features) and save its artifacts")
    train.add_argument("--data", type=Path, default=Path("data"), help="directory with fraudTrain.csv, fraudTest.csv")
    train.add_argument("--out", type=Path, default=Path("models"), help="output directory for the artifacts")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--epochs", type=int, default=20, help="20 as in the bachelor listing; lower only for probes")
    train.add_argument("--sample-frac", type=float, default=None,
                       help="subsample fit rows (timing probes only)")

    skew = sub.add_parser("skew", help="check that the batch and serving feature paths agree")
    skew.add_argument("--data", type=Path, default=Path("data"))
    skew.add_argument("--model", type=Path, default=Path("models"), help="artifact directory")
    skew.add_argument("--cards", type=int, default=50, help="number of sampled cards")
    skew.add_argument("--seed", type=int, default=42)
    skew.add_argument("--atol", type=float, default=1e-4,
                      help="tolerance of the float aggregate columns only (mean_amt_*, sum_1h, amt_ratio_30d)")
    skew.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")
    skew.add_argument("--out", type=Path, default=None, help="also write the report as JSON (for audit --skew-report)")

    audit = sub.add_parser("audit", help="episode audit and quality gate: every attack x seed of gate.yaml; "
                                         "exit 1 if a check fails")
    audit.add_argument("--data", type=Path, default=Path("data"))
    audit.add_argument("--model", type=Path, default=Path("models"), help="artifact directory")
    audit.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    audit.add_argument("--gate", type=Path, default=Path("gate.yaml"),
                       help="gate rules: thresholds, attacks and attack seeds (the worst run decides)")
    audit.add_argument("--policy", type=Path, default=None,
                       help="bank policy file (policies/*.yaml: max_declines, corridor κ and q) overlaid on the threat "
                            "model; default: the threat file's own policy")
    audit.add_argument("--delta", type=float, default=None,
                       help="the bar δ′ (share of the episode amount) instead of the gate file's δ; the report names "
                            "the source")
    audit.add_argument("--skew-report", type=Path, default=None,
                       help="JSON of `fraudaudit skew --out`; adds the skew check")
    audit.add_argument("--single-threat", type=Path, default=Path("threat-models/single-transaction.yaml"),
                       help="threat model of the informational single-transaction baseline (conventional audit)")
    audit.add_argument("--no-baseline", action="store_true",
                       help="skip the single-transaction baseline comparison (it never changes the verdict)")
    audit.add_argument("--out", type=Path, default=Path("out"), help="writes audit.json and summary.md")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")

    cal = sub.add_parser("calibrate", help="resolution of the gate on the baseline model: the gate's ensemble, "
                                           "uplift mean / upper bound / SE per run and the verdict for several δ; "
                                           "never changes gate.yaml")
    cal.add_argument("--data", type=Path, default=Path("data"))
    cal.add_argument("--model", type=Path, default=Path("models"), help="artifact directory (baseline model)")
    cal.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    cal.add_argument("--gate", type=Path, default=Path("gate.yaml"),
                     help="attacks, attack seeds, confidence, one_sided, spares_period (read only)")
    cal.add_argument("--out", type=Path, default=Path("out/calibration"),
                     help="writes calibration.json and calibration.md")
    cal.add_argument("--deltas", type=_deltas, default=[0.02, 0.05, 0.10, 0.15, 0.20],
                     help="comma-separated δ values for the verdict table (default 0.02,0.05,0.10,0.15,0.20)")
    cal.add_argument("--seed", type=int, default=42)
    cal.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")

    hard = sub.add_parser("harden", help="round-based adversarial retraining (ES and random-search examples on the "
                                          "H half of the fit episodes); writes every round's model and "
                                          "harden-report.json/.md")
    hard.add_argument("--data", type=Path, default=Path("data"))
    hard.add_argument("--model", type=Path, default=Path("models"), help="input model (artifact directory)")
    hard.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    hard.add_argument("--gate", type=Path, default=Path("gate.yaml"),
                      help="δ (early stop), attacks of the val mini-audit, confidence, one_sided; read only")
    hard.add_argument("--out", type=Path, default=Path("models-harden"),
                      help="round-{r}/ model directories and harden-report.json/.md")
    hard.add_argument("--rounds", type=int, default=5, help="at most this many rounds (early stop at val ≤ δ)")
    hard.add_argument("--seed", type=int, default=42)
    hard.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")
    hard.add_argument("--epochs", type=int, default=20, help="20 as in train; lower only for probes")
    hard.add_argument("--budget-ops", type=int, default=20000,
                      help="reconnaissance budget per generation unit, ES and random alike (harden_budget_ops)")
    hard.add_argument("--share-among-fraud", type=float, default=0.5,
                      help="adversarial rows / all fraud rows of fit before balancing (design §11)")
    hard.add_argument("--no-baseline-audit", action="store_true", help="skip the round-0 mini-audit of the input model")
    hard.add_argument("--gen-attacks", type=_names, default=["es", "random"],
                      help="comma-separated attacks that generate hardening data (default es,random); the held-out "
                           "stepwise attack is refused")

    cs = sub.add_parser("corridor-sweep", help="Pareto front of the soft spending corridor: guaranteed loss bound vs "
                                              "owner-confirmation burden over q x tau (no retraining); writes "
                                              "corridor-sweep.json/.md/.png")
    cs.add_argument("--data", type=Path, default=Path("data"))
    cs.add_argument("--model", type=Path, default=Path("models"), help="artifact directory")
    cs.add_argument("--threat", type=Path, default=Path("threat-models/realistic-corridor.yaml"),
                    help="threat model with the base corridor section (q is replaced by the grid)")
    cs.add_argument("--gate", type=Path, default=Path("gate.yaml"), help="attacks, attack seeds, spares (read only)")
    cs.add_argument("--q", type=_deltas, default=[0.98, 0.99, 0.995],
                    help="comma-separated corridor quantiles (default 0.98,0.99,0.995)")
    cs.add_argument("--target-fpr", type=_deltas, default=[0.01],
                    help="comma-separated target FPRs, must include the model's own; tau of each from clean val "
                         "(default 0.01)")
    cs.add_argument("--variants", type=_names, default=["base", "v1:1.5", "v1:2", "v1:3", "v2", "v1:2+v2"],
                    help="comma-separated corridor variants: base, v1:KAPPA, v2, v1:KAPPA+v2 "
                         "(default base,v1:1.5,v1:2,v1:3,v2,v1:2+v2)")
    cs.add_argument("--no-reference", action="store_true", help="skip the cells without a corridor")
    cs.add_argument("--out", type=Path, default=Path("out/corridor-sweep"))
    cs.add_argument("--seed", type=int, default=42)
    cs.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")

    ph = sub.add_parser("corridor-posthoc", help="V2 cells: post-hoc bound M'(c) with the owner's real operations "
                                                "replayed around each target episode (review I1); optional merge "
                                                "with a corridor-sweep.json into the post-hoc Pareto front")
    ph.add_argument("--data", type=Path, default=Path("data"))
    ph.add_argument("--model", type=Path, default=Path("models"), help="artifact directory (its own tau)")
    ph.add_argument("--threat", type=Path, default=Path("threat-models/realistic-corridor.yaml"))
    ph.add_argument("--gate", type=Path, default=Path("gate.yaml"), help="spares period (target selection)")
    ph.add_argument("--q", type=_deltas, default=[0.98, 0.99, 0.995])
    ph.add_argument("--variants", type=_names, default=["v2", "v1:2+v2"])
    ph.add_argument("--sweep", type=Path, default=None, help="corridor-sweep.json to merge")
    ph.add_argument("--out", type=Path, default=Path("out/corridor-posthoc"))
    ph.add_argument("--seed", type=int, default=42)

    ps = sub.add_parser("policy-sweep", help="bank-policy grid (Task 19): tau x bank reaction x corridor on one or "
                                            "more models; U, U$, clean quality and friction per cell; resumable "
                                            "<out>/<label>/policy-sweep.json/.md")
    ps.add_argument("--data", type=Path, default=Path("data"))
    ps.add_argument("--model", type=Path, action="append", required=True, help="artifact directory (repeatable)")
    ps.add_argument("--label", action="append", required=True, help="label of each --model, in the same order")
    ps.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"),
                    help="threat model of the cells (max_declines / tau_after_decline / corridor are set per cell)")
    ps.add_argument("--corridor-threat", type=Path, default=Path("threat-models/realistic-corridor-v1v2.yaml"),
                    help="its corridor section is the corridor of the corridor-on cells")
    ps.add_argument("--gate", type=Path, default=Path("gate.yaml"), help="attacks, attack seeds, δ (read only)")
    ps.add_argument("--target-fpr", type=_deltas, default=[0.005, 0.01, 0.02],
                    help="comma-separated target FPRs on clean val (default 0.005,0.01,0.02)")
    ps.add_argument("--reactions", type=_names, default=["md1", "md3", "md3+esc"],
                    help="comma-separated bank reactions md<N> | mdinf, optionally +esc (default md1,md3,md3+esc)")
    ps.add_argument("--esc-fpr", type=float, default=0.05, help="val FPR of the escalation threshold (default 0.05)")
    ps.add_argument("--corridor", type=_names, default=["off", "on"], help="comma-separated off,on (default both)")
    ps.add_argument("--corridor-kappa", type=_deltas, default=None,
                    help="comma-separated kappa of the V1 daily split for the corridor-on cells (Task 19b variants; "
                         "default: the corridor threat file's)")
    ps.add_argument("--corridor-q", type=_deltas, default=None,
                    help="comma-separated corridor quantiles q for the corridor-on cells (default: the file's)")
    ps.add_argument("--recon-budget-ops", type=int, default=None,
                    help="reconnaissance budget of the gate's attacker (operations on spare cards) instead of the "
                         "threat files' value, for both --threat and --corridor-threat (R33 sensitivity check; "
                         "delta, thresholds and the rest of the threat model stay)")
    ps.add_argument("--r19", action="store_true", help="also the R19 cells md2 and mdinf at the own tau, corridor off")
    ps.add_argument("--role", choices=["grid", "confirmation"], default="grid",
                    help="confirmation: a confirmation run with fresh seeds (Task 19 T2: --gate gate-seeds-456.yaml; "
                         "Task 19b front: --gate gate-seeds-4-9.yaml)")
    stage = ps.add_mutually_exclusive_group()
    stage.add_argument("--no-friction", action="store_true", help="ensemble cells only")
    stage.add_argument("--friction-only", action="store_true", help="friction of the cells already in the JSON only")
    ps.add_argument("--out", type=Path, default=Path("out/t19"), help="writes <out>/<label>/policy-sweep.json/.md")
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")

    pm = sub.add_parser("policy-summary", help="Task 19 summary: Pareto fronts, T1 / T2 / policy-vs-model "
                                              "hypotheses, R19 bias and the T2 confirmation plan")
    pm.add_argument("sweeps", type=Path, nargs="+", help="policy-sweep.json files (grid and confirmation)")
    pm.add_argument("--out", type=Path, default=Path("out/t19/summary"))
    pm.add_argument("--resolution", type=float, default=0.036, help="R of T1 (default 0.036, M3 calibration)")
    pm.add_argument("--confirm-rule", choices=["t2", "front"], default="t2",
                    help="t2: Task 19 green-cell confirmation plan (confirmations.tsv); front: Task 19b – the Pareto "
                         "front confirmed on fresh seeds, margin X, risk-appetite curve (confirmations-front.tsv)")
    pm.add_argument("--merge-commits", type=_names, default=None,
                    help="comma-separated git SHAs whose cells may be merged (Task 19 + Task 19b after the "
                         "reproduction check); default: one commit only")
    pm.add_argument("--check-merge", action="store_true",
                    help="only load, merge and check provenance (a preflight); writes nothing")

    bs = sub.add_parser("budget-sensitivity", help="R33: U of the same cells at several reconnaissance budgets against "
                                                  "the budget-2000 reference; pre-registered hypothesis H")
    bs.add_argument("--reference", type=Path, required=True, help="directory with <label>/policy-sweep.json at budget 2000")
    bs.add_argument("--budget", action="append", required=True, metavar="B=DIR", help="budget=directory (repeatable)")
    bs.add_argument("--labels", type=_names, required=True, help="comma-separated model labels")
    bs.add_argument("--out", type=Path, required=True)

    decide = sub.add_parser("decide", help="decide JSONL operations with the serving path")
    decide.add_argument("--data", type=Path, default=Path("data"), help="card history (before the first input ts)")
    decide.add_argument("--model", type=Path, default=Path("models"))
    decide.add_argument("--input", type=Path, required=True,
                        help="JSONL: cc_num, ts, amt, merchant, category, merch_lat, merch_long; in time order")
    decide.add_argument("--output", type=Path, default=None, help="JSONL output (default: stdout)")
    return parser


def _names(text: str) -> list[str]:
    """``--gen-attacks``: comma-separated names (validated by ``harden.check_generation_attack``)."""
    vals = [x.strip() for x in text.split(",") if x.strip()]
    if not vals:
        raise argparse.ArgumentTypeError(f"expected comma-separated attack names, got {text!r}")
    return vals


def _deltas(text: str) -> list[float]:
    """``--deltas``: comma-separated non-negative numbers."""
    try:
        vals = [float(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected comma-separated numbers, got {text!r}") from None
    if not vals or any(not (v >= 0) or v == float("inf") for v in vals):
        raise argparse.ArgumentTypeError(f"expected finite numbers >= 0, got {text!r}")
    return vals


def git_info(cwd: Path | None = None) -> dict:
    """Code version of a run: env GIT_SHA / GIT_DIRTY (CI, Docker without git), else git, else "unknown"."""
    sha = os.environ.get("GIT_SHA")
    if sha:
        dirty_env = os.environ.get("GIT_DIRTY")
        dirty = "unknown" if dirty_env is None else dirty_env.strip().lower() in ("1", "true", "yes")
        return {"git_sha": sha, "git_dirty": dirty}
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True,
                                text=True, check=True).stdout
        return {"git_sha": sha, "git_dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"git_sha": "unknown", "git_dirty": "unknown"}


def _model_sha256(model_dir: Path) -> str:
    """SHA-256 of ``model.json`` (it holds the manifest hashes of every artifact): identifies the model."""
    import hashlib
    return hashlib.sha256((model_dir / "model.json").read_bytes()).hexdigest()


def cmd_train(args: argparse.Namespace) -> int:
    from fraudaudit.data import load_raw
    from fraudaudit.model.pipeline import train_model

    df = load_raw(args.data)
    print(f"loaded {len(df)} rows", flush=True)
    model, report = train_model(df, seed=args.seed, epochs=args.epochs,
                                sample_frac=args.sample_frac, verbose=2)
    meta = dict(report, platform={"system": platform.system(), "machine": platform.machine(),
                                  "platform": platform.platform()},
                **git_info())
    model.save(args.out, meta=meta)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


def cmd_skew(args: argparse.Namespace) -> int:
    from fraudaudit.data import load_raw
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.skew import skew_check, skew_to_json, summary_markdown

    df = load_raw(args.data)
    model = FraudModel.from_artifacts(load_artifacts(args.model))
    rep = skew_check(df, model, n_cards=args.cards, seed=args.seed, atol=args.atol, n_jobs=args.n_jobs,
                     model_dir=args.model)
    data = skew_to_json(rep, seed=args.seed, atol=args.atol, model_sha256=_model_sha256(args.model),
                        **git_info())
    print(json.dumps(data, indent=2, sort_keys=True), flush=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(summary_markdown(rep, args.atol))
    return 0 if rep.passed else 1


def cmd_audit(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from fraudaudit.bankpolicy import BankPolicy, file_sha256
    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig, decide, passed, run_ensemble
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.report import write_error_outputs, write_gate_outputs
    from fraudaudit.skew import skew_from_json
    from fraudaudit.threat import MerchantTable, SingleTxThreatModel, ThreatModel

    t0 = time.perf_counter()
    try:
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        policy = None if args.policy is None else BankPolicy.from_yaml(args.policy)
        if policy is not None:
            tm = policy.apply(tm)
        if args.delta is not None:
            if not math.isfinite(args.delta) or not 0 <= args.delta <= 1:
                raise ValueError(f"--delta must be a share of the episode amount in [0, 1] (a bar above 1 can never "
                                 f"fail), got {args.delta}")
            cfg = dataclasses.replace(cfg, delta=float(args.delta))
        tms = None if args.no_baseline else SingleTxThreatModel.from_yaml(args.single_threat)
        skew_json = None if args.skew_report is None else json.loads(args.skew_report.read_text(encoding="utf-8"))
        skew = None if skew_json is None else skew_from_json(skew_json)
        if skew_json is not None:
            want, got = skew_json.get("model_sha256"), _model_sha256(args.model)
            if want != got:
                raise ValueError(f"skew report {args.skew_report} was made for another model "
                                 f"(model_sha256 {want}, audited model {got}): rerun `fraudaudit skew --out`")
        model_fpr = _check_model_fpr(args.model, cfg, args.gate)
        df = load_raw(args.data)
        model = FraudModel.from_artifacts(load_artifacts(args.model))  # SHA-256 before deserialisation
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"error: audit configuration/artifacts: {msg}", file=sys.stderr)
        md = write_error_outputs(args.out, msg)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a", encoding="utf-8") as f:
                f.write(md)
        return 2
    t1 = time.perf_counter()
    results = run_ensemble(df, model, tm, cfg, args.seed, n_jobs=args.n_jobs, model_dir=args.model,
                           threat_path=args.threat, false_blocks=True, log=lambda m: print(m, file=sys.stderr, flush=True))
    checks = decide(cfg, results, skew)
    baseline, baseline_error = None, None
    if tms is not None:  # informational: the conventional audit on the same targets, not a gate check
        from fraudaudit import audit as audit_mod
        rules = [tms.amount_rule] + [r for r in audit_mod.AMOUNT_RULES if r != tms.amount_rule]  # primary first
        try:
            baseline = {}
            for rule in rules:
                b = audit_mod.run_single_tx_baseline(df, model, tms, tm, args.seed, n_jobs=args.n_jobs,
                                                     model_dir=args.model, single_path=args.single_threat,
                                                     confidence=cfg.confidence, amount_rule=rule,
                                                     spares_period=cfg.spares_period)
                baseline[rule] = b
                print(f"baseline {rule}: evaded ops {b.evaded_tx_share:.4f}, amount {b.evaded_amount_share:.4f}, "
                      f"chosen/A {b.chosen_amount_ratio:.4f}, replay VER {b.replay_ver[0]:.4f}, "
                      f"{b.meta['wall_seconds']} s", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 – informational: never changes the verdict or the exit code
            baseline, baseline_error = None, f"{type(e).__name__}: {e}"
            print(f"warning: single-transaction baseline failed: {baseline_error}", file=sys.stderr, flush=True)
    meta = {"seed": int(args.seed), **git_info(), "gate_file": str(args.gate),
            "model_target_fpr": None if model_fpr is None else float(model_fpr),
            "gate_sha256": hashlib.sha256(args.gate.read_bytes()).hexdigest(),
            "policy": None if policy is None else {"file": str(args.policy), "sha256": file_sha256(args.policy),
                                                   **policy.public()},
            "delta_source": "--delta" if args.delta is not None else str(args.gate),
            "skew_report": None if args.skew_report is None else str(args.skew_report),
            "wall_seconds": round(time.perf_counter() - t1, 1),
            "wall_seconds_with_loading": round(time.perf_counter() - t0, 1)}
    data = write_gate_outputs(args.out, cfg, checks, results, meta, MerchantTable.from_df(df), skew_json,
                              baseline, baseline_error)
    print(json.dumps({"verdict": data["verdict"], "worst_run": data["worst_run"],
                      "checks": {c.name: {"value": c.value, "threshold": c.threshold, "passed": c.passed}
                                 for c in checks},
                      "uplift_upper": {k: r.uplift[2] for k, r in results.items()},
                      "baseline": _baseline_brief(data["baseline"]),
                      "wall_seconds": meta["wall_seconds"], "out": str(args.out)}, indent=2, ensure_ascii=False),
          flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0 if passed(checks) else 1


def _check_model_fpr(model_dir: Path, cfg, gate_path: Path):
    """The model's ``target_fpr`` (``model.json`` meta, may be None); ValueError if it differs from the gate."""
    model_meta = json.loads((model_dir / "model.json").read_text(encoding="utf-8")).get("meta", {})
    model_fpr = model_meta.get("target_fpr")
    if model_fpr is not None and abs(float(model_fpr) - cfg.target_fpr) > 1e-12:
        raise ValueError(f"model τ was chosen for target_fpr {model_fpr}, {gate_path} asks for {cfg.target_fpr}")
    return model_fpr


def cmd_calibrate(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from fraudaudit.calibrate import calibration, write_calibration
    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig, run_ensemble
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.threat import ThreatModel

    t0 = time.perf_counter()
    try:
        gate_bytes = args.gate.read_bytes()
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        _check_model_fpr(args.model, cfg, args.gate)
        df = load_raw(args.data)
        model = FraudModel.from_artifacts(load_artifacts(args.model))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        print(f"error: calibration configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    t1 = time.perf_counter()
    results = run_ensemble(df, model, tm, cfg, args.seed, n_jobs=args.n_jobs, model_dir=args.model,
                           threat_path=args.threat, log=lambda m: print(m, file=sys.stderr, flush=True))
    meta = {"seed": int(args.seed), **git_info(), "gate_file": str(args.gate),
            "gate_sha256": hashlib.sha256(gate_bytes).hexdigest(),
            "wall_seconds": round(time.perf_counter() - t1, 1),
            "wall_seconds_with_loading": round(time.perf_counter() - t0, 1)}
    data = calibration(results, cfg, args.deltas, meta)
    write_calibration(args.out, data)
    if args.gate.read_bytes() != gate_bytes:  # this command only reads the gate
        raise RuntimeError(f"{args.gate} changed during calibration")
    print(json.dumps({"resolution": data["resolution"], "worst_run": data["worst_run"],
                      "worst_upper": data["worst_upper"],
                      "verdict_vs_delta": {str(r["delta"]): r["baseline_verdict"] for r in data["verdict_vs_delta"]},
                      "wall_seconds": meta["wall_seconds"], "out": str(args.out)}, indent=2), flush=True)
    return 0


def cmd_harden(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from fraudaudit.audit import _versions
    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig
    from fraudaudit.harden import check_generation_attack, harden, write_report
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.threat import ThreatModel

    t0 = time.perf_counter()
    try:
        for a in args.gen_attacks:
            check_generation_attack(a)
        gate_bytes = args.gate.read_bytes()
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        _check_model_fpr(args.model, cfg, args.gate)
        df = load_raw(args.data)
        model = FraudModel.from_artifacts(load_artifacts(args.model))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        print(f"error: harden configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    final, reports, info = harden(df, model, tm, cfg, rounds=args.rounds, seed=args.seed, model_dir=args.model,
                                  out_dir=args.out, n_jobs=args.n_jobs, budget_ops=args.budget_ops,
                                  epochs=args.epochs, baseline_audit=not args.no_baseline_audit,
                                  share_among_fraud=args.share_among_fraud,
                                  threat_path=args.threat, log=lambda m: print(m, file=sys.stderr, flush=True),
                                  gen_attacks=tuple(args.gen_attacks))
    meta = {"seed": int(args.seed), **git_info(), "input_model": str(args.model),
            "input_model_sha256": _model_sha256(args.model), "threat_file": str(args.threat),
            "threat_sha256": hashlib.sha256(args.threat.read_bytes()).hexdigest(), "gate_file": str(args.gate),
            "gate_sha256": hashlib.sha256(gate_bytes).hexdigest(), "delta": cfg.delta, "rounds": int(args.rounds),
            "platform": platform.platform(), "versions": _versions(),
            "wall_seconds": round(time.perf_counter() - t0, 1)}
    data = write_report(args.out, reports, info, meta)
    if args.gate.read_bytes() != gate_bytes:  # this command only reads the gate
        raise RuntimeError(f"{args.gate} changed during hardening")
    print(json.dumps({"final_model_dir": data["final_model_dir"],
                      "rounds": [{"round": r["round"], "val_uplift_worst_upper": r["val_uplift_worst_upper"],
                                  "val_uplift_mean": r["val_uplift_mean"], "n_adv_rows": r["n_adv_rows"],
                                  "tau": r["tau"], "val_recall": r["val_quality"]["recall"],
                                  "val_fpr": r["val_quality"]["fpr"]} for r in data["rounds"]],
                      "wall_seconds": meta["wall_seconds"], "out": str(args.out)}, indent=2), flush=True)
    return 0


def cmd_corridor_sweep(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from fraudaudit.audit import _versions
    from fraudaudit.corridor_sweep import corridor_sweep, write_outputs
    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.threat import ThreatModel

    try:
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        if tm.corridor is None:
            raise ValueError(f"{args.threat} has no corridor section")
        df = load_raw(args.data)
        model = FraudModel.from_artifacts(load_artifacts(args.model))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        print(f"error: corridor-sweep configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    data = corridor_sweep(df, model, args.model, tm, cfg, qs=tuple(args.q), fprs=tuple(args.target_fpr),
                          variants=tuple(args.variants),
                          seed=args.seed, n_jobs=args.n_jobs, out_dir=args.out, threat_path=args.threat,
                          reference=not args.no_reference, log=lambda m: print(m, file=sys.stderr, flush=True))
    data["meta"] = {"seed": int(args.seed), **git_info(), "model": str(args.model),
                    "model_sha256": _model_sha256(args.model), "threat_file": str(args.threat),
                    "threat_sha256": hashlib.sha256(args.threat.read_bytes()).hexdigest(),
                    "gate_file": str(args.gate), "platform": platform.platform(), "versions": _versions()}
    write_outputs(data, args.out)
    print((args.out / "corridor-sweep.md").read_text(encoding="utf-8"), flush=True)
    return 0


def cmd_corridor_posthoc(args: argparse.Namespace) -> int:
    import hashlib

    import yaml

    from fraudaudit.audit import _versions
    from fraudaudit.corridor_sweep import merge_posthoc, owner_posthoc_sweep, posthoc_markdown
    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.threat import ThreatModel

    try:
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        if tm.corridor is None:
            raise ValueError(f"{args.threat} has no corridor section")
        sweep = None if args.sweep is None else json.loads(args.sweep.read_text(encoding="utf-8"))
        df = load_raw(args.data)
        model = FraudModel.from_artifacts(load_artifacts(args.model))
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        print(f"error: corridor-posthoc configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    data = owner_posthoc_sweep(df, model, tm, cfg, qs=tuple(args.q), variants=tuple(args.variants), seed=args.seed,
                               log=lambda m: print(m, file=sys.stderr, flush=True))
    data["meta"] = {"seed": int(args.seed), **git_info(), "model": str(args.model),
                    "model_sha256": _model_sha256(args.model), "threat_file": str(args.threat),
                    "threat_sha256": hashlib.sha256(args.threat.read_bytes()).hexdigest(),
                    "sweep_file": None if args.sweep is None else str(args.sweep),
                    "platform": platform.platform(), "versions": _versions()}
    merged = None if sweep is None else merge_posthoc(sweep, data)
    if merged is not None:
        data["merged_cells"] = merged["cells"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "corridor-posthoc.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                                                    encoding="utf-8")
    md = posthoc_markdown(data, merged)
    (args.out / "corridor-posthoc.md").write_text(md, encoding="utf-8")
    print(md, flush=True)
    return 0


def cmd_policy_sweep(args: argparse.Namespace) -> int:
    import yaml

    from fraudaudit.data import load_raw
    from fraudaudit.gate import GateConfig
    from fraudaudit.model.artifacts import ArtifactIntegrityError, load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.policy_sweep import parse_reaction, policy_sweep, write_markdown
    from fraudaudit.threat import ThreatModel

    try:
        if len(args.model) != len(args.label) or len(set(args.label)) != len(args.label):
            raise ValueError("give one distinct --label per --model")
        for r in args.reactions:
            parse_reaction(r)
        cfg = GateConfig.from_yaml(args.gate)
        tm = ThreatModel.from_yaml(args.threat)
        ctm = ThreatModel.from_yaml(args.corridor_threat)  # also for runs without corridor cells: earlier ones may have
        if args.recon_budget_ops is not None:
            if args.recon_budget_ops < 1:
                raise ValueError(f"--recon-budget-ops must be a positive integer, got {args.recon_budget_ops}")
            tm = dataclasses.replace(tm, recon_budget_ops=args.recon_budget_ops)
            ctm = dataclasses.replace(ctm, recon_budget_ops=args.recon_budget_ops)
        if args.corridor_kappa or args.corridor_q:
            from fraudaudit.policy_sweep import corridor_variants
            if "on" not in args.corridor or ctm.corridor is None:
                raise ValueError("--corridor-kappa / --corridor-q need --corridor on and a corridor threat model")
            corridor_variants(ctm.corridor, args.corridor_kappa, args.corridor_q)
        df = load_raw(args.data)
        models = [FraudModel.from_artifacts(load_artifacts(m)) for m in args.model]  # SHA-256 before deserialisation
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError, ArtifactIntegrityError) as e:
        print(f"error: policy-sweep configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    for mdir, label, model in zip(args.model, args.label, models):
        data = policy_sweep(df, model, mdir, label, tm, ctm, cfg, fprs=tuple(args.target_fpr),
                            reactions=tuple(args.reactions), corridors=tuple(args.corridor),
                            kappas=args.corridor_kappa, qs=args.corridor_q, esc_fpr=args.esc_fpr,
                            r19=args.r19, seed=args.seed, n_jobs=args.n_jobs, out_dir=args.out / label,
                            ensemble=not args.friction_only, friction=not args.no_friction, role=args.role,
                            threat_path=args.threat, corridor_path=args.corridor_threat,
                            gate_path=args.gate, log=lambda m: print(m, file=sys.stderr, flush=True))
        write_markdown(data, args.out / label)
        print((args.out / label / "policy-sweep.md").read_text(encoding="utf-8"), flush=True)
    return 0


def cmd_budget_sensitivity(args: argparse.Namespace) -> int:
    from fraudaudit.budget_sensitivity import compare, markdown
    try:
        budgets = {}
        for item in args.budget:
            b, sep, p = item.partition("=")
            if not sep:
                raise ValueError(f"--budget expects B=DIR, got {item!r}")
            if int(b) in budgets:
                raise ValueError(f"--budget {b} given twice")
            budgets[int(b)] = Path(p)
        res = compare(args.reference, budgets, args.labels)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: budget-sensitivity: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "budget-sensitivity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    (args.out / "budget-sensitivity.md").write_text(markdown(res), encoding="utf-8")
    print((args.out / "budget-sensitivity.md").read_text(encoding="utf-8"), flush=True)
    return 0


def cmd_policy_summary(args: argparse.Namespace) -> int:
    from fraudaudit.policy_summary import summarize
    try:
        res = summarize(args.sweeps, args.out, resolution=args.resolution, confirm_rule=args.confirm_rule,
                        merge_commits=args.merge_commits, check_only=args.check_merge)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: policy-summary: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if args.check_merge:
        print(json.dumps(res, indent=2), flush=True)
        return 0
    print((args.out / "hypotheses.md").read_text(encoding="utf-8"), flush=True)
    print(json.dumps({"confirmation_plan": res["confirmation_plan"]}, indent=2), flush=True)
    return 0


def _baseline_brief(b: dict | None) -> dict | None:
    """The headline numbers of ``audit.json["baseline"]`` for the CLI's stdout."""
    if b is None or "error" in b:
        return b
    keys = ("evaded_tx_share", "evaded_amount_share", "chosen_amount_ratio", "replay_ver", "replay_uplift")
    return {"rules": {r: {k: v[k] for k in keys} for r, v in b["rules"].items()},
            "adv_minus_replay": b["comparison"]["adv_minus_replay"]}


def cmd_decide(args: argparse.Namespace) -> int:
    expose = os.environ.get("EXPOSE_SCORE") == "1"
    if expose and os.environ.get("ENV", "").strip().lower() in ("production", "prod"):
        print("error: EXPOSE_SCORE=1 is not allowed with ENV=production: the score p is never exposed "
              "in production (ATLAS AML.M0002)", file=sys.stderr)
        return 2
    import pandas as pd

    from fraudaudit.data import load_raw
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.serving.core import Serving
    from fraudaudit.serving.state import CardState, Tx, profile_of

    try:
        return _decide(args, expose)
    except (KeyError, ValueError, TypeError, OSError) as e:
        msg = f"missing key {e}" if isinstance(e, KeyError) else str(e)
        print(f"error: decide failed: {msg}", file=sys.stderr)
        return 2


def _decide(args: argparse.Namespace, expose: bool) -> int:
    import pandas as pd

    from fraudaudit.data import load_raw
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.serving.core import Serving
    from fraudaudit.serving.state import CardState, Tx, profile_of

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = load_raw(args.data)
    serving = Serving(FraudModel.from_artifacts(load_artifacts(args.model)))
    states: dict[int, CardState] = {}
    for r in rows:  # first ts of a card: its history ends strictly before
        card = int(r["cc_num"])
        if card not in states:
            if not (df["cc_num"] == card).any():
                raise ValueError(f"card {card} is not in the history data")
            states[card] = CardState.from_history(profile_of(df, card), df, at=pd.Timestamp(r["ts"]))
    out = []
    for r in rows:
        state = states[int(r["cc_num"])]
        tx = Tx(ts=pd.Timestamp(r["ts"]), amt=float(r["amt"]), merchant=str(r["merchant"]),
                category=str(r["category"]), merch_lat=float(r["merch_lat"]), merch_long=float(r["merch_long"]))
        d = serving.decide(state, tx)  # raises ValueError if the card's rows are out of time order
        state.push(tx)
        out.append(json.dumps({"p": d.p, "declined": d.declined} if expose else {"declined": d.declined}))
    text = "\n".join(out) + ("\n" if out else "")
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train":
        return cmd_train(args)
    if args.command == "skew":
        return cmd_skew(args)
    if args.command == "audit":
        return cmd_audit(args)
    if args.command == "calibrate":
        return cmd_calibrate(args)
    if args.command == "decide":
        return cmd_decide(args)
    if args.command == "harden":
        return cmd_harden(args)
    if args.command == "corridor-sweep":
        return cmd_corridor_sweep(args)
    if args.command == "corridor-posthoc":
        return cmd_corridor_posthoc(args)
    if args.command == "policy-sweep":
        return cmd_policy_sweep(args)
    if args.command == "budget-sensitivity":
        return cmd_budget_sensitivity(args)
    if args.command == "policy-summary":
        return cmd_policy_summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
