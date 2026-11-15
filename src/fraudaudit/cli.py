import argparse
import json
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

    train = sub.add_parser("train", help="train the baseline model (bachelor architecture, bachelor+hour features) and save its artifacts")
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
    skew.add_argument("--atol", type=float, default=1e-4, help="tolerance of the mean_amt_* columns only")
    skew.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")
    skew.add_argument("--out", type=Path, default=None, help="also write the report as JSON (for audit --skew-report)")

    audit = sub.add_parser("audit", help="episode audit and quality gate: every attack x seed of gate.yaml; "
                                         "exit 1 if a check fails")
    audit.add_argument("--data", type=Path, default=Path("data"))
    audit.add_argument("--model", type=Path, default=Path("models"), help="artifact directory")
    audit.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    audit.add_argument("--gate", type=Path, default=Path("gate.yaml"),
                       help="gate rules: thresholds, attacks and attack seeds (the worst run decides)")
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

    hard = sub.add_parser("harden", help="round-based adversarial retraining (ES examples on the H half of the "
                                          "fit episodes); writes every round's model and harden-report.json/.md")
    hard.add_argument("--data", type=Path, default=Path("data"))
    hard.add_argument("--model", type=Path, default=Path("models"), help="input model (artifact directory)")
    hard.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    hard.add_argument("--gate", type=Path, default=Path("gate.yaml"),
                      help="δ (early stop), confidence, one_sided; read only")
    hard.add_argument("--out", type=Path, default=Path("models-harden"),
                      help="round-{r}/ model directories and harden-report.json/.md")
    hard.add_argument("--rounds", type=int, default=3)
    hard.add_argument("--seed", type=int, default=42)
    hard.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")
    hard.add_argument("--epochs", type=int, default=20, help="20 as in train; lower only for probes")
    hard.add_argument("--budget-ops", type=int, default=20000,
                      help="ES reconnaissance budget per generation unit (harden_budget_ops)")
    hard.add_argument("--no-baseline-audit", action="store_true", help="skip the round-0 mini-audit of the input model")

    decide = sub.add_parser("decide", help="decide JSONL operations with the serving path")
    decide.add_argument("--data", type=Path, default=Path("data"), help="card history (before the first input ts)")
    decide.add_argument("--model", type=Path, default=Path("models"))
    decide.add_argument("--input", type=Path, required=True,
                        help="JSONL: cc_num, ts, amt, merchant, category, merch_lat, merch_long; in time order")
    decide.add_argument("--output", type=Path, default=None, help="JSONL output (default: stdout)")
    return parser


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
                           threat_path=args.threat, log=lambda m: print(m, file=sys.stderr, flush=True))
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
    from fraudaudit.harden import harden, write_report
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
        print(f"error: harden configuration/artifacts: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    final, reports, info = harden(df, model, tm, cfg, rounds=args.rounds, seed=args.seed, model_dir=args.model,
                                  out_dir=args.out, n_jobs=args.n_jobs, budget_ops=args.budget_ops,
                                  epochs=args.epochs, baseline_audit=not args.no_baseline_audit,
                                  threat_path=args.threat, log=lambda m: print(m, file=sys.stderr, flush=True))
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
