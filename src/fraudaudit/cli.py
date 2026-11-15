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

    train = sub.add_parser("train", help="train the bachelor model and save its artifacts")
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

    audit = sub.add_parser("audit", help="episode audit: VER of normal fraud vs VER under attack")
    audit.add_argument("--data", type=Path, default=Path("data"))
    audit.add_argument("--model", type=Path, default=Path("models"), help="artifact directory")
    audit.add_argument("--threat", type=Path, default=Path("threat-models/realistic.yaml"))
    audit.add_argument("--attack", choices=["random"], default="random")
    audit.add_argument("--out", type=Path, default=Path("out"), help="writes audit.json and summary.md")
    audit.add_argument("--seed", type=int, default=42)
    audit.add_argument("--n-jobs", type=int, default=None, help="worker processes (default: cpu count)")

    decide = sub.add_parser("decide", help="decide JSONL operations with the serving path")
    decide.add_argument("--data", type=Path, default=Path("data"), help="card history (before the first input ts)")
    decide.add_argument("--model", type=Path, default=Path("models"))
    decide.add_argument("--input", type=Path, required=True,
                        help="JSONL: cc_num, ts, amt, merchant, category, merch_lat, merch_long; in time order")
    decide.add_argument("--output", type=Path, default=None, help="JSONL output (default: stdout)")
    return parser


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
    from fraudaudit.skew import skew_check, summary_markdown

    df = load_raw(args.data)
    model = FraudModel.from_artifacts(load_artifacts(args.model))
    rep = skew_check(df, model, n_cards=args.cards, seed=args.seed, atol=args.atol, n_jobs=args.n_jobs,
                     model_dir=args.model)
    print(json.dumps({"passed": rep.passed, "n_cards": rep.n_cards, "n_tx": rep.n_tx,
                      "decision_mismatches": rep.decision_mismatches,
                      "max_abs_p_diff": rep.max_abs_p_diff, "max_abs_diff": rep.max_abs_diff,
                      "seed": args.seed, "atol": args.atol, **git_info()}, indent=2, sort_keys=True), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(summary_markdown(rep, args.atol))
    return 0 if rep.passed else 1


def cmd_audit(args: argparse.Namespace) -> int:
    from fraudaudit.attacks.random_search import RandomSearch
    from fraudaudit.audit import run_audit, write_outputs
    from fraudaudit.data import load_raw
    from fraudaudit.model.artifacts import load_artifacts
    from fraudaudit.model.pipeline import FraudModel
    from fraudaudit.threat import MerchantTable, ThreatModel

    attacks = {"random": RandomSearch}
    t0 = time.perf_counter()
    tm = ThreatModel.from_yaml(args.threat)
    df = load_raw(args.data)
    model = FraudModel.from_artifacts(load_artifacts(args.model))
    result = run_audit(df, model, tm, attacks[args.attack](), seed=args.seed, n_jobs=args.n_jobs,
                       model_dir=args.model, threat_path=args.threat)
    result.meta["wall_seconds_with_loading"] = round(time.perf_counter() - t0, 1)  # + load_raw, model load
    data = write_outputs(result, args.out, MerchantTable.from_df(df))
    print(json.dumps({k: data[k] for k in ("ver_clean", "ver_adv", "asr", "asr_upper", "n_targets",
                                           "recon_ops", "feasible_share")}
                     | {"wall_seconds": result.meta["wall_seconds"], "out": str(args.out)}, indent=2), flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write((args.out / "summary.md").read_text(encoding="utf-8"))
    return 0


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
    if args.command == "decide":
        return cmd_decide(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
