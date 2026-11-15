import argparse
import json
import platform
import sys
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
    return parser


def cmd_train(args: argparse.Namespace) -> int:
    from fraudaudit.data import load_raw
    from fraudaudit.model.pipeline import train_model

    df = load_raw(args.data)
    print(f"loaded {len(df)} rows", flush=True)
    model, report = train_model(df, seed=args.seed, epochs=args.epochs,
                                sample_frac=args.sample_frac, verbose=2)
    meta = dict(report, platform={"system": platform.system(), "machine": platform.machine(),
                                  "platform": platform.platform()})
    model.save(args.out, meta=meta)
    print(json.dumps(meta, indent=2, sort_keys=True), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train":
        return cmd_train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
