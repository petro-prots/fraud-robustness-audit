import argparse

from fraudaudit import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraudaudit",
        description="Episode-level adversarial robustness audit of a fraud-detection model.",
    )
    parser.add_argument("--version", action="version", version=f"fraudaudit {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
    return 0
