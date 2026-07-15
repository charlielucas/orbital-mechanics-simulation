"""Command-line interface for generating validation evidence."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from orbital_mechanics.validation import run_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic orbital-mechanics validation and generate evidence artifacts."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/validation"),
        help="directory for JSON, CSV, and PNG outputs (default: artifacts/validation)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_validation(args.output_dir)
    status = "PASS" if summary["overall_pass"] else "FAIL"
    print(f"Validation {status}: {args.output_dir / 'validation_summary.json'}")
    for scenario in summary["scenarios"].values():
        for metric in scenario["metrics"]:
            marker = "PASS" if metric["passed"] else "FAIL"
            print(
                f"  [{marker}] {metric['name']}: {metric['value']:.8g} "
                f"{metric['unit']} {metric['comparison']} {metric['threshold']:.8g}"
            )
    return 0 if summary["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
