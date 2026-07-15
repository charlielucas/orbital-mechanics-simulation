"""Verify regenerated validation evidence against the checked-in snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from orbital_mechanics.evidence import compare_validation_directories


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    arguments = parser.parse_args()

    errors = compare_validation_directories(arguments.reference, arguments.candidate)
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1
    print("Evidence comparison PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
