#!/usr/bin/env python3
"""Draw main figures 2–6 from model results or the supplied result tables."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from plotting import draw_all, load_tables

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURE_DIR = ROOT / "figures"
REFERENCE_DIR = ROOT / "figure_data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=FIGURE_DIR)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    try:
        tables = load_tables(args.results, REFERENCE_DIR)
        draw_all(tables, args.output, args.dpi)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"{error}\n")
    print(f"Five figures written to {args.output}")


if __name__ == "__main__":
    main()
