"""Reconstruct the selected final NCM weights without climate data or training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "replayed_final")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check all reconstructed state members without writing results")
    args = parser.parse_args()
    from model import final_recipe
    result = final_recipe.verify_recipe() if args.verify_only else final_recipe.replay(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
