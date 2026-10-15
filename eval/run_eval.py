"""Compare the dense baseline, hybrid retrieval and the agentic loop on the golden set.

Usage:
    python eval/run_eval.py                      # all three experiments
    python eval/run_eval.py --limit 10           # smoke test
Equivalent to:
    ragdoc eval -e configs/experiments/dense.yaml -e configs/experiments/hybrid.yaml \
                -e configs/experiments/agentic.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ragdoc.evaluation.runner import run_experiments

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "eval/datasets/golden.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument(
        "--experiments",
        nargs="*",
        type=Path,
        default=[ROOT / f"configs/experiments/{n}.yaml" for n in ("dense", "hybrid", "agentic")],
    )
    args = parser.parse_args()
    report = run_experiments(
        dataset=args.dataset,
        experiments=args.experiments,
        out_dir=ROOT / "eval/reports",
        limit=args.limit,
        judge=not args.no_judge,
    )
    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
