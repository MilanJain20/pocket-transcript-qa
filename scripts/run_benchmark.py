#!/usr/bin/env python3
"""Run the synthetic benchmark and optionally refresh the README table."""

from __future__ import annotations

import argparse

from pocket_qa.bench import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the synthetic Pocket Transcript QA benchmark.")
    parser.add_argument("--write-readme", action="store_true", help="Replace the README results block.")
    parser.add_argument(
        "--check-readme",
        action="store_true",
        help="Exit non-zero if the README table disagrees with a fresh run.",
    )
    args = parser.parse_args()
    report = run_benchmark(write_readme=args.write_readme, check_readme=args.check_readme)
    print(report["markdown"])
    misses = report["misses"]
    print(f"\nRecording-level disagreements: {len(misses)}")
    for row in misses:
        print(f"- {row['id']}: gold={row['gold']} pred={row['pred']} ({row['notes']})")


if __name__ == "__main__":
    main()
