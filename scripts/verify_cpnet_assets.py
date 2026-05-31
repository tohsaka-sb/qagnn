#!/usr/bin/env python3
"""Validate CPNet concept and entity embedding consistency for QAGNN."""

from __future__ import annotations

import argparse
import sys

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent-path", required=True, help="Path to tzw entity embedding .npy")
    parser.add_argument("--concept-path", required=True, help="Path to concept.txt")
    parser.add_argument("--expected-rows", type=int, default=799273)
    parser.add_argument("--expected-dim", type=int, default=1024)
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument(
        "--max-zero-row-ratio",
        type=float,
        default=0.2,
        help="Fail if sampled all-zero row ratio is above this threshold.",
    )
    return parser.parse_args()


def count_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as fin:
        for _ in fin:
            n += 1
    return n


def main() -> int:
    args = parse_args()

    concept_rows = count_lines(args.concept_path)
    print(f"[verify] concept rows: {concept_rows}")
    if concept_rows != args.expected_rows:
        print(
            f"[verify][ERROR] concept rows mismatch: got {concept_rows}, expected {args.expected_rows}",
            file=sys.stderr,
        )
        return 2

    try:
        emb = np.load(args.ent_path, mmap_mode="r")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[verify][ERROR] failed to mmap embedding: {exc}", file=sys.stderr)
        return 2

    print(f"[verify] emb shape: {emb.shape}, dtype: {emb.dtype}")
    if emb.shape != (args.expected_rows, args.expected_dim):
        print(
            (
                "[verify][ERROR] embedding shape mismatch: "
                f"got {emb.shape}, expected {(args.expected_rows, args.expected_dim)}"
            ),
            file=sys.stderr,
        )
        return 2
    if emb.dtype != np.float32:
        print(f"[verify][ERROR] embedding dtype mismatch: got {emb.dtype}, expected float32", file=sys.stderr)
        return 2

    sample_size = min(args.sample_size, emb.shape[0])
    idx = np.linspace(0, emb.shape[0] - 1, sample_size, dtype=int)
    zero_row_ratio = float(((emb[idx] == 0).all(axis=1)).mean())
    print(f"[verify] sampled all-zero row ratio: {zero_row_ratio:.6f}")
    if zero_row_ratio > args.max_zero_row_ratio:
        print(
            (
                "[verify][ERROR] too many zero rows in embedding sample "
                f"({zero_row_ratio:.6f} > {args.max_zero_row_ratio:.6f})"
            ),
            file=sys.stderr,
        )
        return 2

    print("[verify] CPNet assets look consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
