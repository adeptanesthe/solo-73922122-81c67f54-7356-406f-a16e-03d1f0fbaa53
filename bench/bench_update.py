"""Measure single-receipt update latency vs number of lab-tested batches.

Run: python bench/bench_update.py
The update must stay O(1): timings should be flat from 1k to 200k batches.
"""
import sys
import time

sys.path.insert(0, ".")
from recycling_backtest.centroids import QUALIFIED, UNQUALIFIED, CentroidModel


def build(n: int) -> CentroidModel:
    m = CentroidModel()
    for i in range(n):
        m.apply_receipt(
            f"B{i}",
            (10.0 + (i % 60) * 0.2, 82.0 - (i % 45) * 0.25),
            QUALIFIED if i % 2 else UNQUALIFIED,
        )
    return m


def time_one_update(n: int, repeats: int = 200) -> float:
    m = build(n)
    # warm + time corrections of an existing batch and a brand-new batch
    t0 = time.perf_counter()
    for r in range(repeats):
        m.apply_receipt("B1000", (13.7 + r * 1e-9, 74.2), QUALIFIED)
        m.apply_receipt(f"NEW_{r}", (14.1, 73.1), UNQUALIFIED)
    dt = (time.perf_counter() - t0) / (2 * repeats)
    return dt * 1e6  # microseconds


if __name__ == "__main__":
    print(f"{'lab batches':>12} | {'avg us/update':>14}")
    for n in (1_000, 10_000, 100_000, 200_000):
        us = time_one_update(n)
        print(f"{n:>12,} | {us:>14.3f}")
