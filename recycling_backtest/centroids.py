"""Nearest-centroid classifiers with constant-time receipt updates.

Label convention
-----------------
``QUALIFIED = 1``   -> 合格 / accepted at intake
``UNQUALIFIED = 0`` -> 不合格 / rejected at intake

A receipt (化验回单) is the ground-truth label for a batch.  When a batch is
re-judged (改判, a newer receipt with a different label), the old receipt's
training contribution is *replaced*: its feature vector is subtracted from the
old class sum before the new one is added.

Only per-class counts and coordinate sums are kept, plus the current
contribution of every known batch.  Applying one receipt is O(1) on average
(dict lookup + a fixed number of arithmetic operations) and does not iterate
over previously lab-tested batches.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

QUALIFIED = 1
UNQUALIFIED = 0

LABEL_NAME = {QUALIFIED: "qualified", UNQUALIFIED: "unqualified"}
LABEL_CN = {QUALIFIED: "合格", UNQUALIFIED: "不合格"}

# Built-in labelled samples shared by both models at start-up.
# Features are (含水率 moisture %, 白度 whiteness).
SEED_SAMPLES: List[Tuple[float, float, int]] = [
    # clearly qualified: lower moisture, higher whiteness
    (10.2, 82.5, QUALIFIED),
    (11.5, 80.1, QUALIFIED),
    (9.8, 84.0, QUALIFIED),
    (12.4, 78.6, QUALIFIED),
    (10.9, 81.3, QUALIFIED),
    (11.1, 83.2, QUALIFIED),
    # clearly unqualified: higher moisture, lower whiteness
    (21.5, 61.2, UNQUALIFIED),
    (20.1, 63.5, UNQUALIFIED),
    (22.8, 58.9, UNQUALIFIED),
    (19.6, 64.8, UNQUALIFIED),
    (21.0, 60.5, UNQUALIFIED),
    (20.7, 62.1, UNQUALIFIED),
]

Point = Tuple[float, float]


class CentroidModel:
    """Two-class nearest-centroid model with incremental sums."""

    def __init__(self, seed_samples=SEED_SAMPLES, name: str = ""):
        self.name = name
        # index 0 -> unqualified, index 1 -> qualified
        self.counts = [0, 0]
        self.sums = [[0.0, 0.0], [0.0, 0.0]]
        # batch -> (label, (moisture, whiteness)) for the *current* receipt
        self.contributions: Dict[str, Tuple[int, Point]] = {}
        self.receipts_applied = 0
        for moisture, whiteness, label in seed_samples:
            self._add(label, (float(moisture), float(whiteness)))

    # ----- basic incremental arithmetic ----------------------------------
    def _add(self, label: int, x: Point) -> None:
        self.counts[label] += 1
        self.sums[label][0] += x[0]
        self.sums[label][1] += x[1]

    def _remove(self, label: int, x: Point) -> None:
        self.counts[label] -= 1
        self.sums[label][0] -= x[0]
        self.sums[label][1] -= x[1]

    def centroid(self, label: int) -> Optional[Point]:
        n = self.counts[label]
        if n <= 0:
            return None
        return (self.sums[label][0] / n, self.sums[label][1] / n)

    def centroids(self) -> Dict[str, Optional[List[float]]]:
        q = self.centroid(QUALIFIED)
        u = self.centroid(UNQUALIFIED)
        return {
            "qualified": None if q is None else [q[0], q[1]],
            "unqualified": None if u is None else [u[0], u[1]],
        }

    # ----- classification -------------------------------------------------
    def squared_distances(self, x: Point) -> Tuple[float, float]:
        """Return (d2_to_qualified, d2_to_unqualified)."""
        cq = self.centroid(QUALIFIED)
        cu = self.centroid(UNQUALIFIED)
        if cq is None or cu is None:
            raise RuntimeError("both class centroids require at least one sample")
        dq = (x[0] - cq[0]) ** 2 + (x[1] - cq[1]) ** 2
        du = (x[0] - cu[0]) ** 2 + (x[1] - cu[1]) ** 2
        return dq, du

    def predict(self, x: Point) -> dict:
        """Nearest centroid; exact equidistance is judged UNQUALIFIED."""
        dq, du = self.squared_distances(x)
        tie = math.isclose(dq, du, rel_tol=1e-12, abs_tol=1e-12)
        if tie:
            label = UNQUALIFIED
        elif dq < du:
            label = QUALIFIED
        else:
            label = UNQUALIFIED
        return {
            "label": label,
            "label_name": LABEL_NAME[label],
            "label_cn": LABEL_CN[label],
            "d2_qualified": dq,
            "d2_unqualified": du,
            "distance_qualified": math.sqrt(max(dq, 0.0)),
            "distance_unqualified": math.sqrt(max(du, 0.0)),
            "tie": tie,
        }

    # ----- receipt update (O(1)) -----------------------------------------
    def apply_receipt(self, batch: str, x: Point, new_label: int) -> dict:
        """Apply the latest receipt for ``batch``.

        If the batch already contributed to the model, its old contribution is
        removed first, so a changed label (改判) *replaces* rather than stacks.
        Cost: O(1) average regardless of how many batches were tested.
        """
        x = (float(x[0]), float(x[1]))
        old = self.contributions.get(batch)
        old_label: Optional[int] = None
        replaced = False
        if old is not None:
            old_label, old_x = old
            self._remove(old_label, old_x)
            replaced = True
        self._add(new_label, x)
        self.contributions[batch] = (new_label, x)
        self.receipts_applied += 1
        return {
            "batch": batch,
            "old_label": old_label,
            "old_label_cn": None if old_label is None else LABEL_CN[old_label],
            "new_label": new_label,
            "new_label_cn": LABEL_CN[new_label],
            "replaced": replaced,
            "correction": replaced and old_label != new_label,
            "same_label_receipt": replaced and old_label == new_label,
        }

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "counts": {
                "qualified": self.counts[QUALIFIED],
                "unqualified": self.counts[UNQUALIFIED],
            },
            "sums": {
                "qualified": [self.sums[QUALIFIED][0], self.sums[QUALIFIED][1]],
                "unqualified": [
                    self.sums[UNQUALIFIED][0],
                    self.sums[UNQUALIFIED][1],
                ],
            },
            "centroids": self.centroids(),
            "distinct_receipt_batches": len(self.contributions),
            "receipts_applied": self.receipts_applied,
        }


def centroid_after_samples(samples: List[Tuple[Point, int]], label: int) -> Optional[Point]:
    """Test/helper: recompute a centroid from scratch (used by self-tests)."""
    xs = [x for x, lab in samples if lab == label]
    if not xs:
        return None
    return (sum(x[0] for x in xs) / len(xs), sum(x[1] for x in xs) / len(xs))
