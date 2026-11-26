"""Minute-stepped replay engine for the two classifier comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .centroids import (
    LABEL_CN,
    LABEL_NAME,
    QUALIFIED,
    SEED_SAMPLES,
    CentroidModel,
)
from .csvio import Dataset, InboundRow, LabRow

WRONG_ACCEPT = "wrong_accept"   # 错收: predicted qualified, lab says unqualified
WRONG_REJECT = "wrong_reject"   # 漏收: predicted unqualified, lab says qualified


@dataclass
class BatchView:
    inbound: InboundRow
    pred_fixed: Optional[dict] = None
    pred_adaptive: Optional[dict] = None

    @property
    def batch(self) -> str:
        return self.inbound.batch


class ReplayEngine:
    """Holds all replay state for one loaded dataset."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.minutes: List[int] = sorted(
            {r.minute for r in dataset.inbound} | {r.minute for r in dataset.lab}
        )
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.fixed = CentroidModel(name="fixed")
        self.adaptive = CentroidModel(name="adaptive")

        self.views: Dict[str, BatchView] = {
            r.batch: BatchView(inbound=r) for r in self.dataset.inbound
        }
        # inbound rows grouped by minute, file order preserved
        self._in_by_minute: Dict[int, List[InboundRow]] = {}
        for r in self.dataset.inbound:
            self._in_by_minute.setdefault(r.minute, []).append(r)

        self._lab_by_minute: Dict[int, List[LabRow]] = {}
        for r in self.dataset.lab:
            self._lab_by_minute.setdefault(r.minute, []).append(r)

        self.cursor = 0                 # number of minutes already played
        self.current_minute: Optional[int] = None
        self.latest: Dict[str, LabRow] = {}
        self.history: Dict[str, List[LabRow]] = {}
        self.minute_log: List[dict] = []

    # ------------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.cursor >= len(self.minutes)

    @property
    def next_minute(self) -> Optional[int]:
        return None if self.done else self.minutes[self.cursor]

    def step(self) -> Optional[dict]:
        """Advance one recorded minute.

        Ordering inside the minute is mandatory:
        1. classify every intake batch arriving at this minute (with the
           centroid state left by earlier minutes only);
        2. only then process this minute's lab receipts, updating the
           adaptive model.
        """
        if self.done:
            return None
        minute = self.minutes[self.cursor]
        self.cursor += 1
        self.current_minute = minute

        block: dict = {"minute": minute, "classifications": [], "receipts": []}

        # 1) intake grading first
        for row in self._in_by_minute.get(minute, []):
            view = self.views[row.batch]
            view.pred_fixed = self.fixed.predict(row.x)
            view.pred_adaptive = self.adaptive.predict(row.x)
            block["classifications"].append(
                {
                    "batch": row.batch,
                    "minute": minute,
                    "fixed": _pred_brief(view.pred_fixed),
                    "adaptive": _pred_brief(view.pred_adaptive),
                }
            )

        # 2) receipts afterwards
        inbound_index = self.dataset.inbound_by_batch
        for rcpt in self._lab_by_minute.get(minute, []):
            self.history.setdefault(rcpt.batch, []).append(rcpt)
            previous = self.latest.get(rcpt.batch)
            self.latest[rcpt.batch] = rcpt

            x = inbound_index[rcpt.batch].x
            update = self.adaptive.apply_receipt(rcpt.batch, x, rcpt.qualified)
            entry = {
                "batch": rcpt.batch,
                "minute": minute,
                "qualified": rcpt.qualified,
                "label_cn": LABEL_CN[rcpt.qualified],
                "line_no": rcpt.line_no,
                "had_previous": previous is not None,
                "previous_minute": None if previous is None else previous.minute,
                "previous_label_cn": None if previous is None else LABEL_CN[previous.qualified],
                "correction": bool(update["correction"]),
                "model_update": update,
            }
            block["receipts"].append(entry)

        self.minute_log.append(block)
        return block

    def run_all(self) -> int:
        n = 0
        while not self.done:
            self.step()
            n += 1
        return n

    # ------------------------------------------------------------------
    def evaluation(self) -> dict:
        """Build per-batch rows and the four error lists.

        Truth is the *latest receipt replayed so far* for each batch.
        Predictions are the ones recorded at each batch's intake minute and
        are never retroactively re-graded.
        """
        rows: List[dict] = []
        lists = {
            "fixed": {WRONG_ACCEPT: [], WRONG_REJECT: []},
            "adaptive": {WRONG_ACCEPT: [], WRONG_REJECT: []},
        }
        counts = {
            "fixed": {WRONG_ACCEPT: 0, WRONG_REJECT: 0, "correct": 0, "pending": 0},
            "adaptive": {WRONG_ACCEPT: 0, WRONG_REJECT: 0, "correct": 0, "pending": 0},
        }

        for in_row in self.dataset.inbound:
            view = self.views[in_row.batch]
            rcpt = self.latest.get(in_row.batch)
            item = {
                "batch": in_row.batch,
                "minute": in_row.minute,
                "moisture": in_row.moisture,
                "whiteness": in_row.whiteness,
                "pred_fixed": None if view.pred_fixed is None else _pred_brief(view.pred_fixed),
                "pred_adaptive": None
                if view.pred_adaptive is None
                else _pred_brief(view.pred_adaptive),
                "receipt": None,
                "receipt_history": [
                    {"minute": r.minute, "label_cn": LABEL_CN[r.qualified],
                     "qualified": r.qualified}
                    for r in self.history.get(in_row.batch, [])
                ],
                "error_fixed": None,
                "error_adaptive": None,
            }
            if rcpt is not None:
                hist = self.history.get(in_row.batch, [])
                label_changed = len(hist) > 1 and hist[0].qualified != hist[-1].qualified
                item["receipt"] = {
                    "minute": rcpt.minute,
                    "qualified": rcpt.qualified,
                    "label_cn": LABEL_CN[rcpt.qualified],
                    "revised": label_changed,
                    "repeat_count": len(hist),
                }
                for key, pred in (
                    ("fixed", view.pred_fixed),
                    ("adaptive", view.pred_adaptive),
                ):
                    if pred is None:
                        continue
                    err = _classify_error(pred["label"], rcpt.qualified)
                    item[f"error_{key}"] = err
                    if err is None:
                        counts[key]["correct"] += 1
                    else:
                        counts[key][err] += 1
                        lists[key][err].append(in_row.batch)
            else:
                pred_by_key = {"fixed": view.pred_fixed, "adaptive": view.pred_adaptive}
                for key in ("fixed", "adaptive"):
                    if pred_by_key[key] is not None:
                        counts[key]["pending"] += 1

            rows.append(item)

        return {"rows": rows, "lists": lists, "counts": counts}

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        ev = self.evaluation()
        processed_receipts = sum(len(b["receipts"]) for b in self.minute_log)
        return {
            "loaded": True,
            "seed_samples": [
                {"moisture": s[0], "whiteness": s[1], "qualified": s[2]}
                for s in SEED_SAMPLES
            ],
            "dataset": {
                "inbound_name": self.dataset.inbound_name,
                "lab_name": self.dataset.lab_name,
                "inbound_count": len(self.dataset.inbound),
                "lab_count": len(self.dataset.lab),
            },
            "minutes": self.minutes,
            "played_minute_count": self.cursor,
            "total_minute_count": len(self.minutes),
            "current_minute": self.current_minute,
            "next_minute": self.next_minute,
            "done": self.done,
            "processed_receipt_count": processed_receipts,
            "models": {
                "fixed": self.fixed.snapshot(),
                "adaptive": self.adaptive.snapshot(),
            },
            "lists": ev["lists"],
            "counts": ev["counts"],
            "batches": ev["rows"],
            "minute_log": self.minute_log,
        }


def _pred_brief(pred: dict) -> dict:
    return {
        "label": pred["label"],
        "label_name": pred["label_name"],
        "label_cn": pred["label_cn"],
        "tie": pred["tie"],
        "distance_qualified": pred["distance_qualified"],
        "distance_unqualified": pred["distance_unqualified"],
    }


def _classify_error(pred_label: int, truth: int) -> Optional[str]:
    if pred_label == QUALIFIED and truth != QUALIFIED:
        return WRONG_ACCEPT
    if pred_label != QUALIFIED and truth == QUALIFIED:
        return WRONG_REJECT
    return None
