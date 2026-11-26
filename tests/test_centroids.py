import math
import time

import pytest

from recycling_backtest.centroids import (
    QUALIFIED,
    UNQUALIFIED,
    CentroidModel,
)


def test_seed_centroids_present():
    m = CentroidModel()
    assert m.centroid(QUALIFIED) is not None
    assert m.centroid(UNQUALIFIED) is not None
    assert m.counts[QUALIFIED] == 6
    assert m.counts[UNQUALIFIED] == 6


def test_equidistant_point_judged_unqualified():
    m = CentroidModel()
    q = m.centroid(QUALIFIED)
    b = m.centroid(UNQUALIFIED)
    mid = ((q[0] + b[0]) / 2.0, (q[1] + b[1]) / 2.0)
    pred = m.predict(mid)
    assert pred["tie"] is True
    assert pred["label"] == UNQUALIFIED
    assert pred["label_cn"] == "不合格"


def test_clear_sides_predict_expected_classes():
    m = CentroidModel()
    assert m.predict((10.5, 82.0))["label"] == QUALIFIED
    assert m.predict((21.0, 61.0))["label"] == UNQUALIFIED


def test_receipt_update_moves_centroid_incrementally():
    m = CentroidModel()
    before = m.centroid(QUALIFIED)
    m.apply_receipt("X1", (15.0, 75.0), QUALIFIED)
    after = m.centroid(QUALIFIED)
    assert after[0] > before[0]
    assert after[1] < before[1]
    assert m.counts[QUALIFIED] == 7


def test_correction_replaces_old_contribution():
    """A 改判 must subtract the old label contribution, not stack it."""
    m = CentroidModel()
    x = (15.6, 72.4)
    m.apply_receipt("A03", x, QUALIFIED)
    q_after_good = m.counts[QUALIFIED]
    b_after_good = m.counts[UNQUALIFIED]

    info = m.apply_receipt("A03", x, UNQUALIFIED)
    assert info["correction"] is True
    assert info["old_label"] == QUALIFIED

    # one removed from qualified, one added to unqualified
    assert m.counts[QUALIFIED] == q_after_good - 1
    assert m.counts[UNQUALIFIED] == b_after_good + 1
    # the batch still has exactly one current contribution
    assert set(m.contributions) == {"A03"}
    assert m.contributions["A03"][0] == UNQUALIFIED

    # centroid must equal a from-scratch model trained on seeds + x as bad
    fresh = CentroidModel()
    fresh.apply_receipt("A03", x, UNQUALIFIED)
    for lab in (QUALIFIED, UNQUALIFIED):
        a = m.centroid(lab)
        c = fresh.centroid(lab)
        assert a == pytest.approx(c, rel=1e-12, abs=1e-12)


def test_same_label_receipt_is_idempotent_contribution():
    m = CentroidModel()
    x = (12.0, 80.0)
    m.apply_receipt("B1", x, QUALIFIED)
    q1 = m.centroid(QUALIFIED)
    n1 = m.counts[QUALIFIED]
    m.apply_receipt("B1", x, QUALIFIED)  # duplicate-ish same label update
    assert m.counts[QUALIFIED] == n1      # remove + add keeps count
    assert m.centroid(QUALIFIED)[0] == pytest.approx(q1[0])
    assert len(m.contributions) == 1


def test_single_update_is_constant_time_vs_history():
    """One receipt update after many batches must stay effectively O(1)."""
    m = CentroidModel()
    n = 100_000
    for i in range(n):
        m.apply_receipt(f"B{i}", (10.0 + (i % 50) * 0.2, 80.0 - (i % 40) * 0.3),
                        QUALIFIED if i % 2 else UNQUALIFIED)
    assert len(m.contributions) == n
    t0 = time.perf_counter()
    m.apply_receipt("B50000", (13.7, 74.2), QUALIFIED)  # correction of existing
    m.apply_receipt("NEW", (14.1, 73.1), UNQUALIFIED)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0  # extremely generous; dict O(1) should be microseconds
