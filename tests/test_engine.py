from recycling_backtest.csvio import build_dataset
from recycling_backtest.engine import ReplayEngine


def _engine(inbound: str, lab: str) -> ReplayEngine:
    return ReplayEngine(build_dataset(inbound, lab))


def test_minutes_are_union_sorted():
    eng = _engine(
        "批号,入厂分钟,含水率,白度\nA,5,10,82\nB,1,21,61\n",
        "批号,出单分钟,合格与否\nA,9,合格\n",
    )
    assert eng.minutes == [1, 5, 9]


def test_same_minute_classify_before_receipt():
    """A batch arriving and receiving a verdict in the same minute must be
    graded with the pre-receipt centroids."""
    eng = _engine(
        "批号,入厂分钟,含水率,白度\nA,1,15.0,73.0\n",
        "批号,出单分钟,合格与否\nA,1,合格\n",
    )
    pre = eng.adaptive.predict((15.0, 73.0))["label"]
    block = eng.step()
    assert block["minute"] == 1
    assert len(block["classifications"]) == 1
    # recorded prediction equals pre-receipt prediction
    assert block["classifications"][0]["adaptive"]["label"] == pre
    # receipt applied after classification
    assert len(eng.adaptive.contributions) == 1
    assert eng.views["A"].pred_adaptive["label"] == pre


def test_step_then_run_and_reset():
    eng = _engine(
        open("/workspace/samples/demo_inbound.csv", encoding="utf-8").read(),
        open("/workspace/samples/demo_lab.csv", encoding="utf-8").read(),
    )
    assert eng.current_minute is None
    b1 = eng.step()
    assert b1["minute"] == 1
    assert eng.cursor == 1
    remaining = len(eng.minutes) - 1
    n = eng.run_all()
    assert n == remaining
    assert eng.done
    state = eng.state_dict()
    assert state["lists"]["fixed"]["wrong_accept"] == ["A03"]
    assert state["lists"]["fixed"]["wrong_reject"] == ["B01", "C01"]
    assert state["lists"]["adaptive"]["wrong_accept"] == ["A03"]
    assert state["lists"]["adaptive"]["wrong_reject"] == ["C01"]

    eng.reset()
    assert eng.cursor == 0
    assert eng.current_minute is None
    assert eng.state_dict()["lists"]["fixed"]["wrong_accept"] == []


def test_predictions_are_retained_after_learning():
    eng = _engine(
        open("/workspace/samples/demo_inbound.csv", encoding="utf-8").read(),
        open("/workspace/samples/demo_lab.csv", encoding="utf-8").read(),
    )
    eng.run_all()
    # B01 was graded qualified by adaptive at minute 3 and that retained
    # grade must not be rewritten by later learning.
    assert eng.views["B01"].pred_adaptive["label_cn"] == "合格"
    # identical features B02 after the A03 correction is graded unqualified
    assert eng.views["B02"].pred_adaptive["label_cn"] == "不合格"


def test_lists_follow_latest_replayed_receipt_only():
    eng = _engine(
        open("/workspace/samples/demo_inbound.csv", encoding="utf-8").read(),
        open("/workspace/samples/demo_lab.csv", encoding="utf-8").read(),
    )
    # through minute 2: A03 currently qualified and predicted qualified
    eng.step(); eng.step()
    lists = eng.evaluation()["lists"]
    assert "A03" not in lists["fixed"]["wrong_accept"]
    # minute 5 correction moves A03 into wrong-accept lists
    while eng.next_minute != 5:
        eng.step()
    eng.step()  # minute 5
    lists = eng.evaluation()["lists"]
    assert "A03" in lists["fixed"]["wrong_accept"]
    assert "A03" in lists["adaptive"]["wrong_accept"]


def test_correction_centroid_matches_scratch_recompute():
    eng = _engine(
        open("/workspace/samples/demo_inbound.csv", encoding="utf-8").read(),
        open("/workspace/samples/demo_lab.csv", encoding="utf-8").read(),
    )
    eng.run_all()
    # adaptive centroid after full replay must equal sums/counts snapshot
    snap = eng.adaptive.snapshot()
    for lab in ("qualified", "unqualified"):
        n = snap["counts"][lab]
        s = snap["sums"][lab]
        c = snap["centroids"][lab]
        assert c[0] == s[0] / n
        assert c[1] == s[1] / n
