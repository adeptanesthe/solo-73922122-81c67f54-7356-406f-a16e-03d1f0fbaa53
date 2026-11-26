import pytest

from recycling_backtest.csvio import (
    UploadError,
    build_dataset,
    parse_lab_csv,
)

INBOUND = """批号,入厂分钟,含水率,白度
A1,1,10.5,82.0
A2,2,21.0,61.0
"""

LAB = """批号,出单分钟,合格与否
A1,3,合格
A2,4,不合格
"""


def test_build_dataset_ok():
    ds = build_dataset(INBOUND, LAB)
    assert len(ds.inbound) == 2
    assert len(ds.lab) == 2
    assert ds.inbound[0].batch == "A1"
    assert ds.inbound[0].moisture == 10.5


def test_english_headers_and_tokens():
    inbound = "batch,minute,moisture,whiteness\nX1,0,11,81\n"
    lab = "batch,report_minute,result\nX1,2,PASS\n"
    ds = build_dataset(inbound, lab)
    assert ds.lab[0].qualified == 1


def test_gbk_encoding():
    text = "批号,入厂分钟,含水率,白度\n甲,1,11,81\n".encode("gbk")
    lab = "批号,出单分钟,合格与否\n甲,2,合格\n".encode("gbk")
    ds = build_dataset(text, lab)
    assert ds.inbound[0].batch == "甲"


def test_duplicate_receipt_same_batch_minute_rejected():
    bad = (
        "批号,出单分钟,合格与否\n"
        "A1,5,合格\n"
        "A1,5,不合格\n"
    )
    with pytest.raises(UploadError, match="同批同分钟限一张回单"):
        parse_lab_csv(bad)


def test_correction_at_different_minute_allowed():
    lab = (
        "批号,出单分钟,合格与否\n"
        "A1,3,合格\n"
        "A1,8,不合格\n"
    )
    rows = parse_lab_csv(lab)
    assert len(rows) == 2


def test_missing_column_rejected():
    with pytest.raises(UploadError, match="缺少必需列"):
        build_dataset("批号,入厂分钟,含水率\nA1,1,10\n", LAB)


def test_bad_number_rejected():
    with pytest.raises(UploadError, match="含水率"):
        build_dataset("批号,入厂分钟,含水率,白度\nA1,1,很高,82\n", LAB)


def test_bad_label_rejected():
    with pytest.raises(UploadError, match="合格与否"):
        build_dataset(INBOUND, "批号,出单分钟,合格与否\nA1,3,也许\n")


def test_unknown_lab_batch_rejected():
    with pytest.raises(UploadError, match="在入厂 CSV 中找不到"):
        build_dataset(INBOUND, "批号,出单分钟,合格与否\nZZ,3,合格\n")


def test_duplicate_inbound_batch_rejected():
    bad_in = (
        "批号,入厂分钟,含水率,白度\n"
        "A1,1,10,82\n"
        "A1,2,11,81\n"
    )
    with pytest.raises(UploadError, match="批号"):
        build_dataset(bad_in, LAB)
