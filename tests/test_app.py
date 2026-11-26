import io
import json

import pytest

import app as app_module
from app import create_app


@pytest.fixture()
def client():
    # ensure a clean global state for each test
    with app_module.STATE.lock:
        app_module.STATE.dataset = None
        app_module.STATE.engine = None
        app_module.STATE.sample_case = None
    application = create_app()
    application.config["TESTING"] = True
    with application.test_client() as c:
        yield c


DEMO_IN = open("/workspace/samples/demo_inbound.csv", "rb").read()
DEMO_LAB = open("/workspace/samples/demo_lab.csv", "rb").read()


def test_health_and_empty_state(client):
    assert client.get("/healthz").get_json()["status"] == "ok"
    assert client.get("/health").get_json()["status"] == "ok"
    data = client.get("/api/state").get_json()
    assert data["loaded"] is False
    html = client.get("/").get_data(as_text=True)
    assert "尚未载入" in html


def test_seeds_endpoint(client):
    d = client.get("/api/seeds").get_json()
    assert d["ok"] is True
    assert len(d["seed_samples"]) == 12
    assert d["initial_centroids"]["qualified"] is not None
    assert "equidistant" in d["rule"]


def test_sample_demo_run_all_json(client):
    r = client.post("/api/sample/load", json={"name": "demo"})
    assert r.status_code == 200
    data = r.get_json()
    assert data["loaded"] is True
    assert data["done"] is False

    r = client.post("/api/run")
    data = r.get_json()
    assert data["done"] is True
    assert data["lists"]["fixed"]["wrong_accept"] == ["A03"]
    assert data["lists"]["fixed"]["wrong_reject"] == ["B01", "C01"]
    assert data["lists"]["adaptive"]["wrong_reject"] == ["C01"]
    assert data["lists"]["adaptive"]["wrong_accept"] == ["A03"]


def test_step_endpoint_advances_one_minute(client):
    client.post("/api/sample/load/demo")
    r = client.post("/api/step")
    data = r.get_json()
    assert data["current_minute"] == 1
    assert data["played_minute_count"] == 1
    assert data["step"]["minute"] == 1
    # second step -> minute 2
    data = client.post("/api/step").get_json()
    assert data["current_minute"] == 2


def test_reset(client):
    client.post("/api/sample/load/demo")
    client.post("/api/run")
    data = client.post("/api/reset").get_json()
    assert data["played_minute_count"] == 0
    assert data["done"] is False


def test_tie_sample_marks_equidistant_unqualified(client):
    client.post("/api/sample/load/tie")
    data = client.post("/api/step").get_json()  # minute 1 classifies TIE01
    tie_rows = [b for b in data["batches"] if b["batch"] == "TIE01"]
    assert tie_rows
    pred = tie_rows[0]["pred_fixed"]
    assert pred["tie"] is True
    assert pred["label_cn"] == "不合格"


def test_multipart_upload_and_failure_preserves_state(client):
    # 1) load good demo
    data = {
        "inbound": (io.BytesIO(DEMO_IN), "demo_inbound.csv"),
        "lab": (io.BytesIO(DEMO_LAB), "demo_lab.csv"),
    }
    r = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    client.post("/api/run")
    before = client.get("/api/state").get_json()
    assert before["done"] is True

    # 2) broken replacement upload (duplicate receipt key)
    bad_lab = (
        "批号,出单分钟,合格与否\n"
        "A01,2,合格\nA01,2,不合格\n"
    ).encode()
    data = {
        "inbound": (io.BytesIO(DEMO_IN), "in.csv"),
        "lab": (io.BytesIO(bad_lab), "bad.csv"),
    }
    r = client.post("/api/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 400
    body = r.get_json()
    assert body["ok"] is False
    assert "同批同分钟" in body["error"]
    assert body["state_preserved"] is True

    # 3) old replay fully intact
    after = client.get("/api/state").get_json()
    assert after["done"] is True
    assert after["current_minute"] == before["current_minute"]
    assert after["lists"] == before["lists"]


def test_form_upload_failure_redirects_with_error(client):
    client.post("/api/sample/load/demo")
    data = {
        "inbound": (io.BytesIO(DEMO_IN), "in.csv"),
        "lab": (io.BytesIO("批号,出单分钟,合格与否\n".encode()), "empty.csv"),
    }
    r = client.post("/upload", data=data, content_type="multipart/form-data")
    assert r.status_code == 303
    assert "error=" in r.headers["Location"]
    # state still queryable and loaded
    assert client.get("/api/state").get_json()["loaded"] is True


def test_json_text_upload(client):
    payload = {
        "inbound_csv": "批号,入厂分钟,含水率,白度\nZ1,1,11,81\n",
        "lab_csv": "批号,出单分钟,合格与否\nZ1,2,合格\n",
    }
    r = client.post("/api/upload", json=payload)
    assert r.status_code == 200
    assert r.get_json()["dataset"]["inbound_count"] == 1


def test_header_based_file_detection_with_arbitrary_field_names(client):
    data = {
        "file1": (io.BytesIO(DEMO_LAB), "whatever1.csv"),
        "file2": (io.BytesIO(DEMO_IN), "whatever2.csv"),
    }
    r = client.post("/api/upload", data=data, content_type="multipart/form-data")
    # headers disambiguate even though field order/name is arbitrary
    assert r.status_code == 200, r.get_data(as_text=True)[:300]


def test_page_renders_lists_and_hooks_after_run(client):
    client.post("/api/sample/load/demo")
    client.post("/api/run")
    html = client.get("/").get_data(as_text=True)
    assert 'id="list-fixed-wrong-accept"' in html
    assert 'id="list-adaptive-wrong-reject"' in html
    assert "B01" in html
    assert "改判" in html
    assert 'id="btn-step"' in html
    assert "已全部跑完" in html


def test_sample_download_route(client):
    r = client.get("/samples/demo_inbound.csv")
    assert r.status_code == 200
    assert "批号" in r.get_data(as_text=True)


def test_query_string_autoload_demo(client):
    r = client.get("/?sample=demo&run=1")
    assert r.status_code == 200
    assert "已全部跑完" in r.get_data(as_text=True)
    state = client.get("/api/state").get_json()
    assert state["loaded"] is True
    assert state["done"] is True
