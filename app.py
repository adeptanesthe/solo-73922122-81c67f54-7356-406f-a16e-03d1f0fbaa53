"""Flask back-test web app for the recycled-paper receiving comparison.

Run locally:  python app.py   (or: gunicorn -w 1 --threads 8 app:app)
The single global replay state is protected by a lock; uploads are parsed and
validated completely *before* the current state is replaced, so a failed
upload keeps the running replay intact.
"""

from __future__ import annotations

import json
import os
import threading
import traceback
from typing import Optional, Tuple

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from recycling_backtest.csvio import UploadError, build_dataset, detect_kind
from recycling_backtest.engine import ReplayEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "samples")

SAMPLE_CASES = {
    "demo": ("demo_inbound.csv", "demo_lab.csv", "改判演示"),
    "recheck": ("demo_inbound.csv", "demo_lab.csv", "改判演示"),
    "tie": ("tie_inbound.csv", "tie_lab.csv", "等距判不合格演示"),
    "equidistant": ("tie_inbound.csv", "tie_lab.csv", "等距判不合格演示"),
}


class GlobalState:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.dataset = None
        self.engine: Optional[ReplayEngine] = None
        self.sample_case: Optional[str] = None


STATE = GlobalState()


# ---------------------------------------------------------------------------
# Atomic state replacement
# ---------------------------------------------------------------------------
def load_pair(
    inbound_bytes: bytes,
    lab_bytes: bytes,
    inbound_name: str = "inbound.csv",
    lab_name: str = "lab.csv",
    sample_case: Optional[str] = None,
) -> ReplayEngine:
    # Parse + validate fully first; any UploadError leaves STATE untouched.
    dataset = build_dataset(inbound_bytes, lab_bytes, inbound_name, lab_name)
    engine = ReplayEngine(dataset)
    with STATE.lock:
        STATE.dataset = dataset
        STATE.engine = engine
        STATE.sample_case = sample_case
    return engine


def load_sample(case: str) -> ReplayEngine:
    if case not in SAMPLE_CASES:
        raise UploadError(f"没有内置样例：{case}")
    in_name, lab_name, _ = SAMPLE_CASES[case]
    with open(os.path.join(SAMPLE_DIR, in_name), "rb") as f:
        inbound_bytes = f.read()
    with open(os.path.join(SAMPLE_DIR, lab_name), "rb") as f:
        lab_bytes = f.read()
    return load_pair(inbound_bytes, lab_bytes, in_name, lab_name, sample_case=case)


# ---------------------------------------------------------------------------
# Extract the two CSVs from arbitrary request shapes
# ---------------------------------------------------------------------------
_INBOUND_FIELD_HINTS = {
    "inbound", "inbound_csv", "inboundfile", "in", "ruchang", "ruchang_csv",
    "intake", "intake_csv", "file_in", "csv_inbound", "inboundcsv",
}
_LAB_FIELD_HINTS = {
    "lab", "lab_csv", "labfile", "huayan", "huayan_csv", "assay",
    "assay_csv", "file_lab", "csv_lab", "labcsv",
}


def _kind_from_field(field: str) -> Optional[str]:
    k = (field or "").lower().replace("-", "_")
    if k in _INBOUND_FIELD_HINTS or "inbound" in k or "ruchang" in k or "intake" in k:
        return "inbound"
    if k in _LAB_FIELD_HINTS or "lab" in k or "huayan" in k or "assay" in k:
        return "lab"
    return None


def _collect_multipart() -> Tuple[bytes, bytes, str, str]:
    uploads = []  # (field, filename, bytes)
    for field in request.files:
        for f in request.files.getlist(field):
            if f is not None and f.filename:
                uploads.append((field, f.filename, f.read()))
    if not uploads:
        raise UploadError("没有收到任何 CSV 文件，请选择入厂 CSV 和化验 CSV 后再上传")

    inbound = None  # tuple (content, filename)
    lab = None
    inbound_src = None
    lab_src = None
    for item in uploads:
        field, filename, content = item
        kind = _kind_from_field(field) or detect_kind(filename, content)
        if kind == "inbound" and inbound is None:
            inbound = (content, filename)
            inbound_src = item
        elif kind == "lab" and lab is None:
            lab = (content, filename)
            lab_src = item

    # Fallbacks for ambiguous field names / headers: fill the missing side by
    # upload order so two-file forms under arbitrary input names still work.
    if inbound is None or lab is None:
        unused = [u for u in uploads if u is not inbound_src and u is not lab_src]
        if inbound is None:
            pick = unused[0] if unused else (uploads[0] if uploads else None)
            if pick is not None and pick is not lab_src:
                inbound = (pick[2], pick[1])
                inbound_src = pick
        if lab is None:
            unused = [u for u in uploads if u is not inbound_src and u is not lab_src]
            if unused:
                pick = unused[0]
                lab = (pick[2], pick[1])
            elif len(uploads) == 2:
                pick = uploads[1] if uploads[0] is inbound_src else uploads[0]
                lab = (pick[2], pick[1])

    if inbound is None:
        raise UploadError("未能识别入厂 CSV：需要包含 批号、入厂分钟、含水率、白度 列")
    if lab is None:
        raise UploadError("未能识别化验 CSV：需要包含 批号、出单分钟、合格与否 列")
    return inbound[0], lab[0], inbound[1], lab[1]


_JSON_INBOUND_KEYS = [
    "inbound_csv", "inboundCsv", "inbound", "ruchang_csv", "ruchang",
    "intake_csv", "入厂csv", "入厂",
]
_JSON_LAB_KEYS = [
    "lab_csv", "labCsv", "lab", "huayan_csv", "huayan",
    "assay_csv", "化验csv", "化验",
]


def _first_present(data: dict, keys):
    for k in keys:
        if k in data and data[k] not in (None, ""):
            return data[k]
    return None


def _collect_json_or_form() -> Tuple[bytes, bytes, str, str]:
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict(flat=True)
    in_text = _first_present(data, _JSON_INBOUND_KEYS)
    lab_text = _first_present(data, _JSON_LAB_KEYS)
    if in_text is None or lab_text is None:
        raise UploadError("请求中缺少 inbound_csv / lab_csv 两段 CSV 文本")
    if not isinstance(in_text, str) or not isinstance(lab_text, str):
        raise UploadError("inbound_csv 与 lab_csv 必须是 CSV 文本")
    return in_text.encode("utf-8"), lab_text.encode("utf-8"), "inbound.csv", "lab.csv"


def _collect_upload() -> Tuple[bytes, bytes, str, str]:
    if request.files:
        return _collect_multipart()
    return _collect_json_or_form()


def wants_json() -> bool:
    if request.path.startswith("/api/"):
        return True
    if request.is_json:
        return True
    accept = request.headers.get("Accept", "")
    requested = request.headers.get("X-Requested-With", "")
    return "application/json" in accept or requested == "fetch" or requested == "XMLHttpRequest"


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    try:
        app.json.ensure_ascii = False  # readable Chinese in JSON responses
    except AttributeError:
        app.config["JSON_AS_ASCII"] = False

    @app.get("/healthz")
    @app.get("/health")
    def healthz():
        return jsonify(status="ok")

    # ---------------- page ----------------
    @app.get("/")
    def index():
        sample = request.args.get("sample")
        error = request.args.get("error")
        notice = request.args.get("notice")
        autorun = request.args.get("run") in ("1", "true", "yes")
        if sample:
            try:
                eng = load_sample(sample)
                if autorun:
                    eng.run_all()
            except UploadError as exc:
                error = str(exc)

        with STATE.lock:
            state = STATE.engine.state_dict() if STATE.engine is not None else None
            sample_case = STATE.sample_case
        return render_template(
            "index.html",
            state=state,
            error=error,
            notice=notice,
            sample_case=sample_case,
        )

    @app.get("/api/state")
    def api_state():
        with STATE.lock:
            if STATE.engine is None:
                return jsonify({"ok": True, "loaded": False})
            return jsonify({"ok": True, **STATE.engine.state_dict()})

    @app.get("/api/seeds")
    def api_seeds():
        """Built-in labelled samples and initial centroids (same for both)."""
        from recycling_backtest.centroids import SEED_SAMPLES, CentroidModel

        m = CentroidModel()
        return jsonify(
            {
                "ok": True,
                "features": ["moisture", "whiteness"],
                "label_convention": {"1": "qualified/合格", "0": "unqualified/不合格"},
                "seed_samples": [
                    {"moisture": s[0], "whiteness": s[1], "qualified": s[2]}
                    for s in SEED_SAMPLES
                ],
                "initial_centroids": m.centroids(),
                "rule": "nearest centroid; equidistant -> unqualified(0)",
            }
        )

    # ---------------- upload ----------------
    def _handle_upload():
        try:
            in_b, lab_b, in_name, lab_name = _collect_upload()
            eng = load_pair(in_b, lab_b, in_name, lab_name)
            if wants_json():
                return jsonify({"ok": True, **eng.state_dict()}), 200
            return redirect(url_for("index") + "#workspace", code=303)
        except UploadError as exc:
            if wants_json():
                return jsonify({"ok": False, "error": str(exc), "state_preserved": True}), 400
            return redirect("/?error=" + _q(str(exc)) + "#upload", code=303)

    for rule in ("/upload", "/api/upload", "/api/backtest/load", "/api/backtest/upload"):
        app.add_url_rule(rule, endpoint="upload" + rule.replace("/", "_"),
                         view_func=_handle_upload, methods=["POST"])

    # ---------------- sample loading ----------------
    def _handle_sample(case: str):
        try:
            eng = load_sample(case)
            if wants_json():
                return jsonify({"ok": True, **eng.state_dict()})
            return redirect(url_for("index") + "#workspace", code=303)
        except UploadError as exc:
            if wants_json():
                return jsonify({"ok": False, "error": str(exc)}), 404
            return redirect("/?error=" + _q(str(exc)) + "#upload", code=303)

    @app.post("/sample/<case>")
    def sample_form(case: str):
        return _handle_sample(case)

    @app.post("/api/sample/load")
    def sample_api_load():
        data = request.get_json(silent=True) if request.is_json else request.form
        case = (data or {}).get("name", "demo")
        return _handle_sample(case)

    @app.post("/api/sample/load/<case>")
    def sample_api_named(case: str):
        return _handle_sample(case)

    # ---------------- replay controls ----------------
    def _require_engine():
        if STATE.engine is None:
            raise UploadError("当前还没有可回放的数据，请先上传两个 CSV 或载入演示样例")
        return STATE.engine

    def _handle_step():
        try:
            with STATE.lock:
                eng = _require_engine()
                block = eng.step()
                state = eng.state_dict()
            if wants_json():
                return jsonify({"ok": True, "step": block, **state})
            return redirect(url_for("index") + "#workspace", code=303)
        except UploadError as exc:
            return _action_error(exc)

    def _handle_run():
        try:
            with STATE.lock:
                eng = _require_engine()
                n = eng.run_all()
                state = eng.state_dict()
            if wants_json():
                return jsonify({"ok": True, "processed_minutes": n, **state})
            return redirect(url_for("index") + "#workspace", code=303)
        except UploadError as exc:
            return _action_error(exc)

    def _handle_reset():
        try:
            with STATE.lock:
                eng = _require_engine()
                eng.reset()
                state = eng.state_dict()
            if wants_json():
                return jsonify({"ok": True, **state})
            return redirect(url_for("index") + "#workspace", code=303)
        except UploadError as exc:
            return _action_error(exc)

    def _action_error(exc: UploadError):
        if wants_json():
            return jsonify({"ok": False, "error": str(exc)}), 400
        return redirect("/?error=" + _q(str(exc)) + "#workspace", code=303)

    for rule in ("/step", "/api/step", "/api/replay/step", "/api/tick", "/next", "/api/next"):
        app.add_url_rule(rule, endpoint="step" + rule.replace("/", "_"),
                         view_func=_handle_step, methods=["POST"])
    for rule in ("/run", "/api/run", "/api/replay/run"):
        app.add_url_rule(rule, endpoint="run" + rule.replace("/", "_"),
                         view_func=_handle_run, methods=["POST"])
    for rule in ("/reset", "/api/reset", "/api/replay/reset"):
        app.add_url_rule(rule, endpoint="reset" + rule.replace("/", "_"),
                         view_func=_handle_reset, methods=["POST"])

    # ---------------- sample file download ----------------
    @app.get("/samples/<path:filename>")
    def download_sample(filename: str):
        return send_from_directory(SAMPLE_DIR, filename, as_attachment=True)

    @app.errorhandler(413)
    def too_large(_e):
        msg = "上传文件过大（总上限 32MB）"
        if wants_json():
            return jsonify({"ok": False, "error": msg}), 413
        return redirect("/?error=" + _q(msg) + "#upload", code=303)

    @app.errorhandler(500)
    def internal_error(e):
        traceback.print_exc()
        if wants_json():
            return jsonify({"ok": False, "error": f"服务器内部错误：{e}"}), 500
        return redirect("/?error=" + _q("服务器内部错误，请查看日志") + "#upload", code=303)

    return app


def _q(s: str) -> str:
    from urllib.parse import quote
    return quote(s, safe="")


app = create_app()


if __name__ == "__main__":
    # Default to Flask's conventional port 5000 for direct `python app.py`;
    # Docker/gunicorn binds 8000 inside the container and Compose publishes
    # both host 8000 and 5000 onto it.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, threaded=True)
