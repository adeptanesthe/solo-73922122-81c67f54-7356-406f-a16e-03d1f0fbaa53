"""CSV loading and strict validation for intake / lab files.

The parser is deliberately tolerant about encoding and header wording
(Chinese / English aliases, spaces, BOM, units in headers) but strict about
data integrity:

* every required column must exist;
* minutes must be integral;
* moisture / whiteness must be finite numbers;
* labels must be recognisable 合格 / 不合格 tokens;
* intake batch ids must be unique (one feature vector per batch);
* at most one lab receipt for the same (batch, minute).

Any :class:`UploadError` is raised *before* the caller touches the currently
loaded replay, so a failed upload preserves the running back-test.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class UploadError(ValueError):
    """User-facing upload/validation failure (message is Chinese)."""


# ---------------------------------------------------------------------------
# Header normalisation / aliases
# ---------------------------------------------------------------------------
def _norm_header(value: object) -> str:
    s = str(value if value is not None else "")
    s = s.strip().lstrip("﻿")
    # drop bracketed units such as (%) / （%）
    for open_ch, close_ch in (("(", ")"), ("（", "）")):
        while open_ch in s and close_ch in s:
            i = s.index(open_ch)
            j = s.index(close_ch, i + 1)
            s = s[:i] + s[j + 1 :]
    for ch in (" ", "　", "\t", "\n", "\r", "_", "-", "/", ".", ":", "："):
        s = s.replace(ch, "")
    return s.lower()


_INBOUND_ALIASES: Dict[str, str] = {}
_LAB_ALIASES: Dict[str, str] = {}


def _register(target: Dict[str, str], canonical: str, aliases: List[str]) -> None:
    for a in [canonical] + aliases:
        target[_norm_header(a)] = canonical


_register(
    _INBOUND_ALIASES,
    "batch",
    ["批号", "批次号", "批次", "编号", "车号", "batch", "batchid", "batchno", "id"],
)
_register(
    _INBOUND_ALIASES,
    "minute",
    [
        "入厂分钟",
        "进厂分钟",
        "到厂分钟",
        "入厂时间分钟",
        "入厂时刻",
        "minute",
        "inboundminute",
        "arrivalminute",
        "inminute",
        "time",
    ],
)
_register(
    _INBOUND_ALIASES,
    "moisture",
    ["含水率", "水份", "水分", "含水率pct", "moisture", "moisturecontent", "watercontent", "water"],
)
_register(
    _INBOUND_ALIASES,
    "whiteness",
    ["白度", "白度值", "whiteness", "brightness"],
)

_register(
    _LAB_ALIASES,
    "batch",
    ["批号", "批次号", "批次", "编号", "batch", "batchid", "batchno", "id"],
)
_register(
    _LAB_ALIASES,
    "minute",
    [
        "出单分钟",
        "出报告分钟",
        "化验出单分钟",
        "报告分钟",
        "发单分钟",
        "化验分钟",
        "出单时间分钟",
        "minute",
        "reportminute",
        "labminute",
        "issueminute",
        "time",
    ],
)
_register(
    _LAB_ALIASES,
    "qualified",
    [
        "合格与否",
        "是否合格",
        "合格不合格",
        "合格",
        "结果",
        "化验结果",
        "判定",
        "质量判定",
        "qualified",
        "pass",
        "passed",
        "result",
        "label",
        "isqualified",
        "status",
    ],
)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------
_QUALIFIED_TOKENS = {
    "合格", "合格的", "是", "真", "ok", "yes", "y", "true", "1", "pass", "passed", "good", "q",
}
_UNQUALIFIED_TOKENS = {
    "不合格", "不合格的", "否", "假", "ng", "no", "n", "false", "0", "fail", "failed", "bad",
    "unqualified", "reject", "rejected",
}


def parse_label(raw: object, row_no: int, column: str = "合格与否") -> int:
    from .centroids import QUALIFIED, UNQUALIFIED

    s = str(raw if raw is not None else "").strip().lower()
    compact = s.replace(" ", "")
    if compact in _QUALIFIED_TOKENS or s in _QUALIFIED_TOKENS:
        return QUALIFIED
    if compact in _UNQUALIFIED_TOKENS or s in _UNQUALIFIED_TOKENS:
        return UNQUALIFIED
    # tolerant fallback for phrases such as "判定：不合格"
    if "不合格" in compact or "不达标" in compact or "fail" in compact:
        return UNQUALIFIED
    if "合格" in compact or "达标" in compact or "pass" in compact:
        return QUALIFIED
    raise UploadError(f"第 {row_no} 行“{column}”取值无法识别：{raw!r}（请用 合格/不合格）")


def _to_number(s: str) -> Optional[float]:
    try:
        return float(s)
    except ValueError:
        m = _NUMBER_RE.search(s)
        return float(m.group(0)) if m else None


def parse_minute(raw: object, row_no: int, column: str) -> int:
    s = str(raw if raw is not None else "").strip()
    if s == "":
        raise UploadError(f"第 {row_no} 行缺少“{column}”")
    f = _to_number(s)
    if f is None or not math.isfinite(f) or abs(f - round(f)) > 1e-9:
        raise UploadError(f"第 {row_no} 行“{column}”不是整数分钟：{raw!r}")
    return int(round(f))


def parse_float(raw: object, row_no: int, column: str) -> float:
    s = str(raw if raw is not None else "").strip().replace("%", "")
    if s == "":
        raise UploadError(f"第 {row_no} 行缺少“{column}”数值")
    v = _to_number(s)
    if v is None or not math.isfinite(v):
        raise UploadError(f"第 {row_no} 行“{column}”不是数值：{raw!r}")
    return v


def parse_batch(raw: object, row_no: int) -> str:
    b = str(raw if raw is not None else "").strip()
    if not b:
        raise UploadError(f"第 {row_no} 行批号为空")
    return b


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def decode_bytes(content: bytes) -> str:
    if isinstance(content, str):
        return content
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return content.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UploadError("文件编码无法识别，请另存为 UTF-8 编码的 CSV 后重试")


def _map_headers(fieldnames: List[Optional[str]], alias_map: Dict[str, str], kind: str):
    mapping: Dict[str, int] = {}
    headers: List[str] = []
    for idx, h in enumerate(fieldnames or []):
        if h is None:
            continue
        key = _norm_header(h)
        headers.append(key)
        canonical = alias_map.get(key)
        if canonical and canonical not in mapping:
            mapping[canonical] = idx
    required = {
        "inbound": ["batch", "minute", "moisture", "whiteness"],
        "lab": ["batch", "minute", "qualified"],
    }[kind]
    cn = {
        "batch": "批号",
        "minute": "分钟",
        "moisture": "含水率",
        "whiteness": "白度",
        "qualified": "合格与否",
    }
    missing = [cn[c] for c in required if c not in mapping]
    if missing:
        raise UploadError(
            f"{'入厂' if kind == 'inbound' else '化验'} CSV 缺少必需列：{ '、'.join(missing) }"
            f"（实际表头：{', '.join(str(h) for h in (fieldnames or []) if h)}）"
        )
    return mapping


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InboundRow:
    batch: str
    minute: int
    moisture: float
    whiteness: float
    line_no: int

    @property
    def x(self) -> Tuple[float, float]:
        return (self.moisture, self.whiteness)


@dataclass(frozen=True)
class LabRow:
    batch: str
    minute: int
    qualified: int
    label_raw: str
    line_no: int


@dataclass
class Dataset:
    inbound: List[InboundRow]
    lab: List[LabRow]
    inbound_name: str = "inbound.csv"
    lab_name: str = "lab.csv"

    @property
    def inbound_by_batch(self) -> Dict[str, InboundRow]:
        return {r.batch: r for r in self.inbound}


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def _data_rows(text: str):
    reader = csv.reader(io.StringIO(text))
    try:
        header = next(reader)
    except StopIteration:
        raise UploadError("CSV 为空，缺少表头行")
    return header, reader


def parse_inbound_csv(content: bytes | str, filename: str = "inbound.csv") -> List[InboundRow]:
    text = decode_bytes(content) if isinstance(content, bytes) else content
    header, reader = _data_rows(text)
    mapping = _map_headers(header, _INBOUND_ALIASES, "inbound")
    rows: List[InboundRow] = []
    seen: Dict[str, int] = {}
    for raw_row in reader:
        line_no = reader.line_num
        if not raw_row or all(str(c).strip() == "" for c in raw_row):
            continue  # skip blank lines

        def cell(key: str) -> str:
            idx = mapping[key]
            return raw_row[idx] if idx < len(raw_row) else ""

        batch = parse_batch(cell("batch"), line_no)
        if batch in seen:
            raise UploadError(
                f"入厂 CSV 批号 {batch!r} 重复（第 {seen[batch]} 行与第 {line_no} 行）："
                "每批只能有一条入厂特征记录，回单改判才能定位并替换其训练贡献"
            )
        seen[batch] = line_no
        minute = parse_minute(cell("minute"), line_no, "入厂分钟")
        moisture = parse_float(cell("moisture"), line_no, "含水率")
        whiteness = parse_float(cell("whiteness"), line_no, "白度")
        rows.append(InboundRow(batch, minute, moisture, whiteness, line_no))
    if not rows:
        raise UploadError("入厂 CSV 没有任何数据行")
    return rows


def parse_lab_csv(content: bytes | str, filename: str = "lab.csv") -> List[LabRow]:
    text = decode_bytes(content) if isinstance(content, bytes) else content
    header, reader = _data_rows(text)
    mapping = _map_headers(header, _LAB_ALIASES, "lab")
    rows: List[LabRow] = []
    seen_key: Dict[Tuple[str, int], int] = {}
    for raw_row in reader:
        line_no = reader.line_num
        if not raw_row or all(str(c).strip() == "" for c in raw_row):
            continue

        def cell(key: str) -> str:
            idx = mapping[key]
            return raw_row[idx] if idx < len(raw_row) else ""

        batch = parse_batch(cell("batch"), line_no)
        minute = parse_minute(cell("minute"), line_no, "出单分钟")
        key = (batch, minute)
        if key in seen_key:
            raise UploadError(
                f"化验 CSV 违反“同批同分钟限一张回单”：批号 {batch!r} 在第 {minute} 分钟"
                f"出现多张回单（第 {seen_key[key]} 行与第 {line_no} 行）"
            )
        seen_key[key] = line_no
        label_raw = cell("qualified")
        qualified = parse_label(label_raw, line_no)
        rows.append(LabRow(batch, minute, qualified, str(label_raw).strip(), line_no))
    if not rows:
        raise UploadError("化验 CSV 没有任何数据行")
    return rows


def build_dataset(
    inbound_content: bytes | str,
    lab_content: bytes | str,
    inbound_name: str = "inbound.csv",
    lab_name: str = "lab.csv",
) -> Dataset:
    """Parse and cross-validate both files; raises UploadError atomically."""
    inbound = parse_inbound_csv(inbound_content, inbound_name)
    lab = parse_lab_csv(lab_content, lab_name)

    inbound_batches = {r.batch for r in inbound}
    missing = sorted({r.batch for r in lab if r.batch not in inbound_batches})
    if missing:
        shown = "、".join(missing[:5]) + ("……" if len(missing) > 5 else "")
        raise UploadError(
            f"化验 CSV 中有 {len(missing)} 个批号在入厂 CSV 中找不到，"
            f"无法取得特征用于模型更新：{shown}"
        )

    return Dataset(inbound=inbound, lab=lab, inbound_name=inbound_name, lab_name=lab_name)


def detect_kind(filename: str, content: bytes) -> Optional[str]:
    """Best-effort detection of whether an uploaded CSV is intake or lab."""
    name = (filename or "").lower()
    if any(k in name for k in ("lab", "huayan", "化验")):
        return "lab"
    if any(k in name for k in ("in", "ruchang", "intake", "入厂", "进厂")):
        return "inbound"
    try:
        text = decode_bytes(content[:4096])
    except Exception:
        return None
    first_line = text.splitlines()[0] if text.splitlines() else ""
    headers = next(csv.reader([first_line])) if first_line else []
    keys = {_norm_header(h) for h in headers}
    in_score = sum(1 for k in keys if _INBOUND_ALIASES.get(k))
    lab_score = sum(1 for k in keys if _LAB_ALIASES.get(k))
    if in_score > lab_score:
        return "inbound"
    if lab_score > in_score:
        return "lab"
    return None
