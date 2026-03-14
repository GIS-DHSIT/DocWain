from __future__ import annotations

import hashlib
from src.utils.logging_utils import get_logger
import math
import re
from typing import Any, Dict, Iterable, List, Optional

from src.api.pipeline_models import ExtractedDocument, Table
from src.embedding.chunking.section_chunker import normalize_text

logger = get_logger(__name__)

_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[\.)])\s+")

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:  # noqa: BLE001
        return default

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:  # noqa: BLE001
        return default

def _text_tokens(text: str) -> List[str]:
    return [tok for tok in (text or "").split() if tok]

def _is_all_caps(text: str) -> bool:
    clean = (text or "").strip()
    return bool(clean) and clean.isupper()

def _hash_id(seed: str, *, prefix: str = "b") -> str:
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{digest}"

def _bbox_union(bboxes: Iterable[List[float]]) -> List[float]:
    xs0, ys0, xs1, ys1 = [], [], [], []
    for box in bboxes:
        if not box or len(box) != 4:
            continue
        xs0.append(_safe_float(box[0]))
        ys0.append(_safe_float(box[1]))
        xs1.append(_safe_float(box[2]))
        ys1.append(_safe_float(box[3]))
    if not xs0:
        return [0.0, 0.0, 0.0, 0.0]
    return [min(xs0), min(ys0), max(xs1), max(ys1)]

def _column_split(points: List[float], page_width: float) -> Optional[float]:
    if not points or page_width <= 0:
        return None
    sorted_points = sorted(points)
    gaps = [(sorted_points[i + 1] - sorted_points[i], i) for i in range(len(sorted_points) - 1)]
    if not gaps:
        return None
    gap, idx = max(gaps, key=lambda g: g[0])
    if gap < max(20.0, page_width * 0.18):
        return None
    return (sorted_points[idx] + sorted_points[idx + 1]) / 2.0

def _assign_columns(blocks: List[Dict[str, Any]], page_width: float) -> None:
    x0s = [float(block["bbox"][0]) for block in blocks if block.get("bbox")]
    split = _column_split(x0s, page_width)
    if split is None:
        for block in blocks:
            block["_column"] = 0
        return
    for block in blocks:
        x0 = float(block["bbox"][0]) if block.get("bbox") else 0.0
        block["_column"] = 0 if x0 <= split else 1

def _reading_order(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(blocks, key=lambda b: (b.get("_column", 0), _safe_float(b["bbox"][1] if b.get("bbox") else 0), _safe_float(b["bbox"][0] if b.get("bbox") else 0)))

def _overlap_ratio(a0: float, a1: float, b0: float, b1: float) -> float:
    left = max(a0, b0)
    right = min(a1, b1)
    if right <= left:
        return 0.0
    return (right - left) / max(a1 - a0, b1 - b0, 1.0)

def _neighbors(blocks: List[Dict[str, Any]]) -> None:
    for block in blocks:
        block["neighbors"] = {"above": [], "below": [], "left": [], "right": []}

    for idx, block in enumerate(blocks):
        bbox = block.get("bbox") or [0, 0, 0, 0]
        x0, y0, x1, y1 = [float(v) for v in bbox]
        above = None
        below = None
        left = None
        right = None
        min_above = math.inf
        min_below = math.inf
        min_left = math.inf
        min_right = math.inf
        for jdx, other in enumerate(blocks):
            if idx == jdx:
                continue
            obox = other.get("bbox") or [0, 0, 0, 0]
            ox0, oy0, ox1, oy1 = [float(v) for v in obox]
            y_overlap = _overlap_ratio(y0, y1, oy0, oy1)
            x_overlap = _overlap_ratio(x0, x1, ox0, ox1)
            if oy1 <= y0 and x_overlap > 0.2:
                dist = y0 - oy1
                if dist < min_above:
                    min_above = dist
                    above = other
            if oy0 >= y1 and x_overlap > 0.2:
                dist = oy0 - y1
                if dist < min_below:
                    min_below = dist
                    below = other
            if ox1 <= x0 and y_overlap > 0.2:
                dist = x0 - ox1
                if dist < min_left:
                    min_left = dist
                    left = other
            if ox0 >= x1 and y_overlap > 0.2:
                dist = ox0 - x1
                if dist < min_right:
                    min_right = dist
                    right = other
        if above:
            block["neighbors"]["above"] = [above["block_id"]]
        if below:
            block["neighbors"]["below"] = [below["block_id"]]
        if left:
            block["neighbors"]["left"] = [left["block_id"]]
        if right:
            block["neighbors"]["right"] = [right["block_id"]]

def _infer_heading_level(blocks: List[Dict[str, Any]]) -> None:
    sizes = [b.get("style", {}).get("font_size") for b in blocks if b.get("style", {}).get("font_size")]
    if sizes:
        sizes_sorted = sorted(set(round(float(s), 1) for s in sizes))
        max_size = sizes_sorted[-1]
        mid_size = sizes_sorted[-2] if len(sizes_sorted) > 1 else max_size
        for block in blocks:
            size = block.get("style", {}).get("font_size")
            level = None
            if size:
                if float(size) >= max_size:
                    level = 1
                elif float(size) >= mid_size:
                    level = 2
            if level is None:
                text = (block.get("text") or "").strip()
                if len(text.split()) <= 8 and (_is_all_caps(text) or block.get("style", {}).get("font_weight") == "bold"):
                    level = 2
            block["structure"]["heading_level"] = level
    else:
        for block in blocks:
            text = (block.get("text") or "").strip()
            block["structure"]["heading_level"] = 2 if len(text.split()) <= 8 and _is_all_caps(text) else None

def _infer_list_level(blocks: List[Dict[str, Any]]) -> None:
    x0s = [float(b["bbox"][0]) for b in blocks if b.get("bbox")]
    if not x0s:
        for block in blocks:
            block["structure"]["list_level"] = 1 if _BULLET_RE.match((block.get("text") or "").strip()) else None
        return
    min_x = min(x0s)
    for block in blocks:
        text = (block.get("text") or "").strip()
        if not _BULLET_RE.match(text):
            block["structure"]["list_level"] = None
            continue
        x0 = float(block["bbox"][0]) if block.get("bbox") else min_x
        indent = max(0.0, x0 - min_x)
        level = 1 + int(indent / max(30.0, (max(x0s) - min_x) * 0.25))
        block["structure"]["list_level"] = max(1, level)

def _infer_headers_footers(pages: List[Dict[str, Any]]) -> None:
    if not pages:
        return
    top_counts: Dict[str, int] = {}
    bottom_counts: Dict[str, int] = {}
    for page in pages:
        blocks = page.get("blocks") or []
        if not blocks:
            continue
        first_text = (blocks[0].get("text") or "").strip()
        last_text = (blocks[-1].get("text") or "").strip()
        if first_text:
            top_counts[first_text] = top_counts.get(first_text, 0) + 1
        if last_text:
            bottom_counts[last_text] = bottom_counts.get(last_text, 0) + 1
    threshold = max(1, int(math.ceil(len(pages) * 0.6)))
    headers = {text for text, count in top_counts.items() if count >= threshold}
    footers = {text for text, count in bottom_counts.items() if count >= threshold}
    for page in pages:
        for block in page.get("blocks") or []:
            text = (block.get("text") or "").strip()
            if text in headers:
                block["type"] = "header"
            if text in footers:
                block["type"] = "footer"

def _build_table_payload(table: Table, *, table_id: str) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    if table.csv:
        lines = [ln for ln in table.csv.splitlines() if ln.strip()]
        for line in lines:
            cells = [cell.strip() for cell in line.split(",")]
            rows.append({"cells": [{"text": cell} for cell in cells]})
    elif table.text:
        lines = [ln for ln in table.text.splitlines() if ln.strip()]
        for line in lines:
            cells = [cell.strip() for cell in re.split(r"\s{2,}|\t|\|", line) if cell.strip()]
            rows.append({"cells": [{"text": cell} for cell in cells]})
    return {
        "table_id": table_id,
        "bbox": [0.0, 0.0, 0.0, 0.0],
        "rows": rows,
    }

def _synthetic_blocks_from_text(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for page in pages:
        page_num = _safe_int(page.get("page_number") or page.get("page") or 1, default=1)
        page_text = str(page.get("text") or page.get("content") or "")
        if not page_text.strip():
            continue
        y = 20.0
        for para in [p for p in page_text.split("\n") if p.strip()]:
            height = max(12.0, min(40.0, 8.0 + len(para) / 20.0))
            bbox = [40.0, y, 560.0, y + height]
            blocks.append(
                {
                    "page": page_num,
                    "bbox": bbox,
                    "text": para,
                    "block_type": "list" if _BULLET_RE.match(para.strip()) else "text",
                    "style": {
                        "font_size": None,
                        "font_weight": "normal",
                        "is_all_caps": _is_all_caps(para),
                        "line_count": len(para.splitlines()),
                    },
                }
            )
            y += height + 8.0
    return blocks

def build_layout_graph(
    extracted: Any,
    *,
    document_id: str,
    file_name: str,
) -> Dict[str, Any]:
    """Build a document-agnostic LayoutGraph from extracted payloads."""
    if isinstance(extracted, dict) and extracted.get("layout_graph"):
        graph = extracted.get("layout_graph") or {}
        graph["doc_id"] = graph.get("doc_id") or str(document_id)
        graph["file_name"] = graph.get("file_name") or file_name
        return graph

    layout_blocks: List[Dict[str, Any]] = []
    page_dims: Dict[int, Dict[str, float]] = {}
    tables: List[Table] = []
    full_text = ""
    ocr_used = False
    ocr_chars = 0
    if isinstance(extracted, ExtractedDocument):
        full_text = extracted.full_text or ""
        _raw_tables = extracted.tables
        tables = list(_raw_tables) if isinstance(_raw_tables, (list, tuple)) else []
        _raw_candidates = extracted.chunk_candidates
        if isinstance(_raw_candidates, (list, tuple)):
            ocr_used = any(cand.chunk_type == "ocr_text" for cand in _raw_candidates)
            ocr_chars = sum(len(cand.text or "") for cand in _raw_candidates if cand.chunk_type == "ocr_text")
        canon = extracted.canonical_json or {}
        _raw_blocks = canon.get("layout_blocks")
        layout_blocks = list(_raw_blocks) if isinstance(_raw_blocks, (list, tuple)) else []
        _raw_dims = canon.get("page_dims")
        page_dims = {int(k): v for k, v in _raw_dims.items()} if isinstance(_raw_dims, dict) else {}
        _raw_pages = canon.get("pages")
        if not layout_blocks and isinstance(_raw_pages, (list, tuple)) and _raw_pages:
            layout_blocks = _synthetic_blocks_from_text(_raw_pages)
    elif isinstance(extracted, dict):
        full_text = str(extracted.get("full_text") or extracted.get("text") or "")
        if isinstance(extracted.get("tables"), list):
            for idx, table in enumerate(extracted.get("tables") or []):
                if isinstance(table, Table):
                    tables.append(table)
                elif isinstance(table, dict):
                    tables.append(Table(page=_safe_int(table.get("page"), 1), text=str(table.get("text") or ""), csv=table.get("csv")))
        _raw_pages = extracted.get("pages")
        if isinstance(_raw_pages, (list, tuple)) and _raw_pages:
            layout_blocks = _synthetic_blocks_from_text(_raw_pages)
    elif isinstance(extracted, str):
        full_text = extracted
        layout_blocks = _synthetic_blocks_from_text([{"page_number": 1, "text": extracted}])

    if not layout_blocks:
        text = full_text or ""
        layout_blocks = _synthetic_blocks_from_text([{"page_number": 1, "text": text}])

    blocks_by_page: Dict[int, List[Dict[str, Any]]] = {}
    for block in layout_blocks:
        page_num = _safe_int(block.get("page") or 1, default=1)
        text = normalize_text(str(block.get("text") or ""))
        if not text:
            continue
        bbox = block.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        style = block.get("style") or {}
        block_type = (block.get("block_type") or block.get("type") or "text").lower()
        block_id = _hash_id(f"{document_id}|{page_num}|{bbox}|{text[:64]}", prefix="b")
        blocks_by_page.setdefault(page_num, []).append(
            {
                "block_id": block_id,
                "type": block_type,
                "bbox": [float(v) for v in bbox],
                "text": text,
                "tokens": _text_tokens(text),
                "style": {
                    "font_size": style.get("font_size"),
                    "font_weight": style.get("font_weight"),
                    "is_all_caps": style.get("is_all_caps") if style.get("is_all_caps") is not None else _is_all_caps(text),
                    "line_count": style.get("line_count") if style.get("line_count") is not None else len(text.splitlines()),
                },
                "neighbors": {"above": [], "below": [], "left": [], "right": []},
                "structure": {"list_level": None, "table_ref": None, "heading_level": None},
            }
        )

    pages: List[Dict[str, Any]] = []
    table_payloads: Dict[int, List[Dict[str, Any]]] = {}
    for idx, table in enumerate(tables, start=1):
        table_id = _hash_id(f"{document_id}|table|{idx}", prefix="t")
        table_payload = _build_table_payload(table, table_id=table_id)
        page_num = _safe_int(getattr(table, "page", None), default=1)
        table_payloads.setdefault(page_num, []).append(table_payload)

    for page_num in sorted(blocks_by_page.keys()):
        blocks = blocks_by_page[page_num]
        page_dim = page_dims.get(page_num) or {}
        page_width = _safe_float(page_dim.get("width"), default=600.0)
        page_height = _safe_float(page_dim.get("height"), default=800.0)
        _assign_columns(blocks, page_width)
        blocks = _reading_order(blocks)
        for block in blocks:
            block["structure"] = block.get("structure") or {"list_level": None, "table_ref": None, "heading_level": None}
        _infer_heading_level(blocks)
        _infer_list_level(blocks)
        _neighbors(blocks)
        # Attach table refs based on type or proximity.
        if table_payloads.get(page_num):
            for block in blocks:
                if block["type"] == "table":
                    block["structure"]["table_ref"] = table_payloads[page_num][0]["table_id"]
        pages.append(
            {
                "page": page_num,
                "width": page_width,
                "height": page_height,
                "blocks": blocks,
                "tables": table_payloads.get(page_num, []),
            }
        )

    _infer_headers_footers(pages)

    dominant_sizes: List[float] = []
    sizes = [b.get("style", {}).get("font_size") for p in pages for b in p.get("blocks") or [] if b.get("style", {}).get("font_size")]
    if sizes:
        rounded = {}
        for size in sizes:
            rounded_size = round(float(size), 1)
            rounded[rounded_size] = rounded.get(rounded_size, 0) + 1
        dominant_sizes = [size for size, _count in sorted(rounded.items(), key=lambda kv: kv[1], reverse=True)[:3]]

    total_chars = 0
    page_digests: List[Dict[str, Any]] = []
    has_lists = False
    has_tables = any(tables)
    for page in pages:
        texts = [blk.get("text") or "" for blk in page.get("blocks") or []]
        page_text = "\n".join([t for t in texts if t]).strip()
        total_chars += len(page_text)
        if any((blk.get("type") or "") == "list" for blk in page.get("blocks") or []):
            has_lists = True
        if any((blk.get("type") or "") == "table" for blk in page.get("blocks") or []):
            has_tables = True
        page_digests.append(
            {
                "page": page.get("page"),
                "chars": len(page_text),
                "sha1": hashlib.sha1(page_text.encode("utf-8")).hexdigest() if page_text else None,
            }
        )

    coverage_ratio = None
    if full_text:
        coverage_ratio = total_chars / max(len(full_text), 1)

    doc_signals = {
        "has_tables": has_tables,
        "has_lists": has_lists,
        "dominant_font_sizes": dominant_sizes,
        "ocr_used": ocr_used,
    }

    extraction_metrics = {
        "extracted_chars": total_chars,
        "coverage_ratio": coverage_ratio,
        "ocr_ratio": (ocr_chars / total_chars) if total_chars else 0.0,
        "table_count": len(tables),
    }

    return {
        "doc_id": str(document_id),
        "file_name": file_name,
        "page_count": len(pages),
        "pages": pages,
        "doc_signals": doc_signals,
        "page_digests": page_digests,
        "extraction_metrics": extraction_metrics,
    }

__all__ = ["build_layout_graph"]
