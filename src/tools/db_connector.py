from __future__ import annotations

from src.utils.logging_utils import get_logger
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field, field_validator

from src.tools.base import ToolError, generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record

logger = get_logger(__name__)

router = APIRouter(prefix="/db", tags=["Tools-DB"])

ALLOWED_DB_TYPES = {"postgres", "mysql", "mssql", "sqlite"}
MAX_ROWS = 200

class DBConnectRequest(BaseModel):
    db_type: str = Field(..., description="Database type", pattern="^(postgres|mysql|mssql|sqlite)$")
    connection_string: Optional[str] = Field(default=None, description="DSN or SQLite path")
    database: Optional[str] = Field(default=None, description="Database name")

    @field_validator("connection_string", mode="before")
    def _require_connection(cls, value: Optional[str], info):  # noqa: N805
        data = info.data
        if data.get("db_type") == "sqlite":
            return value or ":memory:"
        return value

class DBQueryRequest(DBConnectRequest):
    query: str = Field(..., description="SELECT-only query")
    limit: int = Field(default=50, ge=1, le=MAX_ROWS)

class DBIngestRequest(DBQueryRequest):
    chunk_prefix: Optional[str] = None

def _validate_db_type(db_type: str) -> None:
    if db_type not in ALLOWED_DB_TYPES:
        raise ToolError(f"Unsupported database type '{db_type}'", code="db_type_not_allowed")

def _validate_select(query: str) -> None:
    normalized = query.strip().lower()
    if not normalized.startswith("select"):
        raise ToolError("Only SELECT queries are allowed", code="invalid_query")
    forbidden = [";--", "drop ", "delete ", "update ", "insert "]
    if any(token in normalized for token in forbidden):
        raise ToolError("Dangerous query detected", code="invalid_query")

def _connect_sqlite(conn_str: str) -> sqlite3.Connection:
    try:
        return sqlite3.connect(conn_str, check_same_thread=False)
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Failed to connect to sqlite database: {exc}", code="connection_failed") from exc

def _connect(db_type: str, conn_str: str) -> Tuple[Any, str]:
    _validate_db_type(db_type)
    if db_type == "sqlite":
        return _connect_sqlite(conn_str), "sqlite"
    raise ToolError(f"{db_type} driver is not installed in this environment", code="driver_missing", status_code=501)

def _execute_select(conn: sqlite3.Connection, query: str, limit: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    _validate_select(query)
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]
        rows = []
        for idx, row in enumerate(cursor.fetchall()):
            if idx >= limit:
                break
            rows.append({col: row[i] for i, col in enumerate(columns)})
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"Query failed: {exc}", code="query_failed") from exc
    return rows, columns

def _summarize_rows(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "No rows returned."
    preview = rows[:3]
    return f"Retrieved {len(rows)} row(s). Preview: {preview}"

@register_tool("db_connector")
async def db_connector_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    input_payload = payload.get("input") or payload
    action = (input_payload.get("action") or "query").lower()
    warnings: list[str] = []

    if action == "connect":
        req = DBConnectRequest(**input_payload)
        _validate_db_type(req.db_type)
        return {
            "result": {"connected": True, "db_type": req.db_type},
            "sources": [build_source_record("db", correlation_id or "db", title=req.db_type)],
            "context_found": True,
            "grounded": True,
        }

    req = DBQueryRequest(**input_payload)
    conn, backend = _connect(req.db_type, req.connection_string or "")
    try:
        rows, columns = _execute_select(conn, req.query, req.limit)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    summary = _summarize_rows(rows)
    result = {
        "rows": rows,
        "columns": columns,
        "summary": summary,
        "backend": backend,
    }
    return {
        "result": result,
        "sources": [build_source_record("db", correlation_id or "db", title=req.db_type, metadata={"backend": backend})],
        "context_found": True,
        "grounded": True,
        "warnings": warnings,
    }

@router.post("/connect")
async def connect(request: DBConnectRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        _validate_db_type(request.db_type)
        return standard_response(
            "db_connector",
            result={"connected": True, "db_type": request.db_type},
            sources=[build_source_record("db", cid, title=request.db_type)],
            grounded=True,
            context_found=True,
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "db_connector",
            status="error",
            grounded=False,
            context_found=False,
            error=exc.as_dict(),
            correlation_id=cid,
        )

@router.post("/query")
async def query(request: DBQueryRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        conn, backend = _connect(request.db_type, request.connection_string or "")
        try:
            rows, columns = _execute_select(conn, request.query, request.limit)
        finally:
            try:
                conn.close()
            except Exception:
                pass
        summary = _summarize_rows(rows)
        return standard_response(
            "db_connector",
            result={"rows": rows, "columns": columns, "summary": summary, "backend": backend},
            sources=[build_source_record("db", cid, title=request.db_type)],
            grounded=True,
            context_found=True,
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "db_connector",
            status="error",
            grounded=False,
            context_found=False,
            error=exc.as_dict(),
            correlation_id=cid,
        )

@router.post("/ingest")
async def ingest(request: DBIngestRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    try:
        conn, backend = _connect(request.db_type, request.connection_string or "")
        try:
            rows, columns = _execute_select(conn, request.query, request.limit)
        finally:
            try:
                conn.close()
            except Exception:
                pass

        chunks = []
        prefix = request.chunk_prefix or "db_row"
        for idx, row in enumerate(rows):
            snippet = "; ".join(f"{k}: {v}" for k, v in row.items())
            chunks.append({"id": f"{prefix}_{idx}", "text": snippet, "metadata": {"row_index": idx}})

        return standard_response(
            "db_connector",
            result={"chunks": chunks, "columns": columns, "backend": backend},
            sources=[build_source_record("db", cid, title=request.db_type)],
            grounded=True,
            context_found=True,
            correlation_id=cid,
        )
    except ToolError as exc:
        return standard_response(
            "db_connector",
            status="error",
            grounded=False,
            context_found=False,
            error=exc.as_dict(),
            correlation_id=cid,
        )
