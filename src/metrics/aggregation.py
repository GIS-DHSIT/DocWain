from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python < 3.9 fallback
    ZoneInfo = None


def normalize_timezone(tz_name: str) -> dt.tzinfo:
    if not tz_name:
        return dt.timezone.utc
    if tz_name.upper() == "UTC":
        return dt.timezone.utc
    if ZoneInfo is None:
        return dt.timezone.utc
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return dt.timezone.utc


def normalize_week_start(value: str) -> int:
    if not value:
        return 0
    value = value.strip().upper()
    if value == "SUN":
        return 6
    return 0


def compute_date_range(
    days: int,
    tz: dt.tzinfo,
    today: Optional[dt.date] = None,
) -> Tuple[dt.date, dt.date, dt.datetime, dt.datetime]:
    if days <= 0:
        raise ValueError("days must be positive")
    now = dt.datetime.now(tz)
    end_date = today or now.date()
    start_date = end_date - dt.timedelta(days=days - 1)
    range_start = dt.datetime.combine(start_date, dt.time.min, tzinfo=tz)
    range_end = dt.datetime.combine(end_date, dt.time.max, tzinfo=tz)
    return start_date, end_date, range_start, range_end


def iter_hour_keys(start_utc: dt.datetime, end_utc: dt.datetime) -> Iterator[str]:
    cursor = start_utc.replace(minute=0, second=0, microsecond=0)
    end_utc = end_utc.replace(minute=0, second=0, microsecond=0)
    while cursor < end_utc:
        yield cursor.strftime("%Y%m%d%H")
        cursor += dt.timedelta(hours=1)


def daily_boundaries(
    start_date: dt.date,
    days: int,
    tz: dt.tzinfo,
) -> List[Tuple[str, dt.datetime, dt.datetime]]:
    buckets = []
    for offset in range(days):
        day = start_date + dt.timedelta(days=offset)
        start_local = dt.datetime.combine(day, dt.time.min, tzinfo=tz)
        end_local = start_local + dt.timedelta(days=1)
        buckets.append(
            (
                day.strftime("%Y-%m-%d"),
                start_local.astimezone(dt.timezone.utc),
                end_local.astimezone(dt.timezone.utc),
            )
        )
    return buckets


def weekly_boundaries(
    daily: List[Tuple[str, dt.datetime, dt.datetime]],
    week_start: int,
) -> List[Tuple[str, str, dt.datetime, dt.datetime]]:
    grouped: Dict[dt.date, List[Tuple[str, dt.datetime, dt.datetime]]] = defaultdict(list)
    for date_str, start_utc, end_utc in daily:
        date_val = dt.date.fromisoformat(date_str)
        offset = (date_val.weekday() - week_start) % 7
        week_start_date = date_val - dt.timedelta(days=offset)
        grouped[week_start_date].append((date_str, start_utc, end_utc))

    buckets: List[Tuple[str, str, dt.datetime, dt.datetime]] = []
    for week_start_date in sorted(grouped.keys()):
        days = grouped[week_start_date]
        week_end_date = dt.date.fromisoformat(days[-1][0])
        start_utc = days[0][1]
        end_utc = days[-1][2]
        buckets.append(
            (
                week_start_date.strftime("%Y-%m-%d"),
                week_end_date.strftime("%Y-%m-%d"),
                start_utc,
                end_utc,
            )
        )
    return buckets


def empty_aggregate() -> Dict[str, Any]:
    return {
        "totals": defaultdict(float),
        "mins": {},
        "maxs": {},
        "dists": defaultdict(lambda: defaultdict(float)),
        "hists": defaultdict(lambda: defaultdict(float)),
    }


def merge_aggregate(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in (source.get("totals") or {}).items():
        target["totals"][key] += value
    for key, value in (source.get("mins") or {}).items():
        if key not in target["mins"] or value < target["mins"][key]:
            target["mins"][key] = value
    for key, value in (source.get("maxs") or {}).items():
        if key not in target["maxs"] or value > target["maxs"][key]:
            target["maxs"][key] = value
    for name, bins in (source.get("dists") or {}).items():
        for label, amount in bins.items():
            target["dists"][name][label] += amount
    for name, bins in (source.get("hists") or {}).items():
        for label, amount in bins.items():
            target["hists"][name][label] += amount
    return target


def aggregate_range(
    hourly: Dict[str, Dict[str, Any]],
    start_utc: dt.datetime,
    end_utc: dt.datetime,
) -> Dict[str, Any]:
    agg = empty_aggregate()
    for hour_key in iter_hour_keys(start_utc, end_utc):
        if hour_key in hourly:
            merge_aggregate(agg, hourly[hour_key])
    return agg


def _avg(totals: Dict[str, float], name: str) -> Optional[float]:
    total = totals.get(f"sum:{name}", 0.0)
    count = totals.get(f"count:{name}", 0.0)
    if not count:
        return None
    return total / count


def _rate(totals: Dict[str, float], numer: str, denom: str) -> Optional[float]:
    num = totals.get(f"count:{numer}", 0.0)
    den = totals.get(f"count:{denom}", 0.0)
    if not den:
        return None
    return num / den


def hist_quantile(hist: Dict[str, float], quantile: float, buckets: Tuple[int, ...]) -> Optional[float]:
    if not hist:
        return None
    labels = [f"le_{edge}" for edge in buckets] + [f"gt_{buckets[-1]}"]
    counts = [hist.get(label, 0.0) for label in labels]
    total = sum(counts)
    if total <= 0:
        return None
    target = total * quantile
    cumulative = 0.0
    for label, edge, count in zip(labels, list(buckets) + [buckets[-1]], counts):
        cumulative += count
        if cumulative >= target:
            return float(edge)
    return float(buckets[-1])


def build_metrics_payload(
    aggregate: Dict[str, Any],
    *,
    latency_buckets: Tuple[int, ...],
) -> Dict[str, Any]:
    totals = aggregate.get("totals") or {}
    mins = aggregate.get("mins") or {}
    maxs = aggregate.get("maxs") or {}
    dists = aggregate.get("dists") or {}
    hists = aggregate.get("hists") or {}

    notes = defaultdict(list)

    def _with_note(group: str, message: str) -> None:
        if message not in notes[group]:
            notes[group].append(message)

    doc_accuracy = _avg(totals, "text_extraction_accuracy_pct")
    if doc_accuracy is None:
        _with_note("document_extraction", "No extraction signals available for this bucket.")
    else:
        _with_note("document_extraction", "Derived from extraction artifacts; ground truth not available.")

    structure_accuracy = _avg(totals, "structure_extraction_accuracy_pct")
    if structure_accuracy is None:
        _with_note("document_extraction", "No structure extraction signals available for this bucket.")

    retrieval_accuracy = None
    _with_note("retrieval", "Retrieval accuracy requires labeled evaluation data.")

    response_consistency = _avg(totals, "response_consistency_score")
    if response_consistency is None:
        _with_note("llm_performance", "Response consistency signal missing for this bucket.")

    temp_sensitivity = _avg(totals, "temperature_sensitivity_score")
    if temp_sensitivity is None:
        _with_note("llm_performance", "Temperature sensitivity signal missing for this bucket.")

    request_latency_hist = hists.get("request_latency_ms", {})
    ingest_p50 = hist_quantile(request_latency_hist, 0.50, latency_buckets)
    ingest_p95 = hist_quantile(request_latency_hist, 0.95, latency_buckets)
    if ingest_p50 is None:
        _with_note("system_health", "Request latency histogram missing for this bucket.")
    else:
        _with_note("system_health", "Ingest-to-answer latency uses request processing time as proxy.")

    cold_hist = hists.get("cold_start_latency_ms", {})
    warm_hist = hists.get("warm_start_latency_ms", {})

    cold_p50 = hist_quantile(cold_hist, 0.50, latency_buckets)
    cold_p95 = hist_quantile(cold_hist, 0.95, latency_buckets)
    warm_p50 = hist_quantile(warm_hist, 0.50, latency_buckets)
    warm_p95 = hist_quantile(warm_hist, 0.95, latency_buckets)
    if cold_p50 is None and warm_p50 is None:
        _with_note("system_health", "Cold/warm latency split unavailable for this bucket.")

    token_total = totals.get("sum:prompt_tokens", 0.0) + totals.get("sum:completion_tokens", 0.0)
    token_efficiency = (totals.get("sum:completion_tokens", 0.0) / token_total) if token_total else None

    tool_usage = dists.get("tool_usage", {})
    tool_total = sum(tool_usage.values())
    tool_usage_pct = {
        tool: round((count / tool_total) * 100, 2)
        for tool, count in tool_usage.items()
    } if tool_total else {}
    if tool_total == 0:
        _with_note("tool_usage", "No tool calls recorded for this bucket.")

    model_usage = dists.get("model_usage", {})

    return {
        "counts": {
            "requests": int(totals.get("count:requests_total", 0)),
            "documents": int(totals.get("count:documents_processed", 0)),
            "retrievals": int(totals.get("count:retrieval_total", 0)),
            "llm_calls": int(
                max(
                    totals.get("count:llm_request_count", 0),
                    totals.get("count:requests_total", 0),
                )
            ),
            "tool_calls": int(totals.get("count:tool_calls", 0)),
            "failures": int(sum(dists.get("failure_type", {}).values())),
        },
        "document_extraction": {
            "text_extraction_accuracy_pct": round(doc_accuracy * 100, 2) if doc_accuracy is not None else None,
            "ocr_confidence_avg": round(_avg(totals, "ocr_confidence"), 2) if _avg(totals, "ocr_confidence") is not None else None,
            "ocr_confidence_min": round(mins.get("ocr_confidence"), 2) if mins.get("ocr_confidence") is not None else None,
            "ocr_confidence_max": round(maxs.get("ocr_confidence"), 2) if maxs.get("ocr_confidence") is not None else None,
            "structure_extraction_accuracy_pct": round(structure_accuracy * 100, 2) if structure_accuracy is not None else None,
            "missing_content_ratio": round(_avg(totals, "missing_content_ratio"), 4) if _avg(totals, "missing_content_ratio") is not None else None,
            "corrupted_page_ratio": round(_avg(totals, "corrupted_page_ratio"), 4) if _avg(totals, "corrupted_page_ratio") is not None else None,
            "note": "; ".join(notes["document_extraction"]) if notes["document_extraction"] else None,
        },
        "semantic_quality": {
            "sentence_transformation_similarity_score": round(_avg(totals, "sentence_transformation_similarity_score"), 4)
            if _avg(totals, "sentence_transformation_similarity_score") is not None else None,
            "semantic_preservation_score": round(_avg(totals, "semantic_preservation_score"), 4)
            if _avg(totals, "semantic_preservation_score") is not None else None,
            "embedding_coherence_score": round(_avg(totals, "embedding_coherence_score"), 4)
            if _avg(totals, "embedding_coherence_score") is not None else None,
            "chunk_semantic_drift_score": round(_avg(totals, "chunk_semantic_drift_score"), 4)
            if _avg(totals, "chunk_semantic_drift_score") is not None else None,
            "hallucination_risk_score": round(_avg(totals, "hallucination_risk_score"), 4)
            if _avg(totals, "hallucination_risk_score") is not None else None,
            "answer_confidence_score": round(_avg(totals, "answer_confidence_score"), 4)
            if _avg(totals, "answer_confidence_score") is not None else None,
            "evidence_support_score": round(_avg(totals, "evidence_support_score"), 4)
            if _avg(totals, "evidence_support_score") is not None else None,
            "citation_coverage_score": round(_avg(totals, "citation_coverage_score"), 4)
            if _avg(totals, "citation_coverage_score") is not None else None,
            "numeric_support_rate": round(_avg(totals, "numeric_support_rate"), 4)
            if _avg(totals, "numeric_support_rate") is not None else None,
            "note": "; ".join(notes["semantic_quality"]) if notes["semantic_quality"] else None,
        },
        "retrieval": {
            "retrieval_accuracy_pct": retrieval_accuracy,
            "top_k_hit_rate": round(_rate(totals, "retrieval_hits", "retrieval_total"), 4)
            if _rate(totals, "retrieval_hits", "retrieval_total") is not None else None,
            "mean_reciprocal_rank": round(_avg(totals, "mean_reciprocal_rank"), 4)
            if _avg(totals, "mean_reciprocal_rank") is not None else None,
            "context_grounding_score": round(_avg(totals, "context_grounding_score"), 4)
            if _avg(totals, "context_grounding_score") is not None else None,
            "answer_faithfulness_score": round(_avg(totals, "answer_faithfulness_score"), 4)
            if _avg(totals, "answer_faithfulness_score") is not None else None,
            "answer_relevance_score": round(_avg(totals, "answer_relevance_score"), 4)
            if _avg(totals, "answer_relevance_score") is not None else None,
            "note": "; ".join(notes["retrieval"]) if notes["retrieval"] else None,
        },
        "tool_usage": {
            "tool_usage_distribution_pct": tool_usage_pct,
            "tool_success_rate": round(_rate(totals, "tool_success", "tool_calls"), 4)
            if _rate(totals, "tool_success", "tool_calls") is not None else None,
            "tool_failure_rate": round(_rate(totals, "tool_failure", "tool_calls"), 4)
            if _rate(totals, "tool_failure", "tool_calls") is not None else None,
            "agent_execution_distribution": dists.get("agent_execution", {}),
            "avg_tool_latency_ms": round(_avg(totals, "tool_latency_ms"), 2)
            if _avg(totals, "tool_latency_ms") is not None else None,
            "llm_without_tool_fallback_rate": round(_rate(totals, "llm_without_tool_fallback", "requests_with_tools"), 4)
            if _rate(totals, "llm_without_tool_fallback", "requests_with_tools") is not None else None,
            "note": "; ".join(notes["tool_usage"]) if notes["tool_usage"] else None,
        },
        "llm_performance": {
            "prompt_tokens_avg": round(_avg(totals, "prompt_tokens"), 2) if _avg(totals, "prompt_tokens") is not None else None,
            "completion_tokens_avg": round(_avg(totals, "completion_tokens"), 2) if _avg(totals, "completion_tokens") is not None else None,
            "token_efficiency_score": round(token_efficiency, 4) if token_efficiency is not None else None,
            "latency_p50_ms": hist_quantile(hists.get("llm_latency_ms", {}), 0.50, latency_buckets),
            "latency_p95_ms": hist_quantile(hists.get("llm_latency_ms", {}), 0.95, latency_buckets),
            "latency_p99_ms": hist_quantile(hists.get("llm_latency_ms", {}), 0.99, latency_buckets),
            "model_usage_breakdown": model_usage,
            "response_consistency_score": round(response_consistency, 4) if response_consistency is not None else None,
            "temperature_sensitivity_score": round(temp_sensitivity, 4) if temp_sensitivity is not None else None,
            "note": "; ".join(notes["llm_performance"]) if notes["llm_performance"] else None,
        },
        "system_health": {
            "end_to_end_success_rate": round(_rate(totals, "requests_success", "requests_total"), 4)
            if _rate(totals, "requests_success", "requests_total") is not None else None,
            "ingest_to_answer_latency_ms": {
                "p50_ms": ingest_p50,
                "p95_ms": ingest_p95,
            },
            "failure_type_distribution": dists.get("failure_type", {}),
            "cold_start_vs_warm_start_latency": {
                "cold_start_p50_ms": cold_p50,
                "cold_start_p95_ms": cold_p95,
                "warm_start_p50_ms": warm_p50,
                "warm_start_p95_ms": warm_p95,
            },
            "retry_rate": round(_rate(totals, "llm_retry_count", "llm_request_count"), 4)
            if _rate(totals, "llm_retry_count", "llm_request_count") is not None else None,
            "note": "; ".join(notes["system_health"]) if notes["system_health"] else None,
        },
    }
