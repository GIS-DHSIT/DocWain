import datetime as dt

import pytest

from src.metrics.aggregation import (
    aggregate_range,
    build_metrics_payload,
    compute_date_range,
    daily_boundaries,
    hist_quantile,
    normalize_week_start,
    weekly_boundaries,
)


def test_daily_boundaries_count():
    tz = dt.timezone.utc
    start_date, end_date, _, _ = compute_date_range(7, tz, today=dt.date(2024, 1, 10))
    assert start_date == dt.date(2024, 1, 4)
    assert end_date == dt.date(2024, 1, 10)
    daily = daily_boundaries(start_date, 7, tz)
    assert len(daily) == 7
    assert daily[0][0] == "2024-01-04"
    assert daily[-1][0] == "2024-01-10"


def test_weekly_bucketing_mon_sun():
    tz = dt.timezone.utc
    start_date = dt.date(2024, 1, 1)  # Monday
    daily = daily_boundaries(start_date, 10, tz)

    mon_week = weekly_boundaries(daily, normalize_week_start("MON"))
    assert mon_week[0][0] == "2024-01-01"
    assert mon_week[0][1] == "2024-01-07"
    assert mon_week[1][0] == "2024-01-08"
    assert mon_week[1][1] == "2024-01-10"

    sun_week = weekly_boundaries(daily, normalize_week_start("SUN"))
    assert sun_week[0][0] == "2023-12-31"
    assert sun_week[0][1] == "2024-01-06"
    assert sun_week[1][0] == "2024-01-07"
    assert sun_week[1][1] == "2024-01-10"


def test_partial_week_handling():
    tz = dt.timezone.utc
    start_date = dt.date(2024, 1, 3)
    daily = daily_boundaries(start_date, 3, tz)
    weekly = weekly_boundaries(daily, normalize_week_start("MON"))
    assert weekly[0][0] == "2024-01-01"
    assert weekly[0][1] == "2024-01-05"


def test_zero_fill_bucket():
    tz = dt.timezone.utc
    start_date = dt.date(2024, 1, 1)
    daily = daily_boundaries(start_date, 1, tz)
    date_str, start_utc, end_utc = daily[0]
    aggregate = aggregate_range({}, start_utc, end_utc)
    payload = build_metrics_payload(aggregate, latency_buckets=(100, 200, 300))
    assert payload["counts"]["requests"] == 0
    assert payload["document_extraction"]["ocr_confidence_avg"] is None
    assert payload["document_extraction"]["note"] is not None
    assert payload["retrieval"]["retrieval_accuracy_pct"] is None


def test_hist_quantile():
    hist = {
        "le_100": 1,
        "le_200": 1,
        "le_300": 2,
        "gt_300": 0,
    }
    result = hist_quantile(hist, 0.50, (100, 200, 300))
    assert result == 200.0
