from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

MAX_PRIVATE_FINDINGS = 50


@dataclass
class PrivateDataMatch:
    subcategory: str
    start: int
    end: int
    match_text: str
    evidence: Dict[str, object]


KEYWORD_GROUPS = {
    "CONTRACT": {
        "weight": 18,
        "keywords": [
            "confidential",
            "internal use only",
            "do not distribute",
            "nda",
            "non-disclosure",
            "statement of work",
            "sow",
            "contract",
            "agreement",
        ],
    },
    "PRICING": {
        "weight": 20,
        "keywords": [
            "pricing",
            "rate card",
            "margin",
            "cost breakdown",
            "invoice terms",
        ],
    },
    "CUSTOMER_DATA": {
        "weight": 16,
        "keywords": [
            "customer list",
            "client list",
            "pipeline",
            "proposal",
            "quote",
            "customer",
            "client",
        ],
    },
    "HR": {
        "weight": 16,
        "keywords": [
            "salary",
            "compensation",
            "offer letter",
            "performance review",
            "resume",
            "cv",
        ],
    },
    "INTERNAL_INFRA": {
        "weight": 16,
        "keywords": [
            "architecture",
            "internal endpoint",
            "vpn",
            "ssh",
            "prod",
            "staging",
            "internal url",
        ],
    },
}

CODE_KEYWORDS = re.compile(r"\b(class|def|import|from|#include|public|private|function|var|const|SELECT|INSERT|UPDATE)\b")
CODE_BLOCK_RE = re.compile(r"```[\s\S]{0,5000}?```", re.MULTILINE)
IP_RE = re.compile(r"\b(\d{1,3})(?:\.(\d{1,3})){3}\b")
IP_PORT_RE = re.compile(r"\b(\d{1,3})(?:\.(\d{1,3})){3}:\d{2,5}\b")


def _compile_keyword_patterns() -> List[Tuple[str, int, re.Pattern[str]]]:
    patterns: List[Tuple[str, int, re.Pattern[str]]] = []
    for subcategory, info in KEYWORD_GROUPS.items():
        keywords = [re.escape(keyword) for keyword in info["keywords"]]
        pattern = re.compile(r"\b(?:" + "|".join(keywords) + r")\b", re.IGNORECASE)
        patterns.append((subcategory, int(info["weight"]), pattern))
    return patterns


KEYWORD_PATTERNS = _compile_keyword_patterns()


def _is_private_ip(octets: List[int]) -> bool:
    if len(octets) != 4:
        return False
    if octets[0] == 10:
        return True
    if octets[0] == 172 and 16 <= octets[1] <= 31:
        return True
    if octets[0] == 192 and octets[1] == 168:
        return True
    return False


def _internal_domain_patterns() -> List[re.Pattern[str]]:
    raw = os.getenv("DOCWAIN_ORG_DOMAINS") or ""
    domains = [d.strip().lower() for d in raw.split(",") if d.strip()]
    patterns: List[re.Pattern[str]] = []
    for domain in domains:
        escaped = re.escape(domain)
        patterns.append(re.compile(r"\b(?:https?://)?[^\s]*" + escaped + r"\b", re.IGNORECASE))
    return patterns


def _code_like_line(line: str) -> bool:
    if len(line) < 30:
        return False
    symbol_count = sum(1 for ch in line if not ch.isalnum() and ch not in {" ", "_"})
    ratio = symbol_count / max(len(line), 1)
    return ratio >= 0.3 and bool(CODE_KEYWORDS.search(line))


def detect_private_data(text: str) -> Tuple[int, List[PrivateDataMatch]]:
    matches: List[PrivateDataMatch] = []
    total_score = 0
    seen_spans = set()

    for subcategory, weight, pattern in KEYWORD_PATTERNS:
        for match in pattern.finditer(text):
            if len(matches) >= MAX_PRIVATE_FINDINGS:
                break
            span_key = (match.start(), match.end(), subcategory)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            total_score += weight
            matches.append(
                PrivateDataMatch(
                    subcategory=subcategory,
                    start=match.start(),
                    end=match.end(),
                    match_text=match.group(0),
                    evidence={"keyword_hits": [match.group(0).lower()]},
                )
            )
            break

    for pattern in _internal_domain_patterns():
        for match in pattern.finditer(text):
            if len(matches) >= MAX_PRIVATE_FINDINGS:
                break
            span_key = (match.start(), match.end(), "INTERNAL_INFRA")
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            total_score += KEYWORD_GROUPS["INTERNAL_INFRA"]["weight"]
            matches.append(
                PrivateDataMatch(
                    subcategory="INTERNAL_INFRA",
                    start=match.start(),
                    end=match.end(),
                    match_text=match.group(0),
                    evidence={"pattern": "internal_domain"},
                )
            )
            break

    for match in IP_RE.finditer(text):
        if len(matches) >= MAX_PRIVATE_FINDINGS:
            break
        octets = [int(part) for part in match.group(0).split(".") if part.isdigit()]
        if not _is_private_ip(octets):
            continue
        span_key = (match.start(), match.end(), "INTERNAL_INFRA")
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        total_score += KEYWORD_GROUPS["INTERNAL_INFRA"]["weight"]
        matches.append(
            PrivateDataMatch(
                subcategory="INTERNAL_INFRA",
                start=match.start(),
                end=match.end(),
                match_text=match.group(0),
                evidence={"pattern": "private_ip"},
            )
        )

    for match in IP_PORT_RE.finditer(text):
        if len(matches) >= MAX_PRIVATE_FINDINGS:
            break
        octets = [int(part) for part in match.group(0).split(":")[0].split(".") if part.isdigit()]
        if not _is_private_ip(octets):
            continue
        span_key = (match.start(), match.end(), "INTERNAL_INFRA")
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        total_score += KEYWORD_GROUPS["INTERNAL_INFRA"]["weight"]
        matches.append(
            PrivateDataMatch(
                subcategory="INTERNAL_INFRA",
                start=match.start(),
                end=match.end(),
                match_text=match.group(0),
                evidence={"pattern": "private_ip_port"},
            )
        )

    for match in CODE_BLOCK_RE.finditer(text):
        if len(matches) >= MAX_PRIVATE_FINDINGS:
            break
        span_key = (match.start(), match.end(), "SOURCE_CODE")
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        total_score += 22
        matches.append(
            PrivateDataMatch(
                subcategory="SOURCE_CODE",
                start=match.start(),
                end=match.end(),
                match_text="code_block",
                evidence={"pattern": "code_block"},
            )
        )
        break

    if not any(m.subcategory == "SOURCE_CODE" for m in matches):
        for line in text.splitlines():
            if _code_like_line(line):
                total_score += 18
                index = text.find(line)
                if index >= 0:
                    matches.append(
                        PrivateDataMatch(
                            subcategory="SOURCE_CODE",
                            start=index,
                            end=index + len(line),
                            match_text=line[:120],
                            evidence={"pattern": "code_like"},
                        )
                    )
                break

    total_score = min(100, total_score)
    return total_score, matches
