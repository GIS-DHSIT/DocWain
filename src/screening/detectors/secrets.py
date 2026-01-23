from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

MAX_SECRET_FINDINGS = 50
ENTROPY_THRESHOLD = 3.5


@dataclass(frozen=True)
class SecretPattern:
    regex_id: str
    pattern: re.Pattern[str]
    category: str
    severity: str
    confidence: float
    value_group: Optional[int] = None
    mask_mode: str = "value"


@dataclass
class SecretMatch:
    category: str
    severity: str
    confidence: float
    start: int
    end: int
    mask_start: int
    mask_end: int
    regex_id: str
    match_text: str
    evidence: Dict[str, object]
    mask_mode: str = "value"


PRIVATE_KEY_BLOCK = re.compile(
    r"-----BEGIN (?:RSA|EC|OPENSSH|PGP) PRIVATE KEY-----[\s\S]{0,5000}?-----END (?:RSA|EC|OPENSSH|PGP) PRIVATE KEY-----",
    re.IGNORECASE,
)
JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b")
AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA|A3T[A-Z0-9])[A-Z0-9]{16}\b")
AWS_SECRET_RE = re.compile(r"\baws_secret_access_key\b\s*[:=]\s*([A-Za-z0-9/+=]{35,})", re.IGNORECASE)
AZURE_STORAGE_CONN_RE = re.compile(
    r"DefaultEndpointsProtocol=[^;\s]+;AccountName=[^;\s]+;AccountKey=[^;\s]+;?",
    re.IGNORECASE,
)
AUTH_BEARER_RE = re.compile(r"\bAuthorization\b\s*[:=]\s*Bearer\s+([A-Za-z0-9._\-/+=]{16,})", re.IGNORECASE)
GENERIC_API_KEY_RE = re.compile(
    r"\b(api_key|apikey|secret_key|token|bearer)\b\s*[:=]\s*([A-Za-z0-9._\-]{16,})",
    re.IGNORECASE,
)
GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")
SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b")
STRIPE_KEY_RE = re.compile(r"\b(?:sk|rk)_live_[A-Za-z0-9]{16,}\b")
PASSWORD_RE = re.compile(r"\b(password|pwd|passphrase|secret)\b\s*[:=]\s*([^\s;]+)", re.IGNORECASE)
CONNECTION_URL_RE = re.compile(
    r"\b(?:postgres(?:ql)?|mysql|mssql|mongodb(?:\+srv)?|redis|amqp|sqlserver)://[^\s]+",
    re.IGNORECASE,
)
CONNECTION_KV_RE = re.compile(r"\b(?:Server|Host)=[^;]+;[^;\n]+?(?:Password|Pwd)=[^;\n]+", re.IGNORECASE)
PASSPHRASE_WORD_RE = re.compile(r"\bpassphrase\b", re.IGNORECASE)

KEYWORD_RE = re.compile(r"\b(api|key|token|secret|password|pwd|passphrase|auth|bearer)\b", re.IGNORECASE)
BASE64_RE = re.compile(r"\b[A-Za-z0-9+/]{20,}={0,2}\b")
HEX_RE = re.compile(r"\b[A-Fa-f0-9]{32,}\b")
USERNAME_RE = re.compile(r"\b(username|login|user\s+id)\b", re.IGNORECASE)

PATTERNS: List[SecretPattern] = [
    SecretPattern(
        regex_id="private_key_block",
        pattern=PRIVATE_KEY_BLOCK,
        category="SECRET_PRIVATE_KEY",
        severity="CRITICAL",
        confidence=0.95,
        mask_mode="block",
    ),
    SecretPattern(regex_id="jwt", pattern=JWT_RE, category="SECRET_JWT", severity="HIGH", confidence=0.9),
    SecretPattern(
        regex_id="aws_access_key",
        pattern=AWS_ACCESS_KEY_RE,
        category="SECRET_API_KEY",
        severity="HIGH",
        confidence=0.9,
    ),
    SecretPattern(
        regex_id="aws_secret_key",
        pattern=AWS_SECRET_RE,
        category="SECRET_API_KEY",
        severity="HIGH",
        confidence=0.85,
        value_group=1,
    ),
    SecretPattern(
        regex_id="azure_storage_connection",
        pattern=AZURE_STORAGE_CONN_RE,
        category="SECRET_CONNECTION_STRING",
        severity="HIGH",
        confidence=0.85,
    ),
    SecretPattern(
        regex_id="auth_bearer",
        pattern=AUTH_BEARER_RE,
        category="SECRET_AUTH_HEADER",
        severity="HIGH",
        confidence=0.85,
        value_group=1,
    ),
    SecretPattern(
        regex_id="generic_api_key",
        pattern=GENERIC_API_KEY_RE,
        category="SECRET_API_KEY",
        severity="MED",
        confidence=0.7,
        value_group=2,
    ),
    SecretPattern(
        regex_id="github_token",
        pattern=GITHUB_TOKEN_RE,
        category="SECRET_API_KEY",
        severity="HIGH",
        confidence=0.9,
    ),
    SecretPattern(
        regex_id="slack_token",
        pattern=SLACK_TOKEN_RE,
        category="SECRET_API_KEY",
        severity="HIGH",
        confidence=0.9,
    ),
    SecretPattern(
        regex_id="stripe_key",
        pattern=STRIPE_KEY_RE,
        category="SECRET_API_KEY",
        severity="HIGH",
        confidence=0.9,
    ),
    SecretPattern(
        regex_id="password_assignment",
        pattern=PASSWORD_RE,
        category="SECRET_PASSWORD",
        severity="MED",
        confidence=0.65,
        value_group=2,
    ),
    SecretPattern(
        regex_id="connection_url",
        pattern=CONNECTION_URL_RE,
        category="SECRET_CONNECTION_STRING",
        severity="HIGH",
        confidence=0.75,
    ),
    SecretPattern(
        regex_id="connection_kv",
        pattern=CONNECTION_KV_RE,
        category="SECRET_CONNECTION_STRING",
        severity="HIGH",
        confidence=0.75,
    ),
]


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: Dict[str, int] = {}
    for char in value:
        counts[char] = counts.get(char, 0) + 1
    total = len(value)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _line_span(text: str, start: int, end: int) -> Tuple[str, int]:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end], line_start


def _boost_password_severity(text: str, start: int, end: int, value: str) -> Tuple[str, Dict[str, object]]:
    line, _ = _line_span(text, start, end)
    window_start = max(0, start - 40)
    window_end = min(len(text), end + 40)
    window = text[window_start:window_end]
    factors: List[str] = []

    if USERNAME_RE.search(window):
        factors.append("username_proximity")
    if "://" in line or ("Password" in line and ";" in line):
        factors.append("connection_string")
    if re.search(r"^[A-Z0-9_]+\s*=", line.strip()):
        factors.append("env_style")

    entropy = shannon_entropy(value) if len(value) >= 16 else 0.0
    if entropy >= ENTROPY_THRESHOLD:
        factors.append("high_entropy")

    severity = "MED"
    if len(factors) >= 2:
        severity = "CRITICAL"
    elif len(factors) == 1:
        severity = "HIGH"

    evidence: Dict[str, object] = {"password_factors": factors}
    if entropy:
        evidence["entropy"] = round(entropy, 2)
    return severity, evidence


def _keyword_context_matches(text: str) -> Iterable[Tuple[int, int, str, float, List[str]]]:
    for match in KEYWORD_RE.finditer(text):
        keyword = match.group(0)
        window_start = max(0, match.start() - 10)
        window_end = min(len(text), match.end() + 80)
        window = text[window_start:window_end]
        for token_re in (BASE64_RE, HEX_RE):
            for token in token_re.finditer(window):
                token_value = token.group(0)
                if len(token_value) < 16:
                    continue
                entropy = shannon_entropy(token_value)
                start = window_start + token.start()
                end = window_start + token.end()
                yield start, end, token_value, entropy, [keyword]


def detect_secrets(text: str) -> List[SecretMatch]:
    matches: List[SecretMatch] = []
    seen_spans = set()
    ranges: List[Tuple[int, int]] = []

    for pattern in PATTERNS:
        for match in pattern.pattern.finditer(text):
            if len(matches) >= MAX_SECRET_FINDINGS:
                return matches
            start, end = match.span()
            if pattern.value_group is not None:
                mask_start, mask_end = match.start(pattern.value_group), match.end(pattern.value_group)
            else:
                mask_start, mask_end = start, end
            span_key = (mask_start, mask_end, pattern.category)
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            match_text = match.group(0)
            evidence: Dict[str, object] = {"regex_id": pattern.regex_id}

            severity = pattern.severity
            confidence = pattern.confidence
            if pattern.category == "SECRET_PASSWORD" and pattern.value_group is not None:
                value = match.group(pattern.value_group)
                severity, extra = _boost_password_severity(text, start, end, value)
                evidence.update(extra)

            if pattern.value_group is not None:
                value = match.group(pattern.value_group)
                entropy = shannon_entropy(value) if len(value) >= 16 else 0.0
                if entropy >= ENTROPY_THRESHOLD and pattern.category in {"SECRET_API_KEY", "SECRET_PASSWORD"}:
                    evidence["entropy"] = round(entropy, 2)
                    if severity == "MED":
                        severity = "HIGH"

            matches.append(
                SecretMatch(
                    category=pattern.category,
                    severity=severity,
                    confidence=confidence,
                    start=start,
                    end=end,
                    mask_start=mask_start,
                    mask_end=mask_end,
                    regex_id=pattern.regex_id,
                    match_text=match_text,
                    evidence=evidence,
                    mask_mode=pattern.mask_mode,
                )
            )
            ranges.append((start, end))

    for start, end, token_value, entropy, keywords in _keyword_context_matches(text):
        if len(matches) >= MAX_SECRET_FINDINGS:
            break
        span_key = (start, end, "SECRET_API_KEY")
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        severity = "MED" if entropy >= ENTROPY_THRESHOLD else "LOW"
        matches.append(
            SecretMatch(
                category="SECRET_API_KEY",
                severity=severity,
                confidence=0.55,
                start=start,
                end=end,
                mask_start=start,
                mask_end=end,
                regex_id="keyword_context",
                match_text=token_value,
                evidence={"keyword_hits": keywords, "entropy": round(entropy, 2)},
            )
        )
        ranges.append((start, end))

    for match in PASSPHRASE_WORD_RE.finditer(text):
        if len(matches) >= MAX_SECRET_FINDINGS:
            break
        start, end = match.span()
        if any(not (end <= r_start or start >= r_end) for r_start, r_end in ranges):
            continue
        span_key = (start, end, "SECRET_PASSWORD")
        if span_key in seen_spans:
            continue
        seen_spans.add(span_key)
        matches.append(
            SecretMatch(
                category="SECRET_PASSWORD",
                severity="LOW",
                confidence=0.4,
                start=start,
                end=end,
                mask_start=start,
                mask_end=end,
                regex_id="passphrase_word",
                match_text=match.group(0),
                evidence={"keyword_hits": ["passphrase"]},
            )
        )

    return matches
