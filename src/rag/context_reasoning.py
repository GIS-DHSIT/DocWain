from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_INTENT_KEYWORDS = {
    "summarize": {"summarize", "summary", "overview", "recap"},
    "extract": {"extract", "pull out", "find", "locate"},
    "compute": {"compute", "calculate", "total", "sum", "average", "subtotal", "balance"},
    "compare": {"compare", "difference", "vs", "versus", "across", "between"},
    "explain": {"explain", "why", "how", "reason"},
    "list": {"list", "show", "give me", "details", "items"},
}

_OUTPUT_TABLE_CUES = {"table", "tabular", "spreadsheet", "grid", "matrix"}
_OUTPUT_BULLET_CUES = {"bullet", "bullets", "list", "steps"}
_COMPARE_CUES = {"compare", "difference", "vs", "versus", "across", "between"}
_MULTI_DOC_CUES = {"documents", "invoices", "files", "across", "compare", "all"}
_MULTI_DOC_PHRASES = {"these invoices", "these documents", "all documents", "all invoices"}
_PRONOUN_CUES = {"this", "that", "the above", "previous", "same document", "it", "that one"}

_BLOCKED_NUMERIC_LABELS = {
    "phone",
    "fax",
    "tel",
    "telephone",
    "address",
    "zip",
    "postal",
    "vin",
    "ssn",
    "date",
}

_NUMERIC_RE = re.compile(r"\b(?:\$|usd|eur|gbp)?\s*\d[\d,]*(?:\.\d+)?\b", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b(?:\+?\d[\d\s().-]{6,}\d)\b")
_DATE_RE = re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b")


def _normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s._-]", " ", text)
    return re.sub(r"\s+", " ", text)


def _tokenize(text: str) -> List[str]:
    return [t for t in _normalize_text(text).split() if t]


def _strip_extension(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"\.[a-z0-9]{1,5}$", "", name, flags=re.IGNORECASE).strip()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    text = text.lower()
    return any(n in text for n in needles)


def _intent_from_query(query: str) -> str:
    q = query.lower()
    if _contains_any(q, _INTENT_KEYWORDS["compare"]):
        return "compare"
    if _contains_any(q, _INTENT_KEYWORDS["summarize"]):
        return "summarize"
    if _contains_any(q, _INTENT_KEYWORDS["extract"]):
        return "extract"
    if _contains_any(q, _INTENT_KEYWORDS["compute"]):
        return "compute"
    if _contains_any(q, _INTENT_KEYWORDS["list"]):
        return "list"
    if _contains_any(q, _INTENT_KEYWORDS["explain"]):
        return "explain"
    return "lookup"


def _output_mode_from_query(query: str) -> str:
    q = query.lower()
    if _contains_any(q, _OUTPUT_TABLE_CUES):
        return "table"
    if _contains_any(q, _OUTPUT_BULLET_CUES):
        return "bullets"
    return "narrative"


def _query_is_short(query: str) -> bool:
    return len(_tokenize(query)) <= 4


def _extract_entity_keywords(query: str) -> List[str]:
    if not query:
        return []
    candidates: List[str] = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9_-]{1,}\b", query):
        candidates.append(match.group(0))
    for match in re.finditer(r"\"([^\"]{2,})\"", query):
        candidates.append(match.group(1))
    return list(dict.fromkeys(candidates))


def _rank_doc_candidates(query: str, available_sources: Sequence[str]) -> List[Tuple[str, float]]:
    q_norm = _normalize_text(query)
    q_tokens = set(_tokenize(query))
    ranked: List[Tuple[str, float]] = []
    for source in available_sources or []:
        cleaned = _strip_extension(str(source))
        if not cleaned:
            continue
        s_norm = _normalize_text(cleaned)
        s_tokens = set(_tokenize(cleaned))
        overlap = len(q_tokens & s_tokens)
        score = 0.0
        if s_norm and s_norm in q_norm:
            score += 0.6
        if s_tokens:
            score += min(0.4, overlap / max(len(s_tokens), 1))
        if score > 0:
            ranked.append((source, round(score, 4)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def _mentions_pronoun(query: str) -> bool:
    q = query.lower()
    return any(p in q for p in _PRONOUN_CUES)


@dataclass
class AnalyzerOutput:
    intent: str
    output_mode: str
    scope: str
    target_hint: List[str]
    assumptions: List[str]
    clarification_needed: bool
    clarification_question: Optional[str]


class ContextAwareQueryAnalyzer:
    """Local, heuristic query analyzer. No remote LLM usage."""

    def analyze(
        self,
        *,
        query: str,
        conversation_history: str,
        available_sources: Sequence[str],
        last_active_document: Optional[Dict[str, str]] = None,
    ) -> AnalyzerOutput:
        intent = _intent_from_query(query)
        output_mode = _output_mode_from_query(query)
        if intent == "compare":
            output_mode = "table"
        assumptions: List[str] = []
        clarification_needed = False
        clarification_question = None

        ranked_docs = _rank_doc_candidates(query, available_sources)
        top_doc = ranked_docs[0][0] if ranked_docs else None
        doc_candidates = [doc for doc, _ in ranked_docs[:3]]

        convo_docs = _rank_doc_candidates(conversation_history or "", available_sources)
        convo_doc = convo_docs[0][0] if convo_docs else None

        lowered_query = query.lower()
        has_compare = _contains_any(lowered_query, _COMPARE_CUES)
        has_multi_doc = _contains_any(lowered_query, _MULTI_DOC_CUES) or any(
            phrase in lowered_query for phrase in _MULTI_DOC_PHRASES
        )
        has_pronoun = _mentions_pronoun(query)

        scope = "unknown"
        target_hint: List[str] = []

        if has_compare or has_multi_doc:
            scope = "multi_doc"
        elif doc_candidates:
            scope = "single_doc_default"
            target_hint = doc_candidates
        elif has_pronoun and last_active_document:
            scope = "single_doc_default"
            target_hint = [last_active_document.get("doc_name") or "previous document"]
            assumptions.append("Assuming you mean the previously referenced document.")
        elif has_pronoun and convo_doc:
            scope = "single_doc_default"
            target_hint = [convo_doc]
            assumptions.append("Assuming you mean the document mentioned earlier in this conversation.")
        elif len(available_sources or []) == 1:
            scope = "single_doc_default"
            target_hint = [available_sources[0]]
            assumptions.append("Only one document is available, so I used it by default.")
        elif top_doc and _query_is_short(query):
            scope = "single_doc_default"
            target_hint = [top_doc]
            assumptions.append("Assuming the best-matching document based on coverage.")

        if scope == "unknown":
            target_hint = _extract_entity_keywords(query)

        if scope != "multi_doc" and len(doc_candidates) >= 2 and not target_hint and _query_is_short(query):
            score_gap = abs((ranked_docs[0][1] if ranked_docs else 0) - (ranked_docs[1][1] if len(ranked_docs) > 1 else 0))
            if score_gap < 0.1:
                clarification_needed = True
                clarification_question = (
                    "Which document should I use? Please share the document name."
                )

        if output_mode == "table" and scope == "unknown" and not has_compare:
            scope = "single_doc_default"
            assumptions.append("Assuming you want the most relevant document in table form.")

        return AnalyzerOutput(
            intent=intent,
            output_mode=output_mode,
            scope=scope,
            target_hint=[t for t in target_hint if t],
            assumptions=assumptions,
            clarification_needed=clarification_needed,
            clarification_question=clarification_question,
        )


@dataclass
class EvidencePlan:
    retrieval_mode: str
    doc_selection_policy: str
    required_evidence_types: List[str]
    min_chunks: int
    max_chunks: int
    adjacent_expand: bool
    rerank_policy: str


class EvidencePlanBuilder:
    def build(self, analysis: AnalyzerOutput, query: str) -> EvidencePlan:
        intent = analysis.intent
        output_mode = analysis.output_mode

        retrieval_mode = "precision"
        if intent in {"compute", "list"} or output_mode == "table":
            retrieval_mode = "coverage"
        if intent in {"summarize", "compare"}:
            retrieval_mode = "coverage"

        doc_selection_policy = "single_best"
        if analysis.scope == "multi_doc" or intent == "compare":
            doc_selection_policy = "multi_doc_balanced"

        required_types = ["narrative"]
        if output_mode == "table" or intent in {"compute", "compare"}:
            required_types = ["table_rows", "key_value"]
        if intent == "summarize":
            required_types = ["summary_section"]
        if intent == "extract":
            required_types = ["key_value"]

        min_chunks = 5 if retrieval_mode == "precision" else 8
        max_chunks = 12 if retrieval_mode == "precision" else 18
        if intent == "compare":
            min_chunks = max(min_chunks, 8)
            max_chunks = max(max_chunks, 20)

        adjacent_expand = retrieval_mode == "coverage"
        rerank_policy = "filter_and_order" if retrieval_mode == "precision" else "order_only"

        return EvidencePlan(
            retrieval_mode=retrieval_mode,
            doc_selection_policy=doc_selection_policy,
            required_evidence_types=required_types,
            min_chunks=min_chunks,
            max_chunks=max_chunks,
            adjacent_expand=adjacent_expand,
            rerank_policy=rerank_policy,
        )


class EvidenceQualityScorer:
    def score_chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> float:
        text = text or ""
        meta = metadata or {}
        score = 0.0
        tokens = _tokenize(text)
        if len(tokens) >= 12:
            score += 0.2
        if text.strip().endswith((".", "!", "?")):
            score += 0.1
        section = str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "").lower()
        if any(word in section for word in ("summary", "total", "results", "breakdown", "table", "overview")):
            score += 0.25
        label_density = len(re.findall(r"[A-Za-z][A-Za-z0-9\s]{1,25}\s*[:\-]\s*\S+", text))
        if label_density >= 2:
            score += 0.2
        if _PHONE_RE.search(text) and label_density == 0:
            score -= 0.15
        if _DATE_RE.search(text) and label_density == 0:
            score -= 0.1
        if len(tokens) > 0:
            unique_ratio = len(set(tokens)) / max(len(tokens), 1)
            if unique_ratio < 0.35:
                score -= 0.15
        return max(score, 0.0)

    def score_documents(self, chunks: Sequence[Any], query: str) -> Dict[str, float]:
        doc_scores: Dict[str, float] = {}
        doc_sections: Dict[str, set] = {}
        for chunk in chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            doc_id = str(meta.get("document_id") or meta.get("doc_id") or meta.get("docId") or "")
            if not doc_id:
                continue
            section = str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "")
            doc_sections.setdefault(doc_id, set()).add(section)
            score = self.score_chunk(getattr(chunk, "text", ""), meta)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + score
        for doc_id, sections in doc_sections.items():
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + min(0.4, 0.05 * len(sections))
        if _contains_any(query.lower(), {"total", "summary", "overview"}):
            for doc_id in doc_scores:
                doc_scores[doc_id] += 0.05
        return doc_scores


@dataclass
class KeyFact:
    label: str
    value: str
    doc_name: str
    section: str
    chunk_id: str


@dataclass
class TableData:
    headers: List[str]
    rows: List[List[str]]
    doc_name: str
    chunk_id: str


@dataclass
class NumericClaim:
    label: str
    value: str
    unit: Optional[str]
    doc_name: str
    chunk_id: str


@dataclass
class Contradiction:
    label: str
    values: List[str]
    docs: List[str]


@dataclass
class WorkingContext:
    resolved_scope: List[str]
    key_facts: List[KeyFact]
    tables: List[TableData]
    numeric_claims: List[NumericClaim]
    contradictions: List[Contradiction]
    missing_fields: List[str]
    citations_map: Dict[str, List[str]] = field(default_factory=dict)

    def brief_text(self, max_items: int = 8) -> str:
        parts: List[str] = []
        for fact in self.key_facts[:max_items]:
            parts.append(f"{fact.label}: {fact.value}")
        for claim in self.numeric_claims[:max_items]:
            parts.append(f"{claim.label}: {claim.value}")
        return " | ".join(parts)


def _chunk_doc_name(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", {}) or {}
    return (
        meta.get("source_file")
        or meta.get("file_name")
        or meta.get("filename")
        or meta.get("document_name")
        or meta.get("source")
        or getattr(chunk, "source", None)
        or "Document"
    )


def _chunk_section(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", {}) or {}
    return str(meta.get("section_title") or meta.get("section_path") or meta.get("section") or "Section")


def _chunk_id(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", {}) or {}
    return str(meta.get("chunk_id") or getattr(chunk, "id", ""))


def _extract_label_value_pairs(text: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or len(line) > 200:
            continue
        match = re.match(r"([A-Za-z][A-Za-z0-9\s/._-]{1,40})\s*[:\-]\s*(.+)", line)
        if match:
            label = match.group(1).strip()
            value = match.group(2).strip()
            if label and value:
                pairs.append((label, value))
    for match in re.finditer(r"([A-Za-z][A-Za-z0-9\s/._-]{1,40})\s+(?:is|are|was|were|=)\s+([^\n.;]{1,80})", text):
        label = match.group(1).strip()
        value = match.group(2).strip()
        if label and value:
            pairs.append((label, value))
    return pairs


def _extract_tables(text: str) -> List[Tuple[List[str], List[List[str]]]]:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    tables: List[Tuple[List[str], List[List[str]]]] = []
    if not lines:
        return tables
    pipe_lines = [line for line in lines if line.count("|") >= 2]
    if len(pipe_lines) >= 2:
        header = [c.strip() for c in pipe_lines[0].split("|") if c.strip()]
        rows: List[List[str]] = []
        for line in pipe_lines[1:]:
            if re.match(r"^[\-\s|]+$", line):
                continue
            row = [c.strip() for c in line.split("|") if c.strip()]
            if row:
                rows.append(row)
        if header and rows:
            tables.append((header, rows))
    if tables:
        return tables

    spaced_lines = [line for line in lines if re.search(r"\S+\s{2,}\S+", line)]
    if len(spaced_lines) >= 2:
        header = [c.strip() for c in re.split(r"\s{2,}", spaced_lines[0]) if c.strip()]
        rows = []
        for line in spaced_lines[1:]:
            row = [c.strip() for c in re.split(r"\s{2,}", line) if c.strip()]
            if row:
                rows.append(row)
        if header and rows:
            tables.append((header, rows))
    return tables


def _numeric_unit(value: str) -> Optional[str]:
    if "$" in value:
        return "$"
    if "usd" in value.lower():
        return "USD"
    if "eur" in value.lower():
        return "EUR"
    if "gbp" in value.lower():
        return "GBP"
    return None


def _is_blocked_label(label: str) -> bool:
    label_norm = _normalize_text(label)
    return any(word in label_norm for word in _BLOCKED_NUMERIC_LABELS)


def _dedupe_items(items: Iterable[Any], key_fn) -> List[Any]:
    seen = set()
    output = []
    for item in items:
        key = key_fn(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


class WorkingContextAssembler:
    def assemble(self, *, query: str, chunks: Sequence[Any], analysis: AnalyzerOutput) -> WorkingContext:
        resolved_scope = sorted({str(_chunk_doc_name(c)) for c in chunks if _chunk_doc_name(c)})

        key_facts: List[KeyFact] = []
        tables: List[TableData] = []
        numeric_claims: List[NumericClaim] = []
        citations_map: Dict[str, List[str]] = {}

        for chunk in chunks:
            text = getattr(chunk, "text", "") or ""
            doc_name = str(_chunk_doc_name(chunk))
            section = str(_chunk_section(chunk))
            chunk_id = _chunk_id(chunk)

            for label, value in _extract_label_value_pairs(text):
                fact = KeyFact(label=label, value=value, doc_name=doc_name, section=section, chunk_id=chunk_id)
                key_facts.append(fact)

            for header, rows in _extract_tables(text):
                tables.append(TableData(headers=header, rows=rows, doc_name=doc_name, chunk_id=chunk_id))

        key_facts = _dedupe_items(key_facts, lambda f: (_normalize_text(f.label), _normalize_text(f.value), f.doc_name))

        for fact in key_facts:
            if _is_blocked_label(fact.label):
                continue
            if not _NUMERIC_RE.search(fact.value or ""):
                continue
            numeric_claims.append(
                NumericClaim(
                    label=fact.label,
                    value=fact.value,
                    unit=_numeric_unit(fact.value),
                    doc_name=fact.doc_name,
                    chunk_id=fact.chunk_id,
                )
            )

        for table in tables:
            headers = table.headers
            for row in table.rows:
                for idx, cell in enumerate(row):
                    if not cell or not _NUMERIC_RE.search(cell):
                        continue
                    label = headers[idx] if idx < len(headers) else "Value"
                    if _is_blocked_label(label):
                        continue
                    numeric_claims.append(
                        NumericClaim(
                            label=label,
                            value=cell,
                            unit=_numeric_unit(cell),
                            doc_name=table.doc_name,
                            chunk_id=table.chunk_id,
                        )
                    )

        numeric_claims = _dedupe_items(numeric_claims, lambda c: (_normalize_text(c.label), _normalize_text(c.value), c.doc_name))

        contradictions: List[Contradiction] = []
        by_label: Dict[str, Dict[str, set]] = {}
        label_display: Dict[str, str] = {}
        for claim in numeric_claims:
            label_key = _normalize_text(claim.label)
            if label_key not in label_display:
                label_display[label_key] = claim.label
            by_label.setdefault(label_key, {}).setdefault(claim.value, set()).add(claim.doc_name)
        for label_key, value_map in by_label.items():
            if len(value_map) > 1:
                values = list(value_map.keys())
                docs = sorted({doc for docs in value_map.values() for doc in docs})
                contradictions.append(
                    Contradiction(label=label_display.get(label_key, label_key or "value"), values=values, docs=docs)
                )

        missing_fields = _determine_missing_fields(query, key_facts, numeric_claims)

        for idx, fact in enumerate(key_facts):
            citations_map[f"fact:{idx}"] = [fact.chunk_id]
        for idx, table in enumerate(tables):
            citations_map[f"table:{idx}"] = [table.chunk_id]
        for idx, claim in enumerate(numeric_claims):
            citations_map[f"num:{idx}"] = [claim.chunk_id]

        return WorkingContext(
            resolved_scope=resolved_scope,
            key_facts=key_facts,
            tables=tables,
            numeric_claims=numeric_claims,
            contradictions=contradictions,
            missing_fields=missing_fields,
            citations_map=citations_map,
        )


def _determine_missing_fields(query: str, key_facts: Sequence[KeyFact], numeric_claims: Sequence[NumericClaim]) -> List[str]:
    q = query.lower()
    requested = []
    if "total" in q:
        requested.append("total")
    if "price" in q or "amount" in q or "cost" in q:
        requested.append("amount")
    if "date" in q:
        requested.append("date")
    if "summary" in q:
        requested.append("summary")
    if not requested:
        return []
    available_labels = { _normalize_text(f.label) for f in key_facts }
    available_labels.update({ _normalize_text(c.label) for c in numeric_claims })
    missing = []
    for req in requested:
        if not any(req in label for label in available_labels):
            missing.append(req)
    return missing


@dataclass
class AnswerRenderResult:
    text: str
    used_chunk_ids: List[str]


class AnswerRenderer:
    @staticmethod
    def _primary_doc(context: WorkingContext, analysis: AnalyzerOutput) -> Optional[str]:
        if analysis.scope == "single_doc_default" and context.resolved_scope:
            return context.resolved_scope[0]
        return None

    def render(self, *, query: str, analysis: AnalyzerOutput, context: WorkingContext) -> AnswerRenderResult:
        if analysis.output_mode == "table":
            text, used = self._render_table(query, analysis, context)
            return AnswerRenderResult(text=text, used_chunk_ids=used)
        if analysis.output_mode == "bullets":
            text, used = self._render_bullets(query, analysis, context)
            return AnswerRenderResult(text=text, used_chunk_ids=used)
        text, used = self._render_narrative(query, analysis, context)
        return AnswerRenderResult(text=text, used_chunk_ids=used)

    def _render_table(
        self,
        query: str,
        analysis: AnalyzerOutput,
        context: WorkingContext,
    ) -> Tuple[str, List[str]]:
        lead = []
        if analysis.assumptions:
            lead.append(analysis.assumptions[0])
        elif context.resolved_scope:
            lead.append(f"Using {context.resolved_scope[0]} for this response.")

        rows: List[List[str]] = []
        used_chunk_ids: List[str] = []
        primary_doc = self._primary_doc(context, analysis)
        if context.tables:
            table = context.tables[0]
            headers = table.headers
            seen = set()
            for row in table.rows:
                key = tuple(row)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
            used_chunk_ids.append(table.chunk_id)
            table_text = _format_table(headers, rows)
            return "\n".join([*lead, table_text]).strip(), used_chunk_ids

        if context.numeric_claims:
            headers = ["Document", "Label", "Value"]
            seen = set()
            for claim in context.numeric_claims:
                if primary_doc and claim.doc_name != primary_doc:
                    continue
                key = (claim.doc_name, claim.label, claim.value)
                if key in seen:
                    continue
                seen.add(key)
                rows.append([claim.doc_name, claim.label, claim.value])
                used_chunk_ids.append(claim.chunk_id)
            table_text = _format_table(headers, rows)
            return "\n".join([*lead, table_text]).strip(), used_chunk_ids

        headers = ["Document", "Detail", "Value"]
        seen = set()
        for fact in context.key_facts[:10]:
            if primary_doc and fact.doc_name != primary_doc:
                continue
            key = (fact.doc_name, fact.label, fact.value)
            if key in seen:
                continue
            seen.add(key)
            rows.append([fact.doc_name, fact.label, fact.value])
            used_chunk_ids.append(fact.chunk_id)
        table_text = _format_table(headers, rows) if rows else "No tabular data was found in the retrieved sections."
        return "\n".join([*lead, table_text]).strip(), used_chunk_ids

    def _render_bullets(
        self,
        query: str,
        analysis: AnalyzerOutput,
        context: WorkingContext,
    ) -> Tuple[str, List[str]]:
        lines = []
        used_chunk_ids: List[str] = []
        primary_doc = self._primary_doc(context, analysis)
        if analysis.assumptions:
            lines.append(analysis.assumptions[0])
        if context.resolved_scope:
            lines.append(f"- Document focus: {', '.join(context.resolved_scope[:2])}")
        if context.numeric_claims:
            for claim in context.numeric_claims[:8]:
                if primary_doc and claim.doc_name != primary_doc:
                    continue
                lines.append(f"- {claim.label}: {claim.value}")
                used_chunk_ids.append(claim.chunk_id)
        elif context.key_facts:
            for fact in context.key_facts[:8]:
                if primary_doc and fact.doc_name != primary_doc:
                    continue
                lines.append(f"- {fact.label}: {fact.value}")
                used_chunk_ids.append(fact.chunk_id)
        else:
            lines.append("- I could not find labeled facts in the retrieved sections.")
        if context.contradictions:
            contra = context.contradictions[0]
            lines.append(f"- Conflicts: {contra.label} has multiple values ({', '.join(contra.values)}).")
        if context.missing_fields:
            lines.append(f"- Missing in retrieved sections: {', '.join(context.missing_fields)}.")
        return "\n".join(lines).strip(), used_chunk_ids

    def _render_narrative(
        self,
        query: str,
        analysis: AnalyzerOutput,
        context: WorkingContext,
    ) -> Tuple[str, List[str]]:
        sentences: List[str] = []
        used_chunk_ids: List[str] = []
        primary_doc = self._primary_doc(context, analysis)

        if analysis.assumptions:
            sentences.append(analysis.assumptions[0])
        elif context.resolved_scope:
            sentences.append(f"Based on {context.resolved_scope[0]}, here is what I found.")

        primary = _select_primary_claim(query, analysis, context)
        if primary:
            label, value, chunk_id = primary
            sentences.append(f"The {label} is {value}.")
            used_chunk_ids.append(chunk_id)
        elif context.key_facts:
            fact = next((f for f in context.key_facts if not primary_doc or f.doc_name == primary_doc), context.key_facts[0])
            sentences.append(f"{fact.label} is listed as {fact.value}.")
            used_chunk_ids.append(fact.chunk_id)
        else:
            sentences.append("I did not find a clear labeled answer in the retrieved sections.")

        if analysis.intent == "compare" and context.numeric_claims:
            comparisons = []
            for claim in context.numeric_claims[:5]:
                comparisons.append(f"{claim.doc_name}: {claim.label} {claim.value}")
                used_chunk_ids.append(claim.chunk_id)
            sentences.append("Comparisons by document: " + "; ".join(comparisons) + ".")

        if context.contradictions:
            contra = context.contradictions[0]
            sentences.append(
                f"I found conflicting values for {contra.label}: {', '.join(contra.values)}."
            )

        if context.missing_fields:
            missing = ", ".join(context.missing_fields)
            sentences.append(f"I could not find the following in the retrieved sections: {missing}.")

        if analysis.scope == "single_doc_default" and len(context.resolved_scope) > 1:
            sentences.append("If you meant a different document, tell me which one.")

        sentences = [s.strip() for s in sentences if s.strip()]
        filler = [
            "If you want a comparison across documents, tell me which ones to include.",
            "If you need more detail, tell me the section or field to focus on.",
        ]
        while len(sentences) < 4 and filler:
            sentences.append(filler.pop(0))
        sentences = sentences[:8]
        return " ".join(sentences).strip(), used_chunk_ids


def _select_primary_claim(
    query: str,
    analysis: AnalyzerOutput,
    context: WorkingContext,
) -> Optional[Tuple[str, str, str]]:
    if not context.numeric_claims:
        return None
    q_tokens = set(_tokenize(query))
    for claim in context.numeric_claims:
        label_tokens = set(_tokenize(claim.label))
        if q_tokens & label_tokens:
            return claim.label, claim.value, claim.chunk_id
    return context.numeric_claims[0].label, context.numeric_claims[0].value, context.numeric_claims[0].chunk_id


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    if not headers or not rows:
        return ""
    safe_rows = rows[:40]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in safe_rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return "\n".join(lines)


__all__ = [
    "AnalyzerOutput",
    "ContextAwareQueryAnalyzer",
    "EvidencePlan",
    "EvidencePlanBuilder",
    "EvidenceQualityScorer",
    "WorkingContextAssembler",
    "WorkingContext",
    "AnswerRenderer",
    "AnswerRenderResult",
]
