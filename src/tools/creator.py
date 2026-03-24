from __future__ import annotations

from src.utils.logging_utils import get_logger
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field

from src.tools.base import generate_correlation_id, register_tool, standard_response
from src.tools.common.grounding import build_source_record
from src.tools.common.text_extract import sanitize_text

logger = get_logger(__name__)

router = APIRouter(prefix="/creator", tags=["Tools-Creator"])

class CreatorRequest(BaseModel):
    content_type: str = Field(..., pattern="^(summary|blog|sop|faq|slide_outline)$")
    tone: str = Field(default="neutral")
    length: str = Field(default="medium", description="rough size hint: short|medium|long")
    context: Optional[Dict[str, Any]] = None
    text: Optional[str] = Field(default=None, description="Reference text to ground generation")

# ── Type-specific instructions ──────────────────────────────────────

_TYPE_INSTRUCTIONS = {
    "summary": "Write a concise summary with key takeaways.",
    "blog": "Write a blog post with introduction, body sections, and conclusion.",
    "sop": "Write a standard operating procedure with numbered steps.",
    "faq": "Generate 5-8 FAQ questions and answers as JSON: [{\"q\": \"...\", \"a\": \"...\"}]",
    "slide_outline": "Create a slide deck outline with title + bullet points per slide.",
}

_EXPECTED_FIELDS = ["content"]

# ── LLM generation ──────────────────────────────────────────────────

def _llm_generate(content_type: str, tone: str, length: str, reference: str) -> Optional[Dict[str, Any]]:
    """LLM-powered content generation. Returns None on failure."""
    try:
        from src.tools.llm_tools import build_generation_prompt, tool_generate, tool_generate_structured

        type_instruction = _TYPE_INSTRUCTIONS.get(content_type, "Generate the requested content.")
        instructions = (
            f"Generate a {content_type.replace('_', ' ')} with these specifications:\n"
            f"- Tone: {tone}\n"
            f"- Length: {length}\n"
            f"- Ground ALL content in the reference material.\n\n"
            f"{type_instruction}"
        )
        prompt = build_generation_prompt("creator", instructions, reference)

        if content_type == "faq":
            result = tool_generate_structured(prompt, domain="creative")
            if result:
                faqs = result.get("faqs") or result.get("questions") or []
                if isinstance(result, dict) and not faqs:
                    # LLM may return a list at top level wrapped in a key
                    for key, val in result.items():
                        if isinstance(val, list) and val:
                            faqs = val
                            break
                return {"content": "", "faqs": faqs}
            return None

        text = tool_generate(prompt, domain="creative")
        if text:
            return {"content": text}
        return None
    except Exception as exc:
        logger.debug("Creator LLM generation failed: %s", exc)
        return None

# ── Template fallback ───────────────────────────────────────────────

import re

# Sentence-level helpers

def _split_sentences(text: str) -> List[str]:
    """Split text into clean sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in raw if len(s.strip()) > 10]


def _extract_key_sentences(sentences: List[str], max_count: int) -> List[str]:
    """Score and rank sentences by informational density."""
    _NUM_DATE = re.compile(r'\b(\d{1,4}[%$.,]?|\d{4}|\d+\s*(seconds?|minutes?|hours?|days?|years?))\b', re.I)
    _PROPER = re.compile(r'\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})*\b')
    _ACTION = re.compile(r'\b(increase|decrease|grow|support|process|generat|extract|enable|improv|reduc|add|launch|complet|integrat|deploy|analyz|identif)\w*\b', re.I)

    scored: List[tuple] = []
    for s in sentences:
        score = 0
        score += len(_NUM_DATE.findall(s)) * 3
        score += len(_PROPER.findall(s)) * 2
        score += len(_ACTION.findall(s)) * 1
        scored.append((score, s))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [s for _, s in scored[:max_count]]
    # Re-order to preserve original narrative sequence
    idx_map = {s: i for i, s in enumerate(sentences)}
    top.sort(key=lambda s: idx_map.get(s, 999))
    return top


_STOP_TOPICS = frozenset({
    "The", "This", "These", "Those", "That", "A", "An", "All", "Any",
    "For", "In", "On", "At", "By", "As", "Is", "Are", "Was", "Were",
    "And", "Or", "But", "If", "Of", "To", "From", "With", "Without",
    "OCR", "API", "PDF", "ID", "IDs", "UI",
})


def _extract_topics(sentences: List[str], max_topics: int = 6) -> List[str]:
    """Extract meaningful proper-noun-like topics from the most informative sentences."""
    _PROPER = re.compile(r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b')
    seen: List[str] = []
    for s in sentences:
        for match in _PROPER.findall(s):
            if match in _STOP_TOPICS:
                continue
            if match not in seen:
                seen.append(match)
            if len(seen) >= max_topics:
                return seen
    # Fallback: grab first meaningful word from each sentence
    if not seen:
        for s in sentences[:max_topics]:
            words = [w.strip(",.;:") for w in s.split() if len(w) > 3]
            if words:
                seen.append(words[0])
    return seen[:max_topics]


def _build_outline(content: str) -> List[str]:
    if not content:
        return []
    sentences = _split_sentences(content)
    return sentences[:6]


# ── Specialized template generators ─────────────────────────────────

def _template_summary(text: str, tone: str, length: str) -> str:
    """Produce a structured summary: Overview + Key Findings + Conclusion."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No content available to summarize."

    length_map = {"short": 3, "long": 10, "medium": 5}
    target = length_map.get(length.lower(), 5)

    key = _extract_key_sentences(sentences, max_count=target)

    # Overview: first 1-2 key sentences
    overview_count = 1 if target <= 3 else 2
    overview_sentences = key[:overview_count]
    overview = " ".join(overview_sentences)

    # Key Findings: remaining key sentences as bullets
    finding_sentences = key[overview_count:]
    if not finding_sentences and len(sentences) > overview_count:
        finding_sentences = sentences[overview_count:overview_count + (target - overview_count)]

    # Conclusion: last substantive sentence not already used
    used = set(key)
    remaining = [s for s in sentences if s not in used]
    conclusion_sentence = remaining[-1] if remaining else (sentences[-1] if sentences else "")

    tone_label = tone.capitalize()
    lines = [
        f"**Overview** _{tone_label} tone_",
        overview,
        "",
        "**Key Findings**",
    ]
    for i, finding in enumerate(finding_sentences, 1):
        lines.append(f"  {i}. {finding}")

    if conclusion_sentence:
        lines += ["", "**Conclusion**", conclusion_sentence]

    return "\n".join(lines)


def _template_sop(text: str, tone: str) -> str:
    """Produce a structured SOP: Purpose, Scope, and numbered Procedure steps."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No content available to generate an SOP."

    _MODAL = re.compile(
        r'\b(must|shall|should|need to|required to|have to|is to|are to)\b', re.I
    )
    _ACTION_VERB = re.compile(
        r'^\s*(upload|extract|process|generate|check|verify|store|review|flag|submit|send|create|validate|ensure|confirm|run|execute|deploy|configure|set|update|delete|remove|add|enable|disable)\b',
        re.I
    )

    procedure_steps: List[str] = []
    context_sentences: List[str] = []

    # Matches: single-word subject (or "The <word>") + modal, producing an imperative.
    # e.g. "Users must upload X" → "Upload X"
    # e.g. "The system shall extract Y" → "Extract Y"
    # Does NOT strip "Documents should be..." because "be" is not an active-voice verb.
    _IMPERATIVE_STRIP = re.compile(
        r'^(?:(?:The\s+)?\w+\s+)(must|shall)\s+(?=\w)',
        re.I,
    )
    _ACTIVE_VERB = re.compile(r'^[a-z]', re.I)

    for s in sentences:
        if _MODAL.search(s) or _ACTION_VERB.search(s):
            candidate = _IMPERATIVE_STRIP.sub('', s).strip()
            # Accept the stripped version only if it starts with an active verb (not "be", "have")
            first_word = candidate.split()[0].lower() if candidate else ''
            if first_word and first_word not in ('be', 'have', 'get', 'do') and len(candidate) > 8:
                step = candidate[0].upper() + candidate[1:]
            else:
                step = s
            procedure_steps.append(step)
        else:
            context_sentences.append(s)

    if not procedure_steps:
        procedure_steps = sentences

    # Derive purpose from first non-procedural sentence or first sentence overall
    purpose_sentence = context_sentences[0] if context_sentences else sentences[0]

    # Scope: topics found across all sentences
    topics = _extract_topics(sentences, max_topics=4)
    scope_text = ", ".join(topics) if topics else "all relevant operations"

    tone_label = tone.capitalize()
    lines = [
        f"**Standard Operating Procedure** — _{tone_label} tone_",
        "",
        "**1. Purpose**",
        f"  {purpose_sentence}",
        "",
        "**2. Scope**",
        f"  This procedure applies to: {scope_text}.",
        "",
        "**3. Procedure**",
    ]
    for i, step in enumerate(procedure_steps, 1):
        lines.append(f"  Step {i}: {step}")

    return "\n".join(lines)


def _template_faq(text: str, tone: str) -> List[Dict[str, str]]:
    """Generate FAQ pairs from content using topic-matched question patterns."""
    sentences = _split_sentences(text)
    if not sentences:
        return [{"q": "What is this about?", "a": "No content was provided."}]

    key_sentences = _extract_key_sentences(sentences, max_count=8)
    topics = _extract_topics(key_sentences or sentences, max_topics=8)

    _NUM = re.compile(r'\b\d[\d,.%$]*\b')
    _TIME = re.compile(r'\b(\d+\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?))\b', re.I)

    # Question pattern functions: take a topic string, return a question string
    _Q_PATTERNS = [
        lambda t: f"What is {t}?",
        lambda t: f"How does {t} work?",
        lambda t: f"When should {t} be used?",
        lambda t: f"Why is {t} important?",
        lambda t: f"What are the benefits of {t}?",
        lambda t: f"How long does {t} take?",
        lambda t: f"Who uses {t}?",
        lambda t: f"What types of {t} are supported?",
    ]

    faqs: List[Dict[str, str]] = []
    used_sentences: set = set()

    for i, topic in enumerate(topics):
        pattern_fn = _Q_PATTERNS[i % len(_Q_PATTERNS)]
        question = pattern_fn(topic)

        # Find the most relevant answer sentence
        best: Optional[str] = None
        best_score = -1
        topic_words = set(topic.lower().split())
        for s in key_sentences:
            if s in used_sentences:
                continue
            overlap = len(topic_words & set(s.lower().split()))
            has_num = bool(_NUM.search(s)) or bool(_TIME.search(s))
            score = overlap * 2 + (2 if has_num else 0)
            if score > best_score:
                best_score = score
                best = s

        if best is None:
            # Fall back to the sentence that mentions the topic by substring
            for s in sentences:
                if s not in used_sentences and topic.lower() in s.lower():
                    best = s
                    break

        answer = best if best else sentences[i % len(sentences)]
        used_sentences.add(answer)
        faqs.append({"q": question, "a": answer})

    # If fewer than 5 FAQs, add generic ones from remaining key sentences
    remaining = [s for s in key_sentences if s not in used_sentences]
    generic_q_prefixes = ["What should you know about", "Can you explain", "What does this mean for"]
    while len(faqs) < 5 and remaining:
        s = remaining.pop(0)
        prefix = generic_q_prefixes[len(faqs) % len(generic_q_prefixes)]
        # Pick a topic word from the sentence
        words = [w for w in s.split() if len(w) > 4 and w[0].isupper()]
        topic_word = words[0] if words else "this"
        faqs.append({"q": f"{prefix} {topic_word}?", "a": s})
        used_sentences.add(s)

    return faqs[:8]


def _template_slides(text: str, tone: str) -> str:
    """Produce a slide deck outline: title slide + 4-6 content slides."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No content available to generate a slide outline."

    topics = _extract_topics(sentences, max_topics=6)
    key_sentences = _extract_key_sentences(sentences, max_count=min(24, len(sentences)))

    # Title slide: derive from most prominent topic or first sentence
    title_topic = topics[0] if topics else "Document Overview"
    subtitle = sentences[0][:80] if sentences else ""

    # Partition key sentences into groups for each content slide
    content_topics = topics[1:] if len(topics) > 1 else topics
    num_slides = max(2, min(5, len(content_topics)))
    chunk_size = max(1, len(key_sentences) // num_slides)

    tone_label = tone.capitalize()
    lines = [
        f"**Slide Deck Outline** — _{tone_label} tone_",
        "",
        "---",
        "**Slide 1: Title**",
        f"  Title: {title_topic}",
        f"  Subtitle: {subtitle}",
    ]

    for slide_idx in range(num_slides):
        slide_num = slide_idx + 2
        slide_topic = content_topics[slide_idx] if slide_idx < len(content_topics) else f"Section {slide_idx + 1}"
        start = slide_idx * chunk_size
        bullet_sentences = key_sentences[start: start + 4]
        if not bullet_sentences:
            bullet_sentences = sentences[start: start + 4]

        lines += [
            "",
            "---",
            f"**Slide {slide_num}: {slide_topic}**",
        ]
        for bullet in bullet_sentences[:4]:
            short = bullet[:100] + ("…" if len(bullet) > 100 else "")
            lines.append(f"  • {short}")

    lines += [
        "",
        "---",
        f"**Slide {num_slides + 2}: Summary & Takeaways**",
        "  • Key points reviewed",
        "  • Next steps and recommendations",
        "  • Q&A",
    ]

    return "\n".join(lines)


def _template_blog(text: str, tone: str, length: str) -> str:
    """Produce a blog post: hook intro + body sections with headers + conclusion."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No content available to generate a blog post."

    length_map = {"short": 2, "long": 4, "medium": 3}
    num_sections = length_map.get(length.lower(), 3)

    key_sentences = _extract_key_sentences(sentences, max_count=min(20, len(sentences)))
    topics = _extract_topics(sentences, max_topics=num_sections + 1)

    # Hook / intro: first key sentence or first original sentence
    hook = key_sentences[0] if key_sentences else sentences[0]
    intro_extras = key_sentences[1:3] if len(key_sentences) > 1 else []
    intro = hook + (" " + " ".join(intro_extras) if intro_extras else "")

    # Body sections
    body_topics = topics[1:] if len(topics) > 1 else topics
    body_topics = body_topics[:num_sections]
    remaining_sentences = [s for s in key_sentences if s != hook and s not in intro_extras]
    chunk_size = max(1, len(remaining_sentences) // max(1, num_sections))

    tone_label = tone.capitalize()
    title_topic = topics[0] if topics else "Document Insights"

    lines = [
        f"# {title_topic}",
        f"_{tone_label} tone_",
        "",
        "## Introduction",
        intro,
    ]

    for sec_idx, sec_topic in enumerate(body_topics):
        start = sec_idx * chunk_size
        sec_sentences = remaining_sentences[start: start + chunk_size + 1]
        sec_body = " ".join(sec_sentences) if sec_sentences else sentences[sec_idx % len(sentences)]
        lines += [
            "",
            f"## {sec_topic}",
            sec_body,
        ]

    # Conclusion: last key sentence not used in intro
    conclusion_pool = [s for s in sentences if s not in {hook} and s not in intro_extras]
    conclusion = conclusion_pool[-1] if conclusion_pool else sentences[-1]

    # Takeaways: up to 3 key sentences not used elsewhere
    used = {hook, conclusion} | set(intro_extras)
    takeaways = [s for s in key_sentences if s not in used][:3]

    lines += [
        "",
        "## Conclusion",
        conclusion,
    ]
    if takeaways:
        lines += ["", "**Key Takeaways**"]
        for t in takeaways:
            lines.append(f"  - {t}")

    return "\n".join(lines)


def _template_generate(req: CreatorRequest) -> Dict[str, Any]:
    """Dispatch to specialized template generators based on content_type."""
    reference = sanitize_text(req.text or "Provided context", max_chars=2400)
    header = f"{req.content_type.replace('_', ' ').title()} in a {req.tone} tone ({req.length})"
    outline = _build_outline(reference)
    faqs: List[Dict[str, str]] = []
    content = ""

    ct = req.content_type
    if ct == "summary":
        content = _template_summary(reference, req.tone, req.length)
    elif ct == "sop":
        content = _template_sop(reference, req.tone)
    elif ct == "faq":
        faqs = _template_faq(reference, req.tone)
        content = "\n".join(f"Q: {f['q']}\nA: {f['a']}" for f in faqs)
    elif ct == "slide_outline":
        content = _template_slides(reference, req.tone)
    elif ct == "blog":
        content = _template_blog(reference, req.tone, req.length)
    else:
        content = f"{header}:\n{reference}"

    return {
        "header": header,
        "outline": outline,
        "content": content,
        "faqs": faqs,
    }

# ── Unified generation ──────────────────────────────────────────────

def _generate_content(req: CreatorRequest) -> Dict[str, Any]:
    """Generate content using LLM first, falling back to template."""
    from src.tools.llm_tools import score_tool_response

    reference = sanitize_text(req.text or "Provided context", max_chars=2400)
    llm_result = _llm_generate(req.content_type, req.tone, req.length, reference)

    if llm_result and llm_result.get("content") or (llm_result and llm_result.get("faqs")):
        result = llm_result
        # Ensure all expected keys exist
        result.setdefault("header", f"{req.content_type.replace('_', ' ').title()} in a {req.tone} tone ({req.length})")
        result.setdefault("outline", [])
        result.setdefault("content", "")
        result.setdefault("faqs", [])
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="llm")
    else:
        result = _template_generate(req)
        iq = score_tool_response(result, domain="general", expected_fields=_EXPECTED_FIELDS, source="template")

    result["iq_score"] = iq.as_dict()
    return result

@register_tool("creator")
async def creator_handler(payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
    req = CreatorRequest(**(payload.get("input") or payload))
    result = _generate_content(req)
    sources = [build_source_record("tool", correlation_id or "creator", title=req.content_type)]
    return {"result": result, "sources": sources, "grounded": True, "context_found": True}

@router.post("/generate")
async def generate(request: CreatorRequest, x_correlation_id: str | None = Header(None)):
    cid = generate_correlation_id(x_correlation_id)
    result = _generate_content(request)
    sources = [build_source_record("tool", cid, title=request.content_type)]
    return standard_response(
        "creator",
        grounded=True,
        context_found=True,
        result=result,
        sources=sources,
        warnings=[],
        correlation_id=cid,
    )
