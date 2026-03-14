"""Flow diagram structural extraction.

Detects flow diagrams, process flows, decision trees from OCR'd image text
and extracts structured node/edge representations using LLM reasoning.
"""

import json
from src.utils.logging_utils import get_logger
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = get_logger(__name__)

@dataclass
class DiagramNode:
    """A node in a flow diagram."""
    label: str
    node_type: str = "process"  # decision, process, start_end, io, connector

@dataclass
class DiagramEdge:
    """An edge connecting two nodes."""
    from_label: str
    to_label: str
    edge_label: Optional[str] = None

@dataclass
class DiagramStructure:
    """Parsed structure of a flow diagram."""
    diagram_type: str = "flowchart"  # flowchart, process_flow, decision_tree, state_diagram
    nodes: List[DiagramNode] = field(default_factory=list)
    edges: List[DiagramEdge] = field(default_factory=list)
    description: str = ""
    raw_ocr_text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_type": self.diagram_type,
            "nodes": [{"label": n.label, "node_type": n.node_type} for n in self.nodes],
            "edges": [{"from": e.from_label, "to": e.to_label, "label": e.edge_label} for e in self.edges],
            "description": self.description,
            "confidence": self.confidence,
        }

    def to_text(self) -> str:
        """Render as readable text for embedding."""
        parts = [f"[{self.diagram_type.upper()}]"]
        if self.description:
            parts.append(self.description)
        if self.nodes:
            parts.append(f"Nodes ({len(self.nodes)}):")
            for n in self.nodes:
                parts.append(f"  - [{n.node_type}] {n.label}")
        if self.edges:
            parts.append(f"Flow ({len(self.edges)} connections):")
            for e in self.edges:
                label_part = f" ({e.edge_label})" if e.edge_label else ""
                parts.append(f"  {e.from_label} → {e.to_label}{label_part}")
        return "\n".join(parts)

# ── Detection heuristics ────────────────────────────────────────────

_ARROW_CHARS = {"→", "←", "↑", "↓", "↔", "⟶", "⟵", "▶", "◀", "➜", "➔"}
_ARROW_TEXT_RE = re.compile(
    r"(?:->|-->|=>|==>|<-|<--|<=>|->|→|←|↓|↑)", re.IGNORECASE
)
_DECISION_KEYWORDS = re.compile(
    r"\b(?:yes|no|true|false|if|else|decision|condition|branch|start|end|begin|stop|terminate)\b",
    re.IGNORECASE,
)
_FLOW_KEYWORDS = re.compile(
    r"\b(?:step\s*\d|phase\s*\d|stage\s*\d|process|input|output|action|loop|repeat|return|next)\b",
    re.IGNORECASE,
)
_SHAPE_INDICATORS = re.compile(
    r"(?:\[.*?\]|\(.*?\)|<.*?>|\{.*?\})",
)

def is_likely_diagram(ocr_text: str, image_dims: Tuple[int, int] = (0, 0)) -> bool:
    """Cheap heuristic: detect whether OCR text likely came from a diagram.

    Looks for: arrow characters, short text fragments, decision/flow keywords,
    low text density relative to image area.
    """
    if not ocr_text or len(ocr_text.strip()) < 10:
        return False

    text = ocr_text.strip()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    score = 0.0
    total_checks = 5

    # Check 1: Arrow characters or text arrows
    has_arrows = bool(_ARROW_TEXT_RE.search(text)) or any(c in text for c in _ARROW_CHARS)
    if has_arrows:
        score += 1.0

    # Check 2: Decision keywords (yes/no, if/else, start/end)
    decision_matches = len(_DECISION_KEYWORDS.findall(text))
    if decision_matches >= 2:
        score += 1.0
    elif decision_matches >= 1:
        score += 0.5

    # Check 3: Flow keywords (step, phase, process, input, output)
    flow_matches = len(_FLOW_KEYWORDS.findall(text))
    if flow_matches >= 2:
        score += 1.0
    elif flow_matches >= 1:
        score += 0.5

    # Check 4: Short fragmented text (diagram labels tend to be short)
    avg_line_len = sum(len(l) for l in lines) / len(lines) if lines else 100
    if avg_line_len < 40:
        score += 0.5
    if avg_line_len < 25:
        score += 0.5

    # Check 5: Shape indicators in text [box], (oval), {diamond}
    shape_matches = len(_SHAPE_INDICATORS.findall(text))
    if shape_matches >= 2:
        score += 1.0
    elif shape_matches >= 1:
        score += 0.5

    threshold = 0.5
    try:
        from src.api.config import Config
        threshold = getattr(Config.DiagramExtraction, "DETECTION_THRESHOLD", 0.5)
    except Exception:
        pass

    return (score / total_checks) >= threshold

# ── LLM-based structure extraction ──────────────────────────────────

_DIAGRAM_EXTRACTION_PROMPT = """Analyze this text that was OCR'd from a flow diagram/process chart image.
Extract the structural elements:

TEXT:
{ocr_text}

Return a JSON object with exactly these fields:
{{
  "diagram_type": "flowchart" or "process_flow" or "decision_tree" or "state_diagram",
  "nodes": [
    {{"label": "node text", "node_type": "process|decision|start_end|io|connector"}}
  ],
  "edges": [
    {{"from": "source node label", "to": "target node label", "label": "edge label or null"}}
  ],
  "description": "One paragraph natural language description of the flow"
}}

Rules:
- node_type: "start_end" for Start/End/Begin/Stop, "decision" for Yes/No/If branches, "process" for action steps, "io" for Input/Output, "connector" for links
- Infer edges from the spatial and textual flow even if arrows are not explicit
- description should explain the overall process in plain language
- Return ONLY valid JSON, no markdown fencing"""

def extract_diagram_with_vision(
    image: Any,
    use_thinking: bool = False,
) -> Optional[DiagramStructure]:
    """Extract diagram structure directly from image using glm-ocr vision.

    Args:
        image: PIL.Image.Image or bytes — the diagram image.
        use_thinking: Whether to use lfm2.5-thinking for post-processing.

    Returns:
        DiagramStructure if extraction succeeded, None otherwise.
    """
    try:
        from src.llm.vision_ocr import get_vision_ocr_client
        client = get_vision_ocr_client()
        if client is None or not client.is_available():
            return None

        analysis, confidence = client.analyze_image(image, analysis_type="diagram")
        if not analysis or len(analysis.strip()) < 20:
            return None

        # Parse the vision analysis into structured diagram
        result = _parse_diagram_response(analysis, analysis)
        if result and confidence:
            result.confidence = max(result.confidence, (confidence or 0) / 100.0)
        return result
    except Exception as exc:
        logger.debug("Vision diagram extraction failed: %s", exc)
        return None

def extract_diagram_structure(
    ocr_text: str,
    use_thinking: bool = False,
    image: Any = None,
) -> Optional[DiagramStructure]:
    """Extract structured diagram from OCR text using LLM.

    MoE routing:
      1. If ``image`` provided → try glm-ocr vision-first (rich visual analysis)
      2. If vision fails or no image → use lfm2.5-thinking for text-based extraction
         (faster reasoning than DocWain-Agent for structure extraction)
      3. Fallback to pattern-based text extraction

    Args:
        ocr_text: OCR'd text from the diagram image.
        use_thinking: Whether to use thinking/reasoning mode for complex diagrams.
        image: Optional image (PIL.Image or bytes) for vision-first extraction.

    Returns:
        DiagramStructure if extraction succeeded, None otherwise.
    """
    if not ocr_text or len(ocr_text.strip()) < 15:
        return None

    try:
        from src.api.config import Config
        if not getattr(Config.DiagramExtraction, "ENABLED", True):
            return None
    except Exception:
        pass

    # Strategy 1: Vision-first (glm-ocr) if image available
    if image is not None:
        vision_result = extract_diagram_with_vision(image, use_thinking=use_thinking)
        if vision_result:
            return vision_result

    # Strategy 2: LLM text-based extraction (lfm2.5-thinking preferred for structure)
    prompt = _DIAGRAM_EXTRACTION_PROMPT.format(ocr_text=ocr_text.strip()[:3000])

    try:
        from src.llm.clients import get_default_client
        client = get_default_client()

        options: Dict[str, Any] = {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 4096,
        }
        if use_thinking:
            options["think"] = True
            options["num_ctx"] = 8192
            options["num_predict"] = 4096

        # Use task_scope to route to lfm2.5-thinking for structure extraction
        try:
            from src.llm.task_router import TaskType, task_scope
            with task_scope(TaskType.AGENT_REASONING):
                response, metadata = client.generate_with_metadata(prompt, options=options)
        except ImportError:
            response, metadata = client.generate_with_metadata(prompt, options=options)
    except Exception as exc:
        logger.debug("Diagram LLM extraction failed: %s", exc)
        # Fallback: create basic structure from text patterns
        return _fallback_text_extraction(ocr_text)

    return _parse_diagram_response(response, ocr_text)

def _parse_diagram_response(response: str, ocr_text: str) -> Optional[DiagramStructure]:
    """Parse LLM JSON response into DiagramStructure."""
    # Strip markdown fencing if present
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                logger.debug("Failed to parse diagram JSON")
                return _fallback_text_extraction(ocr_text)
        else:
            return _fallback_text_extraction(ocr_text)

    nodes = []
    for n in data.get("nodes", []):
        if isinstance(n, dict) and n.get("label"):
            nodes.append(DiagramNode(
                label=str(n["label"]),
                node_type=str(n.get("node_type", "process")),
            ))

    edges = []
    for e in data.get("edges", []):
        if isinstance(e, dict) and e.get("from") and e.get("to"):
            edges.append(DiagramEdge(
                from_label=str(e["from"]),
                to_label=str(e["to"]),
                edge_label=str(e["label"]) if e.get("label") else None,
            ))

    if not nodes and not edges:
        return _fallback_text_extraction(ocr_text)

    confidence = min(1.0, (len(nodes) + len(edges)) / 10.0)
    if len(nodes) >= 3 and len(edges) >= 2:
        confidence = max(confidence, 0.7)

    return DiagramStructure(
        diagram_type=data.get("diagram_type", "flowchart"),
        nodes=nodes,
        edges=edges,
        description=data.get("description", ""),
        raw_ocr_text=ocr_text,
        confidence=confidence,
    )

def _fallback_text_extraction(ocr_text: str) -> Optional[DiagramStructure]:
    """Fallback: extract basic structure from text patterns without LLM."""
    lines = [l.strip() for l in ocr_text.split("\n") if l.strip()]
    if len(lines) < 2:
        return None

    nodes = []
    seen_labels = set()
    for line in lines:
        # Clean up line
        label = line.strip("[](){}<>-=>#* ")
        if not label or len(label) > 80 or label.lower() in seen_labels:
            continue
        seen_labels.add(label.lower())

        # Classify node type
        lower = label.lower()
        if lower in ("start", "end", "begin", "stop", "terminate"):
            node_type = "start_end"
        elif any(kw in lower for kw in ("yes", "no", "if ", "decision", "?")):
            node_type = "decision"
        elif any(kw in lower for kw in ("input", "output", "read", "write", "display")):
            node_type = "io"
        else:
            node_type = "process"

        nodes.append(DiagramNode(label=label, node_type=node_type))

    if len(nodes) < 2:
        return None

    # Create sequential edges (best guess for linear flow)
    edges = []
    for i in range(len(nodes) - 1):
        edges.append(DiagramEdge(from_label=nodes[i].label, to_label=nodes[i + 1].label))

    return DiagramStructure(
        diagram_type="process_flow",
        nodes=nodes,
        edges=edges,
        description=f"Process flow with {len(nodes)} steps extracted from diagram text.",
        raw_ocr_text=ocr_text,
        confidence=0.3,
    )
