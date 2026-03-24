"""Relationship graph / flow diagram renderer using networkx + matplotlib + plotly.

Parses Mermaid-like graph syntax from LLM output and renders as both:
  - Static PNG (matplotlib) for Teams / fallback
  - Interactive HTML (Plotly) for web channels

Both renders run in parallel via ThreadPoolExecutor.
Integrates with the existing chart pipeline through RenderedChart.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

from src.utils.logging_utils import get_logger
from src.visualization.chart_renderer import DOCWAIN_THEME, RenderedChart

logger = get_logger(__name__)


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class GraphNode:
    node_id: str
    label: str
    shape: str = "box"  # box, circle, diamond, rounded


@dataclass
class GraphEdge:
    source: str
    target: str
    label: str = ""
    style: str = "solid"  # solid, dashed, bold


@dataclass
class GraphData:
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    direction: str = "TD"  # TD (top-down), LR (left-right)
    title: str = ""


# ── Query Detection ───────────────────────────────────────────────────────────

# Keywords that signal the user wants a relationship/structure diagram
_RELATIONSHIP_KEYWORDS = re.compile(
    r"\b("
    r"relationship|relationships|relation|relations|"
    r"flow diagram|flowchart|flow chart|"
    r"diagram of|structure diagram|"
    r"entity relationship|erd|"
    r"map the connection|map the connections|"
    r"organisational chart|organizational chart|org chart|"
    r"process flow|workflow|"
    r"show.*connection|connections between|"
    r"how.*connect|how.*relate|"
    r"architecture diagram|dependency|dependencies"
    r")\b",
    re.IGNORECASE,
)

# "diagram of", "flow diagram", etc. without data-chart indicators
_DIAGRAM_PHRASE = re.compile(
    r"\b(diagram|flow diagram|flowchart|process flow|workflow|org chart|map)\b",
    re.IGNORECASE,
)

# Keywords that reveal the user wants a data chart (not a relationship graph)
_DATA_CHART_KEYWORDS = re.compile(
    r"\b("
    r"over time|sales|revenue|expenses?|budget|cost|costs|"
    r"trend|monthly|yearly|annually|quarterly|"
    r"percentage|percent|proportion|distribution of (?:values|numbers|amounts)|"
    r"bar graph of|pie chart of|line graph of|bar chart of"
    r")\b",
    re.IGNORECASE,
)

# Explicit relationship-graph triggers (even if "graph" word is absent)
_EXPLICIT_GRAPH_TRIGGERS = re.compile(
    r"\b("
    r"show.*relationship|relationship.*between|"
    r"diagram.*contract|contract.*diagram|"
    r"create.*flow diagram|create.*flowchart|"
    r"show.*structure|show.*connections|"
    r"map.*relationships|relationship.*map|"
    r"graph of the|graph of (?:a|an|the|this|that)"
    r")\b",
    re.IGNORECASE,
)


def is_graph_query(query: str) -> bool:
    """Return True when the query is asking for a relationship/flow diagram.

    Key distinction:
    - "graph" + relationship/structure/flow/connection keywords → True
    - "graph" + data/numeric keywords → False (data chart)
    - Explicit "diagram of X structure", "flow diagram", "org chart" etc. → True
    """
    q = query.strip()

    # Explicit triggers always win
    if _EXPLICIT_GRAPH_TRIGGERS.search(q):
        return True

    # Plain diagram / flowchart phrases (with no data-chart context)
    if _DIAGRAM_PHRASE.search(q) and not _DATA_CHART_KEYWORDS.search(q):
        return True

    # "graph" or "chart" together with relationship/structure signals
    has_graph_word = bool(re.search(r"\b(graph|chart|visuali[sz]e)\b", q, re.IGNORECASE))
    has_relationship = bool(_RELATIONSHIP_KEYWORDS.search(q))
    has_data_chart = bool(_DATA_CHART_KEYWORDS.search(q))

    if has_graph_word and has_relationship and not has_data_chart:
        return True

    return False


# ── Mermaid Parser ────────────────────────────────────────────────────────────

# Matches the opening of a mermaid/flowchart block
_MERMAID_HEADER = re.compile(
    r"^\s*(graph|flowchart)\s+(TD|TB|LR|RL|BT)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Node definition patterns  (order matters – most specific first)
# NodeId([Label])  — stadium/rounded-rectangle
_NODE_STADIUM = re.compile(r"""^(\w[\w\s]*?)\s*\(\[([^\]]*)\]\)\s*$""")
# NodeId[Label]    — rectangle  (label may not contain ']')
_NODE_RECT = re.compile(r"""^(\w[\w\s]*?)\s*\[([^\]]*)\]\s*$""")
# NodeId(Label)    — rounded rectangle  (label may not contain ')')
_NODE_ROUND = re.compile(r"""^(\w[\w\s]*?)\s*\(([^)]*)\)\s*$""")
# NodeId{Label}    — diamond
_NODE_DIAMOND = re.compile(r"""^(\w[\w\s]*?)\s*\{([^}]*)\}\s*$""")
# NodeId>Label]    — asymmetric
_NODE_ASYMM = re.compile(r"""^(\w[\w\s]*?)\s*>([^\]]*)\]\s*$""")

# Node token: bare id optionally followed by shape brackets
# e.g. Client, Client[Label], Client(Label), Client{Label}, Client([Label])
_NODE_TOKEN_RE = r"""(\w[\w\d]*(?:\([^\)]*\)|\[[^\]]*\]|\{[^}]*\}|\(\[[^\]]*\]\))?)"""

# Edge patterns (with optional label)
# A -->|Label| B   or   A --> B
_EDGE_LABELLED = re.compile(
    _NODE_TOKEN_RE + r"""\s*"""
    r"""(-{1,2}>{1,2}|\.{0,2}-{0,2}>{1,2}|={1,2}>{1,2}|-{1,2}[^|>-]+-{1,2}>{1,2})"""
    r"""\|(.*?)\|\s*""" + _NODE_TOKEN_RE + r"""\s*$"""
)
_EDGE_PLAIN = re.compile(
    _NODE_TOKEN_RE + r"""\s*"""
    r"""(-{1,2}>{1,2}|\.{0,2}-{0,2}>{1,2}|={1,2}>{1,2}|-{1,2}[^|>\s]+-{1,2}>{1,2}|---?)"""
    r"""\s*""" + _NODE_TOKEN_RE + r"""\s*$"""
)


def _classify_edge_style(arrow: str) -> str:
    if "==>" in arrow or "==" in arrow:
        return "bold"
    if ".->" in arrow or "-.-" in arrow or "-.>" in arrow:
        return "dashed"
    return "solid"


def _parse_node_token(token: str) -> Optional[GraphNode]:
    """Try to parse a bare node token (no arrow present)."""
    token = token.strip()
    if not token:
        return None
    for pattern, shape in [
        (_NODE_STADIUM, "rounded"),
        (_NODE_RECT, "box"),
        (_NODE_ROUND, "rounded"),
        (_NODE_DIAMOND, "diamond"),
        (_NODE_ASYMM, "box"),
    ]:
        m = pattern.match(token)
        if m:
            nid = m.group(1).strip()
            label = m.group(2).strip()
            return GraphNode(node_id=nid, label=label or nid, shape=shape)
    # Bare identifier
    if re.match(r"^\w[\w\s]*$", token):
        return GraphNode(node_id=token, label=token, shape="box")
    return None


def parse_mermaid(text: str) -> Optional[GraphData]:
    """Parse Mermaid graph syntax from LLM response text.

    Handles:
    - ```mermaid ... ``` code fences
    - graph TD / graph LR / flowchart TD headers
    - Node shapes: [], (), {}, ([]), >]
    - Edges: -->, -->|label|, ---, -.->  , ==>, --text-->
    """
    # Extract code from ```mermaid ... ``` fences first
    fence_match = re.search(
        r"```(?:mermaid)?\s*\n?(graph|flowchart)\s+(TD|TB|LR|RL|BT)(.*?)```",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if fence_match:
        direction_str = fence_match.group(2).upper()
        body = fence_match.group(3)
    else:
        # Look for bare mermaid header anywhere in text
        header_match = _MERMAID_HEADER.search(text)
        if not header_match:
            return None
        direction_str = header_match.group(2).upper()
        # Take everything from the header to end (or next blank section)
        start = header_match.end()
        body = text[start:]

    # Normalize direction
    direction = "TD" if direction_str in ("TD", "TB") else "LR"

    nodes: dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        # Skip empty lines and style/class declarations
        if not line or line.startswith(("classDef", "class ", "%%", "style ", "linkStyle", "subgraph", "end")):
            continue

        # Handle subgraph-like groupings: just skip the header
        if re.match(r"^subgraph\s", line, re.IGNORECASE):
            continue

        # Try labelled edge first  A -->|label| B
        m = _EDGE_LABELLED.match(line)
        if m:
            src_token, arrow, edge_label, tgt_token = m.group(1), m.group(2), m.group(3), m.group(4)
            _register_node_from_token(src_token.strip(), nodes)
            _register_node_from_token(tgt_token.strip(), nodes)
            edges.append(GraphEdge(
                source=_bare_id(src_token),
                target=_bare_id(tgt_token),
                label=edge_label.strip(),
                style=_classify_edge_style(arrow),
            ))
            continue

        # Try plain edge  A --> B
        m = _EDGE_PLAIN.match(line)
        if m:
            src_token, arrow, tgt_token = m.group(1), m.group(2), m.group(3)
            _register_node_from_token(src_token.strip(), nodes)
            _register_node_from_token(tgt_token.strip(), nodes)
            edges.append(GraphEdge(
                source=_bare_id(src_token),
                target=_bare_id(tgt_token),
                label="",
                style=_classify_edge_style(arrow),
            ))
            continue

        # Try standalone node definition
        node = _parse_node_token(line)
        if node and node.node_id not in nodes:
            nodes[node.node_id] = node

    if not nodes and not edges:
        return None

    # Ensure all edge endpoints have a node entry
    for edge in edges:
        for nid in (edge.source, edge.target):
            if nid not in nodes:
                nodes[nid] = GraphNode(node_id=nid, label=nid, shape="box")

    return GraphData(
        nodes=list(nodes.values()),
        edges=edges,
        direction=direction,
    )


def _bare_id(token: str) -> str:
    """Extract the bare node id from a token that may contain shape syntax."""
    token = token.strip()
    # Strip shape brackets
    for pat in (_NODE_STADIUM, _NODE_RECT, _NODE_ROUND, _NODE_DIAMOND, _NODE_ASYMM):
        m = pat.match(token)
        if m:
            return m.group(1).strip()
    return token


def _register_node_from_token(token: str, nodes: dict) -> None:
    """Parse a node definition out of an edge endpoint token and register it."""
    token = token.strip()
    node = _parse_node_token(token)
    if node:
        if node.node_id not in nodes:
            nodes[node.node_id] = node
    else:
        bare = _bare_id(token)
        if bare and bare not in nodes:
            nodes[bare] = GraphNode(node_id=bare, label=bare, shape="box")


# ── Fallback Relationship Extractor ──────────────────────────────────────────

_VERB_PATTERNS = re.compile(
    # Subject: starts with capital, allow dots (for "Ltd.", "Inc.", etc.), stop at verb
    r"([A-Z][^\n,]+?)\s+(?:engages?|delivers?(?:\s+to)?|pays?|sends?|"
    r"receives?|connects?(?:\s+to)?|links?(?:\s+to)?|flows?(?:\s+to)?|"
    r"reports?(?:\s+to)?|uses?|provides?|manages?|owns?|contains?)\s+"
    # Object: starts with capital, runs to end of sentence
    r"([A-Z][^\n,\.]+)",
    re.MULTILINE,
)

_ARROW_PATTERN = re.compile(
    r"([A-Za-z][^\n→\->]+?)\s*(?:→|->|-->)\s*([A-Za-z][^\n→\->]+)",
)

_TABLE_ROW = re.compile(r"\|([^|]+)\|([^|]+)\|([^|]+)\|")


def extract_relationships(text: str) -> Optional[GraphData]:
    """Fallback parser for natural-language or arrow relationship patterns."""
    nodes: dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    def _add(src: str, tgt: str, label: str = "") -> None:
        src, tgt = src.strip()[:50], tgt.strip()[:50]
        if not src or not tgt or src == tgt:
            return
        for nid, lbl in ((src, src), (tgt, tgt)):
            if nid not in nodes:
                nodes[nid] = GraphNode(node_id=nid, label=lbl, shape="box")
        edges.append(GraphEdge(source=src, target=tgt, label=label))

    # Arrow patterns
    for m in _ARROW_PATTERN.finditer(text):
        _add(m.group(1).strip(), m.group(2).strip())

    # Verb-phrase patterns
    for m in _VERB_PATTERNS.finditer(text):
        verb_phrase = m.group(0)
        verb_match = re.search(
            r"\b(engages?|delivers?(?:\s+to)?|pays?|sends?|receives?|"
            r"connects?(?:\s+to)?|flows?(?:\s+to)?|reports?(?:\s+to)?|"
            r"uses?|provides?|manages?|owns?|contains?)\b",
            verb_phrase, re.IGNORECASE
        )
        edge_label = verb_match.group(0) if verb_match else ""
        _add(m.group(1), m.group(2), edge_label)

    # Markdown tables with 3+ columns (try to find "A → B" in cells)
    for m in _TABLE_ROW.finditer(text):
        cells = [c.strip() for c in [m.group(1), m.group(2), m.group(3)]]
        # Heuristic: if second column looks like a relationship verb, treat as A → C
        if re.match(r"^[a-z ]+$", cells[1], re.IGNORECASE) and cells[0] and cells[2]:
            _add(cells[0], cells[2], cells[1])

    if not edges:
        return None

    return GraphData(nodes=list(nodes.values()), edges=edges, direction="TD")


# ── Relationship Table Extractor ─────────────────────────────────────────────

_REL_TABLE_HEADER_RE = re.compile(
    r"\|\s*(?:source|from|entity|party|node)\s*\|"
    r"\s*(?:relationship|relation|type|action|link|role|connects?)\s*\|"
    r"\s*(?:target|to|destination|object|party)\s*\|",
    re.IGNORECASE,
)

_TABLE_DATA_ROW_RE = re.compile(r"^\|([^|]+)\|([^|]+)\|([^|]+)\|", re.MULTILINE)


def extract_relationship_table(text: str) -> Optional[GraphData]:
    """Parse a Source|Relationship|Target markdown table into a GraphData.

    The system prompt instructs the LLM to produce this format instead of
    Mermaid code blocks.
    """
    if not _REL_TABLE_HEADER_RE.search(text):
        return None

    nodes: dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []

    for m in _TABLE_DATA_ROW_RE.finditer(text):
        cells = [c.strip().replace("**", "") for c in [m.group(1), m.group(2), m.group(3)]]
        src, rel, tgt = cells

        # Skip header row and separator rows
        if src.lower() in ("source", "from", "entity", "party", "node"):
            continue
        if re.match(r"^[-:\s]+$", src):
            continue
        if not src or not tgt:
            continue

        src_id = re.sub(r"[^A-Za-z0-9_]", "_", src)[:30]
        tgt_id = re.sub(r"[^A-Za-z0-9_]", "_", tgt)[:30]

        if src_id not in nodes:
            nodes[src_id] = GraphNode(node_id=src_id, label=src, shape="box")
        if tgt_id not in nodes:
            nodes[tgt_id] = GraphNode(node_id=tgt_id, label=tgt, shape="box")

        edges.append(GraphEdge(source=src_id, target=tgt_id, label=rel.strip()))

    if not edges:
        return None

    return GraphData(nodes=list(nodes.values()), edges=edges, direction="TD")


# ── Graph Renderer ────────────────────────────────────────────────────────────


def _build_nx_graph(graph_data: GraphData) -> Tuple[Any, Dict[str, Any], Dict[str, "GraphNode"]]:
    """Build a networkx DiGraph with layout positions.

    Returns (G, pos, node_map) so both renderers can share the same
    topology and layout without duplicating computation.
    """
    import networkx as nx

    G = nx.DiGraph()
    node_map = {n.node_id: n for n in graph_data.nodes}

    for node in graph_data.nodes:
        G.add_node(node.node_id, label=node.label, shape=node.shape)
    for edge in graph_data.edges:
        G.add_edge(edge.source, edge.target, label=edge.label, style=edge.style)

    n_nodes = len(G.nodes)
    if n_nodes <= 2:
        pos = nx.spring_layout(G, seed=42)
    elif n_nodes <= 8:
        try:
            pos = nx.kamada_kawai_layout(G)
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5)
    else:
        pos = nx.spring_layout(G, seed=42, k=2.0, iterations=100)

    if graph_data.direction == "LR":
        pos = {n: (y, -x) for n, (x, y) in pos.items()}
    else:
        pos = {n: (x, -y) for n, (x, y) in pos.items()}

    return G, pos, node_map


# ── Matplotlib (static PNG) ──────────────────────────────────────────────────


def _render_matplotlib(
    G: Any,
    pos: Dict[str, Any],
    node_map: Dict[str, "GraphNode"],
    graph_data: GraphData,
    title: str,
) -> str:
    """Render the graph to a base64-encoded PNG using matplotlib.

    Thread-safe: creates its own figure and closes it before returning.
    """
    import networkx as nx

    palette = DOCWAIN_THEME["palette"]
    primary = DOCWAIN_THEME["primary"]
    secondary = DOCWAIN_THEME["secondary"]
    bg = DOCWAIN_THEME["background"]
    text_color = DOCWAIN_THEME["text"]
    n_nodes = len(G.nodes)

    fig_w = max(10, min(18, n_nodes * 1.8))
    fig_h = max(7, min(14, n_nodes * 1.4))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.axis("off")

    # Edges
    edge_labels: dict = {}
    for u, v, data in G.edges(data=True):
        style = data.get("style", "solid")
        lbl = data.get("label", "")
        if lbl:
            edge_labels[(u, v)] = lbl

        linestyle = "--" if style == "dashed" else "-"
        lw = 2.5 if style == "bold" else 1.8

        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            edge_color=secondary, arrows=True, arrowsize=20,
            arrowstyle="-|>", style=linestyle, width=lw,
            connectionstyle="arc3,rad=0.05",
            node_size=_node_size(n_nodes),
            min_source_margin=25, min_target_margin=25,
        )

    # Nodes
    node_colors = []
    for i, node_id in enumerate(G.nodes()):
        shape = node_map.get(node_id, GraphNode(node_id, node_id)).shape
        color = palette[i % len(palette)] if shape == "diamond" else primary
        node_colors.append(color)

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=node_colors,
        node_size=_node_size(n_nodes), node_shape="o",
        alpha=0.92, linewidths=1.5, edgecolors="white",
    )

    # Node labels
    node_labels = {n.node_id: _wrap_label(n.label, 16) for n in graph_data.nodes}
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=max(6, 9 - n_nodes // 4),
        font_color="white", font_weight="bold",
    )

    # Edge labels
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, ax=ax,
            font_size=max(5, 8 - n_nodes // 5),
            font_color=text_color,
            bbox={"boxstyle": "round,pad=0.2", "fc": "#F0F4FF", "ec": secondary, "alpha": 0.85},
            rotate=False,
        )

    ax.set_title(title, fontsize=14, color=text_color, fontweight="bold", pad=12)

    # Legend for mixed edge styles
    edge_styles = {d.get("style", "solid") for _, _, d in G.edges(data=True)}
    if len(edge_styles) > 1:
        patches = []
        if "solid" in edge_styles:
            patches.append(mpatches.Patch(color=secondary, label="Direct relationship"))
        if "dashed" in edge_styles:
            patches.append(mpatches.Patch(color=secondary, linestyle="--", label="Indirect / optional"))
        if "bold" in edge_styles:
            patches.append(mpatches.Patch(color=primary, linewidth=3, label="Strong relationship"))
        if patches:
            ax.legend(handles=patches, loc="lower right", fontsize=8,
                      framealpha=0.8, facecolor=bg, edgecolor=secondary)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=bg)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ── Plotly (interactive HTML + JSON) ─────────────────────────────────────────


def _render_plotly(
    G: Any,
    pos: Dict[str, Any],
    node_map: Dict[str, "GraphNode"],
    graph_data: GraphData,
    title: str,
    width: int = 900,
    height: int = 650,
) -> Dict[str, Any]:
    """Render the graph as interactive Plotly HTML and JSON spec.

    Returns {"html": str, "json": dict}.
    Thread-safe: Plotly figures are independent objects.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        logger.warning("plotly not installed — skipping interactive graph render")
        return {"html": "", "json": {}}

    palette = DOCWAIN_THEME["palette"]
    primary = DOCWAIN_THEME["primary"]
    secondary = DOCWAIN_THEME["secondary"]
    bg = DOCWAIN_THEME["background"]
    text_color = DOCWAIN_THEME["text"]
    font_family = DOCWAIN_THEME["font"]

    # ── Edge traces (lines + arrows + labels) ────────────────────────────
    edge_traces = []
    edge_annotations = []
    edge_label_traces = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        style = data.get("style", "solid")
        label = data.get("label", "")

        dash = "dash" if style == "dashed" else ("solid" if style != "bold" else "solid")
        lw = 3.0 if style == "bold" else 1.8

        # Edge line
        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line={"width": lw, "color": secondary, "dash": dash},
            hoverinfo="none",
            showlegend=False,
        ))

        # Arrowhead annotation
        edge_annotations.append({
            "ax": x0, "ay": y0,
            "x": x1, "y": y1,
            "xref": "x", "yref": "y",
            "axref": "x", "ayref": "y",
            "showarrow": True,
            "arrowhead": 3,
            "arrowsize": 1.5,
            "arrowwidth": lw,
            "arrowcolor": secondary,
            "standoff": 28,
            "startstandoff": 20,
        })

        # Edge label at midpoint
        if label:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            edge_label_traces.append(go.Scatter(
                x=[mx], y=[my],
                mode="text",
                text=[label],
                textfont={"size": 10, "color": text_color, "family": font_family},
                hoverinfo="none",
                showlegend=False,
            ))

    # ── Node trace ───────────────────────────────────────────────────────
    node_x, node_y, node_text, node_hover, node_colors_list = [], [], [], [], []
    for i, node in enumerate(graph_data.nodes):
        x, y = pos[node.node_id]
        node_x.append(x)
        node_y.append(y)
        node_text.append(_wrap_label(node.label, 18))
        node_hover.append(f"<b>{node.label}</b><br>ID: {node.node_id}")

        # Colour per node — cycle through palette for variety
        color = palette[i % len(palette)] if node.shape == "diamond" else primary
        node_colors_list.append(color)

    # Determine node marker size based on count
    n = len(graph_data.nodes)
    marker_size = max(30, min(55, 60 - n * 3))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont={"size": max(8, 12 - n // 3), "color": "white", "family": font_family},
        hovertext=node_hover,
        hoverinfo="text",
        marker={
            "size": marker_size,
            "color": node_colors_list,
            "line": {"width": 2, "color": "white"},
            "symbol": "circle",
        },
        showlegend=False,
    )

    # ── Assemble figure ──────────────────────────────────────────────────
    fig = go.Figure(
        data=[*edge_traces, *edge_label_traces, node_trace],
        layout=go.Layout(
            title={"text": title, "font": {"size": 16, "color": text_color, "family": font_family}},
            font={"family": font_family, "color": text_color},
            showlegend=False,
            hovermode="closest",
            plot_bgcolor=bg,
            paper_bgcolor=bg,
            width=width,
            height=height,
            annotations=edge_annotations,
            xaxis={
                "showgrid": False, "zeroline": False, "showticklabels": False,
                "showline": False, "visible": False,
            },
            yaxis={
                "showgrid": False, "zeroline": False, "showticklabels": False,
                "showline": False, "visible": False,
            },
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
        ),
    )

    try:
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        json_spec = fig.to_dict()
        return {"html": html, "json": json_spec}
    except Exception as exc:
        logger.warning("Plotly graph render failed: %s", exc)
        return {"html": "", "json": {}}


# ── Unified render — parallel matplotlib + plotly ────────────────────────────


def render_graph(
    graph_data: GraphData,
    title: str = "Relationship Diagram",
    channel: str = "web",
) -> RenderedChart:
    """Render a GraphData as both static PNG and interactive Plotly in parallel.

    Both renderers share the same networkx layout so the diagram looks
    consistent across channels.  A ThreadPoolExecutor runs them concurrently.

    Returns a RenderedChart with:
      - matplotlib_png_base64  (always attempted)
      - plotly_html / plotly_json  (web channel only)
    """
    import networkx as nx

    effective_title = graph_data.title or title
    G, pos, node_map = _build_nx_graph(graph_data)

    n_nodes = len(G.nodes)
    fig_w = max(10, min(18, n_nodes * 1.8))
    fig_h = max(7, min(14, n_nodes * 1.4))

    # ── Launch both renderers in parallel ────────────────────────────────
    png_b64 = ""
    plotly_html = ""
    plotly_json: dict = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # Always render matplotlib PNG
        mpl_future = executor.submit(
            _render_matplotlib, G, pos, node_map, graph_data, effective_title,
        )

        # Render Plotly for web channel
        plotly_future = None
        if channel == "web":
            plotly_future = executor.submit(
                _render_plotly, G, pos, node_map, graph_data, effective_title,
                int(fig_w * 100), int(fig_h * 100),
            )

        # Collect matplotlib result
        try:
            png_b64 = mpl_future.result(timeout=10.0)
        except Exception as exc:
            logger.warning("Matplotlib graph render failed: %s", exc)

        # Collect Plotly result
        if plotly_future is not None:
            try:
                plotly_result = plotly_future.result(timeout=10.0)
                plotly_html = plotly_result.get("html", "")
                plotly_json = plotly_result.get("json", {})
            except Exception as exc:
                logger.warning("Plotly graph render failed: %s", exc)

    node_count = len(graph_data.nodes)
    edge_count = len(graph_data.edges)
    data_summary = f"{effective_title}: {node_count} nodes, {edge_count} relationships"

    return RenderedChart(
        chart_type="graph",
        title=effective_title,
        matplotlib_png_base64=png_b64,
        plotly_html=plotly_html,
        plotly_json=plotly_json,
        width=int(fig_w * 100),
        height=int(fig_h * 100),
        data_summary=data_summary,
    )


def _node_size(n: int) -> int:
    """Scale node size with graph density."""
    if n <= 4:
        return 3500
    if n <= 8:
        return 2800
    if n <= 15:
        return 2000
    return 1400


def _wrap_label(label: str, max_chars: int = 16) -> str:
    """Wrap a long label into multiple lines."""
    words = label.split()
    lines: List[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = (current + " " + word).strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines) if lines else label


# ── Top-Level Pipeline Entry Point ───────────────────────────────────────────

def try_graph_rendering(
    response_text: str,
    query: str,
    channel: str = "web",
) -> Optional[RenderedChart]:
    """Try to render a relationship graph from LLM response text.

    Called by the enhancer before the standard chart pipeline.

    Steps:
    1. Check is_graph_query(query)
    2. Parse Mermaid syntax from response_text
    3. Fallback to extract_relationships if Mermaid not found
    4. Render and return RenderedChart, or None if not applicable

    Args:
        response_text: The LLM's full response text (may contain Mermaid blocks)
        query:         The original user query
        channel:       "web" or "teams"

    Returns:
        RenderedChart with chart_type="graph", or None.
    """
    try:
        if not is_graph_query(query):
            return None

        # Try Mermaid parse first
        graph_data = parse_mermaid(response_text)

        # Fallback to natural-language relationship extraction
        if graph_data is None:
            graph_data = extract_relationships(response_text)

        if graph_data is None or not graph_data.edges:
            logger.debug("Graph rendering skipped: no parseable graph structure found")
            return None

        # Use query words as title hint
        title = _infer_title(query)
        if graph_data.title:
            title = graph_data.title

        rendered = render_graph(graph_data, title=title, channel=channel)
        logger.info(
            "Graph rendered: %s nodes, %s edges, %s chars base64",
            len(graph_data.nodes),
            len(graph_data.edges),
            len(rendered.matplotlib_png_base64),
        )
        return rendered

    except Exception as exc:
        logger.warning("Graph rendering failed: %s", exc, exc_info=True)
        return None


def _infer_title(query: str) -> str:
    """Generate a human-readable title from the query."""
    q = query.strip().rstrip("?.")
    # Remove leading verbs
    q = re.sub(r"^(show me|create|generate|draw|display|give me|can you|please)\s+", "", q, flags=re.IGNORECASE)
    q = re.sub(r"^(a|an|the)\s+", "", q, flags=re.IGNORECASE)
    if len(q) > 60:
        q = q[:57] + "..."
    return q.title() if q else "Relationship Diagram"
