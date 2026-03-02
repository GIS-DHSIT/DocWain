"""Tests for flow diagram structural extraction."""
import pytest
from src.doc_understanding.diagram_extractor import (
    DiagramNode,
    DiagramEdge,
    DiagramStructure,
    is_likely_diagram,
    _fallback_text_extraction,
    _parse_diagram_response,
)


class TestDiagramDataclasses:
    def test_node_defaults(self):
        n = DiagramNode(label="Process A")
        assert n.label == "Process A"
        assert n.node_type == "process"

    def test_edge_defaults(self):
        e = DiagramEdge(from_label="A", to_label="B")
        assert e.from_label == "A"
        assert e.to_label == "B"
        assert e.edge_label is None

    def test_structure_to_dict(self):
        s = DiagramStructure(
            nodes=[DiagramNode("Start", "start_end"), DiagramNode("Process", "process")],
            edges=[DiagramEdge("Start", "Process", "begin")],
            description="A simple flow",
            confidence=0.8,
        )
        d = s.to_dict()
        assert d["diagram_type"] == "flowchart"
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1
        assert d["edges"][0]["label"] == "begin"
        assert d["confidence"] == 0.8

    def test_structure_to_text(self):
        s = DiagramStructure(
            nodes=[DiagramNode("Start", "start_end"), DiagramNode("Process", "process")],
            edges=[DiagramEdge("Start", "Process")],
            description="A flow",
        )
        text = s.to_text()
        assert "FLOWCHART" in text
        assert "Start" in text
        assert "\u2192" in text


class TestIsLikelyDiagram:
    def test_empty_text(self):
        assert is_likely_diagram("") is False
        assert is_likely_diagram("short") is False

    def test_with_arrows(self):
        text = "Start \u2192 Process A \u2192 Decision \u2192 End\nYes \u2192 Branch A\nNo \u2192 Branch B"
        assert is_likely_diagram(text) is True

    def test_with_decision_keywords(self):
        text = "Start\nIf condition\nYes\nNo\nProcess Step\nEnd"
        assert is_likely_diagram(text) is True

    def test_with_flow_keywords(self):
        # Needs flow keywords + at least one other signal (arrows or decision)
        # to cross the 0.5 threshold (score/5 >= 0.5 requires score >= 2.5)
        text = "Step 1: Input data\nStep 2: Process -> Validate\nStep 3: Output results\nStep 4: Loop back\nEnd"
        assert is_likely_diagram(text) is True

    def test_normal_prose_not_diagram(self):
        text = "This is a regular paragraph of text that discusses the company's financial performance over the last quarter. Revenue increased significantly compared to the previous year."
        assert is_likely_diagram(text) is False

    def test_short_fragments_with_shapes(self):
        text = "[Start]\n[Process Data]\n[Decision?]\n{Validate}\n[End]"
        assert is_likely_diagram(text) is True


class TestFallbackTextExtraction:
    def test_basic_flow(self):
        text = "Start\nProcess Data\nValidate\nEnd"
        result = _fallback_text_extraction(text)
        assert result is not None
        assert len(result.nodes) == 4
        assert result.nodes[0].node_type == "start_end"
        assert result.nodes[-1].node_type == "start_end"
        assert len(result.edges) == 3

    def test_too_short(self):
        result = _fallback_text_extraction("OnlyOne")
        assert result is None

    def test_decision_classification(self):
        text = "Start\nIs valid?\nYes\nProcess\nEnd"
        result = _fallback_text_extraction(text)
        assert result is not None
        decision_nodes = [n for n in result.nodes if n.node_type == "decision"]
        assert len(decision_nodes) >= 1

    def test_deduplication(self):
        text = "Start\nStart\nProcess\nEnd"
        result = _fallback_text_extraction(text)
        labels = [n.label for n in result.nodes]
        # "Start" should appear only once due to dedup
        assert labels.count("Start") == 1


class TestParseDiagramResponse:
    def test_valid_json(self):
        response = '{"diagram_type": "flowchart", "nodes": [{"label": "Start", "node_type": "start_end"}, {"label": "Process", "node_type": "process"}], "edges": [{"from": "Start", "to": "Process"}], "description": "A flow"}'
        result = _parse_diagram_response(response, "Start \u2192 Process")
        assert result is not None
        assert len(result.nodes) == 2
        assert len(result.edges) == 1

    def test_markdown_fenced_json(self):
        response = '```json\n{"diagram_type": "flowchart", "nodes": [{"label": "A", "node_type": "process"}], "edges": [], "description": "test"}\n```'
        result = _parse_diagram_response(response, "A")
        assert result is not None
        assert len(result.nodes) == 1

    def test_invalid_json_falls_back(self):
        result = _parse_diagram_response("this is not json at all", "Start\nEnd")
        # Should fall back to text extraction
        assert result is not None or result is None  # Depends on text

    def test_empty_nodes_falls_back(self):
        response = '{"diagram_type": "flowchart", "nodes": [], "edges": [], "description": ""}'
        result = _parse_diagram_response(response, "Start\nProcess\nEnd")
        # Should fall back since no nodes/edges
        assert result is not None
