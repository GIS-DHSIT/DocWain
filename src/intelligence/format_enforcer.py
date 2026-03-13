"""Deterministic post-processing for response formatting."""

import re

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class FormatEnforcer:
    """Enforces clean, consistent markdown formatting on LLM responses."""

    _FORMAT_HANDLERS = {
        "table": "_enforce_table",
        "bullets": "_enforce_bullets",
        "numbered": "_enforce_numbered",
        "sections": "_enforce_sections",
    }

    def enforce(self, response: str, output_format: str) -> str:
        """Apply deterministic formatting fixes to a response.

        Args:
            response: The raw LLM response text.
            output_format: One of 'table', 'bullets', 'numbered', 'sections', or empty.

        Returns:
            Cleaned and formatted response.
        """
        # Step 1: always fix general markdown issues
        result = self._fix_markdown(response)

        # Step 2: format-specific enforcement
        handler_name = self._FORMAT_HANDLERS.get(output_format)
        if handler_name:
            handler = getattr(self, handler_name)
            result = handler(result)

        # Step 3: always normalize citations
        result = self._normalize_citations(result)

        return result

    def _fix_markdown(self, text: str) -> str:
        """Fix common markdown issues."""
        # Fix table separator rows (ensure correct column count)
        lines = text.split("\n")
        fixed_lines = []
        for i, line in enumerate(lines):
            if re.match(r"^\s*\|[\s\-:|]+\|?\s*$", line) and i > 0:
                # Count columns in the header row above
                header = fixed_lines[-1] if fixed_lines else ""
                col_count = header.count("|") - 1 if header.startswith("|") else header.count("|") + 1
                if col_count > 0:
                    separator = "| " + " | ".join(["---"] * col_count) + " |"
                    fixed_lines.append(separator)
                    continue
            fixed_lines.append(line)
        text = "\n".join(fixed_lines)

        # Close unclosed bold ** markers
        # Count ** on each line; if odd, append closing **
        lines = text.split("\n")
        result_lines = []
        for line in lines:
            count = line.count("**")
            if count % 2 != 0:
                line = line + "**"
            result_lines.append(line)
        text = "\n".join(result_lines)

        # Remove triple+ blank lines -> double
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Fix orphaned list items (- without content)
        text = re.sub(r"^-\s*$", "", text, flags=re.MULTILINE)

        return text

    def _enforce_table(self, text: str) -> str:
        """Validate and fix table formatting."""
        lines = text.split("\n")
        result_lines = []
        in_table = False
        expected_cols = None

        for line in lines:
            stripped = line.strip()

            # Detect table rows
            if stripped.startswith("|") and stripped.endswith("|"):
                cells = [c.strip() for c in stripped.split("|")[1:-1]]
                col_count = len(cells)

                if not in_table:
                    in_table = True
                    expected_cols = col_count

                # Fix separator rows
                if all(re.match(r"^[-:]+$", c) for c in cells if c):
                    separator = "| " + " | ".join(["---"] * expected_cols) + " |"
                    result_lines.append(separator)
                    continue

                # Pad or trim columns to match expected
                if expected_cols and col_count != expected_cols:
                    while len(cells) < expected_cols:
                        cells.append("")
                    cells = cells[:expected_cols]
                    line = "| " + " | ".join(cells) + " |"

                result_lines.append(line)
            else:
                if in_table and stripped:
                    in_table = False
                    expected_cols = None
                result_lines.append(line)

        return "\n".join(result_lines)

    def _enforce_bullets(self, text: str) -> str:
        """Normalize mixed bullet styles to dashes and ensure proper line breaks."""
        # Normalize * and bullet to -
        text = re.sub(r"^(\s*)[*\u2022]\s+", r"\1- ", text, flags=re.MULTILINE)

        # Ensure each bullet is on its own line
        # Split cases where multiple bullets are on one line
        text = re.sub(r"(\S)\s+-\s+", r"\1\n- ", text)

        return text

    def _enforce_numbered(self, text: str) -> str:
        """Re-number broken sequences in numbered lists."""
        lines = text.split("\n")
        result_lines = []
        counter = 0
        in_list = False

        for line in lines:
            match = re.match(r"^(\s*)\d+\.\s+(.+)$", line)
            if match:
                indent = match.group(1)
                content = match.group(2)
                if not in_list:
                    in_list = True
                    counter = 0
                counter += 1
                result_lines.append(f"{indent}{counter}. {content}")
            else:
                if in_list and line.strip() == "":
                    # Blank line may continue list or end it
                    in_list = False
                    counter = 0
                result_lines.append(line)

        return "\n".join(result_lines)

    def _enforce_sections(self, text: str) -> str:
        """Ensure ## headers are properly spaced with a blank line before them."""
        # Add blank line before ## headers if not already present
        text = re.sub(r"([^\n])\n(##\s)", r"\1\n\n\2", text)
        return text

    def _normalize_citations(self, text: str) -> str:
        """Clean up and merge citations."""
        # Merge adjacent citations: [SOURCE-1][SOURCE-2] -> [SOURCE-1, SOURCE-2]
        def merge_adjacent(match):
            full = match.group(0)
            sources = re.findall(r"SOURCE-(\d+)", full)
            if sources:
                merged = ", ".join(f"SOURCE-{s}" for s in sources)
                return f"[{merged}]"
            return full

        text = re.sub(
            r"(\[SOURCE-\d+\])(\[SOURCE-\d+\])+",
            merge_adjacent,
            text,
        )

        # Collect all valid source numbers referenced in the document
        # Look for source definitions like [SOURCE-1]: or SOURCE-1 in a sources section
        all_source_refs = set(re.findall(r"SOURCE-(\d+)", text))

        # Find citations in brackets and remove those referencing non-existent sources
        # We consider a source "existent" if it appears more than once (definition + citation)
        source_counts: dict[str, int] = {}
        for s in re.findall(r"SOURCE-(\d+)", text):
            source_counts[s] = source_counts.get(s, 0) + 1

        # Sources that appear only once in a citation bracket are potentially orphaned
        # but we keep them if they exist anywhere in the text at least twice
        def clean_citation(match):
            content = match.group(1)
            sources = re.findall(r"SOURCE-(\d+)", content)
            valid = [s for s in sources if source_counts.get(s, 0) >= 1]
            if not valid:
                return ""
            merged = ", ".join(f"SOURCE-{s}" for s in valid)
            return f"[{merged}]"

        text = re.sub(r"\[(SOURCE-[\d,\s]+(?:,\s*SOURCE-\d+)*)\]", clean_citation, text)

        # Clean up orphaned brackets (empty [] or [, ])
        text = re.sub(r"\[\s*,?\s*\]", "", text)

        return text
