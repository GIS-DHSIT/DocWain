"""Evaluation prompts used by the Observer to probe model weaknesses."""

from __future__ import annotations

from typing import Dict, List


def get_eval_prompts() -> List[Dict[str, str]]:
    """Return 30+ evaluation prompts spanning all target subcategories.

    Each prompt dict has keys: query, category, subcategory.
    Categories: document_understanding, interaction_quality.
    """
    return [
        # --- document_understanding / table_extraction ---
        {"query": "What are the Q3 revenue figures from the financial summary table?", "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Extract the top 5 rows from the employee compensation table.", "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Compare the 2024 and 2025 budget columns from the spending table.", "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "List all items in the inventory table that have quantity below 10.", "category": "document_understanding", "subcategory": "table_extraction"},
        {"query": "Summarize the data in the performance metrics table on page 7.", "category": "document_understanding", "subcategory": "table_extraction"},

        # --- document_understanding / layout_parsing ---
        {"query": "What is the title of the document and where does the introduction end?", "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "Identify the header and footer content on page 3.", "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "How many columns does the text layout use in the methodology section?", "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "Describe the layout structure of the first page including any sidebars.", "category": "document_understanding", "subcategory": "layout_parsing"},
        {"query": "Where are the figure captions positioned relative to their images?", "category": "document_understanding", "subcategory": "layout_parsing"},

        # --- document_understanding / cross_reference ---
        {"query": "What does the reference in Section 3.2 to Appendix B describe?", "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "Follow the citation [12] and summarize the referenced finding.", "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "The footnote on page 5 references a table. Which table is it and what does it show?", "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "Link the abbreviation defined in the glossary to its usage in section 4.", "category": "document_understanding", "subcategory": "cross_reference"},
        {"query": "What clause does the contract term in paragraph 8 refer back to?", "category": "document_understanding", "subcategory": "cross_reference"},

        # --- document_understanding / section_hierarchy ---
        {"query": "Outline the document's section hierarchy down to three levels.", "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "Which subsection falls under Chapter 2 in the report?", "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "List all top-level headings in the policy document.", "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "How is section 5 subdivided and what topics do the sub-sections cover?", "category": "document_understanding", "subcategory": "section_hierarchy"},
        {"query": "Describe the nesting depth of the table of contents.", "category": "document_understanding", "subcategory": "section_hierarchy"},

        # --- document_understanding / multi_page_reasoning ---
        {"query": "Combine information from pages 3, 7, and 12 to explain the project timeline.", "category": "document_understanding", "subcategory": "multi_page_reasoning"},
        {"query": "The conclusion on page 20 contradicts a statement on page 4. Explain the discrepancy.", "category": "document_understanding", "subcategory": "multi_page_reasoning"},
        {"query": "Trace the evolution of the budget from the proposal on page 2 through the revision on page 15.", "category": "document_understanding", "subcategory": "multi_page_reasoning"},
        {"query": "Using data from multiple sections, calculate the total cost mentioned across the document.", "category": "document_understanding", "subcategory": "multi_page_reasoning"},

        # --- interaction_quality / uncertainty_handling ---
        {"query": "What is the exact delivery date if the contract does not specify one?", "category": "interaction_quality", "subcategory": "uncertainty_handling"},
        {"query": "Can you confirm the author's intent behind the ambiguous clause in section 6?", "category": "interaction_quality", "subcategory": "uncertainty_handling"},
        {"query": "What are the projected savings for next year based on incomplete data?", "category": "interaction_quality", "subcategory": "uncertainty_handling"},
        {"query": "Is the product recall mandatory based on the regulatory notice?", "category": "interaction_quality", "subcategory": "uncertainty_handling"},

        # --- interaction_quality / adaptive_tone ---
        {"query": "Explain the legal implications of clause 14 in simple terms.", "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "Provide a technical summary of the architecture diagram for the engineering team.", "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "Write a brief executive summary of this compliance report.", "category": "interaction_quality", "subcategory": "adaptive_tone"},
        {"query": "Describe the findings in a way suitable for a non-technical stakeholder.", "category": "interaction_quality", "subcategory": "adaptive_tone"},
    ]
