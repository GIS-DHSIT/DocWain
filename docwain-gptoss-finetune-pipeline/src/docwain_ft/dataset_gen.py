import argparse
import json
import logging
import random
import uuid
from datetime import date
from pathlib import Path
from typing import Iterable

from docwain_ft.config import CONFIG
from docwain_ft.prompts import DOCWAIN_SYSTEM_ANCHOR, OUTPUT_RULES_DEFAULT
from docwain_ft.utils import ensure_dir, setup_logging, write_jsonl

SCENARIOS = [
    "resume_summary",
    "rank_candidates",
    "invoice_extraction",
    "medical_admission",
    "procurement_po",
    "multi_doc_compare",
    "missing_field_inference",
    "conflicting_field_resolution",
]

DOC_TYPES = ["invoice", "resume", "medical", "procurement", "summary"]
SECTIONS = ["Overview", "Details", "Line Items", "Summary", "Notes"]
CHUNK_KINDS = ["text", "table", "header", "footer"]
PROFILE_NAMES = ["A. Patel", "B. Ruiz", "C. Nguyen", "D. Smith"]


def _random_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _chunk_meta(rng: random.Random) -> dict:
    return {
        "doc_name": f"Doc_{rng.randint(100,999)}",
        "page": rng.randint(1, 12),
        "section": rng.choice(SECTIONS),
        "chunk_kind": rng.choice(CHUNK_KINDS),
        "profile_name": rng.choice(PROFILE_NAMES),
    }


def _noisy_prefix(rng: random.Random) -> str:
    return rng.choice(["", "B)", "B)", "(i)", ""])


def _context_chunk(rng: random.Random, text: str) -> dict:
    meta = _chunk_meta(rng)
    noisy = _noisy_prefix(rng)
    chunk_id = _random_uuid(rng)
    return {
        **meta,
        "chunk_id": chunk_id,
        "text": f"{noisy} {text} [chunk_id:{chunk_id}]",
    }


def _render_context(chunks: list[dict]) -> str:
    lines = []
    for chunk in chunks:
        meta = (
            f"doc_name={chunk['doc_name']} page={chunk['page']} "
            f"section={chunk['section']} chunk_kind={chunk['chunk_kind']} "
            f"profile_name={chunk['profile_name']}"
        )
        lines.append(f"- {meta}\n  {chunk['text']}")
    return "Retrieved Context:\n" + "\n".join(lines)


def _citations(chunks: list[dict]) -> str:
    sources = {(c["doc_name"], c["page"]) for c in chunks}
    return "\n".join([f"Source: {doc} p{page}" for doc, page in sorted(sources)])


def _assistant_output(body: str, chunks: list[dict]) -> str:
    return f"{body}\n\n{_citations(chunks)}"


def _base_user_query(scenario: str) -> str:
    if scenario == "resume_summary":
        return "User Query: Provide a concise resume summary for the candidate."
    if scenario == "rank_candidates":
        return "User Query: Rank the top 2 candidates by years of experience (descending)."
    if scenario == "invoice_extraction":
        return "User Query: Extract invoice number, invoice date, terms, and total amount."
    if scenario == "medical_admission":
        return "User Query: Extract admission date, diagnosis, and attending physician."
    if scenario == "procurement_po":
        return "User Query: Extract PO number, vendor, and line items with quantity and unit price."
    if scenario == "multi_doc_compare":
        return "User Query: Compare totals across the two invoices and state the difference."
    if scenario == "missing_field_inference":
        return "User Query: Provide the payment due date."
    if scenario == "conflicting_field_resolution":
        return "User Query: Resolve the correct invoice total."
    return "User Query: Summarize."


def _schema_rules(rng: random.Random, scenario: str) -> str:
    if scenario in {"invoice_extraction", "procurement_po", "medical_admission"}:
        return (
            "Output Rules:\n"
            "Return JSON with keys: invoice_number, invoice_date, terms, total_amount.\n"
            "If a key is missing, use 'Unknown from provided documents'.\n"
            "Provide citations after JSON."
        )
    if scenario == "rank_candidates":
        return (
            "Output Rules:\n"
            "Return a table with columns: Candidate | YearsExperience | Rank.\n"
            "Provide citations after the table."
        )
    if scenario == "missing_field_inference":
        return (
            "Output Rules:\n"
            "Respond with a single line: Payment Due Date: <value>.\n"
            "If inferred, append ' (Inference)'.\n"
            "Provide citations after the line."
        )
    if scenario == "conflicting_field_resolution":
        return (
            "Output Rules:\n"
            "Respond with: Resolved Total: <value> and brief rationale.\n"
            "Provide citations after the rationale."
        )
    if scenario == "multi_doc_compare":
        return (
            "Output Rules:\n"
            "Respond with: Invoice A Total, Invoice B Total, Difference.\n"
            "Provide citations after the values."
        )
    return OUTPUT_RULES_DEFAULT


def _scenario_chunks(rng: random.Random, scenario: str) -> list[dict]:
    chunks: list[dict] = []
    today = date(2024, 6, 1)
    if scenario == "resume_summary":
        chunks.append(_context_chunk(rng, "Candidate has 6 years of data engineering experience at Orion Labs."))
        chunks.append(_context_chunk(rng, "Skills include Python, Spark, AWS, and ETL design."))
    elif scenario == "rank_candidates":
        chunks.append(_context_chunk(rng, "A. Patel has 8 years experience in ML engineering."))
        chunks.append(_context_chunk(rng, "B. Ruiz has 5 years experience in backend systems."))
        chunks.append(_context_chunk(rng, "C. Nguyen has 10 years experience in data science."))
    elif scenario == "invoice_extraction":
        chunks.append(_context_chunk(rng, "Invoice #INV-1002 dated 2024-05-12 with Net 30 terms."))
        chunks.append(_context_chunk(rng, "Total due is $12,450.00 including tax."))
    elif scenario == "medical_admission":
        chunks.append(_context_chunk(rng, "Admission Date: 2024-04-18. Diagnosis: Pneumonia."))
        chunks.append(_context_chunk(rng, "Attending Physician: Dr. J. Lee."))
    elif scenario == "procurement_po":
        chunks.append(_context_chunk(rng, "PO-7788 Vendor: Nova Supplies."))
        chunks.append(_context_chunk(rng, "Line Item: Surgical masks, Qty 200, Unit Price $0.45."))
        chunks.append(_context_chunk(rng, "Line Item: Nitrile gloves, Qty 100, Unit Price $0.30."))
    elif scenario == "multi_doc_compare":
        chunks.append(_context_chunk(rng, "Invoice A Total: $8,200.00 (Doc A)."))
        chunks.append(_context_chunk(rng, "Invoice B Total: $9,750.00 (Doc B)."))
    elif scenario == "missing_field_inference":
        chunks.append(_context_chunk(rng, "Invoice date: 2024-05-01. Terms: Net 15."))
        chunks.append(_context_chunk(rng, "Payment instructions indicate due within 15 days."))
    elif scenario == "conflicting_field_resolution":
        chunks.append(_context_chunk(rng, "Total due shows $4,900.00 in summary."))
        chunks.append(_context_chunk(rng, "Line items sum to $4,850.00."))
        chunks.append(_context_chunk(rng, "Final total on signature page: $4,900.00."))
    else:
        chunks.append(_context_chunk(rng, "General summary text."))

    # Add noise and IDs
    for _ in range(rng.randint(1, 3)):
        chunks.append(_context_chunk(rng, f"Reference ID {_random_uuid(rng)} for internal tracking."))

    rng.shuffle(chunks)
    return chunks[: rng.randint(3, 8)]


def _assistant_body(scenario: str, chunks: list[dict]) -> str:
    if scenario == "resume_summary":
        return "Candidate has 6 years of data engineering experience with strong Python, Spark, AWS, and ETL skills."
    if scenario == "rank_candidates":
        return (
            "Candidate | YearsExperience | Rank\n"
            "C. Nguyen | 10 | 1\n"
            "A. Patel | 8 | 2"
        )
    if scenario == "invoice_extraction":
        payload = {
            "invoice_number": "INV-1002",
            "invoice_date": "2024-05-12",
            "terms": "Net 30",
            "total_amount": "$12,450.00",
        }
        return json.dumps(payload, ensure_ascii=False)
    if scenario == "medical_admission":
        payload = {
            "invoice_number": "Unknown from provided documents",
            "invoice_date": "Unknown from provided documents",
            "terms": "Unknown from provided documents",
            "total_amount": "Unknown from provided documents",
        }
        return json.dumps(payload, ensure_ascii=False)
    if scenario == "procurement_po":
        payload = {
            "invoice_number": "PO-7788",
            "invoice_date": "Unknown from provided documents",
            "terms": "Unknown from provided documents",
            "total_amount": "Unknown from provided documents",
        }
        return json.dumps(payload, ensure_ascii=False)
    if scenario == "multi_doc_compare":
        return "Invoice A Total: $8,200.00; Invoice B Total: $9,750.00; Difference: $1,550.00"
    if scenario == "missing_field_inference":
        return "Payment Due Date: 2024-05-16 (Inference)"
    if scenario == "conflicting_field_resolution":
        return "Resolved Total: $4,900.00. Rationale: summary and signature page align."
    return "Summary unavailable."


def build_samples(rng: random.Random, size: int) -> Iterable[dict]:
    for i in range(size):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        chunks = _scenario_chunks(rng, scenario)
        user_parts = [
            _base_user_query(scenario),
            _render_context(chunks),
            _schema_rules(rng, scenario),
        ]
        user_content = "\n\n".join(user_parts)
        assistant_content = _assistant_output(_assistant_body(scenario, chunks), chunks)
        yield {
            "messages": [
                {"role": "system", "content": DOCWAIN_SYSTEM_ANCHOR},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "scenario": scenario,
        }


def build_eval_samples(rng: random.Random, size: int) -> Iterable[dict]:
    for i in range(size):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        chunks = _scenario_chunks(rng, scenario)
        user_parts = [
            _base_user_query(scenario),
            _render_context(chunks),
            _schema_rules(rng, scenario),
        ]
        user_content = "\n\n".join(user_parts)
        expected_output = _assistant_output(_assistant_body(scenario, chunks), chunks)
        yield {
            "prompt": user_content,
            "scenario": scenario,
            "expected_output": expected_output,
            "context": "\n".join([c["text"] for c in chunks]),
            "output_rules": _schema_rules(rng, scenario),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DocWain synthetic datasets.")
    parser.add_argument("--out-sft", default="data/docwain_sft.jsonl")
    parser.add_argument("--out-eval", default="data/docwain_eval.jsonl")
    parser.add_argument("--size", type=int, default=CONFIG.dataset_size)
    parser.add_argument("--eval-size", type=int, default=CONFIG.evalset_size)
    parser.add_argument("--seed", type=int, default=CONFIG.seed)
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("dataset_gen")
    rng = random.Random(args.seed)

    sft_samples = list(build_samples(rng, args.size))
    eval_samples = list(build_eval_samples(rng, args.eval_size))

    ensure_dir(Path(args.out_sft).parent)
    ensure_dir(Path(args.out_eval).parent)

    write_jsonl(args.out_sft, sft_samples)
    write_jsonl(args.out_eval, eval_samples)

    logger.info("Generated %s SFT samples and %s eval samples.", len(sft_samples), len(eval_samples))


if __name__ == "__main__":
    main()
