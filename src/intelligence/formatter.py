from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.intelligence.deterministic_router import DeterministicRoute


def _collect_attribute_values(
    facts: Iterable[Dict[str, Any]],
    key: str,
    *,
    section_kinds: Optional[List[str]] = None,
) -> List[str]:
    values: List[str] = []
    for fact in facts:
        if section_kinds and str(fact.get("section_kind")) not in set(section_kinds):
            continue
        attrs = fact.get("attributes") or {}
        if key not in attrs:
            continue
        val = attrs.get(key)
        if isinstance(val, list):
            values.extend([str(v) for v in val if v])
        elif val:
            values.append(str(val))
    seen = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _collect_entities_by_type(
    facts: Iterable[Dict[str, Any]],
    entity_type: str,
    *,
    section_kinds: Optional[List[str]] = None,
) -> List[str]:
    values: List[str] = []
    for fact in facts:
        if section_kinds and str(fact.get("section_kind")) not in set(section_kinds):
            continue
        for ent in fact.get("entities") or []:
            if str(ent.get("type")).upper() == entity_type.upper():
                val = ent.get("value") or ent.get("normalized")
                if val:
                    values.append(str(val))
    seen = set()
    return [v for v in values if not (v in seen or seen.add(v))]


def _pick_first(values: List[str]) -> Optional[str]:
    return values[0] if values else None


def _collect_evidence_sources(
    facts: Iterable[Dict[str, Any]],
    catalog: Dict[str, Any],
) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    seen = set()
    doc_lookup = {str(d.get("document_id")): d.get("source_name") for d in catalog.get("documents") or []}
    for fact in facts:
        doc_id = (fact.get("provenance") or {}).get("document_id")
        for span in fact.get("evidence_spans") or []:
            chunk_id = span.get("chunk_id")
            if not chunk_id:
                continue
            key = str(chunk_id)
            if key in seen:
                continue
            seen.add(key)
            entry = {
                "page": span.get("page"),
            }
            if doc_id and doc_lookup.get(str(doc_id)):
                entry["source_name"] = doc_lookup[str(doc_id)]
            sources.append(entry)
    return sources


def _format_sources_line(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return ""
    first = sources[0]
    name = first.get("source_name") or "source document"
    page = first.get("page")
    extras = []
    if page is not None:
        extras.append(f"page {page}")
    extra_text = f" ({', '.join(extras)})" if extras else ""
    return f"Source: {name}{extra_text}"


def _format_resume_contact(
    facts: List[Dict[str, Any]],
    *,
    catalog: Dict[str, Any],
    person_hint: Optional[str],
) -> Optional[str]:
    identity = [f for f in facts if f.get("section_kind") == "identity_contact"]
    names = _collect_attribute_values(identity, "names") or _collect_entities_by_type(identity, "PERSON")
    emails = _collect_attribute_values(identity, "emails") or _collect_entities_by_type(identity, "EMAIL")
    phones = _collect_attribute_values(identity, "phones") or _collect_entities_by_type(identity, "PHONE")
    locations = _collect_attribute_values(identity, "locations") or _collect_entities_by_type(identity, "LOCATION")

    if person_hint:
        matches = [name for name in names if person_hint.lower() in name.lower()]
        if matches:
            names = matches

    if not any([names, emails, phones, locations]):
        return None

    sources = _collect_evidence_sources(identity, catalog)
    lines = [
        f"Contact Details{f' for {person_hint}' if person_hint else ''}:",
    ]
    if names:
        lines.append(f"- Name: {', '.join(names[:3])}")
    if emails:
        lines.append(f"- Email: {', '.join(emails[:3])}")
    if phones:
        lines.append(f"- Phone: {', '.join(phones[:3])}")
    if locations:
        lines.append(f"- Location: {', '.join(locations[:3])}")
    source_line = _format_sources_line(sources)
    if source_line:
        lines.append(f"- {source_line}")
    return "\n".join(lines)


def _format_resume_skills(facts: List[Dict[str, Any]]) -> Optional[str]:
    skills = _collect_attribute_values(
        facts,
        "items",
        section_kinds=["skills_technical", "skills_functional", "tools_technologies"],
    )
    if not skills:
        return None
    return "Skills:\n- " + "\n- ".join(skills[:20])


def _format_resume_education(facts: List[Dict[str, Any]]) -> Optional[str]:
    education = [f for f in facts if f.get("section_kind") == "education"]
    entities = _collect_entities_by_type(education, "ORG")
    dates = _collect_entities_by_type(education, "DATE")
    if not entities and not dates:
        return None
    lines = ["Education:"]
    if entities:
        lines.append(f"- Institutions: {', '.join(entities[:5])}")
    if dates:
        lines.append(f"- Dates: {', '.join(dates[:5])}")
    return "\n".join(lines)


def _format_patient_details(
    facts: List[Dict[str, Any]],
    *,
    catalog: Dict[str, Any],
) -> Optional[str]:
    identity = [f for f in facts if f.get("section_kind") == "identity_contact"]
    diagnosis = [f for f in facts if f.get("section_kind") == "diagnoses_procedures"]
    medications = [f for f in facts if f.get("section_kind") == "medications"]
    notes = [f for f in facts if f.get("section_kind") == "notes"]

    names = _collect_attribute_values(identity, "names") or _collect_entities_by_type(identity, "PERSON")
    ids = _collect_attribute_values(identity, "ids") or _collect_entities_by_type(identity, "ID")
    ages = _collect_attribute_values(identity, "age")
    sex = _collect_attribute_values(identity, "sex")
    diagnoses = _collect_attribute_values(diagnosis, "terms")
    meds = _collect_attribute_values(medications, "terms")
    encounter_dates = _collect_attribute_values(notes + diagnosis, "dates")

    if not any([names, ids, ages, sex, diagnoses, meds, encounter_dates]):
        return None

    sources = _collect_evidence_sources(facts, catalog)
    lines = ["Patient Details:"]
    if names or ids:
        lines.append(f"- Name/ID: {', '.join([v for v in [(_pick_first(names) or ''), (_pick_first(ids) or '')] if v])}")
    if ages or sex:
        lines.append(f"- Age/Sex: {', '.join([v for v in [(_pick_first(ages) or ''), (_pick_first(sex) or '')] if v])}")
    if diagnoses:
        lines.append(f"- Diagnosis: {', '.join(diagnoses[:5])}")
    if meds:
        lines.append(f"- Medications: {', '.join(meds[:5])}")
    if encounter_dates:
        lines.append(f"- Encounter Date: {_pick_first(encounter_dates)}")
    source_line = _format_sources_line(sources)
    if source_line:
        lines.append(f"- {source_line}")
    return "\n".join(lines)


def _format_patient_list(
    facts: List[Dict[str, Any]],
    *,
    condition: str,
    catalog: Dict[str, Any],
) -> Optional[str]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for fact in facts:
        doc_id = (fact.get("provenance") or {}).get("document_id")
        if not doc_id:
            continue
        grouped.setdefault(str(doc_id), []).append(fact)

    doc_lookup = {str(d.get("document_id")): d.get("source_name") for d in catalog.get("documents") or []}
    lines = [f"Patients with {condition}:"]
    matched = False

    for doc_id, doc_facts in grouped.items():
        diagnosis = [f for f in doc_facts if f.get("section_kind") in {"diagnoses_procedures", "notes"}]
        mentions_condition = False
        for fact in diagnosis:
            for term in (fact.get("attributes") or {}).get("terms", []):
                if condition.lower() in str(term).lower():
                    mentions_condition = True
            for span in fact.get("evidence_spans") or []:
                if condition.lower() in str(span.get("value") or "").lower():
                    mentions_condition = True
        if not mentions_condition:
            continue

        identity = [f for f in doc_facts if f.get("section_kind") == "identity_contact"]
        names = _collect_attribute_values(identity, "names") or _collect_entities_by_type(identity, "PERSON")
        name = _pick_first(names) or "Unknown patient"
        source_name = doc_lookup.get(str(doc_id)) or "source document"
        lines.append(f"- {name} (source: {source_name})")
        matched = True

    if not matched:
        return None
    return "\n".join(lines)


def _format_generic_sections(
    facts: List[Dict[str, Any]],
    *,
    section_focus: Optional[List[str]] = None,
    domain_hint: Optional[str] = None,
) -> Optional[str]:
    if not facts:
        return None
    lines: List[str] = []
    focus_set = set(section_focus or [])
    domain_hint = (domain_hint or "").lower()
    label_map = {
        "items": "Key items",
        "amounts": "Amounts mentioned",
        "dates": "Dates mentioned",
        "terms": "Terms mentioned",
        "invoice_number": "Invoice number",
        "purchase_order_number": "Purchase order number",
        "account_number": "Account number",
        "due_date": "Due date",
        "total_amount": "Total amount",
    }
    for fact in facts:
        kind = fact.get("section_kind") or "misc"
        if focus_set and kind not in focus_set:
            continue
        attributes = fact.get("attributes") or {}
        entities = fact.get("entities") or []
        if domain_hint == "resume":
            attributes.pop("amounts", None)
        if not attributes and not entities:
            continue
        lines.append(f"{kind.replace('_', ' ').title()}:")
        for key, value in attributes.items():
            if isinstance(value, list):
                if value:
                    label = label_map.get(key, key.replace("_", " ").title())
                    lines.append(f"- {label}: {', '.join([str(v) for v in value[:10]])}")
            elif value:
                label = label_map.get(key, key.replace("_", " ").title())
                lines.append(f"- {label}: {value}")
        if entities and not attributes:
            values = ", ".join({ent.get("value") for ent in entities if ent.get("value")})
            if values:
                lines.append(f"- {values}")
    return "\n".join(lines) if lines else None


def format_facts_response(
    *,
    query: str,
    route: DeterministicRoute,
    facts: List[Dict[str, Any]],
    catalog: Dict[str, Any],
) -> Optional[str]:
    domain = route.domain_hint
    section_focus = route.section_focus
    response: Optional[str] = None

    if route.task_type == "greet":
        return None

    if route.task_type == "list" and domain == "medical":
        condition = "diabetes" if "diabetes" in (query or "").lower() else "the requested condition"
        response = _format_patient_list(facts, condition=condition, catalog=catalog)
    elif route.task_type in {"extract", "qa"} and domain == "medical":
        response = _format_patient_details(facts, catalog=catalog)
    elif domain == "resume":
        if section_focus and "identity_contact" in section_focus:
            response = _format_resume_contact(facts, catalog=catalog, person_hint=route.target_person)
        elif section_focus and any(kind in section_focus for kind in ["skills_technical", "skills_functional", "tools_technologies"]):
            response = _format_resume_skills(facts)
        elif section_focus and "education" in section_focus:
            response = _format_resume_education(facts)
    if response is None:
        if section_focus and any(kind in section_focus for kind in ["skills_technical", "skills_functional", "tools_technologies", "education"]):
            return None
        response = _format_generic_sections(facts, section_focus=section_focus, domain_hint=domain)

    if response:
        sources = _collect_evidence_sources(facts, catalog)
        source_line = _format_sources_line(sources)
        if source_line and "Source:" not in response:
            suffix = f"- {source_line}" if "\n-" in response else source_line
            response = f"{response}\n{suffix}"
    return response


__all__ = ["format_facts_response", "_collect_evidence_sources"]
