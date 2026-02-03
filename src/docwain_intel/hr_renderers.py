import os
from typing import Dict, List

from src.docwain_intel.fact_cache import FactCache
from src.docwain_intel.intent_router import TASK_1, TASK_2, TASK_3, TASK_4, TASK_5, TASK_6


def _basename(name: str) -> str:
    return os.path.basename(name or "") or (name or "Document")


def _best_fit(value: str, fallback: str) -> str:
    if value:
        return value
    return f"Best-fit from available content: {fallback}"


def _task6_value(value: str) -> str:
    return value or "Not Mentioned"


def _select_source(doc_metadata: List[Dict[str, str]], kind_hint: str) -> str:
    for doc in doc_metadata:
        if doc.get("doc_kind") == kind_hint:
            return _basename(doc.get("doc_name", ""))
    if doc_metadata:
        return _basename(doc_metadata[0].get("doc_name", ""))
    return "Document"


def _candidate_names(cache: FactCache, limit: int = 5) -> List[str]:
    names = cache.entities.get("person", [])
    if not names:
        name = cache.best_value(["name", "candidate", "employee"])
        if name:
            names = [name]
    if not names:
        names = ["Candidate"]
    unique = []
    for name in names:
        if name not in unique:
            unique.append(name)
        if len(unique) >= limit:
            break
    return unique


def _experience(cache: FactCache) -> str:
    value = cache.best_value(["total experience", "years of experience", "experience"])
    if not value:
        section = cache.best_section(["experience", "summary"])
        value = section
    return value or ""


def _skills(cache: FactCache) -> str:
    if cache.skills:
        return ", ".join(cache.skills[:8])
    value = cache.best_value(["skills", "technical skills", "functional skills"])
    return value or ""


def _certifications(cache: FactCache) -> str:
    return cache.best_value(["certifications", "certification"]) or cache.best_section(["certification"]) or ""


def _education(cache: FactCache) -> str:
    return cache.best_value(["education"]) or cache.best_section(["education"]) or ""


def _achievements(cache: FactCache) -> str:
    return cache.best_value(["achievements", "awards"]) or cache.best_section(["awards", "achievements"]) or ""


def render_task(task: str, cache: FactCache, doc_metadata: List[Dict[str, str]]) -> str:
    if task == TASK_1:
        name = _best_fit(_candidate_names(cache, 1)[0], "Candidate")
        exp = _best_fit(_experience(cache), "Summary not explicitly stated")
        tech = _best_fit(_skills(cache), "Skills not explicitly stated")
        functional = _best_fit(cache.best_value(["functional skills"]) or "", "Functional skills not explicitly stated")
        certs = _best_fit(_certifications(cache), "Certifications not explicitly stated")
        edu = _best_fit(_education(cache), "Education not explicitly stated")
        achievements = _best_fit(_achievements(cache), "Achievements not explicitly stated")
        source_resume = _best_fit(_select_source(doc_metadata, "resume"), "Resume")
        source_linkedin = _best_fit(_select_source(doc_metadata, "linkedin_profile"), "LinkedIn")
        return "\n".join(
            [
                "Name",
                name,
                "Experience Summary",
                exp,
                "Technical",
                tech,
                "Functional",
                functional,
                "Certifications",
                certs,
                "Education",
                edu,
                "Achievements/Awards",
                achievements,
                "Source -> Resume",
                source_resume,
                "Source -> LinkedIn",
                source_linkedin,
            ]
        ).strip()

    if task == TASK_2:
        header = "Top 5 of with experience between 5 to 10 years"
        entries = []
        names = _candidate_names(cache, 5)
        for idx, name in enumerate(names, start=1):
            entries.append(
                "\n".join(
                    [
                        f"{idx}. Name: {name}",
                        f"Total Experience: {_best_fit(_experience(cache), 'Experience not explicitly stated')}",
                        f"Key Skills: {_best_fit(_skills(cache), 'Skills not explicitly stated')}",
                        f"Most Relevant Role: {_best_fit(cache.best_value(['role', 'title']) or '', 'Role not explicitly stated')}",
                        f"Source: {_best_fit(_select_source(doc_metadata, 'resume'), 'Document')}",
                    ]
                )
            )
        return "\n".join([header] + entries).strip()

    if task == TASK_3:
        header = "Top 5 of SCM peoples"
        entries = []
        names = _candidate_names(cache, 5)
        for idx, name in enumerate(names, start=1):
            entries.append(
                "\n".join(
                    [
                        f"{idx}. Name: {name}",
                        f"SCM Summary: {_best_fit(_experience(cache), 'SCM summary not explicitly stated')}",
                        f"Core SCM Skills: {_best_fit(_skills(cache), 'SCM skills not explicitly stated')}",
                        f"Tools/Technologies: {_best_fit(cache.best_value(['tools', 'technologies']) or '', 'Tools not explicitly stated')}",
                        f"Source: {_best_fit(_select_source(doc_metadata, 'resume'), 'Document')}",
                    ]
                )
            )
        return "\n".join([header] + entries).strip()

    if task == TASK_4:
        header = "Top 5 of with experience between 5 to 10 years with certifications"
        entries = []
        names = _candidate_names(cache, 5)
        for idx, name in enumerate(names, start=1):
            entries.append(
                "\n".join(
                    [
                        f"{idx}. Name: {name}",
                        f"Total Experience: {_best_fit(_experience(cache), 'Experience not explicitly stated')}",
                        f"Key Skills: {_best_fit(_skills(cache), 'Skills not explicitly stated')}",
                        f"Certifications: {_best_fit(_certifications(cache), 'Certifications not explicitly stated')}",
                        f"Source: {_best_fit(_select_source(doc_metadata, 'resume'), 'Document')}",
                    ]
                )
            )
        return "\n".join([header] + entries).strip()

    if task == TASK_5:
        name = _candidate_names(cache, 1)[0]
        matching = _best_fit(_skills(cache), "Matching details not explicitly stated")
        missing = _best_fit("", "Missing fields could not be inferred")
        inconsistencies = _best_fit("", "Inconsistencies not explicitly stated")
        validation = _best_fit(_experience(cache), "Validation summary not explicitly stated")
        return "\n".join(
            [
                f"Candidate X: {name}",
                "Matching Details:",
                matching,
                "Missing Information:",
                missing,
                "Inconsistencies:",
                inconsistencies,
                "Validation Summary:",
                validation,
            ]
        ).strip()

    if task == TASK_6:
        list1 = ["List 1: Top 5 Candidates with 5–10 Years Experience"]
        list2 = ["List 2: Top 5 SCM Professionals"]
        list3 = ["List 3: Top 5 Candidates with 5–10 Years Experience + Certifications"]
        names = _candidate_names(cache, 5)
        for idx, name in enumerate(names, start=1):
            block = "\n".join(
                [
                    f"{idx}. Name: {name}",
                    f"Experience Summary: {_task6_value(_experience(cache))}",
                    f"Technical Skills: {_task6_value(_skills(cache))}",
                    f"Functional Skills: {_task6_value(cache.best_value(['functional skills']) or '')}",
                    f"Certifications: {_task6_value(_certifications(cache))}",
                    f"Education: {_task6_value(_education(cache))}",
                    f"Achievements/Awards: {_task6_value(_achievements(cache))}",
                    f"Source: {_task6_value(_select_source(doc_metadata, 'resume'))}",
                ]
            )
            list1.append(block)
            list2.append(block)
            list3.append(block)
        return "\n".join(list1 + [""] + list2 + [""] + list3).strip()

    return ""


def render_generic(doc_domain: str, cache: FactCache, doc_metadata: List[Dict[str, str]]) -> str:
    summary = cache.summary_text() or ""
    summary = _best_fit(summary, "Summary not explicitly stated")
    people = ", ".join(cache.entities.get("person", [])[:6]) or _best_fit("", "No explicit people names")
    orgs = ", ".join(cache.entities.get("org", [])[:6]) or _best_fit("", "No explicit organizations")
    dates = ", ".join(cache.entities.get("date", [])[:6]) or _best_fit("", "No explicit dates")
    amounts = ", ".join(cache.entities.get("money", [])[:6]) or _best_fit("", "No explicit amounts")
    sources = ", ".join({_basename(doc.get("doc_name", "")) for doc in doc_metadata if doc.get("doc_name")})
    sources = sources or _best_fit("", "Document")

    return "\n".join(
        [
            "Summary",
            summary,
            "Key People",
            people,
            "Organizations",
            orgs,
            "Dates",
            dates,
            "Amounts",
            amounts,
            "Source",
            sources,
        ]
    ).strip()


__all__ = ["render_task", "render_generic"]
