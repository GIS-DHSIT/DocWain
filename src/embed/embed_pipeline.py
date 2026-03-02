from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple

from src.cache.redis_keys import RedisKeys
from src.cache.redis_store import RedisStore
from src.embed.entity_extractor import EntityExtractor
from src.embed.chunk_builder import canonicalize_text
from src.kg.kg_store import EvidencePointer, KGStore


def update_profile_indexes_from_embeddings(
    *,
    embeddings_payload: Dict[str, Any],
    subscription_id: str,
    profile_id: str,
    document_id: str,
    file_name: str,
    redis_client: Optional[Any] = None,
    kg_store: Optional[KGStore] = None,
) -> Dict[str, Any]:
    if not redis_client:
        return {"updated": False, "reason": "redis_unavailable"}

    keys = RedisKeys(subscription_id=str(subscription_id), profile_id=str(profile_id))
    store = RedisStore(redis_client)

    texts = embeddings_payload.get("texts") or []
    chunk_metadata = embeddings_payload.get("chunk_metadata") or []
    doc_domain = embeddings_payload.get("doc_domain") or "generic"
    if doc_domain in {"unknown", "generic", None}:
        try:
            from src.intelligence.domain_indexer import infer_domain

            sample_text = " ".join([t for t in texts[:6] if t])
            doc_domain = infer_domain(sample_text, doc_type=embeddings_payload.get("doc_type"), source_name=file_name)
        except Exception:
            doc_domain = doc_domain or "generic"

    page_count, has_tables, has_lists = _infer_doc_signals(chunk_metadata)
    detected_language = _detect_language_hint(embeddings_payload, texts)
    chunk_count = len(texts)

    catalog = store.get_catalog(keys) or {
        "profile_id": str(profile_id),
        "subscription_id": str(subscription_id),
        "updated_at": int(time.time()),
        "files": [],
    }
    catalog["updated_at"] = int(time.time())

    files = list(catalog.get("files") or [])
    updated_files: List[Dict[str, Any]] = []
    replaced = False
    for entry in files:
        if entry.get("document_id") == str(document_id):
            updated_files.append(
                _build_catalog_entry(
                    document_id=document_id,
                    file_name=file_name,
                    doc_domain_hint=doc_domain,
                    detected_language=detected_language,
                    page_count=page_count,
                    chunk_count=chunk_count,
                    has_tables=has_tables,
                    has_lists=has_lists,
                )
            )
            replaced = True
        else:
            updated_files.append(entry)
    if not replaced:
        updated_files.append(
            _build_catalog_entry(
                document_id=document_id,
                file_name=file_name,
                doc_domain_hint=doc_domain,
                detected_language=detected_language,
                page_count=page_count,
                chunk_count=chunk_count,
                has_tables=has_tables,
                has_lists=has_lists,
            )
        )
    catalog["files"] = updated_files
    store.set_catalog(keys, catalog)

    layout_ref = {
        "blob_pointer_latest": None,
        "version_id": embeddings_payload.get("doc_version_hash"),
        "page_count": page_count,
        "signals": {"has_tables": has_tables, "has_lists": has_lists, "ocr_used": bool(embeddings_payload.get("ocr_used"))},
    }
    store.set_layout_ref(keys, str(document_id), layout_ref)

    section_index = _build_section_index(chunk_metadata)
    store.set_section_index(keys, str(document_id), section_index)

    entity_index_entries, kg_stats = _build_entity_index(
        texts,
        chunk_metadata,
        document_id=document_id,
        file_name=file_name,
        doc_domain=doc_domain,
        kg_store=kg_store,
        subscription_id=subscription_id,
        profile_id=profile_id,
    )
    store.update_entity_index(keys, entity_index_entries)

    if kg_stats:
        store.set_kg_anchor_stats(keys, kg_stats)

    return {
        "updated": True,
        "catalog_files": len(updated_files),
        "sections": len(section_index.get("sections") or []),
        "entities": sum(len(v) for v in (entity_index_entries or {}).values()),
    }


def _build_catalog_entry(
    *,
    document_id: str,
    file_name: str,
    doc_domain_hint: str,
    detected_language: str,
    page_count: int,
    chunk_count: int,
    has_tables: bool,
    has_lists: bool,
) -> Dict[str, Any]:
    return {
        "document_id": str(document_id),
        "file_name": file_name,
        "doc_domain_hint": doc_domain_hint or "generic",
        "detected_language": detected_language or "unknown",
        "page_count": int(page_count or 0),
        "chunk_count": int(chunk_count or 0),
        "has_tables": bool(has_tables),
        "has_lists": bool(has_lists),
    }


def _infer_doc_signals(chunk_metadata: List[Dict[str, Any]]) -> Tuple[int, bool, bool]:
    page_count = 0
    has_tables = False
    has_lists = False
    for meta in chunk_metadata:
        if not isinstance(meta, dict):
            continue
        page = meta.get("page") or meta.get("page_number") or meta.get("page_start")
        if isinstance(page, int):
            page_count = max(page_count, page)
        chunk_kind = str(meta.get("chunk_kind") or meta.get("chunk_type") or "")
        if chunk_kind.startswith("table"):
            has_tables = True
        if chunk_kind.startswith("list"):
            has_lists = True
    if page_count <= 0:
        page_count = 1
    return page_count, has_tables, has_lists


def _build_section_index(chunk_metadata: List[Dict[str, Any]]) -> Dict[str, Any]:
    sections: Dict[str, Dict[str, Any]] = {}
    for meta in chunk_metadata:
        if not isinstance(meta, dict):
            continue
        section_id = str(meta.get("section_id") or meta.get("section") or "")
        title = meta.get("section_title") or meta.get("section") or None
        if not section_id:
            continue
        page_start = meta.get("page_start") or meta.get("page") or meta.get("page_number")
        page_end = meta.get("page_end") or page_start
        entry = sections.get(section_id) or {
            "section_id": section_id,
            "title": title,
            "page_start": page_start,
            "page_end": page_end,
            "kind_structural": True,
        }
        if page_start is not None:
            entry["page_start"] = min(int(entry.get("page_start") or page_start), int(page_start))
        if page_end is not None:
            entry["page_end"] = max(int(entry.get("page_end") or page_end), int(page_end))
        sections[section_id] = entry
    return {
        "sections": list(sections.values()),
        "digest": hashlib.sha1("|".join(sorted(sections.keys())).encode("utf-8")).hexdigest()[:12],
    }


def _build_entity_index(
    texts: List[str],
    chunk_metadata: List[Dict[str, Any]],
    *,
    document_id: str,
    file_name: str,
    doc_domain: str,
    kg_store: Optional[KGStore],
    subscription_id: str,
    profile_id: str,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
    extractor = EntityExtractor()
    entity_index: Dict[str, List[Dict[str, Any]]] = {}
    kg_store = kg_store or KGStore()
    entity_count = 0
    edge_count = 0

    for idx, text in enumerate(texts):
        meta = chunk_metadata[idx] if idx < len(chunk_metadata) else {}
        canonical = canonicalize_text(meta.get("canonical_text") or meta.get("embedding_text") or text or "")
        entities = extractor.extract(canonical)
        if not entities:
            continue
        section_id = str(meta.get("section_id") or "")
        page = meta.get("page") or meta.get("page_number") or meta.get("page_start")
        snippet = _snippet(canonical)
        snippet_sha = hashlib.sha1(snippet.encode("utf-8")).hexdigest()[:12]
        evidence = EvidencePointer(
            file_name=file_name,
            document_id=str(document_id),
            section_id=section_id,
            page=int(page) if page is not None else None,
            snippet=snippet,
            snippet_sha=snippet_sha,
        )
        kg_store.upsert_document(subscription_id, profile_id, str(document_id), file_name, doc_domain=doc_domain)
        if section_id:
            kg_store.upsert_section(subscription_id, profile_id, str(document_id), section_id)
        for ent in entities:
            entity_count += 1
            entry = {
                "file_name": file_name,
                "document_id": str(document_id),
                "section_id": section_id,
                "page": int(page) if page is not None else None,
                "snippet_sha": snippet_sha,
            }
            entity_index.setdefault(ent.entity_norm, []).append(entry)
            kg_store.upsert_entity(subscription_id, profile_id, ent.entity_norm, ent.entity_type, ent.surface_form)
            if section_id:
                kg_store.add_mention(
                    subscription_id,
                    profile_id,
                    ent.entity_norm,
                    ent.entity_type,
                    section_id,
                    str(document_id),
                    evidence,
                    doc_domain=doc_domain,
                )
        for a_idx, ent_a in enumerate(entities):
            for ent_b in entities[a_idx + 1 :]:
                edge_count += 1
                kg_store.add_cooccurrence(
                    subscription_id,
                    profile_id,
                    ent_a.entity_norm,
                    ent_a.entity_type,
                    ent_b.entity_norm,
                    ent_b.entity_type,
                    section_id,
                    str(document_id),
                    evidence,
                    doc_domain=doc_domain,
                )

    stats = {
        "entities": int(entity_count),
        "edges": int(edge_count),
        "last_built_at": int(time.time()),
    }
    return entity_index, stats


def _snippet(text: str, limit: int = 160) -> str:
    if not text:
        return ""
    cleaned = " ".join(text.replace("\n", " ").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip()


def _detect_language_hint(embeddings_payload: Dict[str, Any], texts: List[str]) -> str:
    lang = embeddings_payload.get("detected_language")
    if not lang:
        languages = embeddings_payload.get("languages")
        if isinstance(languages, list) and languages:
            lang = languages[0]
        elif isinstance(languages, str):
            lang = languages
    if lang:
        return str(lang)
    try:
        from src.utils.language import detect_language

        sample = " ".join([t for t in texts[:4] if t])
        lang, _conf = detect_language(sample)
        return lang
    except Exception:
        return "unknown"


__all__ = ["update_profile_indexes_from_embeddings"]
