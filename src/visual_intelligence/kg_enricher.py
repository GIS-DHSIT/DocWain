"""Parallel KG enrichment for visual intelligence entities.

Non-blocking, non-fatal writes — failures are logged but never
propagate to the caller.
"""

from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Dict, List

from src.visual_intelligence.datatypes import (
    VisualEnrichmentResult,
    VisualRegion,
    StructuredTableResult,
    KVPair,
)

logger = logging.getLogger(__name__)


class VisualKGEnricher:
    """Builds and enqueues KG payloads from visual enrichment results."""

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def build_payload(
        self,
        doc_id: str,
        subscription_id: str,
        profile_id: str,
        result: VisualEnrichmentResult,
    ) -> Dict[str, Any]:
        """Create a KG ingest payload from a *VisualEnrichmentResult*.

        Node types produced:
            - ``LayoutRegion``  — one per *VisualRegion*
            - ``StructuredTable`` — one per *StructuredTableResult*
            - ``FormField``     — one per *KVPair*

        Edge types produced:
            - ``Document -[HAS_REGION]-> LayoutRegion``
            - ``LayoutRegion -[CONTAINS]-> StructuredTable``  (bbox match)
            - ``KVPair -[FOUND_IN]-> LayoutRegion``           (same page)
            - ``Document -[HAS_TABLE]-> StructuredTable``     (no region match)
        """
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        # --- Region nodes & Document-[HAS_REGION]->Region edges ----------
        region_ids: Dict[int, List[Dict[str, Any]]] = {}  # page -> region records
        for region in result.regions:
            rid = str(uuid.uuid4())
            node = {
                "id": rid,
                "type": "LayoutRegion",
                "label": region.label,
                "bbox": region.bbox,
                "confidence": region.confidence,
                "page": region.page,
            }
            nodes.append(node)
            edges.append({
                "source": doc_id,
                "target": rid,
                "relation": "HAS_REGION",
            })
            region_ids.setdefault(region.page, []).append(node)

        # --- Table nodes & edges -----------------------------------------
        for table in result.tables:
            tid = str(uuid.uuid4())
            nodes.append({
                "id": tid,
                "type": "StructuredTable",
                "headers": table.headers,
                "row_count": len(table.rows),
                "confidence": table.confidence,
                "page": table.page,
                "bbox": table.bbox,
            })

            # Try to find a matching region on the same page with same bbox.
            matched = False
            for rnode in region_ids.get(table.page, []):
                if rnode["bbox"] == table.bbox:
                    edges.append({
                        "source": rnode["id"],
                        "target": tid,
                        "relation": "CONTAINS",
                    })
                    matched = True
                    break

            if not matched:
                edges.append({
                    "source": doc_id,
                    "target": tid,
                    "relation": "HAS_TABLE",
                })

        # --- FormField nodes & KV-[FOUND_IN]->Region edges ---------------
        for kv in result.kv_pairs:
            kid = str(uuid.uuid4())
            nodes.append({
                "id": kid,
                "type": "FormField",
                "key": kv.key,
                "value": kv.value,
                "confidence": kv.confidence,
                "page": kv.page,
            })

            # Link to the first region on the same page (if any).
            page_regions = region_ids.get(kv.page, [])
            if page_regions:
                edges.append({
                    "source": kid,
                    "target": page_regions[0]["id"],
                    "relation": "FOUND_IN",
                })

        return {
            "doc_id": doc_id,
            "subscription_id": subscription_id,
            "profile_id": profile_id,
            "nodes": nodes,
            "edges": edges,
        }

    def enqueue_enrichment(
        self,
        doc_id: str,
        subscription_id: str,
        profile_id: str,
        result: VisualEnrichmentResult,
    ) -> None:
        """Build and fire-and-forget enqueue a KG payload.

        Skips silently when the result contains no visual entities.
        """
        has_entities = bool(result.regions or result.tables or result.kv_pairs)
        if not has_entities:
            logger.debug("No visual entities for doc %s — skipping KG enrichment", doc_id)
            return

        payload = self.build_payload(doc_id, subscription_id, profile_id, result)
        self._enqueue(payload)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _enqueue(self, payload: Dict[str, Any]) -> None:
        """Fire a daemon thread to write the payload to the KG store."""
        thread = threading.Thread(
            target=self._write_to_kg,
            args=(payload,),
            daemon=True,
        )
        thread.start()

    def _write_to_kg(self, payload: Dict[str, Any]) -> None:
        """Attempt to push *payload* into the graph ingest queue.

        If the KG subsystem is not available the call is silently skipped.
        Any unexpected error is logged as a warning — never raised.
        """
        try:
            from src.kg.ingest import GraphIngestQueue, GraphIngestPayload  # type: ignore[import-untyped]

            ingest_payload = GraphIngestPayload(
                doc_id=payload["doc_id"],
                subscription_id=payload["subscription_id"],
                profile_id=payload["profile_id"],
                nodes=payload["nodes"],
                edges=payload["edges"],
            )
            GraphIngestQueue.push(ingest_payload)
            logger.info("KG enrichment enqueued for doc %s", payload["doc_id"])
        except ImportError:
            logger.debug(
                "KG ingest module unavailable — skipping enrichment for doc %s",
                payload["doc_id"],
            )
        except Exception:
            logger.warning(
                "Failed to enqueue KG enrichment for doc %s",
                payload["doc_id"],
                exc_info=True,
            )
