from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.deterministic.contacts import extract_contacts
from src.deterministic.multi_profile import maybe_answer_multi_profile_deterministic
from src.retrieval.profile_corpus import get_all_profile_ids


@dataclass
class DummyPoint:
    payload: Dict[str, Any]


class DummyQdrantClient:
    """
    Minimal QdrantClient stub for scroll-based deterministic tests.
    """

    def __init__(self, points: List[DummyPoint]):
        self._points = points

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Optional[int] = None,
    ) -> Tuple[List[DummyPoint], Optional[int]]:
        _ = collection_name, with_payload, with_vectors
        start = int(offset or 0)

        filtered = self._apply_filter(self._points, scroll_filter)
        batch = filtered[start : start + int(limit)]
        next_offset = start + len(batch)
        if next_offset >= len(filtered):
            next_offset = None
        return batch, next_offset

    @staticmethod
    def _apply_filter(points: List[DummyPoint], scroll_filter: Any) -> List[DummyPoint]:
        if scroll_filter is None:
            return list(points)

        must = getattr(scroll_filter, "must", None) or []
        constraints = {}
        for cond in must:
            key = getattr(cond, "key", None)
            match = getattr(cond, "match", None)
            value = getattr(match, "value", None)
            if key and value is not None:
                constraints[str(key)] = str(value)

        def ok(pt: DummyPoint) -> bool:
            for key, value in constraints.items():
                if str((pt.payload or {}).get(key) or "") != value:
                    return False
            return True

        return [pt for pt in points if ok(pt)]


def test_get_all_profile_ids_returns_full_unique_sorted_universe():
    points = [
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p3", "text": "x"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p1", "text": "x"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p2", "text": "x"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p1", "text": "x"}),  # dup
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p6", "text": "x"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p5", "text": "x"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p4", "text": "x"}),
        DummyPoint({"subscription_id": "sub-2", "profile_id": "other", "text": "x"}),  # other sub
    ]
    client = DummyQdrantClient(points)
    profile_ids = get_all_profile_ids(client=client, subscription_id="sub-1", collection="sub-1")
    assert profile_ids == ["p1", "p2", "p3", "p4", "p5", "p6"]


def test_extract_contacts_handles_separators_and_linkedin():
    text = "9826110111 | devchouhan1430@gmail.com | In https://www.linkedin.com/in/dev-chouhan-123/"
    contacts = extract_contacts(text)
    assert "9826110111" in contacts.phones
    assert "devchouhan1430@gmail.com" in contacts.emails
    assert any("linkedin.com/in/" in url for url in contacts.linkedins)


def test_contact_response_never_marks_present_fields_as_not_mentioned():
    points = [
        DummyPoint(
            {
                "subscription_id": "sub-1",
                "profile_id": "p-dev",
                "profile_name": "Dev",
                "text": "☎️ +91 9826110111 | devchouhan1430@gmail.com | In https://www.linkedin.com/in/dev-chouhan-123/",
            }
        ),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p-dev", "profile_name": "Dev", "text": "Other chunk"}),
        DummyPoint({"subscription_id": "sub-1", "profile_id": "p2", "profile_name": "Other", "text": "No contacts here"}),
    ]
    client = DummyQdrantClient(points)
    resp = maybe_answer_multi_profile_deterministic(
        client=client,
        subscription_id="sub-1",
        collection="sub-1",
        query="provide contact information of each candidate",
    )
    assert resp is not None
    answer = resp.get("response") or ""
    assert "9826110111" in answer
    assert "devchouhan1430@gmail.com" in answer
    assert "linkedin.com/in/" in answer
    # Ensure the candidate with extracted contacts is not marked missing for its own fields.
    lines = answer.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("- Dev "))
    dev_block = "\n".join(lines[start : start + 4])
    assert "Not Mentioned" not in dev_block


def test_rank_top_3_considers_full_universe():
    points = []
    for pid in ["p1", "p2", "p3", "p4", "p5", "p6"]:
        points.append(
            DummyPoint(
                {
                    "subscription_id": "sub-1",
                    "profile_id": pid,
                    "profile_name": pid.upper(),
                    "text": f"{pid} experience 2020 skills: Python, SQL\n",
                }
            )
        )
    client = DummyQdrantClient(points)
    resp = maybe_answer_multi_profile_deterministic(
        client=client, subscription_id="sub-1", collection="sub-1", query="rank top 3 profiles"
    )
    assert resp is not None
    assert len(resp.get("candidate_universe") or []) == 6
    answer = resp.get("response") or ""
    assert "Top 3 profiles" in answer
    assert answer.count(") ") >= 3
