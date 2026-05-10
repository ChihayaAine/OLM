from __future__ import annotations

from typing import Dict, List

from ..core.memory import apply_resolution_updates, resolution_record
from ..core.store import ClosedMemoryStore, OpenLoopStore
from ..core.types import EvidenceItem, OpenLoop


def verification_evidence(loop: OpenLoop, verification_text: str, source: str) -> EvidenceItem:
    return EvidenceItem(
        content=verification_text,
        polarity="support",
        source=source,
        supports_closure=True,
        metadata={"modality": loop.closure_predicate.get("modality", "tool-verifiable")},
    )


def unresolved_outcome_evidence(session_id: str, query_type: str) -> EvidenceItem:
    if query_type == "memory_reuse":
        return EvidenceItem(
            content="Scope or evidence still unresolved after memory reuse attempt.",
            polarity="support",
            source=session_id,
            supports_closure=False,
            metadata={"query_type": query_type},
        )
    return EvidenceItem(
        content="Decision remained unresolved at end of step.",
        polarity="support",
        source=session_id,
        supports_closure=False,
        metadata={"query_type": query_type},
    )


def close_loop_with_evidence(closer, loop: OpenLoop, evidence: EvidenceItem, closed_store: ClosedMemoryStore) -> Dict[str, object]:
    resolution = closer.apply(loop, evidence)
    apply_resolution_updates(closed_store, resolution.memory_updates)
    closed_store.add(resolution_record(loop, resolution.verdict))
    payload = resolution.to_dict()
    payload["loop_id"] = loop.loop_id
    payload["memory_license_histogram"] = closed_store.license_histogram()
    payload["conditioned_memories"] = list(loop.conditioned_memory_ids)
    return payload


def consolidate_expired_loops(open_store: OpenLoopStore, closed_store: ClosedMemoryStore) -> List[str]:
    archived: List[str] = []
    for loop in open_store.all():
        if loop.state != "expired":
            continue
        closed_store.add(resolution_record(loop, "superseded" if loop.unresolvedness <= 0.2 else "insufficient"))
        archived.append(loop.loop_id)
    return archived


def loop_state_summary(open_store: OpenLoopStore) -> Dict[str, object]:
    return open_store.summary()
