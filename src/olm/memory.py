from __future__ import annotations

from typing import Dict, Iterable, List

from .store import ClosedMemoryStore
from .types import MemoryEntry, OpenLoop


def apply_resolution_updates(store: ClosedMemoryStore, updates: Dict[str, str]) -> None:
    for memory_id, action in updates.items():
        if memory_id not in store.entries:
            continue
        if action == "preserve":
            store.update_license(memory_id, "usable")
        elif action == "downgrade_license":
            store.update_license(memory_id, "requires_qualification")
        elif action == "attach_scope":
            entry = store.get(memory_id)
            entry.metadata["scope_attached"] = True
            store.update_license(memory_id, "context_only")
        elif action == "tombstone":
            entry = store.get(memory_id)
            entry.metadata["tombstoned"] = True
            store.update_license(memory_id, "blocked")


def resolution_record(loop: OpenLoop, verdict: str) -> MemoryEntry:
    return MemoryEntry(
        memory_id=f"resolution-{loop.loop_id}",
        text=f"Loop {loop.loop_id} resolved with verdict {verdict}: {loop.proposition}",
        tags=[loop.loop_type, "resolution"],
        source=loop.provenance.get("session_id", "unknown"),
        factual_confidence=0.9,
        license_status="usable",
        metadata={"loop_id": loop.loop_id, "verdict": verdict},
    )
