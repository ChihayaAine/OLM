from __future__ import annotations

from typing import Dict, Iterable, List

from .types import MemoryEntry, OpenLoop


class ClosedMemoryStore:
    def __init__(self) -> None:
        self.entries: Dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        self.entries[entry.memory_id] = entry

    def get(self, memory_id: str) -> MemoryEntry:
        return self.entries[memory_id]

    def update_license(self, memory_id: str, status: str) -> None:
        if memory_id in self.entries:
            self.entries[memory_id].license_status = status

    def all(self) -> List[MemoryEntry]:
        return list(self.entries.values())

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        query_terms = set(query.lower().split())
        scored = []
        for entry in self.entries.values():
            terms = set(entry.text.lower().split()) | set(tag.lower() for tag in entry.tags)
            score = len(query_terms & terms)
            if score > 0 or any(tag in query.lower() for tag in entry.tags):
                scored.append((score, entry))
        scored.sort(key=lambda item: (item[0], item[1].factual_confidence), reverse=True)
        if scored:
            return [entry for _, entry in scored[:limit]]
        return self.all()[:limit]


class OpenLoopStore:
    def __init__(self) -> None:
        self.loops: Dict[str, OpenLoop] = {}

    def add(self, loop: OpenLoop) -> None:
        self.loops[loop.loop_id] = loop

    def get(self, loop_id: str) -> OpenLoop:
        return self.loops[loop_id]

    def all(self) -> List[OpenLoop]:
        return list(self.loops.values())

    def open_loops(self) -> List[OpenLoop]:
        return [loop for loop in self.loops.values() if loop.state in {"open", "investigating", "stale"}]

    def increment_age(self) -> None:
        for loop in self.loops.values():
            loop.age += 1

    def merge_duplicates(self) -> None:
        seen = {}
        to_delete = []
        for loop in self.loops.values():
            key = (loop.loop_type, loop.proposition.lower())
            if key in seen:
                seen[key].evidence.extend(loop.evidence)
                to_delete.append(loop.loop_id)
            else:
                seen[key] = loop
        for loop_id in to_delete:
            del self.loops[loop_id]

    def expire_stale(self, max_age: int) -> None:
        for loop in self.loops.values():
            if loop.state in {"closed_confirmed", "closed_refuted", "superseded"}:
                continue
            if loop.age >= max_age:
                loop.state = "expired"
            elif loop.age >= max(1, max_age - 1):
                loop.state = "stale"
