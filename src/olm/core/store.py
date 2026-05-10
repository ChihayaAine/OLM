from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

from .types import MemoryEntry, OpenLoop


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


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

    def attach_scope(self, memory_id: str, scope: Dict[str, object]) -> None:
        if memory_id in self.entries:
            self.entries[memory_id].metadata.setdefault("scopes", []).append(scope)

    def tombstone(self, memory_id: str, reason: str) -> None:
        if memory_id in self.entries:
            self.entries[memory_id].license_status = "blocked"
            self.entries[memory_id].metadata["tombstone_reason"] = reason

    def all(self) -> List[MemoryEntry]:
        return list(self.entries.values())

    def license_histogram(self) -> Dict[str, int]:
        histogram: Dict[str, int] = {}
        for entry in self.entries.values():
            histogram[entry.license_status] = histogram.get(entry.license_status, 0) + 1
        return histogram

    def scoped_entries(self, domain: str) -> List[MemoryEntry]:
        domain_lower = domain.lower()
        matches: List[MemoryEntry] = []
        for entry in self.entries.values():
            scopes = entry.metadata.get("scopes", [])
            entry_scope = str(entry.metadata.get("scope", entry.metadata.get("domain", ""))).lower()
            if entry_scope == domain_lower:
                matches.append(entry)
                continue
            if any(str(scope.get("domain", "")).lower() == domain_lower for scope in scopes if isinstance(scope, dict)):
                matches.append(entry)
        return matches

    def retrieve(self, query: str, limit: int = 5) -> List[MemoryEntry]:
        query_terms = set(_tokenize(query))
        scored: List[Tuple[Tuple[float, float, float, float], MemoryEntry]] = []
        for entry in self.entries.values():
            terms = set(_tokenize(entry.text)) | set(_tokenize(" ".join(entry.tags)))
            lexical_overlap = len(query_terms & terms)
            query_lower = query.lower()
            tag_match = sum(1 for tag in entry.tags if tag.lower() in query_lower)
            entry_scope = str(entry.metadata.get("domain", entry.metadata.get("scope", ""))).lower()
            scope_bonus = 1.0 if entry_scope and entry_scope in query_lower else 0.0
            recency_bonus = 0.0
            if entry.metadata.get("kind") == "decision":
                recency_bonus += 0.15
            if entry.license_status == "usable":
                recency_bonus += 0.10
            elif entry.license_status == "requires_qualification":
                recency_bonus -= 0.05
            elif entry.license_status in {"context_only", "blocked"}:
                recency_bonus -= 0.15
            score = (
                float(lexical_overlap + tag_match) + scope_bonus,
                float(entry.factual_confidence),
                recency_bonus,
                float(len(entry.metadata.get("scopes", []))),
            )
            if lexical_overlap > 0 or tag_match > 0 or scope_bonus > 0 or entry.license_status == "usable":
                scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
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

    def active_by_domain(self) -> Dict[str, List[OpenLoop]]:
        grouped: Dict[str, List[OpenLoop]] = {}
        for loop in self.open_loops():
            domain = str(loop.scope.get("domain", "general"))
            grouped.setdefault(domain, []).append(loop)
        return grouped

    def increment_age(self) -> None:
        for loop in self.loops.values():
            loop.age += 1

    def merge_duplicates(self) -> None:
        seen = {}
        to_delete = []
        for loop in self.loops.values():
            key = (loop.loop_type, loop.proposition.lower(), loop.scope.get("domain", "general"))
            if key in seen:
                survivor = seen[key]
                survivor.evidence.extend(loop.evidence)
                survivor.triggers.extend(trigger for trigger in loop.triggers if trigger not in survivor.triggers)
                survivor.conditioned_memory_ids = sorted(set(survivor.conditioned_memory_ids + loop.conditioned_memory_ids))
                survivor.risk_score = max(survivor.risk_score, loop.risk_score)
                survivor.unresolvedness = max(survivor.unresolvedness, loop.unresolvedness)
                survivor.age = min(survivor.age, loop.age)
                to_delete.append(loop.loop_id)
            else:
                seen[key] = loop
        for loop_id in to_delete:
            del self.loops[loop_id]

    def expire_stale(self, max_age: int) -> None:
        for loop in self.loops.values():
            if loop.state in {"closed_confirmed", "closed_refuted", "superseded"}:
                continue
            if loop.state == "stale" and loop.unresolvedness <= 0.2:
                loop.state = "expired"
            elif loop.age >= max_age:
                loop.state = "expired"
            elif loop.age >= max(1, max_age - 1):
                loop.state = "stale"

    def stale_candidates(self, min_age: int) -> List[OpenLoop]:
        return [loop for loop in self.loops.values() if loop.age >= min_age and loop.state in {"open", "investigating", "stale"}]

    def summary(self) -> Dict[str, object]:
        states: Dict[str, int] = {}
        for loop in self.loops.values():
            states[loop.state] = states.get(loop.state, 0) + 1
        return {
            "total": len(self.loops),
            "open": len(self.open_loops()),
            "states": states,
            "domains": {domain: len(items) for domain, items in self.active_by_domain().items()},
        }
