from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from .types import LicenseTransition, MemoryEntry, OpenLoop


@dataclass
class MemoryLineageRecord:
    memory_id: str
    source: str
    current_license: str
    linked_loops: List[str]
    scope: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_memory_lineage(
    memories: Iterable[MemoryEntry],
    loops: Iterable[OpenLoop],
) -> List[MemoryLineageRecord]:
    loop_map: Dict[str, List[str]] = {}
    for loop in loops:
        for memory_id in loop.conditioned_memory_ids:
            loop_map.setdefault(memory_id, []).append(loop.loop_id)
    lineage: List[MemoryLineageRecord] = []
    for memory in memories:
        lineage.append(
            MemoryLineageRecord(
                memory_id=memory.memory_id,
                source=memory.source,
                current_license=memory.license_status,
                linked_loops=sorted(loop_map.get(memory.memory_id, [])),
                scope=str(memory.metadata.get("scope", memory.metadata.get("domain", "general"))),
            )
        )
    lineage.sort(key=lambda item: (item.current_license, item.memory_id))
    return lineage


def lineage_transition_report(
    lineage: Iterable[MemoryLineageRecord],
    transitions: Iterable[LicenseTransition],
) -> Dict[str, object]:
    lineage_list = list(lineage)
    transition_list = list(transitions)
    touched_ids = {transition.memory_id for transition in transition_list}
    return {
        "memory_count": len(lineage_list),
        "touched_count": len(touched_ids),
        "touched_memories": [record.to_dict() for record in lineage_list if record.memory_id in touched_ids],
        "all_memories": [record.to_dict() for record in lineage_list],
    }
