from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from .types import LicenseTransition, MemoryEntry


@dataclass
class LicenseRevision:
    memory_id: str
    prior_status: str
    revised_status: str
    trigger: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def revise_licenses_after_resolution(
    memories: Iterable[MemoryEntry],
    transitions: Iterable[LicenseTransition],
) -> List[LicenseRevision]:
    transition_map = {transition.memory_id: transition for transition in transitions}
    revisions: List[LicenseRevision] = []
    for memory in memories:
        transition = transition_map.get(memory.memory_id)
        if transition is None:
            continue
        revised_status = transition.new_status
        if transition.new_status == "requires_qualification" and memory.factual_confidence < 0.6:
            revised_status = "context_only"
        revisions.append(
            LicenseRevision(
                memory_id=memory.memory_id,
                prior_status=transition.previous_status,
                revised_status=revised_status,
                trigger=transition.reason,
            )
        )
    return revisions


def license_revision_summary(revisions: Iterable[LicenseRevision]) -> Dict[str, object]:
    revision_list = list(revisions)
    by_status: Dict[str, int] = {}
    for revision in revision_list:
        by_status[revision.revised_status] = by_status.get(revision.revised_status, 0) + 1
    return {
        "count": len(revision_list),
        "by_status": by_status,
        "revisions": [revision.to_dict() for revision in revision_list],
    }
