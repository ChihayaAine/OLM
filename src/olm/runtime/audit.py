from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from ..core.types import ActivatedDecision, OpenLoop, StepTrace


@dataclass
class LoopAuditRecord:
    loop_id: str
    loop_type: str
    state: str
    domain: str
    risk_score: float
    unresolvedness: float
    age: int
    conditioned_memory_count: int
    evidence_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def audit_open_loops(loops: Iterable[OpenLoop]) -> List[LoopAuditRecord]:
    records: List[LoopAuditRecord] = []
    for loop in loops:
        records.append(
            LoopAuditRecord(
                loop_id=loop.loop_id,
                loop_type=loop.loop_type,
                state=loop.state,
                domain=str(loop.scope.get("domain", "general")),
                risk_score=float(loop.risk_score),
                unresolvedness=float(loop.unresolvedness),
                age=int(loop.age),
                conditioned_memory_count=len(loop.conditioned_memory_ids),
                evidence_count=len(loop.evidence),
            )
        )
    records.sort(key=lambda item: (item.state, -item.risk_score, -item.unresolvedness, item.loop_id))
    return records


def build_runtime_audit(trace: StepTrace, loop_records: Sequence[LoopAuditRecord]) -> Dict[str, object]:
    active_domains: Dict[str, int] = {}
    for record in loop_records:
        active_domains[record.domain] = active_domains.get(record.domain, 0) + 1
    highest_priority = 0.0
    if trace.activated_records:
        highest_priority = max(record.priority for record in trace.activated_records)
    return {
        "session_id": trace.session_id,
        "activated_loop_count": len(trace.activated_loops),
        "blocked_action_count": len(trace.blocked_actions),
        "highest_priority": highest_priority,
        "active_domains": active_domains,
        "loop_records": [record.to_dict() for record in loop_records],
    }


def summarize_activation_pressure(records: Sequence[ActivatedDecision]) -> Dict[str, object]:
    if not records:
        return {
            "count": 0,
            "avg_priority": 0.0,
            "max_priority": 0.0,
            "relation_mix": {},
        }
    relation_mix: Dict[str, int] = {}
    for record in records:
        relation_mix[record.relation_type] = relation_mix.get(record.relation_type, 0) + 1
    priorities = [record.priority for record in records]
    return {
        "count": len(records),
        "avg_priority": sum(priorities) / len(priorities),
        "max_priority": max(priorities),
        "relation_mix": relation_mix,
    }
