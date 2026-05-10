from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from ..core.types import ActivatedDecision, DecisionObject, OpenLoop


@dataclass
class SessionLedger:
    session_id: str
    retrieved_memory_ids: List[str]
    action_ids: List[str]
    claim_ids: List[str]
    active_loop_ids: List[str]
    verification_targets: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_session_ledger(
    session_id: str,
    retrieved_memory_ids: Iterable[str],
    actions: Sequence[DecisionObject],
    claims: Sequence[DecisionObject],
    active_loops: Sequence[OpenLoop],
    activated_records: Sequence[ActivatedDecision],
) -> SessionLedger:
    verification_targets: List[str] = []
    for record in activated_records:
        if record.relation_type in {"act_precond", "claim_dep"} and record.loop_id not in verification_targets:
            verification_targets.append(record.loop_id)
    return SessionLedger(
        session_id=session_id,
        retrieved_memory_ids=list(retrieved_memory_ids),
        action_ids=[item.object_id for item in actions],
        claim_ids=[item.object_id for item in claims],
        active_loop_ids=[loop.loop_id for loop in active_loops],
        verification_targets=verification_targets,
    )


def ledger_summary(ledger: SessionLedger) -> Dict[str, object]:
    return {
        "session_id": ledger.session_id,
        "retrieved_count": len(ledger.retrieved_memory_ids),
        "action_count": len(ledger.action_ids),
        "claim_count": len(ledger.claim_ids),
        "active_loop_count": len(ledger.active_loop_ids),
        "verification_target_count": len(ledger.verification_targets),
        "ledger": ledger.to_dict(),
    }
