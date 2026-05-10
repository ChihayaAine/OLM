from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

from pydantic import BaseModel

from .types import OpenLoop


class VerificationPlan(BaseModel):
    loop_id: str
    modality: str
    recommended_step: str
    urgency: str
    justification: str

    def to_dict(self) -> Dict[str, object]:
        return self.model_dump()


def plan_verification_steps(loops: Iterable[OpenLoop]) -> List[VerificationPlan]:
    plans: List[VerificationPlan] = []
    for loop in loops:
        modality = str(loop.closure_predicate.get("modality", "tool-verifiable"))
        if modality == "tool-verifiable":
            step = "run_targeted_check"
        elif modality == "external-query-verifiable":
            step = "request_external_confirmation"
        elif modality == "evidence-threshold":
            step = "collect_additional_evidence"
        else:
            step = "monitor_future_consistency"
        urgency = "high" if loop.risk_score >= 0.8 or any(trigger.get("irreversible") for trigger in loop.triggers) else "medium"
        plans.append(
            VerificationPlan(
                loop_id=loop.loop_id,
                modality=modality,
                recommended_step=step,
                urgency=urgency,
                justification=f"{loop.loop_type} remains unresolved in domain {loop.scope.get('domain', 'general')}.",
            )
        )
    plans.sort(key=lambda item: (item.urgency != "high", item.loop_id))
    return plans


def verification_backlog(plans: Sequence[VerificationPlan]) -> Dict[str, object]:
    return {
        "total": len(plans),
        "high_urgency": sum(1 for plan in plans if plan.urgency == "high"),
        "modalities": {
            modality: sum(1 for plan in plans if plan.modality == modality)
            for modality in sorted({plan.modality for plan in plans})
        },
        "plans": [plan.to_dict() for plan in plans],
    }
