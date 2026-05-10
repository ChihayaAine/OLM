from __future__ import annotations

from typing import Dict, Iterable, List

from .types import DecisionObject, GateDecision, LicenseTransition, OpenLoop, RequiresClosedResult


LICENSE_ORDER = {
    "usable": 0,
    "requires_qualification": 1,
    "context_only": 2,
    "blocked": 3,
}


class GateController:
    def memory_license_diagnostics(
        self,
        retrieved_memories: Iterable[DecisionObject],
        activated: Dict[str, RequiresClosedResult],
    ) -> Dict[str, Dict[str, object]]:
        diagnostics: Dict[str, Dict[str, object]] = {}
        for decision in retrieved_memories:
            memory_id = decision.metadata["memory_id"]
            reasons: List[Dict[str, object]] = []
            for key, result in activated.items():
                loop_id, object_id = key.split("::", 1)
                if object_id != decision.object_id or result.relation_type != "mem_license":
                    continue
                reasons.append(
                    {
                        "loop_id": loop_id,
                        "confidence": result.confidence,
                        "requires_closed": result.requires_closed,
                        "rationale": result.rationale,
                    }
                )
            diagnostics[memory_id] = {
                "candidate_count": len(reasons),
                "reasons": sorted(reasons, key=lambda item: item["confidence"], reverse=True),
            }
        return diagnostics

    def apply_memory_licenses(
        self,
        retrieved_memories: Iterable[DecisionObject],
        activated: Dict[str, RequiresClosedResult],
    ) -> Dict[str, str]:
        licenses: Dict[str, str] = {}
        for decision in retrieved_memories:
            memory_id = decision.metadata["memory_id"]
            current = "usable"
            for key, result in activated.items():
                loop_id, object_id = key.split("::", 1)
                if object_id != decision.object_id or not result.requires_closed:
                    continue
                if result.relation_type != "mem_license":
                    continue
                proposed = "requires_qualification"
                if decision.metadata.get("stakes") == "high":
                    proposed = "blocked"
                elif decision.metadata.get("scope_mismatch"):
                    proposed = "context_only"
                elif result.confidence >= 0.8:
                    proposed = "context_only"
                if LICENSE_ORDER[proposed] > LICENSE_ORDER[current]:
                    current = proposed
            licenses[memory_id] = current
        return licenses

    def license_transitions(
        self,
        previous_statuses: Dict[str, str],
        new_statuses: Dict[str, str],
    ) -> List[LicenseTransition]:
        transitions: List[LicenseTransition] = []
        for memory_id, new_status in new_statuses.items():
            previous = previous_statuses.get(memory_id, "usable")
            if previous == new_status:
                continue
            transitions.append(
                LicenseTransition(
                    memory_id=memory_id,
                    previous_status=previous,
                    new_status=new_status,
                    reason=f"license changed from {previous} to {new_status}",
                )
            )
        return transitions

    def gate_action(
        self,
        action: DecisionObject,
        loop: OpenLoop,
        result: RequiresClosedResult,
    ) -> GateDecision:
        if not result.requires_closed:
            return GateDecision(decision="allow", reason="loop not activated")
        low_cost = action.metadata.get("verification_cost", 10) <= action.metadata.get("verify_budget", 4)
        reversible = action.metadata.get("reversible", False)
        if result.relation_type == "act_precond":
            if result.sufficient_evidence and low_cost:
                verify = DecisionObject(
                    object_id=f"verify-{action.object_id}-{loop.loop_id}",
                    object_type="verification",
                    text=result.sufficient_evidence[0],
                    metadata={"targets": [loop.loop_id], "cost": action.metadata.get("verification_cost", 3)},
                )
                return GateDecision(
                    decision="verify",
                    reason=result.rationale or "closure required before action",
                    verification_action=verify,
                )
            if reversible:
                return GateDecision(decision="hedge_or_defer", reason=result.rationale or "reversible action can be deferred")
            return GateDecision(decision="block", reason=result.rationale or "irreversible action blocked by unresolved loop")
        if result.relation_type == "claim_dep":
            if result.sufficient_evidence and low_cost:
                verify = DecisionObject(
                    object_id=f"verify-{action.object_id}-{loop.loop_id}",
                    object_type="verification",
                    text=result.sufficient_evidence[0],
                    metadata={"targets": [loop.loop_id], "cost": action.metadata.get("verification_cost", 3)},
                )
                return GateDecision(
                    decision="verify",
                    reason=result.rationale or "claim requires stronger evidence",
                    verification_action=verify,
                )
            return GateDecision(decision="hedge_or_defer", reason=result.rationale or "claim must be qualified until closure")
        return GateDecision(decision="allow", reason="license-only relation")
