from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from ..core.types import (
    DecisionObject,
    EvidenceItem,
    LoopResolution,
    MemoryEntry,
    OpenLoop,
    RequiresClosedResult,
    Session,
)


LOOP_TYPES = {
    "assumption",
    "evidence_gap",
    "contradiction",
    "deferred_commitment",
    "scope_limit",
    "partial_verification",
}


def high_stake(trigger: Dict[str, str]) -> bool:
    return trigger.get("stakes", "medium") == "high" or trigger.get("irreversible", False)


class OpenOperator:
    def __init__(self) -> None:
        self.counter = 0

    def extract(self, session: Session, closed_memories: Iterable[MemoryEntry]) -> List[OpenLoop]:
        loops: List[OpenLoop] = []
        metadata = session.metadata
        memory_ids = [entry.memory_id for entry in closed_memories]

        mapping = [
            ("has_assumption", "assumption", "Assumption in prior conclusion remains unverified."),
            ("has_evidence_gap", "evidence_gap", "Evidence was insufficient for the reused conclusion."),
            ("has_contradiction", "contradiction", "Conflicting evidence exists and is unresolved."),
            ("has_deferred_commitment", "deferred_commitment", "Deferred decision was later treated as final."),
            ("has_scope_limit", "scope_limit", "Prior memory may not apply outside its original scope."),
            ("has_partial_verification", "partial_verification", "Verification was only partial for the stored result."),
        ]
        for flag, loop_type, proposition in mapping:
            if not metadata.get(flag):
                continue
            if not self._admissible(loop_type, metadata):
                continue
            loops.append(
                OpenLoop(
                    loop_id=f"loop-{self.counter}",
                    loop_type=loop_type,
                    proposition=proposition,
                    evidence=[
                        EvidenceItem(
                            content=session.query,
                            polarity="support",
                            source=session.session_id,
                            supports_closure=False,
                            metadata={"query_type": metadata.get("query_type", "general")},
                        )
                    ],
                    triggers=self._triggers_for(loop_type, metadata),
                    closure_predicate=self._closure_predicate(loop_type),
                    gate=self._gate_for(loop_type),
                    risk_score=float(metadata.get("risk_score", 0.7)),
                    scope={"domain": metadata.get("domain", "general")},
                    conditioned_memory_ids=memory_ids,
                    provenance={"session_id": session.session_id},
                    state="open",
                    unresolvedness=self._unresolvedness(loop_type, metadata),
                )
            )
            self.counter += 1
        return loops

    def _admissible(self, loop_type: str, metadata: Dict[str, object]) -> bool:
        risk_score = float(metadata.get("risk_score", 0.0))
        triggers = self._triggers_for(loop_type, metadata)
        return bool(triggers) and (risk_score > 0.25 or any(high_stake(trigger) for trigger in triggers))

    def _triggers_for(self, loop_type: str, metadata: Dict[str, object]) -> List[Dict[str, object]]:
        if loop_type in {"partial_verification", "evidence_gap"}:
            return [{"relation": "claim_dep", "stakes": metadata.get("stakes", "medium"), "irreversible": False}]
        if loop_type == "deferred_commitment":
            return [{"relation": "act_precond", "stakes": metadata.get("stakes", "high"), "irreversible": True}]
        if loop_type == "scope_limit":
            return [{"relation": "mem_license", "stakes": metadata.get("stakes", "medium"), "irreversible": False}]
        if loop_type == "contradiction":
            return [{"relation": "claim_dep", "stakes": "high", "irreversible": False}]
        return [{"relation": "claim_dep", "stakes": metadata.get("stakes", "medium"), "irreversible": False}]

    def _closure_predicate(self, loop_type: str) -> Dict[str, str]:
        modalities = {
            "assumption": "external-query-verifiable",
            "evidence_gap": "evidence-threshold",
            "contradiction": "tool-verifiable",
            "deferred_commitment": "external-query-verifiable",
            "scope_limit": "monitorable",
            "partial_verification": "tool-verifiable",
        }
        return {"modality": modalities[loop_type], "status": "insufficient"}

    def _gate_for(self, loop_type: str) -> Dict[str, str]:
        mapping = {
            "assumption": {"type": "epistemic"},
            "evidence_gap": {"type": "epistemic"},
            "contradiction": {"type": "epistemic"},
            "deferred_commitment": {"type": "action"},
            "scope_limit": {"type": "license"},
            "partial_verification": {"type": "action"},
        }
        return mapping[loop_type]

    def _unresolvedness(self, loop_type: str, metadata: Dict[str, object]) -> float:
        base = {
            "assumption": 0.7,
            "evidence_gap": 0.8,
            "contradiction": 0.9,
            "deferred_commitment": 0.85,
            "scope_limit": 0.65,
            "partial_verification": 0.95,
        }[loop_type]
        return min(1.0, base + 0.1 * float(metadata.get("risk_score", 0.5)))


class RequiresClosedOperator:
    def __init__(self, activation_threshold: float = 0.45) -> None:
        self.activation_threshold = activation_threshold

    def prefilter(self, decisions: Iterable[DecisionObject], loops: Iterable[OpenLoop]) -> List[Tuple[DecisionObject, OpenLoop]]:
        filtered = []
        for decision in decisions:
            decision_terms = set(decision.text.lower().split())
            for loop in loops:
                loop_terms = set(loop.proposition.lower().split()) | {loop.loop_type.replace("_", " ")}
                if decision.object_type == "memory":
                    if decision.metadata.get("memory_id") in loop.conditioned_memory_ids:
                        filtered.append((decision, loop))
                elif decision_terms & loop_terms or decision.metadata.get("stakes") == "high":
                    filtered.append((decision, loop))
        return filtered

    def evaluate(self, decision: DecisionObject, loop: OpenLoop) -> RequiresClosedResult:
        relation = self._relation_type(decision, loop)
        confidence = self._confidence(decision, loop, relation)
        requires_closed = confidence >= self.activation_threshold
        return RequiresClosedResult(
            requires_closed=requires_closed,
            relation_type=relation,
            confidence=confidence,
            sufficient_evidence=self._sufficient_evidence(loop),
            rationale=self._rationale(decision, loop, relation, requires_closed),
        )

    def _relation_type(self, decision: DecisionObject, loop: OpenLoop) -> str:
        if decision.object_type == "memory":
            return "mem_license"
        if decision.object_type == "action" and loop.loop_type in {"deferred_commitment", "partial_verification"}:
            return "act_precond"
        return "claim_dep"

    def _confidence(self, decision: DecisionObject, loop: OpenLoop, relation: str) -> float:
        score = 0.25 + 0.35 * loop.unresolvedness + 0.2 * loop.risk_score
        if relation == "act_precond":
            score += 0.2
        if relation == "mem_license":
            score += 0.1
        if decision.metadata.get("stakes") == "high":
            score += 0.1
        return min(1.0, score)

    def _sufficient_evidence(self, loop: OpenLoop) -> List[str]:
        return {
            "assumption": ["obtain_external_confirmation"],
            "evidence_gap": ["collect_more_supporting_evidence"],
            "contradiction": ["run_disambiguating_tool_check"],
            "deferred_commitment": ["obtain_explicit_confirmation"],
            "scope_limit": ["confirm_scope_match"],
            "partial_verification": ["run_missing_integration_or_system_check"],
        }[loop.loop_type]

    def _rationale(self, decision: DecisionObject, loop: OpenLoop, relation: str, requires_closed: bool) -> str:
        if not requires_closed:
            return f"{decision.object_type} does not currently depend strongly enough on unresolved {loop.loop_type}."
        if relation == "act_precond":
            return f"Action depends on closing unresolved {loop.loop_type} before execution."
        if relation == "mem_license":
            return f"Memory use is conditioned on unresolved {loop.loop_type} and should be downgraded."
        return f"Claim or reasoning depends on unresolved {loop.loop_type} and should be qualified."


class CloseOperator:
    def apply(self, loop: OpenLoop, new_evidence: EvidenceItem) -> LoopResolution:
        loop.evidence.append(new_evidence)
        verdict = self._verdict(loop)
        state = {
            "confirmed": "closed_confirmed",
            "refuted": "closed_refuted",
            "superseded": "superseded",
            "insufficient": "investigating",
        }[verdict]
        loop.state = state
        loop.closure_predicate["status"] = verdict
        if verdict != "insufficient":
            loop.unresolvedness = 0.0
        else:
            loop.unresolvedness = max(0.15, loop.unresolvedness - 0.15)
        updates = self._memory_updates(loop, verdict)
        return LoopResolution(verdict=verdict, updated_state=state, memory_updates=updates)

    def _verdict(self, loop: OpenLoop) -> str:
        if any(item.metadata.get("supersedes") for item in loop.evidence):
            return "superseded"
        if any(item.supports_closure and item.polarity == "support" for item in loop.evidence):
            return "confirmed"
        if any(item.supports_closure and item.polarity == "refute" for item in loop.evidence):
            return "refuted"
        return "insufficient"

    def _memory_updates(self, loop: OpenLoop, verdict: str) -> Dict[str, str]:
        if verdict == "confirmed":
            return {memory_id: "preserve" for memory_id in loop.conditioned_memory_ids}
        if verdict == "refuted":
            destructive = "tombstone" if loop.loop_type == "contradiction" else "downgrade_license"
            return {memory_id: destructive for memory_id in loop.conditioned_memory_ids}
        if verdict == "superseded":
            return {memory_id: "attach_scope" for memory_id in loop.conditioned_memory_ids}
        return {}
