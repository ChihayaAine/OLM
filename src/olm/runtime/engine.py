from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ..core.gating import GateController
from ..core.lineage import build_memory_lineage, lineage_transition_report
from ..core.licensing import license_revision_summary, revise_licenses_after_resolution
from ..core.planning import plan_verification_steps, verification_backlog
from ..core.retrieval import build_query_profile, rank_memory_candidates, retrieval_audit
from ..core.store import ClosedMemoryStore, OpenLoopStore
from ..core.types import DecisionObject, RequiresClosedResult, Session, StepTrace
from ..operators.components import OLMComponents, build_mock_components, build_provider_components
from .audit import audit_open_loops, build_runtime_audit, summarize_activation_pressure
from .activation import ActivationConfig, activation_summary, select_activations
from .closure import (
    close_loop_with_evidence,
    consolidate_expired_loops,
    loop_state_summary,
    unresolved_outcome_evidence,
    verification_evidence,
)
from .session_ledger import build_session_ledger, ledger_summary


@dataclass
class OLMConfig:
    retrieval_limit: int = 5
    activation_threshold: float = 0.45
    stale_after: int = 3
    max_active_loops: int = 6


class OLMRuntime:
    def __init__(self, config: Optional[OLMConfig] = None, components: Optional[OLMComponents] = None) -> None:
        self.config = config or OLMConfig()
        self.components = components or build_mock_components(self.config.activation_threshold)
        self.closed_store = ClosedMemoryStore()
        self.open_store = OpenLoopStore()
        self.gate_controller = GateController()
        self.activation_config = ActivationConfig(
            activation_threshold=self.config.activation_threshold,
            max_active_loops=self.config.max_active_loops,
        )

    def bootstrap(self, memories: Iterable) -> None:
        for memory in memories:
            self.closed_store.add(memory)

    def run_session(self, session: Session) -> StepTrace:
        self.open_store.increment_age()
        retrieved = self.components.retriever.retrieve(
            self.closed_store,
            session.query,
            limit=self.config.retrieval_limit,
        )
        retrieval_profile = build_query_profile(session.query, session.metadata)
        retrieval_rankings = rank_memory_candidates(retrieved, retrieval_profile)
        new_loops = self.components.extractor.extract(session, retrieved)
        for loop in new_loops:
            self.open_store.add(loop)
        self.open_store.merge_duplicates()

        actions, claims, memory_decisions = self._decision_objects(session, retrieved)
        decisions = actions + claims + memory_decisions
        candidate_pairs = self.components.judge.prefilter(decisions, self.open_store.open_loops())
        verification_plans = plan_verification_steps(self.open_store.open_loops())

        activated, activated_records, activated_loop_ids = select_activations(
            candidate_pairs=candidate_pairs,
            evaluator=self.components.judge.evaluate,
            config=self.activation_config,
        )

        previous_licenses = {entry.memory_id: entry.license_status for entry in retrieved}
        licenses = self.gate_controller.apply_memory_licenses(memory_decisions, activated)
        license_diagnostics = self.gate_controller.memory_license_diagnostics(memory_decisions, activated)
        for memory_id, status in licenses.items():
            self.closed_store.update_license(memory_id, status)
        license_transitions = self.gate_controller.license_transitions(previous_licenses, licenses)
        license_revisions = revise_licenses_after_resolution(retrieved, license_transitions)
        lineage = build_memory_lineage(self.closed_store.all(), self.open_store.all())

        blocked_actions: List[str] = []
        final_state = "allow"
        chosen = self.components.policy.select_action(session, actions)
        for action in [chosen]:
            gate_state = self._gate_action(action, activated, activated_loop_ids)
            if gate_state["decision"] == "block":
                blocked_actions.append(action.object_id)
                final_state = "block"
                continue
            if gate_state["decision"] == "verify":
                verification_notes = self._execute_verification(gate_state["verification_action"])
                final_state = "verify_then_allow"
                break
            if gate_state["decision"] == "hedge_or_defer":
                final_state = "hedge_or_defer"
                break
            final_state = "allow"
            break

        closure_notes = self._close_from_session_outcome(session, activated_loop_ids)
        self.open_store.expire_stale(self.config.stale_after)
        archived_loops = consolidate_expired_loops(self.open_store, self.closed_store)
        loop_records = audit_open_loops(self.open_store.all())
        active_loops = [self.open_store.get(loop_id) for loop_id in activated_loop_ids if loop_id in self.open_store.loops]
        ledger = build_session_ledger(
            session_id=session.session_id,
            retrieved_memory_ids=[entry.memory_id for entry in retrieved],
            actions=actions,
            claims=claims,
            active_loops=active_loops,
            activated_records=activated_records,
        )
        return StepTrace(
            session_id=session.session_id,
            query=session.query,
            retrieved_memories=[entry.memory_id for entry in retrieved],
            activated_loops=activated_loop_ids,
            blocked_actions=blocked_actions,
            chosen_action=chosen.text,
            final_action_state=final_state,
            license_transitions=license_transitions,
            activated_records=activated_records,
            notes={
                "licenses": licenses,
                "license_diagnostics": license_diagnostics,
                "license_revisions": license_revision_summary(license_revisions),
                "memory_lineage": lineage_transition_report(lineage, license_transitions),
                "activation": activation_summary(activated_records),
                "activation_pressure": summarize_activation_pressure(activated_records),
                "closure_notes": closure_notes,
                "archived_loops": archived_loops,
                "session_ledger": ledger_summary(ledger),
                "verification_notes": verification_notes if final_state == "verify_then_allow" else [],
                "verification_backlog": verification_backlog(verification_plans),
                "closed_memory_histogram": self.closed_store.license_histogram(),
                "retrieval_profile": retrieval_profile.to_dict(),
                "retrieval_audit": retrieval_audit(retrieval_rankings, self.config.retrieval_limit),
                "runtime_audit": build_runtime_audit(
                    StepTrace(
                        session_id=session.session_id,
                        query=session.query,
                        retrieved_memories=[entry.memory_id for entry in retrieved],
                        activated_loops=activated_loop_ids,
                        blocked_actions=blocked_actions,
                        chosen_action=chosen.text,
                        final_action_state=final_state,
                    ),
                    loop_records,
                ),
            },
            loop_summary=loop_state_summary(self.open_store),
        )

    def _decision_objects(self, session: Session, retrieved) -> Tuple[List[DecisionObject], List[DecisionObject], List[DecisionObject]]:
        actions = self.components.policy.propose_actions(session, retrieved)
        claims = self.components.policy.draft_claims(session, retrieved)
        memories = [
            DecisionObject(
                object_id=f"memory-{entry.memory_id}",
                object_type="memory",
                text=entry.text,
                metadata={
                    "memory_id": entry.memory_id,
                    "stakes": session.metadata.get("stakes", "medium"),
                    "scope_mismatch": session.metadata.get("scope_mismatch", False),
                },
            )
            for entry in retrieved
        ]
        return actions, claims, memories

    def _gate_action(self, action: DecisionObject, activated: Dict[str, RequiresClosedResult], loop_ids: List[str]) -> Dict[str, object]:
        for loop_id in loop_ids:
            action_key = f"{loop_id}::{action.object_id}"
            if action_key not in activated:
                continue
            gate = self.gate_controller.gate_action(
                action,
                self.open_store.get(loop_id),
                activated[action_key],
            )
            return {
                "decision": gate.decision,
                "reason": gate.reason,
                "verification_action": gate.verification_action,
            }
        return {"decision": "allow", "reason": "no active blocking loop", "verification_action": None}

    def _execute_verification(self, verification_action: Optional[DecisionObject]) -> List[Dict[str, object]]:
        if verification_action is None:
            return []
        notes: List[Dict[str, object]] = []
        for loop_id in verification_action.metadata.get("targets", []):
            loop = self.open_store.get(loop_id)
            notes.append(
                close_loop_with_evidence(
                    self.components.closer,
                    loop,
                    verification_evidence(loop, verification_action.text, verification_action.object_id),
                    self.closed_store,
                )
            )
        return notes

    def _close_from_session_outcome(self, session: Session, loop_ids: List[str]) -> List[Dict[str, object]]:
        if session.metadata.get("scenario") == "clean":
            return []
        notes: List[Dict[str, object]] = []
        for loop_id in loop_ids:
            loop = self.open_store.get(loop_id)
            if loop.state not in {"open", "investigating", "stale"}:
                continue
            notes.append(
                close_loop_with_evidence(
                    self.components.closer,
                    loop,
                    unresolved_outcome_evidence(session.session_id, session.metadata.get("query_type", "general")),
                    self.closed_store,
                )
            )
        return notes

    def summary(self) -> Dict[str, object]:
        return {
            "closed_memories": [asdict(entry) for entry in self.closed_store.all()],
            "open_loops": [asdict(loop) for loop in self.open_store.all()],
            "memory_lineage": [item.to_dict() for item in build_memory_lineage(self.closed_store.all(), self.open_store.all())],
        }


def build_provider_runtime() -> OLMRuntime:
    config = OLMConfig()
    runtime = OLMRuntime(config=config, components=build_provider_components(config.activation_threshold))
    return runtime


def build_mock_runtime() -> OLMRuntime:
    config = OLMConfig()
    return OLMRuntime(config=config, components=build_mock_components(config.activation_threshold))
