from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .components import OLMComponents, build_mock_components, build_openai_components
from .gating import GateController
from .memory import apply_resolution_updates, resolution_record
from .store import ClosedMemoryStore, OpenLoopStore
from .types import DecisionObject, EvidenceItem, RequiresClosedResult, Session, StepTrace


@dataclass
class OLMConfig:
    retrieval_limit: int = 5
    activation_threshold: float = 0.45
    stale_after: int = 3


class OLMRuntime:
    def __init__(self, config: Optional[OLMConfig] = None, components: Optional[OLMComponents] = None) -> None:
        self.config = config or OLMConfig()
        self.components = components or build_mock_components(self.config.activation_threshold)
        self.closed_store = ClosedMemoryStore()
        self.open_store = OpenLoopStore()
        self.gate_controller = GateController()

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
        new_loops = self.components.extractor.extract(session, retrieved)
        for loop in new_loops:
            self.open_store.add(loop)
        self.open_store.merge_duplicates()

        actions, claims, memory_decisions = self._decision_objects(session, retrieved)
        decisions = actions + claims + memory_decisions
        candidate_pairs = self.components.judge.prefilter(decisions, self.open_store.open_loops())

        activated: Dict[str, RequiresClosedResult] = {}
        activated_loop_ids: List[str] = []
        for decision, loop in candidate_pairs:
            result = self.components.judge.evaluate(decision, loop)
            if result.requires_closed:
                key = f"{loop.loop_id}::{decision.object_id}"
                activated[key] = result
                if loop.loop_id not in activated_loop_ids:
                    activated_loop_ids.append(loop.loop_id)

        licenses = self.gate_controller.apply_memory_licenses(memory_decisions, activated)
        for memory_id, status in licenses.items():
            self.closed_store.update_license(memory_id, status)

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
                self._execute_verification(gate_state["verification_action"])
                final_state = "verify_then_allow"
                break
            if gate_state["decision"] == "hedge_or_defer":
                final_state = "hedge_or_defer"
                break
            final_state = "allow"
            break

        self._close_from_session_outcome(session, activated_loop_ids)
        self.open_store.expire_stale(self.config.stale_after)
        return StepTrace(
            session_id=session.session_id,
            query=session.query,
            retrieved_memories=[entry.memory_id for entry in retrieved],
            activated_loops=activated_loop_ids,
            blocked_actions=blocked_actions,
            chosen_action=chosen.text,
            final_action_state=final_state,
            notes={"licenses": licenses},
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

    def _execute_verification(self, verification_action: Optional[DecisionObject]) -> None:
        if verification_action is None:
            return
        for loop_id in verification_action.metadata.get("targets", []):
            loop = self.open_store.get(loop_id)
            resolution = self.components.closer.apply(
                loop,
                EvidenceItem(
                    content=verification_action.text,
                    polarity="support",
                    source=verification_action.object_id,
                    supports_closure=True,
                ),
            )
            apply_resolution_updates(self.closed_store, resolution.memory_updates)
            self.closed_store.add(resolution_record(loop, resolution.verdict))

    def _close_from_session_outcome(self, session: Session, loop_ids: List[str]) -> None:
        if session.metadata.get("scenario") == "clean":
            return
        for loop_id in loop_ids:
            loop = self.open_store.get(loop_id)
            if loop.state not in {"open", "investigating", "stale"}:
                continue
            if session.metadata.get("query_type") == "memory_reuse":
                evidence = EvidenceItem(
                    content="Scope or evidence still unresolved after memory reuse attempt.",
                    polarity="support",
                    source=session.session_id,
                    supports_closure=False,
                )
            else:
                evidence = EvidenceItem(
                    content="Decision remained unresolved at end of step.",
                    polarity="support",
                    source=session.session_id,
                    supports_closure=False,
                )
            self.components.closer.apply(loop, evidence)

    def summary(self) -> Dict[str, object]:
        return {
            "closed_memories": [asdict(entry) for entry in self.closed_store.all()],
            "open_loops": [asdict(loop) for loop in self.open_store.all()],
        }


def build_openai_runtime() -> OLMRuntime:
    config = OLMConfig()
    runtime = OLMRuntime(config=config, components=build_openai_components(config.activation_threshold))
    return runtime


def build_mock_runtime() -> OLMRuntime:
    config = OLMConfig()
    return OLMRuntime(config=config, components=build_mock_components(config.activation_threshold))
