from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import os
from typing import Dict, List, Optional, Sequence, Tuple

from .openai_client import OpenAIClientConfig, OpenAIClientError, OpenAIResponsesClient
from .operators import CloseOperator, OpenOperator, RequiresClosedOperator
from .store import ClosedMemoryStore
from .types import (
    DecisionObject,
    EvidenceItem,
    LoopResolution,
    MemoryEntry,
    OpenLoop,
    RequiresClosedResult,
    Session,
)


class MemoryRetriever(ABC):
    @abstractmethod
    def retrieve(self, store: ClosedMemoryStore, query: str, limit: int = 5) -> List[MemoryEntry]:
        raise NotImplementedError


class LoopExtractor(ABC):
    @abstractmethod
    def extract(self, session: Session, closed_memories: Sequence[MemoryEntry]) -> List[OpenLoop]:
        raise NotImplementedError


class ClosureRequirementJudge(ABC):
    @abstractmethod
    def prefilter(
        self,
        decisions: Sequence[DecisionObject],
        loops: Sequence[OpenLoop],
    ) -> List[Tuple[DecisionObject, OpenLoop]]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, decision: DecisionObject, loop: OpenLoop) -> RequiresClosedResult:
        raise NotImplementedError


class LoopCloser(ABC):
    @abstractmethod
    def apply(self, loop: OpenLoop, new_evidence: EvidenceItem) -> LoopResolution:
        raise NotImplementedError


class BasePolicy(ABC):
    @abstractmethod
    def propose_actions(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        raise NotImplementedError

    @abstractmethod
    def draft_claims(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        raise NotImplementedError

    @abstractmethod
    def select_action(self, session: Session, candidates: Sequence[DecisionObject]) -> DecisionObject:
        raise NotImplementedError


@dataclass
class OLMComponents:
    retriever: MemoryRetriever
    extractor: LoopExtractor
    judge: ClosureRequirementJudge
    closer: LoopCloser
    policy: BasePolicy


class StoreRetriever(MemoryRetriever):
    def retrieve(self, store: ClosedMemoryStore, query: str, limit: int = 5) -> List[MemoryEntry]:
        return store.retrieve(query, limit=limit)


class MockLoopExtractor(LoopExtractor):
    def __init__(self) -> None:
        self.operator = OpenOperator()

    def extract(self, session: Session, closed_memories: Sequence[MemoryEntry]) -> List[OpenLoop]:
        return self.operator.extract(session, closed_memories)


class MockClosureRequirementJudge(ClosureRequirementJudge):
    def __init__(self, activation_threshold: float) -> None:
        self.operator = RequiresClosedOperator(activation_threshold=activation_threshold)

    def prefilter(
        self,
        decisions: Sequence[DecisionObject],
        loops: Sequence[OpenLoop],
    ) -> List[Tuple[DecisionObject, OpenLoop]]:
        return self.operator.prefilter(decisions, loops)

    def evaluate(self, decision: DecisionObject, loop: OpenLoop) -> RequiresClosedResult:
        return self.operator.evaluate(decision, loop)


class MockLoopCloser(LoopCloser):
    def __init__(self) -> None:
        self.operator = CloseOperator()

    def apply(self, loop: OpenLoop, new_evidence: EvidenceItem) -> LoopResolution:
        return self.operator.apply(loop, new_evidence)


class MockPolicy(BasePolicy):
    def propose_actions(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        return [
            DecisionObject(
                object_id=f"action-{session.session_id}",
                object_type="action",
                text=session.query,
                metadata={
                    "stakes": session.metadata.get("stakes", "medium"),
                    "reversible": session.metadata.get("reversible", False),
                    "verification_cost": session.metadata.get("verification_cost", 3),
                    "verify_budget": session.metadata.get("verify_budget", 3),
                },
            )
        ]

    def draft_claims(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        return [
            DecisionObject(
                object_id=f"claim-{session.session_id}",
                object_type="claim",
                text=session.query,
                metadata={"stakes": session.metadata.get("stakes", "medium")},
            )
        ]

    def select_action(self, session: Session, candidates: Sequence[DecisionObject]) -> DecisionObject:
        return list(candidates)[0]


class APINotConfiguredError(RuntimeError):
    pass


def _memory_brief(memory: MemoryEntry) -> Dict[str, object]:
    return {
        "memory_id": memory.memory_id,
        "text": memory.text,
        "tags": memory.tags,
        "source": memory.source,
        "license_status": memory.license_status,
        "metadata": memory.metadata,
    }


def _loop_brief(loop: OpenLoop) -> Dict[str, object]:
    payload = asdict(loop)
    payload["evidence_count"] = len(loop.evidence)
    return payload


def _decision_brief(decision: DecisionObject) -> Dict[str, object]:
    return asdict(decision)


RETRIEVER_SCHEMA = {
    "type": "object",
    "properties": {
        "memory_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["memory_ids"],
    "additionalProperties": False,
}

OPEN_SCHEMA = {
    "type": "object",
    "properties": {
        "loops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "should_open": {"type": "boolean"},
                    "loop_type": {
                        "type": "string",
                        "enum": [
                            "assumption",
                            "evidence_gap",
                            "contradiction",
                            "deferred_commitment",
                            "scope_limit",
                            "partial_verification",
                        ],
                    },
                    "proposition": {"type": "string"},
                    "trigger_relation": {"type": "string", "enum": ["claim_dep", "act_precond", "mem_license"]},
                "trigger_stakes": {"type": "string", "enum": ["low", "medium", "high"]},
                "irreversible": {"type": "boolean"},
                "closure_modality": {
                    "type": "string",
                    "enum": ["tool-verifiable", "external-query-verifiable", "evidence-threshold", "monitorable"],
                },
                    "gate_type": {"type": "string", "enum": ["epistemic", "action", "license"]},
                    "risk_score": {"type": "number"},
                    "unresolvedness": {"type": "number"},
                    "scope_domain": {"type": "string"},
                    "conditioned_memory_ids": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": [
                    "should_open",
                    "loop_type",
                    "proposition",
                    "trigger_relation",
                    "trigger_stakes",
                    "irreversible",
                    "closure_modality",
                    "gate_type",
                    "risk_score",
                    "unresolvedness",
                    "scope_domain",
                    "conditioned_memory_ids",
                    "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["loops"],
    "additionalProperties": False,
}

REQUIRES_CLOSED_SCHEMA = {
    "type": "object",
    "properties": {
        "requires_closed": {"type": "boolean"},
        "relation_type": {"type": "string", "enum": ["claim_dep", "act_precond", "mem_license"]},
        "confidence": {"type": "number"},
        "sufficient_evidence": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["requires_closed", "relation_type", "confidence", "sufficient_evidence"],
    "additionalProperties": False,
}

CLOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["confirmed", "refuted", "superseded", "insufficient"]},
        "updated_state": {
            "type": "string",
            "enum": [
                "open",
                "investigating",
                "closed_confirmed",
                "closed_refuted",
                "superseded",
                "stale",
                "expired",
            ],
        },
        "memory_updates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["preserve", "downgrade_license", "attach_scope", "tombstone"]},
                },
                "required": ["memory_id", "action"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string"},
    },
    "required": ["verdict", "updated_state", "memory_updates", "note"],
    "additionalProperties": False,
}

POLICY_ACTIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "actions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "stakes": {"type": "string", "enum": ["low", "medium", "high"]},
                    "reversible": {"type": "boolean"},
                    "verification_cost": {"type": "number"},
                    "verify_budget": {"type": "number"},
                },
                "required": ["text", "stakes", "reversible", "verification_cost", "verify_budget"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["actions"],
    "additionalProperties": False,
}

POLICY_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "stakes": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["text", "stakes"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}

POLICY_SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {"type": "integer"},
    },
    "required": ["selected_index"],
    "additionalProperties": False,
}


class OpenAIBackedMixin:
    def __init__(self, client: Optional[OpenAIResponsesClient] = None, model_name: Optional[str] = None) -> None:
        if client is not None:
            self.client = client
        else:
            config = OpenAIClientConfig(model=model_name or "gpt-4o-mini")
            self.client = OpenAIResponsesClient(config)

    def _call_json(self, schema_name: str, schema: Dict[str, object], system_prompt: str, payload: Dict[str, object]) -> Dict[str, object]:
        try:
            return self.client.generate_json(schema_name=schema_name, schema=schema, system_prompt=system_prompt, payload=payload)
        except OpenAIClientError as exc:
            raise APINotConfiguredError(str(exc)) from exc


class OpenAIRetriever(MemoryRetriever, OpenAIBackedMixin):
    def __init__(self, client: Optional[OpenAIResponsesClient] = None, model_name: Optional[str] = None) -> None:
        OpenAIBackedMixin.__init__(self, client=client, model_name=model_name)

    def retrieve(self, store: ClosedMemoryStore, query: str, limit: int = 5) -> List[MemoryEntry]:
        lexical_candidates = store.retrieve(query, limit=max(limit * 2, limit))
        if not lexical_candidates:
            lexical_candidates = store.all()[: max(limit * 2, limit)]
        payload = {
            "query": query,
            "limit": limit,
            "candidate_memories": [_memory_brief(entry) for entry in lexical_candidates],
            "instruction": "Return the memory_ids that are most relevant for answering the current session query.",
        }
        result = self._call_json(
            schema_name="olm_retrieval_rerank",
            schema=RETRIEVER_SCHEMA,
            system_prompt=(
                "OLM retrieval component. Select the candidate memory IDs that are most relevant "
                "to the current query. Prefer direct semantic relevance and memories likely to be evidence for the decision."
            ),
            payload=payload,
        )
        selected_ids = result.get("memory_ids", [])
        chosen = [store.get(memory_id) for memory_id in selected_ids if memory_id in store.entries]
        return chosen[:limit] if chosen else lexical_candidates[:limit]


class OpenAILoopExtractor(LoopExtractor, OpenAIBackedMixin):
    def __init__(self, client: Optional[OpenAIResponsesClient] = None, model_name: Optional[str] = None) -> None:
        OpenAIBackedMixin.__init__(self, client=client, model_name=model_name)
        self.counter = 0

    def extract(self, session: Session, closed_memories: Sequence[MemoryEntry]) -> List[OpenLoop]:
        payload = {
            "session": asdict(session),
            "retrieved_memories": [_memory_brief(memory) for memory in closed_memories],
            "instruction": (
                "Extract unresolved epistemic obligations that should become open loops. "
                "Open a loop only when the session depends on unresolved assumptions, incomplete verification, "
                "conflicting evidence, deferred commitments, scope limitations, or evidence gaps."
            ),
        }
        result = self._call_json(
            schema_name="olm_open",
            schema=OPEN_SCHEMA,
            system_prompt=(
                "OLM Open operator. Detect unresolved obligations from the session and retrieved memories. "
                "Use only the allowed loop types. Calibrate risk and unresolvedness to [0,1]."
            ),
            payload=payload,
        )
        loops: List[OpenLoop] = []
        for item in result.get("loops", []):
            if not item.get("should_open"):
                continue
            loops.append(
                OpenLoop(
                    loop_id=f"loop-{self.counter}",
                    loop_type=item["loop_type"],
                    proposition=item["proposition"],
                    evidence=[
                        EvidenceItem(
                            content=session.query,
                            polarity="support",
                            source=session.session_id,
                            supports_closure=False,
                            metadata={"reason": item["reason"], "query_type": session.metadata.get("query_type", "general")},
                        )
                    ],
                    triggers=[
                        {
                            "relation": item["trigger_relation"],
                            "stakes": item["trigger_stakes"],
                            "irreversible": item["irreversible"],
                        }
                    ],
                    closure_predicate={"modality": item["closure_modality"], "status": "insufficient"},
                    gate={"type": item["gate_type"]},
                    risk_score=max(0.0, min(1.0, float(item["risk_score"]))),
                    scope={"domain": item["scope_domain"]},
                    conditioned_memory_ids=item["conditioned_memory_ids"],
                    provenance={"session_id": session.session_id, "reason": item["reason"]},
                    state="open",
                    unresolvedness=max(0.0, min(1.0, float(item["unresolvedness"]))),
                )
            )
            self.counter += 1
        return loops


class OpenAIClosureRequirementJudge(ClosureRequirementJudge, OpenAIBackedMixin):
    def __init__(
        self,
        activation_threshold: float,
        client: Optional[OpenAIResponsesClient] = None,
        model_name: Optional[str] = None,
    ) -> None:
        OpenAIBackedMixin.__init__(self, client=client, model_name=model_name)
        self.activation_threshold = activation_threshold
        self.prefilter_operator = RequiresClosedOperator(activation_threshold=activation_threshold)

    def prefilter(
        self,
        decisions: Sequence[DecisionObject],
        loops: Sequence[OpenLoop],
    ) -> List[Tuple[DecisionObject, OpenLoop]]:
        return self.prefilter_operator.prefilter(decisions, loops)

    def evaluate(self, decision: DecisionObject, loop: OpenLoop) -> RequiresClosedResult:
        payload = {
            "decision": _decision_brief(decision),
            "loop": _loop_brief(loop),
            "activation_threshold": self.activation_threshold,
            "instruction": (
                "Determine whether this decision object requires the loop to be closed first. "
                "Use relation_type=mem_license only for memory evidence use, act_precond for actions blocked by unresolved obligations, "
                "and claim_dep for claims or reasoning that depend on closure."
            ),
        }
        result = self._call_json(
            schema_name="olm_requires_closed",
            schema=REQUIRES_CLOSED_SCHEMA,
            system_prompt=(
                "OLM RequiresClosed operator. Decide whether the decision depends on a loop being closed, "
                "and provide a confidence score in [0,1] plus minimal sufficient evidence hints."
            ),
            payload=payload,
        )
        return RequiresClosedResult(
            requires_closed=bool(result["requires_closed"]),
            relation_type=result["relation_type"],
            confidence=max(0.0, min(1.0, float(result["confidence"]))),
            sufficient_evidence=list(result["sufficient_evidence"]),
        )


class OpenAILoopCloser(LoopCloser, OpenAIBackedMixin):
    def __init__(self, client: Optional[OpenAIResponsesClient] = None, model_name: Optional[str] = None) -> None:
        OpenAIBackedMixin.__init__(self, client=client, model_name=model_name)

    def apply(self, loop: OpenLoop, new_evidence: EvidenceItem) -> LoopResolution:
        payload = {
            "loop": _loop_brief(loop),
            "existing_evidence": [asdict(item) for item in loop.evidence],
            "new_evidence": asdict(new_evidence),
            "instruction": (
                "Given the new evidence, decide whether the loop is now confirmed, refuted, superseded, or still insufficient. "
                "Also decide how conditioned memories should be updated."
            ),
        }
        result = self._call_json(
            schema_name="olm_close",
            schema=CLOSE_SCHEMA,
            system_prompt=(
                "OLM Close operator. Evaluate loop closure conservatively and produce explicit memory update actions."
            ),
            payload=payload,
        )
        loop.evidence.append(new_evidence)
        loop.state = result["updated_state"]
        loop.closure_predicate["status"] = result["verdict"]
        loop.unresolvedness = 0.0 if result["verdict"] != "insufficient" else max(0.15, loop.unresolvedness - 0.15)
        memory_updates = {item["memory_id"]: item["action"] for item in result["memory_updates"]}
        return LoopResolution(
            verdict=result["verdict"],
            updated_state=result["updated_state"],
            memory_updates=memory_updates,
            notes={"model_note": result["note"]},
        )


class OpenAIPolicy(BasePolicy, OpenAIBackedMixin):
    def __init__(self, client: Optional[OpenAIResponsesClient] = None, model_name: Optional[str] = None) -> None:
        OpenAIBackedMixin.__init__(self, client=client, model_name=model_name)
        self.fallback = MockPolicy()

    def propose_actions(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        payload = {
            "session": asdict(session),
            "retrieved_memories": [_memory_brief(item) for item in retrieved],
            "instruction": "Propose up to three candidate next actions for the session.",
        }
        result = self._call_json(
            schema_name="olm_policy_actions",
            schema=POLICY_ACTIONS_SCHEMA,
            system_prompt=(
                "OLM base policy. Propose concrete next actions for the runtime. "
                "Actions should be executable next-step decisions, not long explanations."
            ),
            payload=payload,
        )
        actions = result.get("actions", [])
        if not actions:
            return self.fallback.propose_actions(session, retrieved)
        return [
            DecisionObject(
                object_id=f"action-{session.session_id}-{idx}",
                object_type="action",
                text=item["text"],
                metadata={
                    "stakes": item["stakes"],
                    "reversible": item["reversible"],
                    "verification_cost": item["verification_cost"],
                    "verify_budget": item["verify_budget"],
                },
            )
            for idx, item in enumerate(actions)
        ]

    def draft_claims(self, session: Session, retrieved: Sequence[MemoryEntry]) -> List[DecisionObject]:
        payload = {
            "session": asdict(session),
            "retrieved_memories": [_memory_brief(item) for item in retrieved],
            "instruction": "Draft up to three candidate claims or reasoning statements relevant to the current step.",
        }
        result = self._call_json(
            schema_name="olm_policy_claims",
            schema=POLICY_CLAIMS_SCHEMA,
            system_prompt=(
                "OLM base policy. Draft candidate claims that may be relied on while answering the current query."
            ),
            payload=payload,
        )
        claims = result.get("claims", [])
        if not claims:
            return self.fallback.draft_claims(session, retrieved)
        return [
            DecisionObject(
                object_id=f"claim-{session.session_id}-{idx}",
                object_type="claim",
                text=item["text"],
                metadata={"stakes": item["stakes"]},
            )
            for idx, item in enumerate(claims)
        ]

    def select_action(self, session: Session, candidates: Sequence[DecisionObject]) -> DecisionObject:
        candidate_list = list(candidates)
        if len(candidate_list) == 1:
            return candidate_list[0]
        payload = {
            "session": asdict(session),
            "candidates": [asdict(item) for item in candidate_list],
            "instruction": "Select the best candidate action index for this step.",
        }
        result = self._call_json(
            schema_name="olm_policy_select_action",
            schema=POLICY_SELECT_SCHEMA,
            system_prompt=(
                "OLM base policy. Choose the single best next action index."
            ),
            payload=payload,
        )
        selected = int(result["selected_index"])
        if selected < 0 or selected >= len(candidate_list):
            return candidate_list[0]
        return candidate_list[selected]


def build_mock_components(activation_threshold: float) -> OLMComponents:
    return OLMComponents(
        retriever=StoreRetriever(),
        extractor=MockLoopExtractor(),
        judge=MockClosureRequirementJudge(activation_threshold),
        closer=MockLoopCloser(),
        policy=MockPolicy(),
    )


def build_openai_components(activation_threshold: float, model_name: Optional[str] = None) -> OLMComponents:
    client = OpenAIResponsesClient(
        OpenAIClientConfig(
            model=model_name or os.environ.get("OLM_OPENAI_MODEL", "gpt-4o-mini"),
            timeout_seconds=int(os.environ.get("OLM_OPENAI_TIMEOUT", "60")),
            max_retries=int(os.environ.get("OLM_OPENAI_MAX_RETRIES", "2")),
        )
    )
    return OLMComponents(
        retriever=OpenAIRetriever(client=client),
        extractor=OpenAILoopExtractor(client=client),
        judge=OpenAIClosureRequirementJudge(activation_threshold, client=client),
        closer=OpenAILoopCloser(client=client),
        policy=OpenAIPolicy(client=client),
    )


def build_api_ready_components(activation_threshold: float) -> OLMComponents:
    return build_openai_components(activation_threshold)
