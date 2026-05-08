from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MemoryEntry:
    memory_id: str
    text: str
    tags: List[str]
    source: str
    factual_confidence: float = 0.8
    license_status: str = "usable"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "MemoryEntry":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceItem:
    content: str
    polarity: str
    source: str
    supports_closure: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvidenceItem":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OpenLoop:
    loop_id: str
    loop_type: str
    proposition: str
    evidence: List[EvidenceItem]
    triggers: List[Dict[str, Any]]
    closure_predicate: Dict[str, Any]
    gate: Dict[str, Any]
    risk_score: float
    scope: Dict[str, Any]
    conditioned_memory_ids: List[str]
    provenance: Dict[str, Any]
    state: str = "open"
    unresolvedness: float = 1.0
    age: int = 0

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OpenLoop":
        evidence = [item if isinstance(item, EvidenceItem) else EvidenceItem.from_dict(item) for item in payload.get("evidence", [])]
        return cls(
            loop_id=payload["loop_id"],
            loop_type=payload["loop_type"],
            proposition=payload["proposition"],
            evidence=evidence,
            triggers=payload["triggers"],
            closure_predicate=payload["closure_predicate"],
            gate=payload["gate"],
            risk_score=payload["risk_score"],
            scope=payload["scope"],
            conditioned_memory_ids=payload["conditioned_memory_ids"],
            provenance=payload["provenance"],
            state=payload.get("state", "open"),
            unresolvedness=payload.get("unresolvedness", 1.0),
            age=payload.get("age", 0),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionObject:
    object_id: str
    object_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GateDecision:
    decision: str
    reason: str
    verification_action: Optional[DecisionObject] = None


@dataclass
class RequiresClosedResult:
    requires_closed: bool
    relation_type: str
    confidence: float
    sufficient_evidence: List[str] = field(default_factory=list)


@dataclass
class LoopResolution:
    verdict: str
    updated_state: str
    memory_updates: Dict[str, str] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StepTrace:
    session_id: str
    query: str
    retrieved_memories: List[str]
    activated_loops: List[str]
    blocked_actions: List[str]
    chosen_action: str
    final_action_state: str
    notes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Session:
    session_id: str
    query: str
    metadata: Dict[str, Any]
    gold: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "Session":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
