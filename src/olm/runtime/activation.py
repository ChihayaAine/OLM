from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import Dict, Iterable, List

from ..core.types import ActivatedDecision, DecisionObject, OpenLoop, RequiresClosedResult


@dataclass
class ActivationConfig:
    activation_threshold: float = 0.45
    max_active_loops: int = 6
    age_decay: float = 0.12
    relation_bonus: float = 0.08


def activation_priority(loop: OpenLoop, result: RequiresClosedResult, age_decay: float) -> float:
    trigger_bonus = 0.1 if any(trigger.get("irreversible") for trigger in loop.triggers) else 0.0
    relation_bonus = {
        "act_precond": 0.12,
        "claim_dep": 0.06,
        "mem_license": 0.03,
    }.get(result.relation_type, 0.0)
    return (
        loop.risk_score
        * max(0.05, loop.unresolvedness)
        * max(0.0, result.confidence)
        * exp(-age_decay * max(0, loop.age))
        + trigger_bonus
        + relation_bonus
    )


def select_activations(
    candidate_pairs: Iterable[tuple[DecisionObject, OpenLoop]],
    evaluator,
    config: ActivationConfig,
) -> tuple[Dict[str, RequiresClosedResult], List[ActivatedDecision], List[str]]:
    activated: Dict[str, RequiresClosedResult] = {}
    records: List[ActivatedDecision] = []
    strongest_by_loop: Dict[str, float] = {}
    for decision, loop in candidate_pairs:
        result = evaluator(decision, loop)
        if not result.requires_closed or result.confidence < config.activation_threshold:
            continue
        priority = activation_priority(loop, result, config.age_decay)
        key = f"{loop.loop_id}::{decision.object_id}"
        activated[key] = result
        records.append(
            ActivatedDecision(
                loop_id=loop.loop_id,
                object_id=decision.object_id,
                relation_type=result.relation_type,
                confidence=result.confidence,
                priority=priority,
                sufficient_evidence=list(result.sufficient_evidence),
                rationale=result.rationale,
                age=loop.age,
                risk_score=loop.risk_score,
                unresolvedness=loop.unresolvedness,
            )
        )
        strongest_by_loop[loop.loop_id] = max(strongest_by_loop.get(loop.loop_id, 0.0), priority)
    ranked_loop_ids = [
        loop_id
        for loop_id, _ in sorted(strongest_by_loop.items(), key=lambda item: item[1], reverse=True)[: config.max_active_loops]
    ]
    ranked = set(ranked_loop_ids)
    filtered = {key: result for key, result in activated.items() if key.split("::", 1)[0] in ranked}
    filtered_records = [record for record in records if record.loop_id in ranked]
    filtered_records.sort(key=lambda item: item.priority, reverse=True)
    return filtered, filtered_records, ranked_loop_ids


def activation_summary(records: List[ActivatedDecision]) -> Dict[str, object]:
    relation_counts: Dict[str, int] = {}
    loop_ids: List[str] = []
    for record in records:
        relation_counts[record.relation_type] = relation_counts.get(record.relation_type, 0) + 1
        if record.loop_id not in loop_ids:
            loop_ids.append(record.loop_id)
    return {
        "count": len(records),
        "loops": loop_ids,
        "max_priority": max((record.priority for record in records), default=0.0),
        "avg_confidence": (
            sum(record.confidence for record in records) / len(records)
            if records
            else 0.0
        ),
        "avg_unresolvedness": (
            sum(record.unresolvedness for record in records) / len(records)
            if records
            else 0.0
        ),
        "relation_counts": relation_counts,
    }
