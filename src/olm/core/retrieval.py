from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence

from .types import MemoryEntry


@dataclass
class QueryProfile:
    raw_query: str
    query_terms: List[str]
    domain_hint: str
    stakes: str
    query_type: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class RetrievalDecision:
    memory_id: str
    lexical_overlap: int
    tag_match: int
    scope_bonus: float
    license_penalty: float
    confidence_bonus: float
    final_score: float

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def build_query_profile(query: str, metadata: Dict[str, object]) -> QueryProfile:
    terms = [token.strip(".,!?;:()[]{}").lower() for token in query.split()]
    query_terms = [token for token in terms if token]
    return QueryProfile(
        raw_query=query,
        query_terms=query_terms,
        domain_hint=str(metadata.get("domain", "general")),
        stakes=str(metadata.get("stakes", "medium")),
        query_type=str(metadata.get("query_type", "general")),
    )


def rank_memory_candidates(
    entries: Iterable[MemoryEntry],
    profile: QueryProfile,
) -> List[RetrievalDecision]:
    decisions: List[RetrievalDecision] = []
    query_term_set = set(profile.query_terms)
    raw_query_lower = profile.raw_query.lower()
    for entry in entries:
        terms = set(entry.text.lower().split()) | {tag.lower() for tag in entry.tags}
        lexical_overlap = len(query_term_set & terms)
        tag_match = sum(1 for tag in entry.tags if tag.lower() in raw_query_lower)
        entry_domain = str(entry.metadata.get("domain", entry.metadata.get("scope", ""))).lower()
        scope_bonus = 0.2 if entry_domain and entry_domain == profile.domain_hint.lower() else 0.0
        confidence_bonus = 0.25 * float(entry.factual_confidence)
        license_penalty = {
            "usable": 0.0,
            "requires_qualification": 0.05,
            "context_only": 0.15,
            "blocked": 0.30,
        }.get(entry.license_status, 0.1)
        if profile.query_type == "memory_reuse" and entry.license_status == "requires_qualification":
            license_penalty += 0.05
        final_score = float(lexical_overlap + tag_match) + scope_bonus + confidence_bonus - license_penalty
        decisions.append(
            RetrievalDecision(
                memory_id=entry.memory_id,
                lexical_overlap=lexical_overlap,
                tag_match=tag_match,
                scope_bonus=scope_bonus,
                license_penalty=license_penalty,
                confidence_bonus=confidence_bonus,
                final_score=final_score,
            )
        )
    decisions.sort(key=lambda item: item.final_score, reverse=True)
    return decisions


def retrieval_audit(decisions: Sequence[RetrievalDecision], limit: int) -> Dict[str, object]:
    top = list(decisions[:limit])
    return {
        "candidate_count": len(decisions),
        "returned_count": len(top),
        "top_memory_ids": [item.memory_id for item in top],
        "scores": [item.to_dict() for item in top],
    }
