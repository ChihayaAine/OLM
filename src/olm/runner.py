from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .runtime import OLMConfig, OLMRuntime
from .components import OLMComponents, build_mock_components, build_openai_components
from .benchmarks import BenchmarkDataset
from .evaluation import EvaluationReport, evaluate_sessions
from .types import StepTrace


@dataclass
class ExperimentArtifacts:
    traces: List[StepTrace]
    report: EvaluationReport
    final_state: Dict[str, object]


def build_runtime(
    backend: str,
    config: Optional[OLMConfig] = None,
    components: Optional[OLMComponents] = None,
) -> OLMRuntime:
    resolved_config = config or OLMConfig()
    if components is None:
        if backend == "mock":
            components = build_mock_components(resolved_config.activation_threshold)
        elif backend == "openai":
            components = build_openai_components(resolved_config.activation_threshold)
        else:
            raise ValueError(f"Unsupported backend: {backend}")
    return OLMRuntime(config=resolved_config, components=components)


def run_experiment(
    dataset: BenchmarkDataset,
    backend: str,
    config: Optional[OLMConfig] = None,
    components: Optional[OLMComponents] = None,
) -> ExperimentArtifacts:
    runtime = build_runtime(backend=backend, config=config, components=components)
    runtime.bootstrap(dataset.bootstrap_memories)
    traces = [runtime.run_session(session) for session in dataset.sessions]
    report = evaluate_sessions(dataset.name, dataset.sessions, traces)
    return ExperimentArtifacts(traces=traces, report=report, final_state=runtime.summary())


def write_artifacts(artifacts: ExperimentArtifacts, output_dir: str) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "traces.jsonl", "w", encoding="utf-8") as handle:
        for trace in artifacts.traces:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
    with open(path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.report.to_dict(), handle, indent=2, ensure_ascii=False)
    with open(path / "final_state.json", "w", encoding="utf-8") as handle:
        json.dump(artifacts.final_state, handle, indent=2, ensure_ascii=False)
