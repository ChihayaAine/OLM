from __future__ import annotations

import argparse
import json
import os

from ..evaluation.benchmarks import load_benchmark_dataset
from ..evaluation.runner import run_experiment, write_artifacts
from ..operators.components import APINotConfiguredError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OLM experiments.")
    parser.add_argument("--benchmark", choices=["synthetic", "jsonl"], default="synthetic")
    parser.add_argument("--sessions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sessions-file")
    parser.add_argument("--memories-file")
    parser.add_argument("--backend", choices=["auto", "mock", "provider"], default="auto")
    parser.add_argument("--output-dir")
    parser.add_argument("--json", action="store_true", help="Emit JSON traces and memory state.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    backend = args.backend
    if backend == "auto":
        backend = "provider" if (os.environ.get("OLM_API_KEY") and os.environ.get("OLM_API_BASE")) else "mock"
    dataset = load_benchmark_dataset(
        benchmark_name=args.benchmark,
        session_count=args.sessions,
        seed=args.seed,
        sessions_path=args.sessions_file,
        memories_path=args.memories_file,
    )
    try:
        artifacts = run_experiment(dataset=dataset, backend=backend)
    except APINotConfiguredError as exc:
        raise SystemExit(f"Provider backend is configured incorrectly: {exc}")
    if args.output_dir:
        write_artifacts(artifacts, args.output_dir)
    if args.json:
        payload = {
            "benchmark": dataset.name,
            "metrics": artifacts.report.to_dict(),
            "traces": [trace.to_dict() for trace in artifacts.traces],
            "state": artifacts.final_state,
        }
        print(json.dumps(payload, indent=2))
        return
    for trace in artifacts.traces:
        print(
            f"session={trace.session_id} activated={trace.activated_loops} "
            f"blocked={trace.blocked_actions} final={trace.final_action_state}"
        )
    print(json.dumps(artifacts.report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
