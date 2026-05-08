import unittest
from pathlib import Path

from olm.runtime import build_mock_runtime, build_openai_runtime
from olm.benchmarks import SyntheticObligationBenchmark, load_benchmark_dataset
from olm.components import APINotConfiguredError
from olm.runner import run_experiment


class OLMTests(unittest.TestCase):
    def test_agent_runs_session(self):
        benchmark = SyntheticObligationBenchmark(seed=3)
        runtime = build_mock_runtime()
        dataset = benchmark.build_dataset(count=1)
        runtime.bootstrap(dataset.bootstrap_memories)
        trace = runtime.run_session(dataset.sessions[0])
        self.assertTrue(trace.session_id.startswith("session-"))
        self.assertIsInstance(trace.activated_loops, list)

    def test_olm_creates_or_updates_loops(self):
        benchmark = SyntheticObligationBenchmark(seed=5)
        dataset = benchmark.build_dataset(count=4)
        runtime = build_mock_runtime()
        runtime.bootstrap(dataset.bootstrap_memories)
        for session in dataset.sessions:
            runtime.run_session(session)
        self.assertGreaterEqual(len(runtime.open_store.all()), 1)
        self.assertGreaterEqual(len(runtime.closed_store.all()), 3)

    def test_openai_backend_requires_api_key(self):
        benchmark = SyntheticObligationBenchmark(seed=7)
        dataset = benchmark.build_dataset(count=1)
        runtime = build_openai_runtime()
        runtime.bootstrap(dataset.bootstrap_memories)
        with self.assertRaises(APINotConfiguredError):
            runtime.run_session(dataset.sessions[0])

    def test_jsonl_dataset_and_runner_work(self):
        repo_root = Path(__file__).resolve().parents[1]
        dataset = load_benchmark_dataset(
            benchmark_name="jsonl",
            sessions_path=str(repo_root / "data" / "sample_sessions.jsonl"),
            memories_path=str(repo_root / "data" / "sample_memories.jsonl"),
        )
        artifacts = run_experiment(dataset=dataset, backend="mock")
        self.assertEqual(len(artifacts.traces), 3)
        self.assertEqual(artifacts.report.benchmark_name, "sample_sessions")
        self.assertIn("closed_memories", artifacts.final_state)


if __name__ == "__main__":
    unittest.main()
