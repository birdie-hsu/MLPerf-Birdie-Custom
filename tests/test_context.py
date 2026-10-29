import io
import json
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch

from clientbench.cli import main
from clientbench.context import local_context_plan, resolve_context, workload_context_length
from clientbench.local import LocalServer
from clientbench.runner import Runner
from clientbench.transport import Reply
from clientbench.workloads import UPSTREAM, load_cases


class ContextTests(unittest.TestCase):
    def test_every_pinned_workload_keeps_official_context_and_source_files(self):
        for case in load_cases("all", True):
            if case["mode"] == "autonomous":
                self.assertEqual(case["context_length"], 4096)
                continue
            relative = case["id"].split("#")[0]
            source = json.loads((UPSTREAM / relative).read_text(encoding="utf-8-sig"))
            self.assertEqual(case["context_length"], source["model_config"]["context_length"])
        self.assertEqual({c["context_length"] for c in load_cases("llm", True)}, {4096, 5120, 8192})
        self.assertEqual({c["context_length"] for c in load_cases("replay", True)}, {12000, 15000})

    def test_official_flat_legacy_precedence_and_default(self):
        self.assertEqual(workload_context_length({}), 4096)
        self.assertEqual(workload_context_length({"model": {"context_length": 12000}}), 12000)
        self.assertEqual(workload_context_length({"context_length": 8192,
                                                "model": {"context_length": 12000}}), 8192)

    def test_override_null_default_and_invalid_capacities(self):
        self.assertEqual(resolve_context({}, {}), (4096, "default"))
        self.assertEqual(resolve_context({"context_size": None}, {"context_length": 15000}), (15000, "workload"))
        self.assertEqual(resolve_context({"context_size": 16384}, {"context_length": 15000}), (16384, "override"))
        for value in (0, -1, True, "8192", 8192.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_context({"context_size": value}, {})

    def test_cli_plan_lists_per_case_contexts_and_filtered_suite(self):
        for extra, sizes, policy in (([], {4096, 5120, 8192}, "workload"),
                                     (["--context-size", "16384"], {16384}, "override"),
                                     (["--case-filter", "booksum_8k"], {8192}, "workload")):
            output = io.StringIO()
            with redirect_stdout(output):
                main(["run", "--target", "local", "--suite", "llm", "--extended",
                      "--mock", "--dry-run", *extra])
            plan = json.loads(output.getvalue())
            for target in plan["local_hardware"]:
                self.assertEqual(target["context_policy"], policy)
                self.assertEqual({c["context_size"] for c in target["case_contexts"]}, sizes)

    def test_external_plan_does_not_claim_server_was_reconfigured(self):
        plan = local_context_plan({"context_size": 32768}, load_cases("replay", True))
        self.assertEqual(plan["context_policy"], "external_server")
        self.assertTrue(all(c["context_size"] is None and c["context_source"] == "external_server"
                            for c in plan["case_contexts"]))

    def test_context_and_parallel_flags_cannot_bypass_plan_or_server(self):
        for args in (["-c", "32768"], ["--ctx-size=32768"], ["-c32768"],
                     ["--parallel", "2"], ["-np2"]):
            target = {"server_args": args}
            with self.subTest(args=args):
                with self.assertRaisesRegex(ValueError, "cannot override context"):
                    local_context_plan(target, load_cases("llm"), mock=True)
                with self.assertRaisesRegex(ValueError, "cannot override context"):
                    LocalServer(target, "unused").__enter__()

    def test_server_launch_uses_exact_capacity_and_separate_logs(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            model, exe = root / "model.gguf", root / "server.exe"
            model.write_bytes(b"GGUFfixture")
            exe.write_bytes(b"fixture")
            logs = []
            for size in (4096, 5120, 8192, 12000, 15000):
                target = {"name": "fixture", "model": "fixture", "context_size": size,
                          "metadata": {}, "installation": {"missing_parts": [],
                              "parts": [str(model)], "server": str(exe),
                              "model_path": str(model), "root": str(root)}}
                process = Mock()
                process.poll.return_value = None
                health, props, sock = MagicMock(), MagicMock(), MagicMock()
                health.__enter__.return_value.status = 200
                props.__enter__.return_value.read.return_value = b'{}'
                sock.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 23456)
                with patch("clientbench.local.socket.socket", return_value=sock), \
                     patch("clientbench.local.subprocess.Popen", return_value=process) as popen, \
                     patch("clientbench.local.urllib.request.urlopen", side_effect=[health, props]):
                    with LocalServer(target, root / "run"):
                        args = popen.call_args.args[0]
                        self.assertEqual(args[args.index("-c") + 1], str(size))
                        self.assertEqual(args[args.index("-np") + 1], "1")
                        logs.append(target["metadata"]["server_log"])
                    process.terminate.assert_called_once()
            self.assertEqual(len(set(logs)), 5)


class ContextLifecycleTests(unittest.TestCase):
    def config(self, **changes):
        return {"repeats": 2, "warmups": 1, "order_seed": 7, "max_requests": 1000,
                "task_timeout_s": 20, "request_timeout_s": 10, "replay_tools": False, **changes}

    def run_managed(self, root, *, override=None, fail_size=None, config=None):
        runner = Runner(config or self.config(), root)
        cases = load_cases("llm", True)
        target = {"name": "fixture", "model": "fixture", "backend": "llamacpp",
                  "context_size": override, "installation": {"server": "fixture"}, "metadata": {}}
        sessions, active = [], []

        @contextmanager
        def server(target, run_dir):
            self.assertFalse(active, "servers must never overlap")
            active.append(target["context_size"])
            sessions.append(target["context_size"])
            try:
                if target["context_size"] == fail_size:
                    raise RuntimeError("fixture_load_failure")
                target["metadata"]["server_argv"] = ["fixture", "-c", str(target["context_size"])]
                yield
            finally:
                active.pop()

        with patch("clientbench.runner.LocalServer", side_effect=server), \
             patch("clientbench.runner.HTTPAdapter") as adapter, redirect_stdout(io.StringIO()):
            adapter.return_value.call.return_value = Reply(text="fixture", finish="stop")
            rows = runner.run([target], cases)
        self.assertFalse(active)
        self.assertEqual(target["context_size"], override, "session values must not mutate the template")
        metadata = json.loads((Path(root) / "run.json").read_text(encoding="utf-8"))
        return rows, sessions, metadata

    def test_each_capacity_gets_one_server_with_warmups_and_repeats(self):
        with tempfile.TemporaryDirectory() as root:
            rows, sessions, meta = self.run_managed(root)
            self.assertCountEqual(sessions, [4096, 5120, 8192])
            self.assertEqual(len(rows), 45)  # 15 cases: 1 warmup + 2 measured.
            self.assertEqual(len([r for r in rows if r["phase"] == "warmup"]), 15)
            self.assertEqual(len(meta["execution_order"]), 30)
            self.assertEqual(meta["request_count"], 45)
            for row in rows:
                self.assertEqual(row["context_size"], row["workload_context_length"])
                self.assertEqual(row["context_source"], "workload")
                trace = json.loads((Path(root) / row["trace"]).read_text(encoding="utf-8"))
                self.assertEqual(trace["row"]["context_size"], row["context_size"])
            for session in meta["server_sessions"]:
                self.assertEqual(session["status"], "completed")
                self.assertEqual(session["metadata"]["server_argv"][-1], str(session["context_size"]))

    def test_numeric_override_keeps_one_server_and_records_both_values(self):
        with tempfile.TemporaryDirectory() as root:
            rows, sessions, _ = self.run_managed(root, override=16384)
            self.assertEqual(sessions, [16384])
            self.assertEqual({r["context_size"] for r in rows}, {16384})
            self.assertEqual({r["context_source"] for r in rows}, {"override"})
            self.assertEqual({r["workload_context_length"] for r in rows}, {4096, 5120, 8192})

    def test_load_failure_only_marks_affected_capacity(self):
        with tempfile.TemporaryDirectory() as root:
            rows, sessions, meta = self.run_managed(root, fail_size=5120)
            self.assertEqual(len(sessions), 3)
            failed = [r for r in rows if r["status"] == "load_error"]
            self.assertEqual(len(failed), 6)  # Three code-analysis cases x two repeats.
            self.assertEqual({r["context_size"] for r in failed}, {5120})
            measured = [r for r in rows if r["phase"] == "measured"]
            self.assertEqual(len(measured), 30)
            self.assertEqual(len({(r["case_id"], r["repeat"]) for r in measured}), 30)
            self.assertEqual(sum(s["status"] == "load_error" for s in meta["server_sessions"]), 1)

    def test_request_budget_is_shared_across_capacity_groups(self):
        with tempfile.TemporaryDirectory() as root:
            _, _, meta = self.run_managed(root, config=self.config(repeats=1, warmups=0, max_requests=5))
            self.assertEqual(meta["request_count"], 5)
            self.assertIn("fixture", meta["budget_exhausted_targets"])

    def test_external_server_and_cloud_run_without_managed_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            targets = [{"name": "external", "model": "fixture", "backend": "llamacpp"},
                       {"name": "cloud", "model": "fixture", "backend": "responses"}]
            runner = Runner(self.config(repeats=1, warmups=0), root)
            with patch("clientbench.runner.LocalServer") as server, \
                 patch("clientbench.runner.HTTPAdapter") as adapter, redirect_stdout(io.StringIO()):
                adapter.return_value.call.return_value = Reply(text="fixture", finish="stop")
                rows = runner.run(targets, load_cases("llm", True))
            server.assert_not_called()
            self.assertEqual(len(rows), 30)
            for row in rows:
                if row["target"] == "external":
                    self.assertIsNone(row["context_size"])
                    self.assertEqual(row["context_source"], "external_server")
                else:
                    self.assertNotIn("context_size", row)


if __name__ == "__main__":
    unittest.main()
