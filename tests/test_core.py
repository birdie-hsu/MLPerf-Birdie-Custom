import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from clientbench.local import discover_installations, inventory, expand_local_targets, LocalServer
from clientbench.report import summarize, csv_export
from clientbench.runner import Runner, reply_status
from clientbench.tools import ToolEngine, bounded_path, extract_reference_calls
from clientbench.transport import Reply, TransportError, consume, responses_input, request_body, sse_events, assistant_message
from clientbench.workloads import load_cases, make_data, check_llm, evaluate_agent, SWE_CORRECT, SWE_START, SWE_TESTS


class StreamTests(unittest.TestCase):
    def test_reasoning_only_length_is_truncated_and_preserved(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "Thinking"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}],
             "usage": {"prompt_tokens": 100, "completion_tokens": 128},
             "timings": {"predicted_per_second": 64.5}},
        ]
        reply = consume(events, "llamacpp", lambda: .5)
        self.assertEqual(reply_status(reply), "truncated")
        self.assertEqual(reply.reasoning_text, "Thinking")
        self.assertEqual(reply.text, "")
        self.assertIsNone(reply.metrics()["ttft_visible_s"])
        self.assertEqual(reply.metrics()["ttft_reasoning_s"], .5)
        self.assertEqual(reply.metrics()["server_decode_tps"], 64.5)
        self.assertIsNone(reply.metrics()["native_tps_estimate"])
        self.assertEqual(assistant_message(reply, "llamacpp")["reasoning_content"], "Thinking")

    def test_local_thinking_policy_and_output_budget(self):
        target = {"backend": "llamacpp", "model": "local", "max_output_tokens": 2048}
        off = request_body(target, [], [], 128)
        self.assertEqual(off["max_tokens"], 128)
        self.assertEqual(off["chat_template_kwargs"], {"enable_thinking": False})
        on = request_body({**target, "local_thinking": "on"}, [], [], 128)
        self.assertEqual(on["max_tokens"], 2048)
        self.assertTrue(on["chat_template_kwargs"]["enable_thinking"])
        auto = request_body({**target, "local_thinking": "auto"}, [], [], 128)
        self.assertEqual(auto["max_tokens"], 2048)
        self.assertNotIn("chat_template_kwargs", auto)
        merged = request_body({**target, "extra_body": {"chat_template_kwargs": {"custom": True}}}, [], [], 128)
        self.assertEqual(merged["chat_template_kwargs"], {"custom": True, "enable_thinking": False})
        with self.assertRaises(ValueError):
            request_body({**target, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}, [], [], 128)

    def test_sse_comments_multiline_crlf_and_eof(self):
        raw = [b': keepalive\r\n', b'data: {"a":\r\n', b'data: 1}\r\n', b'\r\n',
               b'data: {"b": 2}']
        self.assertEqual(list(sse_events(raw)), [{"a": 1}, {"b": 2}])
        self.assertEqual(list(sse_events([b'data: [DONE]\n', b'\n'])), [])

    def test_responses_ttft_excludes_metadata_and_reasoning(self):
        events = [
            {"type": "response.created"},
            {"type": "response.reasoning.delta", "delta": "hidden"},
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.output_text.delta", "delta": " world"},
            {"type": "response.completed", "response": {"status": "completed", "model": "snapshot",
                "output": [], "usage": {"input_tokens": 10, "output_tokens": 8,
                    "output_tokens_details": {"reasoning_tokens": 4}}}},
        ]
        counter = iter([.01, .1, .3, .5, .6, .7])
        reply = consume(events, "responses", lambda: next(counter))
        self.assertEqual(reply.first_event_s, .01)
        self.assertEqual(reply.first_text_s, .3)
        self.assertEqual(reply.metrics()["visible_native_tokens"], 4)
        self.assertAlmostEqual(reply.metrics()["native_tps_estimate"], 15)
        self.assertEqual(reply.text, "Hello world")

    def test_tool_fragment_assembly_and_no_fake_visible_tps(self):
        events = [
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {
                "name": "read", "arguments": '{"path":'}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {
                "arguments": '"x.py"}'}}]}, "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 20, "completion_tokens": 9}},
        ]
        reply = consume(events, "llamacpp", lambda: 0.5)
        self.assertEqual(json.loads(reply.calls[0]["arguments"]), {"path": "x.py"})
        self.assertIsNone(reply.first_text_s)
        self.assertIsNone(reply.metrics()["native_tps_estimate"])

    def test_broken_stream_is_an_error(self):
        with self.assertRaises(TransportError):
            consume([{"type": "response.created"}], "responses", lambda: .1)

    def test_missing_usage_and_reasoning_are_not_estimated(self):
        reply = Reply(text="text", first_text_s=.1, last_text_s=.4, text_delta_count=3,
                      usage={"output_tokens": 10}, reasoning_seen=True)
        self.assertIsNone(reply.metrics()["visible_native_tokens"])
        self.assertIsNone(reply.metrics()["native_tps_estimate"])

    def test_stateless_reasoning_and_tool_results_preserved(self):
        reasoning = {"type": "reasoning", "id": "r1", "encrypted_content": "opaque"}
        call = {"type": "function_call", "call_id": "c1", "name": "read", "arguments": "{}"}
        history = [{"role": "user", "content": "go"},
                   {"role": "assistant", "content": "", "_items": [reasoning, call]},
                   {"role": "tool", "tool_call_id": "c1", "content": "result"}]
        result = responses_input(history)
        self.assertEqual(result[1:3], [reasoning, call])
        self.assertEqual(result[-1]["type"], "function_call_output")
        self.assertEqual(result[-1]["call_id"], "c1")

    def test_backend_specific_request_limits(self):
        cloud = request_body({"backend": "responses", "model": "gpt-6.1-sol", "max_output_tokens": 8192},
                             [{"role": "user", "content": "test"}], [], 256)
        self.assertEqual(cloud["max_output_tokens"], 8192)
        self.assertNotIn("temperature", cloud)
        self.assertFalse(cloud["store"])
        with self.assertRaises(ValueError):
            request_body({"backend": "llamacpp", "model": "local", "extra_body": {"stream": False}}, [], [], 128)


class DiscoveryTests(unittest.TestCase):
    def test_nested_models_shards_and_projectors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "llama.cpp"
            (root / "build/bin/Release").mkdir(parents=True)
            (root / "build/bin/Release/llama-server.exe").write_text("fixture")
            models = root / "models"
            (models / "family").mkdir(parents=True)
            for name in ["small-4B.gguf", "large-35B-00001-of-00002.gguf",
                         "large-35B-00002-of-00002.gguf", "mmproj-small.gguf", "readme.txt"]:
                (models / "family" / name).write_bytes(b"GGUFfixture")
            result = discover_installations([str(root)])
            self.assertEqual(len(result["installations"]), 1)
            inv = inventory(result["installations"])
            self.assertEqual(len(inv["models"]), 2)
            self.assertEqual(len(inv["excluded"]), 2)
            self.assertFalse(any(m["missing_parts"] for m in inv["models"]))
            target = expand_local_targets({"name": "local", "model_overrides": {
                "family/small-4B.gguf": {"context_size": 8192}}}, inv)
            self.assertEqual(len({t["name"] for t in target}), 2)
            small = next(t for t in target if "small" in t["name"])
            self.assertEqual(small["context_size"], 8192)

    def test_missing_shard_and_missing_server(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "models").mkdir()
            self.assertTrue(discover_installations([str(root)])["warnings"])
            (root / "llama-server.exe").write_text("x")
            (root / "models/m-00001-of-00002.gguf").write_text("x")
            inv = inventory(discover_installations([str(root)])["installations"])
            self.assertEqual(len(inv["models"][0]["missing_parts"]), 1)

    def test_server_cleanup_terminates_owned_process(self):
        from unittest.mock import Mock
        server = LocalServer({}, Path("unused"))
        server.process = Mock()
        server.process.poll.return_value = None
        server.log = Mock()
        server.__exit__(None, None, None)
        server.process.terminate.assert_called_once()
        server.process.wait.assert_called_once_with(timeout=10)
        server.log.close.assert_called_once()

    def test_server_launch_health_identity_and_cleanup(self):
        from unittest.mock import Mock, MagicMock
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, exe = root / "model.gguf", root / "llama-server.exe"
            model.write_bytes(b"GGUFfixture")
            exe.write_bytes(b"executable fixture")
            target = {"name": "model", "model": "alias", "metadata": {}, "installation": {
                "missing_parts": [], "parts": [str(model)], "server": str(exe),
                "model_path": str(model), "root": str(root)}}
            fake_process = Mock()
            fake_process.poll.return_value = None
            health = MagicMock()
            health.__enter__.return_value.status = 200
            props = MagicMock()
            props.__enter__.return_value.read.return_value = b'{"build_info":"fixture"}'
            fake_socket = MagicMock()
            fake_socket.__enter__.return_value.getsockname.return_value = ("127.0.0.1", 23456)
            with patch("clientbench.local.socket.socket", return_value=fake_socket), \
                 patch("clientbench.local.subprocess.Popen", return_value=fake_process) as popen, \
                 patch("clientbench.local.urllib.request.urlopen", side_effect=[health, props]):
                with LocalServer(target, root / "run") as server:
                    self.assertEqual(server.props["build_info"], "fixture")
                    self.assertEqual(target["base_url"], "http://127.0.0.1:23456/v1")
                    self.assertEqual(len(target["metadata"]["gguf_part_sha256"]), 1)
                    args = popen.call_args.args[0]
                    self.assertIn("--no-context-shift", args)
                    self.assertEqual(args[args.index("-c") + 1], "4096")
                    self.assertIn(str(model), args)
                fake_process.terminate.assert_called_once()


class SafetyAndOracleTests(unittest.TestCase):
    def test_hidden_tests_accept_reference_and_reject_known_bug(self):
        # These are trusted benchmark fixtures, never provider-supplied code.
        import types
        for source, expected in ((SWE_CORRECT, True), (SWE_START, False)):
            module = types.ModuleType("bootstrap_mean_ci")
            exec(source, module.__dict__)
            namespace = {"__name__": "oracle_fixture"}
            with patch.dict("sys.modules", {"bootstrap_mean_ci": module}):
                exec(SWE_TESTS.replace("unittest.main()", ""), namespace)
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(namespace["Tests"])
                result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
                self.assertEqual(result.wasSuccessful(), expected)
                self.assertEqual(result.testsRun, 8)

    def test_path_boundary_and_tool_json_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            for path in ("../secret", "C:/secret", "/etc/passwd"):
                with self.assertRaises(ValueError): bounded_path(temporary, path)
            engine = ToolEngine(temporary, None)
            result = engine.run({"name": "write", "arguments": '{"path":"a.py","content":"x"}'})
            self.assertTrue(result["ok"])
            self.assertFalse(engine.run({"name": "patch", "arguments": {
                "path": "a.py", "old": "absent", "new": "x"}})["ok"])
            self.assertFalse(engine.run({"name": "read", "arguments": "malformed"})["ok"])

    def test_reference_json_parser_ignores_prose_and_unrelated_json(self):
        text = 'Prose {not JSON} {"foo":1}\n```\n{"type":"tool_use","name":"read","input":{"path":"a.py"}}\n```'
        calls = extract_reference_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"], {"path": "a.py"})

    def test_pinned_prompts_have_exact_counts(self):
        self.assertEqual(len(load_cases("llm", True)), 15)
        self.assertEqual(len(load_cases("llm", False)), 13)
        cases = load_cases("replay", True)
        self.assertEqual(sorted(len(c["steps"]) for c in cases), [3, 4])

    def test_data_oracle_rejects_wrong_totals_and_unverified_claim(self):
        case = {"kind": "data", "seed": 11}
        with tempfile.TemporaryDirectory() as temporary:
            _, expected = make_data(11)
            path = Path(temporary) / "report.json"
            path.write_text(json.dumps(expected))
            trace = [{"call": {"name": "execute"}, "result": {"ok": True}}]
            self.assertTrue(evaluate_agent(case, temporary, None, trace, 10)["pass"])
            self.assertFalse(evaluate_agent(case, temporary, None, [], 10)["pass"])
            expected["duplicate_rows"] = True
            path.write_text(json.dumps(expected))
            self.assertFalse(evaluate_agent(case, temporary, None, trace, 10)["pass"])
            expected["duplicate_rows"] = 1
            expected["valid_rows"] = 99
            path.write_text(json.dumps(expected))
            self.assertFalse(evaluate_agent(case, temporary, None, trace, 10)["pass"])

    def test_csv_conversion_requires_semantically_matching_values(self):
        case = next(c for c in load_cases("llm") if "data_csv2json" in c["id"])
        self.assertFalse(check_llm(case, "[]")["pass"])
        self.assertFalse(check_llm(case, "not JSON")["pass"])


class RunnerAndReportTests(unittest.TestCase):
    def test_capped_llm_timings_retained_without_quality_success(self):
        row = {"target": "x", "mode": "llm", "category": "Summarization", "case_id": "a",
               "phase": "measured", "status": "truncated", "measurement_valid": True,
               "e2e_s": 2, "quality": {"pass": True}, "estimated_usd": None,
               "turn_metrics": [{"ttft_visible_s": .1, "server_decode_tps": 64}]}
        summary = summarize([row, {**row, "status": "error", "measurement_valid": False, "e2e_s": 999}])[0]
        self.assertEqual(summary["completed"], 0)
        self.assertEqual(summary["valid_measurements"], 1)
        self.assertEqual(summary["truncated"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["quality_pass_rate"], 0)
        self.assertEqual(summary["e2e_p50_s"], 2)
        self.assertEqual(summary["ttft_visible_p50_s"], .1)
        self.assertEqual(summary["server_decode_tps_p50"], 64)
        agent = summarize([{**row, "mode": "autonomous"}])[0]
        self.assertEqual(agent["valid_measurements"], 0)

    def test_runner_reasoning_only_length_and_warmup_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"max_requests": 20, "task_timeout_s": 20, "request_timeout_s": 10, "replay_tools": False}
            runner = Runner(config, temporary)
            target = {"name": "x", "model": "x", "backend": "llamacpp"}
            reply = Reply(reasoning_seen=True, reasoning_text="thinking", finish="length", request_output_limit=128)
            with patch("clientbench.runner.HTTPAdapter") as adapter:
                adapter.return_value.call.return_value = reply
                measured = runner.run_case(target, load_cases("llm")[0], 0)
                warmup = runner.run_case(target, load_cases("llm")[0], 0, "warmup")
            self.assertEqual(measured["status"], "truncated")
            self.assertTrue(measured["measurement_valid"])
            self.assertEqual(measured["diagnostic"], "reasoning_only_output_budget_exhausted")
            self.assertEqual(measured["request_output_budget"], 128)
            self.assertEqual(warmup["status"], "warmup_truncated")
            self.assertFalse(warmup["measurement_valid"])

    def test_failed_attempts_in_quality_denominator_and_warmups_excluded(self):
        base = {"target": "x", "mode": "autonomous", "category": "SWE Agent", "case_id": "a",
                "repeat": 0, "phase": "measured", "status": "ok", "e2e_s": 2,
                "quality": {"pass": True}, "turn_metrics": [], "estimated_usd": None}
        rows = [base, {**base, "status": "timeout", "quality": {"pass": False}},
                {**base, "phase": "warmup", "e2e_s": 999}]
        result = summarize(rows)[0]
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["completed"], 1)
        self.assertEqual(result["quality_pass_rate"], .5)
        self.assertEqual(result["e2e_p50_s"], 2)

    def test_mock_agents_generate_traces_and_real_oracle_data_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"max_requests": 50, "max_agent_turns": 12, "task_timeout_s": 20,
                      "request_timeout_s": 10, "replay_tools": False}
            runner = Runner(config, temporary, mock=True)
            target = {"name": "local", "backend": "llamacpp", "model": "fixture"}
            case = next(c for c in load_cases("agent") if c["kind"] == "data")
            row = runner.run_case(target, case, 0)
            self.assertEqual(row["status"], "ok")
            self.assertTrue(row["quality"]["pass"])
            self.assertTrue(row["synthetic"])
            self.assertEqual(row["turns"], 4)
            self.assertEqual(row["tool_calls"], 3)
            self.assertTrue((Path(temporary) / row["trace"]).is_file())

    def test_budget_exhaustion_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"max_requests": 1, "max_agent_turns": 12, "task_timeout_s": 20,
                      "request_timeout_s": 10, "replay_tools": False}
            runner = Runner(config, temporary, mock=True)
            target = {"name": "local", "backend": "llamacpp", "model": "fixture"}
            case = load_cases("agent")[0]
            row = runner.run_case(target, case, 0)
            self.assertEqual(row["status"], "budget_exhausted")
            self.assertFalse(row["quality"]["pass"])

    def test_csv_export_escapes_formula_strings(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "out.csv"
            csv_export(path, [{"target": "=HYPERLINK(...)"}])
            with path.open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows[0]["target"].startswith("'="))

    def test_per_target_budget_does_not_skip_next_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = {"max_requests": 1, "max_agent_turns": 12, "task_timeout_s": 20,
                      "request_timeout_s": 10, "replay_tools": False, "order_seed": 7,
                      "repeats": 1, "warmups": 0}
            runner = Runner(config, temporary, mock=True)
            targets = [{"name": name, "backend": "llamacpp", "model": name} for name in ("4B", "35B")]
            rows = runner.run(targets, load_cases("agent")[:1])
            self.assertEqual({r["target"] for r in rows}, {"4B", "35B"})
            self.assertTrue(all(r["status"] == "budget_exhausted" for r in rows))


if __name__ == "__main__":
    unittest.main()
