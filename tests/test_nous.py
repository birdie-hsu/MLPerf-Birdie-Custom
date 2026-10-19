import argparse
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from clientbench.cli import config_load, main, resolve_targets
from clientbench.nous import default_target, expand_targets, fetch_models
from clientbench.transport import HTTPAdapter, Reply, TransportError, assistant_message, consume, request_body


class NousTests(unittest.TestCase):
    def test_router_request_omits_local_and_sampling_fields(self):
        target = {**default_target(), "model": "example/model"}
        body = request_body(target, [{"role": "user", "content": "prompt"}], [{"name": "read", "parameters": {}}], 128)
        self.assertEqual(body["max_tokens"], 8192)
        self.assertEqual(body["model"], "example/model")
        self.assertEqual(body["tools"][0]["function"]["name"], "read")
        self.assertTrue(body["stream_options"]["include_usage"])
        for name in ("seed", "cache_prompt", "chat_template_kwargs", "temperature", "parallel_tool_calls"):
            self.assertNotIn(name, body)
        with self.assertRaises(ValueError):
            request_body({**target, "extra_body": {"model": "other"}}, [], [], 128)

    def test_catalog_key_header_and_text_filtering(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({"data": [
            {"id": "a/chat", "name": "A", "supported_parameters": ["tools"],
             "architecture": {"output_modalities": ["text"]}, "context_length": 1234},
            {"id": "b/chat"}, {"id": "a/chat"},
            {"id": "image/model", "architecture": {"output_modalities": ["image"]}},
            {"id": "embedding/model"}, {"name": "invalid"},
        ]}).encode()
        with patch.dict(os.environ, {"NOUS_API_KEY": "offline-test-secret"}), \
             patch("urllib.request.urlopen", return_value=response) as http:
            catalog = fetch_models(default_target())
        request = http.call_args.args[0]
        self.assertEqual(request.full_url, "https://inference-api.nousresearch.com/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer offline-test-secret")
        self.assertEqual([m["id"] for m in catalog], ["a/chat", "b/chat"])
        self.assertTrue(catalog[0]["tools_supported"])
        self.assertIsNone(catalog[1]["tools_supported"])
        self.assertNotIn("offline-test-secret", json.dumps(catalog))

    def test_catalog_failures_are_safe_and_informative(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(TransportError, "NOUS_API_KEY"):
                fetch_models(default_target())
        error = urllib.error.HTTPError("url", 401, "echo-sensitive-provider-body", {}, None)
        with patch.dict(os.environ, {"NOUS_API_KEY": "offline-test-secret"}), \
             patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(TransportError, "nous_catalog_http_401") as caught:
                fetch_models(default_target())
        self.assertNotIn("echo-sensitive", str(caught.exception))
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"models":[]}'
        with patch.dict(os.environ, {"NOUS_API_KEY": "offline-test-secret"}), \
             patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(TransportError, "invalid_schema"):
                fetch_models(default_target())

    def test_model_selection_is_explicit_unique_and_validated(self):
        template = default_target()
        with self.assertRaisesRegex(ValueError, "--cloud-model"):
            expand_targets(template, None)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            expand_targets(template, ["a", "a"])
        with self.assertRaisesRegex(ValueError, "not found"):
            expand_targets(template, ["unknown"], [{"id": "a"}])
        targets = expand_targets(template, ["a", "b"], [{"id": "a"}, {"id": "b"}])
        self.assertEqual([t["model"] for t in targets], ["a", "b"])
        self.assertEqual({t["api_key_env"] for t in targets}, {"NOUS_API_KEY"})
        self.assertEqual(len({t["name"] for t in targets}), 2)
        self.assertTrue(all(t["metadata"]["catalog_verified"] for t in targets))
        self.assertEqual(template["model"], "")
        self.assertNotIn("metadata", template)

    def test_one_key_authenticates_two_distinct_inference_models(self):
        requests = []
        def fake_http(request, **kwargs):
            requests.append(request)
            stream = MagicMock()
            stream.__enter__.return_value = io.BytesIO(
                b'data: {"model":"resolved","choices":[{"delta":{"content":"Hi"},"finish_reason":"stop"}]}\n\n'
                b'data: {"choices":[],"usage":{"prompt_tokens":2,"completion_tokens":1,"completion_tokens_details":null}}\n\n')
            return stream
        with patch.dict(os.environ, {"NOUS_API_KEY": "offline-test-secret"}), \
             patch("urllib.request.urlopen", side_effect=fake_http):
            for target in expand_targets(default_target(), ["a", "b"]):
                reply = HTTPAdapter(target, 20).call([{"role": "user", "content": "hello"}], cap=128)
                self.assertEqual(reply.text, "Hi")
                self.assertEqual(reply.request_output_limit, 8192)
        self.assertEqual([json.loads(r.data)["model"] for r in requests], ["a", "b"])
        self.assertEqual({r.get_header("Authorization") for r in requests}, {"Bearer offline-test-secret"})
        self.assertTrue(all(r.full_url.endswith("/v1/chat/completions") for r in requests))

    def test_router_reasoning_tool_history_and_provider_metadata(self):
        events = [
            {"model": "resolved", "provider": "upstream", "choices": [{"delta": {
                "reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "Think "}]}}]},
            {"choices": [{"delta": {"reasoning_details": [{"index": 0, "text": "more"}],
                "tool_calls": [{"index": 0, "id": "c1", "function": {"name": "read", "arguments": "{}"}}]},
                "finish_reason": "tool_calls"}]},
            {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 10,
                                      "completion_tokens_details": {"reasoning_tokens": 7}}},
        ]
        reply = consume(events, "chatcompletions", lambda: .1)
        self.assertEqual(reply.provider, "upstream")
        self.assertTrue(reply.reasoning_seen)
        self.assertIsNone(reply.first_text_s)
        self.assertEqual(reply.reasoning_details[0]["text"], "Think more")
        history = [assistant_message(reply, "chatcompletions"), {"role": "tool", "tool_call_id": "c1", "content": "result"}]
        body = request_body({**default_target(), "model": "a"}, history, [], 128)
        self.assertEqual(body["messages"][0]["reasoning_details"], reply.reasoning_details)
        self.assertEqual(body["messages"][0]["tool_calls"][0]["id"], "c1")
        self.assertEqual(body["messages"][1]["tool_call_id"], "c1")

    def test_old_configs_get_opt_in_nous_without_changing_default_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(json.dumps({"targets": [{"name": "existing", "backend": "responses", "model": "m", "base_url": "https://example.test/v1"}]}))
            config = config_load(config_path)
        nous = next(t for t in config["targets"] if t["name"] == "nous")
        self.assertFalse(nous["enabled"])
        args = argparse.Namespace(target=None, cloud_model=None, mock=True, dry_run=False)
        targets, _ = resolve_targets(config, args)
        self.assertEqual([t["name"] for t in targets], ["existing"])
        args.target, args.cloud_model = ["nous"], ["a", "b"]
        with patch("clientbench.cli.fetch_models") as catalog:
            targets, _ = resolve_targets(config, args)
        catalog.assert_not_called()
        self.assertEqual([t["model"] for t in targets], ["a", "b"])

    def test_live_resolution_checks_catalog_before_inference(self):
        config = {"targets": [default_target()], "request_timeout_s": 20}
        args = argparse.Namespace(target=["nous"], cloud_model=["a"], mock=False, dry_run=False)
        with patch("clientbench.cli.fetch_models", return_value=[{"id": "a", "tools_supported": True}]) as catalog:
            targets, _ = resolve_targets(config, args)
        catalog.assert_called_once()
        self.assertEqual(targets[0]["metadata"]["catalog_model"]["id"], "a")

    def test_models_command_filters_catalog(self):
        output = io.StringIO()
        with patch("clientbench.cli.fetch_models", return_value=[
                {"id": "p/flash", "name": "Flash", "tools_supported": True, "context_length": 1000},
                {"id": "p/other", "name": "Other", "tools_supported": None, "context_length": None}]), redirect_stdout(output):
            main(["models", "--target", "nous", "--filter", "flash"])
        self.assertIn("p/flash", output.getvalue())
        self.assertNotIn("p/other", output.getvalue())

    def test_cli_comparison_and_dry_run_do_not_expose_keys_or_call_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "run"
            with patch.dict(os.environ, {"NOUS_API_KEY": "offline-test-secret"}), \
                 patch("urllib.request.urlopen") as http, redirect_stdout(io.StringIO()):
                args = ["run", "--target", "nous", "--cloud-model", "a", "--cloud-model", "b",
                        "--suite", "llm", "--extended", "--repeats", "1", "--warmups", "0", "--out", str(folder)]
                main(args + ["--dry-run"])
                self.assertFalse(folder.exists())
                main(args + ["--mock"])
            http.assert_not_called()
            rows = [json.loads(line) for line in (folder / "results.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 30)
            self.assertEqual({r["requested_model"] for r in rows}, {"a", "b"})
            self.assertTrue(all(r["status"] == "ok" for r in rows))
            for path in folder.rglob("*"):
                if path.is_file():
                    self.assertNotIn("offline-test-secret", path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
