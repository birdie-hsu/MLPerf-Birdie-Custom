import csv
import json
from pathlib import Path
import tempfile
import unittest

from clientbench.report import summarize, write_report
from clientbench.transport import Reply, consume


class ReportTimingsTests(unittest.TestCase):
    def row(self, **changes):
        return {"target": "local", "mode": "llm", "category": "Summarization",
                "case_id": "fixture", "repeat": 0, "phase": "measured", "status": "ok",
                "e2e_s": 9, "quality": {"pass": None}, "estimated_usd": None,
                "turn_metrics": [], **changes}

    def test_stream_preserves_server_prefill_units_cache_and_reasoning_generation(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": "thinking"}}]},
            {"choices": [{"delta": {"content": "answer"}, "finish_reason": "stop"}],
             "usage": {"prompt_tokens": 1200, "completion_tokens": 128},
             "timings": {"cache_n": 1000, "prompt_n": 200, "prompt_ms": 250,
                         "prompt_per_second": 800, "predicted_per_second": 62.07}},
        ]
        metrics = consume(events, "llamacpp", lambda: 9).metrics()
        self.assertEqual(metrics["prefill_s"], .25)
        self.assertEqual(metrics["prefill_tps"], 800)
        self.assertEqual(metrics["prefill_tokens"], 200)
        self.assertEqual(metrics["cached_prompt_tokens"], 1000)
        self.assertEqual(metrics["generation_tps"], 62.07)
        self.assertEqual(metrics["server_decode_tps"], 62.07)
        self.assertIsNone(metrics["native_tps_estimate"])

    def test_missing_server_timings_stay_missing_and_zero_time_is_preserved(self):
        cloud = Reply(usage={"input_tokens": 500, "output_tokens": 128}, first_text_s=2).metrics()
        for name in ("prefill_s", "prefill_tps", "generation_tps"):
            self.assertIsNone(cloud[name])
        cached = Reply(native_timings={"prompt_ms": 0, "prompt_n": 0, "cache_n": 500}).metrics()
        self.assertEqual(cached["prefill_s"], 0)
        self.assertIsNone(cached["prefill_tps"])

    def test_summary_uses_valid_turns_excludes_warmup_failure_and_missing_cloud(self):
        rows = [
            self.row(turn_metrics=[{"prefill_s": .1, "prefill_tps": 100, "generation_tps": 10},
                                   {"prefill_s": .5, "prefill_tps": 300, "generation_tps": 30}]),
            self.row(status="truncated", measurement_valid=True,
                     turn_metrics=[{"prefill_s": .3, "prefill_tps": 200, "server_decode_tps": 20}]),
            self.row(phase="warmup", turn_metrics=[{"prefill_s": 999, "prefill_tps": 999}]),
            self.row(status="error", turn_metrics=[{"prefill_s": 999, "prefill_tps": 999}]),
            self.row(target="cloud", turn_metrics=[{"ttft_visible_s": 2}]),
        ]
        summaries = {r["target"]: r for r in summarize(rows)}
        local = summaries["local"]
        self.assertEqual(local["prefill_p50_s"], .3)
        self.assertEqual(local["prefill_tps_p50"], 200)
        self.assertEqual(local["generation_tps_p50"], 20)
        self.assertEqual(local["server_decode_tps_p50"], 20)
        for name in ("prefill_p50_s", "prefill_tps_p50", "generation_tps_p50"):
            self.assertIsNone(summaries["cloud"][name])

    def test_historical_report_recovers_trace_timings_without_rewriting_raw_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [self.row(trace="trace.json", turn_metrics=[{"server_decode_tps": 62.07}]),
                    self.row(target="cloud", trace="missing.json", turn_metrics=[{}])]
            original = "\n".join(json.dumps(r) for r in rows) + "\n"
            (root / "results.jsonl").write_text(original, encoding="utf-8")
            (root / "run.json").write_text('{"version":"0.2.0"}', encoding="utf-8")
            (root / "trace.json").write_text(json.dumps({"turns": [{"reply": {"native_timings": {
                "prompt_ms": 2586.094, "prompt_per_second": 2938.408, "prompt_n": 7599,
                "cache_n": 0, "predicted_per_second": 62.07}}}]}), encoding="utf-8")
            report = write_report(root)
            self.assertEqual((root / "results.jsonl").read_text(encoding="utf-8"), original)
            summary = {r["target"]: r for r in json.loads((root / "summary.json").read_text(encoding="utf-8"))}
            self.assertAlmostEqual(summary["local"]["prefill_p50_s"], 2.586094)
            self.assertEqual(summary["local"]["prefill_tps_p50"], 2938.408)
            self.assertEqual(summary["local"]["generation_tps_p50"], 62.07)
            self.assertIsNone(summary["cloud"]["prefill_tps_p50"])
            with (root / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
                metric = json.loads(next(csv.DictReader(stream))["turn_metrics"])[0]
            self.assertEqual(metric["prefill_tokens"], 7599)
            self.assertEqual(metric["prefill_tps"], 2938.408)
            with (root / "summary.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertIn("prefill_p50_s", csv.DictReader(stream).fieldnames)
            page = report.read_text(encoding="utf-8")
            for label in ("Prefill p50 秒", "Prefill TPS p50", "Generation TPS p50"):
                self.assertIn(label, page)
            self.assertNotIn("Server TPS", page)
            self.assertIn('colspan="15"', page)


if __name__ == "__main__":
    unittest.main()
