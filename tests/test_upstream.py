import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from clientbench.report import write_report
from clientbench.workloads import UPSTREAM, load_cases


class Upstream201Tests(unittest.TestCase):
    def test_pinned_201_source_integrity_and_workload_loading(self):
        manifest = json.loads((UPSTREAM / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["version"], "2.0.1")
        self.assertEqual(manifest["tag"], "v2.0.1")
        self.assertEqual(manifest["commit"], "3cbc120e753d06b10c4ddbe53b99c461d83c8248")
        for relative, digest in manifest["files"].items():
            with self.subTest(file=relative):
                self.assertEqual(hashlib.sha256((UPSTREAM / relative).read_bytes()).hexdigest(), digest)
        llm = load_cases("llm", True)
        self.assertEqual(len(llm), 15)
        self.assertEqual(len(load_cases("llm")), 13)
        for case in llm:
            relative, index = case["id"].rsplit("#", 1)
            source = json.loads((UPSTREAM / relative).read_text(encoding="utf-8-sig"))
            prompt = source["prompts"][int(index)]
            self.assertEqual(case["messages"], [{"role": k, "content": prompt[k]}
                                                for k in ("system", "user") if k in prompt])
            self.assertEqual(case["output_target"], source["model_config"]["search"]["max_length"])
            self.assertEqual(case["origin"], "MLPerf Client v2.0.1 exact prompt")
        replay = load_cases("replay", True)
        self.assertEqual(sorted(len(c["steps"]) for c in replay), [3, 4])
        self.assertTrue(all(c["origin"] == "MLPerf Client v2.0.1 exact prompts; portable replay" for c in replay))
        self.assertEqual(len(load_cases("all", True)), 23)

    def test_regenerating_old_report_preserves_200_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = {"tool_version": "0.2.1", "upstream": {"tag": "v2.0",
                "commit": "b0e1d4520142e2cf4ee064a9eeed50a36444b30a"}}
            saved = json.dumps(metadata)
            (root / "run.json").write_text(saved, encoding="utf-8")
            (root / "results.jsonl").write_text("", encoding="utf-8")
            page = write_report(root).read_text(encoding="utf-8")
            payload = json.loads(page.split("const data=", 1)[1].split(";const esc=", 1)[0])
            self.assertEqual(payload["meta"]["upstream"], metadata["upstream"])
            self.assertEqual(payload["meta"]["report_version"], "0.3.0")
            self.assertEqual((root / "run.json").read_text(encoding="utf-8"), saved)


if __name__ == "__main__":
    unittest.main()
