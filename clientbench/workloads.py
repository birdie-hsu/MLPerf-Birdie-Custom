from __future__ import annotations

import csv
import hashlib
import io
import inspect
import json
from pathlib import Path
import random
import shutil
import xml.etree.ElementTree as ET

from .context import DEFAULT_CONTEXT_LENGTH, workload_context_length

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "data" / "mlperf_v2"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_cases(suite="all", extended=False):
    cases = []
    manifest = json.loads((UPSTREAM / "manifest.json").read_text(encoding="utf-8"))
    upstream_label = "MLPerf Client " + manifest["tag"]
    if suite in {"all", "llm", "replay"}:
        for path in sorted((UPSTREAM / "prompts").rglob("*.json")):
            if path.name == "template.json":
                continue
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            context_length = workload_context_length(obj.get("model_config", {}))
            agent = path.parent.name in {"swe_agent", "data_agent"}
            if (suite == "llm" and agent) or (suite == "replay" and not agent):
                continue
            if path.parent.name in {"substantial8k", "data_agent"} and not extended:
                continue
            relative = path.relative_to(UPSTREAM).as_posix()
            if agent:
                steps = []
                prompts = obj["prompts"]
                for i in range(1, len(prompts), 2):
                    prompt = prompts[i]
                    messages = [{"role": k, "content": (path.parent / prompt[k]).read_text(encoding="utf-8")}
                                for k in ("system", "user") if k in prompt]
                    reference = (path.parent / prompts[i + 1]["agent"]).read_text(encoding="utf-8")
                    steps.append({"messages": messages, "reference": reference})
                case = {"id": relative, "category": obj["category"], "mode": "replay",
                        "context_length": context_length,
                        "steps": steps, "output_target": obj["model_config"]["search"]["max_length"],
                        "warmup_messages": [{"role": "system", "content":
                            (path.parent / prompts[0]["system"]).read_text(encoding="utf-8")}],
                        "origin": upstream_label + " exact prompts; portable replay"}
                cases.append(case)
            else:
                for i, prompt in enumerate(obj["prompts"]):
                    if not isinstance(prompt, dict):
                        raise ValueError("Only untemplated object prompts are supported")
                    messages = [{"role": k, "content": prompt[k]} for k in ("system", "user") if k in prompt]
                    cases.append({"id": relative + f"#{i}", "category": obj["category"],
                                  "mode": "llm", "messages": messages,
                                  "context_length": context_length,
                                  "output_target": obj["model_config"]["search"]["max_length"],
                                  "origin": upstream_label + " exact prompt"})
    if suite in {"all", "agent"}:
        for seed in (11, 23, 37):
            for kind in ("swe", "data"):
                cases.append({"id": f"autonomous/{kind}/{seed}", "category":
                              "SWE Agent" if kind == "swe" else "Data Analyst Agent",
                              "mode": "autonomous", "kind": kind, "seed": seed,
                              "context_length": DEFAULT_CONTEXT_LENGTH,
                              "output_target": 2048,
                              "origin": "ClientBench synthetic task inspired by MLPerf Client scenario"})
    for case in cases:
        if case["mode"] == "autonomous":
            case["task_spec_sha256"] = digest_json({
                "initial_code": SWE_START, "hidden_tests": SWE_TESTS,
                "setup": inspect.getsource(setup_workspace),
                "data": inspect.getsource(make_data),
                "evaluation": inspect.getsource(evaluate_agent)})
        case["sha256"] = digest_json(case)
    return cases


SWE_START = '''import random
import math


def bootstrap_mean_ci(data, n_iterations=500, ci_level=0.95, seed=None):
    x = list(data)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x)
                   for _ in range(n_iterations))
    # BUG: confidence level was confused with an upper quantile.
    return sum(x) / len(x), means[0], means[int(ci_level * (len(means) - 1))]
'''

SWE_CORRECT = '''import random
import math


def bootstrap_mean_ci(data, n_iterations=500, ci_level=0.95, seed=None):
    x = list(data)
    if not x or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                    or not math.isfinite(v) for v in x):
        raise ValueError("data must contain finite numbers")
    if not isinstance(n_iterations, int) or isinstance(n_iterations, bool) or n_iterations < 1:
        raise ValueError("n_iterations must be a positive integer")
    if not isinstance(ci_level, (int, float)) or not 0 < ci_level < 1:
        raise ValueError("ci_level must be between zero and one")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(x, k=len(x))) / len(x)
                   for _ in range(n_iterations))
    def percentile(p):
        position = p * (len(means) - 1)
        lo = int(position)
        hi = min(lo + 1, len(means) - 1)
        return means[lo] + (position - lo) * (means[hi] - means[lo])
    tail = (1 - ci_level) / 2
    return sum(x) / len(x), percentile(tail), percentile(1 - tail)
'''

SWE_TESTS = '''import sys
sys.path.insert(0, "/workspace")
import unittest
from bootstrap_mean_ci import bootstrap_mean_ci as f
import random

class Tests(unittest.TestCase):
    def test_empty(self):
        with self.assertRaises(ValueError): f([])
    def test_invalid_data(self):
        for values in ([1, float("nan")], [float("inf")], ["x"], [True]):
            with self.assertRaises(ValueError): f(values)
    def test_iterations(self):
        for n in (0, -2, 2.5, True):
            with self.assertRaises(ValueError): f([1, 2], n_iterations=n)
    def test_confidence(self):
        for c in (0, 1, -1, float("nan")):
            with self.assertRaises(ValueError): f([1, 2], ci_level=c)
    def test_constant(self):
        self.assertEqual(f([3, 3], seed=7), (3, 3, 3))
    def test_interpolation(self):
        x, n, c, seed = [1, 3, 7, 11], 21, 0.8, 19
        rng = random.Random(seed)
        means = sorted(sum(rng.choices(x, k=len(x))) / len(x) for _ in range(n))
        def p(q):
            k = (n - 1) * q; i = int(k); j = min(i + 1, n - 1)
            return means[i] + (k - i) * (means[j] - means[i])
        actual = f(x, n_iterations=n, ci_level=c, seed=seed)
        for a, b in zip(actual, (5.5, p((1-c)/2), p((1+c)/2))):
            self.assertAlmostEqual(a, b)
    def test_repeatability(self):
        self.assertEqual(f([1, 2, 9], seed=32), f([1, 2, 9], seed=32))
    def test_single_iteration(self):
        m, lo, hi = f([4], n_iterations=1, seed=3)
        self.assertEqual((m, lo, hi), (4, 4, 4))

unittest.main()
'''


def make_data(seed):
    rng = random.Random(seed)
    rows = []
    for i in range(40):
        rows.append({"month": str(1 + i % 6), "supplier": ("A", "B", "C")[i % 3],
                     "retail": str(rng.randrange(1, 150)), "warehouse": str(rng.randrange(1, 200))})
    rows += [dict(rows[3]), {"month": "2", "supplier": "A", "retail": "", "warehouse": "10"},
             {"month": "3", "supplier": "B", "retail": "-5", "warehouse": "2"},
             {"month": "4", "supplier": "C", "retail": "oops", "warehouse": "8"}]
    valid = rows[:40]
    totals = {s: sum(int(r["retail"]) + int(r["warehouse"]) for r in valid if r["supplier"] == s)
              for s in ("A", "B", "C")}
    monthly = {str(m): sum(int(r["retail"]) + int(r["warehouse"]) for r in valid if int(r["month"]) == m)
               for m in range(1, 7)}
    expected = {"valid_rows": 40, "duplicate_rows": 1, "invalid_rows": 3,
                "totals_by_supplier": totals, "monthly_total": monthly,
                "top_supplier": max(totals, key=totals.get)}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["month", "supplier", "retail", "warehouse"], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue(), expected


def setup_workspace(case, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if case["mode"] == "replay":
        shutil.copytree(UPSTREAM / "tools_sandbox", root / "tools_sandbox", dirs_exist_ok=True)
        # Official SWE reference uses both bare and tools_sandbox-prefixed paths.
        for path in (root / "tools_sandbox").glob("*.py"):
            shutil.copy2(path, root / path.name)
        return []
    if case["kind"] == "swe":
        (root / "bootstrap_mean_ci.py").write_text(SWE_START, encoding="utf-8")
        user = (
            "Fix bootstrap_mean_ci.py. Preserve function name and argument names. Reject empty data, "
            "non-numeric/bool/non-finite values, non-positive or non-integer/bool n_iterations, and "
            "ci_level outside (0,1) with ValueError. Use random.Random(seed).choices(x,k=len(x)) "
            "once per resample. Return (mean, lower, upper) with percentile bounds at "
            "(1-ci_level)/2 and (1+ci_level)/2, linearly interpolating sorted resample means "
            "using index p*(n_iterations-1). Handle a single sample and a single iteration. "
            "Use tools to read, edit, and execute a verification script before finishing. "
            f"Also verify seed={case['seed']} reproducibility. No external libraries."
        )
    else:
        csv_text, _ = make_data(case["seed"])
        (root / "sales.csv").write_text(csv_text, encoding="utf-8")
        user = (
            "Analyze sales.csv. Write and execute analysis.py (stdlib only), and create report.json. "
            "Deduplicate exact rows before validation. An invalid row has blank, non-numeric, or "
            "negative retail/warehouse. Exclude invalid rows from totals. Required JSON keys: "
            "valid_rows, duplicate_rows, invalid_rows, totals_by_supplier (object), "
            "monthly_total (object keyed by month strings), top_supplier. Total means retail+warehouse. "
            "Use tools to inspect the actual data and run your script before giving a short conclusion."
        )
    return [{"role": "system", "content": "Complete the task by using the provided file and Python tools. "
             "Paths are relative to the task workspace. Verify artifacts before finishing. "
             "No network or external packages are available."}, {"role": "user", "content": user}]


def check_llm(case, text):
    # Formatting checks are deliberately NOT a semantic quality score.
    case_id = case["id"]
    if "data_csv2json" in case_id:
        try:
            objects = json.loads(text)
            expected = list(csv.DictReader(io.StringIO(case["messages"][-1]["content"])))
            if isinstance(objects, dict):
                objects = [objects]
            if not isinstance(objects, list) or len(objects) != len(expected):
                return {"type": "csv_conversion", "pass": False}
            for actual, row in zip(objects, expected):
                normalized = {k.strip(): v.strip() for k, v in row.items()}
                if set(actual) != set(normalized):
                    return {"type": "csv_conversion", "pass": False}
                for key, value in normalized.items():
                    if key == "id" or "TTFT" in key or "TPS" in key:
                        if not isinstance(actual[key], (int, float)) or abs(float(actual[key]) - float(value)) > 1e-5:
                            return {"type": "csv_conversion", "pass": False}
                    elif actual[key] != value:
                        return {"type": "csv_conversion", "pass": False}
            return {"type": "csv_conversion", "pass": True}
        except (ValueError, TypeError, KeyError):
            return {"type": "csv_conversion", "pass": False}
    if "ics2xml" in case_id:
        try:
            ET.fromstring(text)
            passed = True
        except ET.ParseError:
            passed = False
        return {"type": "xml_syntax_only", "pass": passed}
    return {"type": "unscored", "pass": None,
            "note": "Use independent blinded human review or external MMLU/IFEval; nonempty is not accuracy."}


def evaluate_agent(case, root, docker, tool_trace, timeout):
    verified = any(t["call"]["name"] == "execute" and t["result"].get("ok") for t in tool_trace)
    if case["kind"] == "swe":
        with __import__("tempfile").TemporaryDirectory(prefix="clientbench-eval-") as directory:
            (Path(directory) / "test_solution.py").write_text(SWE_TESTS, encoding="utf-8")
            result = docker.execute(root, "python /eval/test_solution.py", timeout=timeout, tests=directory)
        return {"type": "hidden_unit_tests", "pass": result["ok"] and verified,
                "agent_verified": verified, "evaluation": result}
    from .tools import bounded_path
    try:
        path = bounded_path(root, "report.json", True)
        if path.stat().st_size > 500000:
            raise ValueError("report_too_large")
        actual = json.loads(path.read_text(encoding="utf-8"))
        _, expected = make_data(case["seed"])
        checks = {}
        for key, value in expected.items():
            candidate = actual.get(key)
            if isinstance(value, dict):
                checks[key] = (isinstance(candidate, dict) and set(candidate) == set(value)
                    and all(isinstance(candidate[k], (int, float)) and not isinstance(candidate[k], bool)
                            and candidate[k] == v for k, v in value.items()))
            elif isinstance(value, int):
                checks[key] = isinstance(candidate, int) and not isinstance(candidate, bool) and candidate == value
            else:
                checks[key] = isinstance(candidate, str) and candidate == value
        passed = all(checks.values()) and verified
    except (OSError, ValueError, TypeError, AttributeError):
        checks, passed = {}, False
    return {"type": "deterministic_data_oracle", "pass": passed,
            "agent_verified": verified, "checks": checks}
