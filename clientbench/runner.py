from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

from . import __version__
from .context import DEFAULT_CONTEXT_LENGTH, resolve_context
from .local import LocalServer
from .mock import MockAdapter, MockDocker
from .tools import DockerPython, HostPython, ToolEngine, extract_reference_calls
from .transport import HTTPAdapter, assistant_message
from .workloads import UPSTREAM, check_llm, evaluate_agent, setup_workspace


class BudgetExceeded(RuntimeError):
    pass


def reply_status(reply):
    # Output-limit termination is not an empty/failed HTTP response. Thinking
    # models can spend their entire generation budget in reasoning_content.
    if reply.finish in {"length", "incomplete"}:
        return "truncated"
    if reply.finish not in {"stop", "completed", "tool_calls"}:
        return "error"
    return "ok" if reply.text or reply.calls else "empty_response"


def is_timing_sample(row):
    if row["status"] == "ok":
        return True
    if row["mode"] == "llm" and row["status"] == "truncated":
        return row.get("measurement_valid", False)
    return False


def sum_usage(replies):
    keys = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens")
    return {key: (sum(r.usage[key] for r in replies) if replies and
                  all(r.usage.get(key) is not None for r in replies) else None) for key in keys}


def estimate_cost(usage, target):
    rates = target.get("price_per_million")
    if not rates or usage.get("input_tokens") is None or usage.get("output_tokens") is None:
        return None
    cached = usage.get("cached_input_tokens")
    if cached is None:
        return None
    return ((usage["input_tokens"] - cached) * rates["input"] + cached * rates["cached_input"]
            + usage["output_tokens"] * rates["output"]) / 1000000


class Runner:
    def __init__(self, config, run_dir, mock=False):
        self.config, self.run_dir, self.mock = config, Path(run_dir), mock
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.requests = 0
        self.target_requests = 0
        self.rows = []

    def save_row(self, row):
        self.rows.append(row)
        with (self.run_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    def call(self, adapter, messages, tools, cap, deadline):
        if self.target_requests >= self.config["max_requests"]:
            raise BudgetExceeded("request_budget_exhausted")
        left = deadline - time.perf_counter()
        if left <= 0:
            raise TimeoutError("task_deadline_exceeded")
        self.requests += 1
        self.target_requests += 1
        return adapter.call(messages, tools=tools, cap=cap, timeout=left)

    def run_case(self, target, case, repeat, phase="measured", warmup_index=0):
        from .tools import TOOL_SCHEMAS
        identity = hashlib.sha256((target["name"] + case["id"]).encode()).hexdigest()[:16]
        key = f"{identity}-r{repeat}-{phase}-{warmup_index}"
        root = self.run_dir / "workspaces" / key
        trace_path = self.run_dir / "traces" / (key + ".json")
        trace_path.parent.mkdir(exist_ok=True)
        row = {"target": target["name"], "requested_model": target["model"],
               "backend": target["backend"], "case_id": case["id"],
               "category": case["category"], "mode": case["mode"], "repeat": repeat,
               "phase": phase, "synthetic": self.mock, "case_sha256": case["sha256"],
               "workload_context_length": case.get("context_length", DEFAULT_CONTEXT_LENGTH),
               "origin": case["origin"], "trace": trace_path.relative_to(self.run_dir).as_posix(),
               "replay_tools_enabled": bool(self.config["replay_tools"] and case["mode"] == "replay"),
               "quality": {"type": "unscored", "pass": None}}
        if target["backend"] == "llamacpp":
            row["local_thinking"] = target.get("local_thinking", "off")
            size, source = resolve_context(target, case)
            managed = bool(target.get("installation") or self.mock)
            row["context_size"] = size if managed else None
            row["context_source"] = target.get("context_source", source) if managed else "external_server"
        if target.get("provider"):
            row["provider"] = target["provider"]
        replies, turns, tool_trace = [], [], []
        started = time.perf_counter()
        try:
            if case["mode"] == "llm":
                messages = case["messages"]
            elif case["mode"] == "replay" and not self.config["replay_tools"]:
                root.mkdir(parents=True, exist_ok=True)
                messages = []
            else:
                messages = setup_workspace(case, root)
            # Like official MLPerf Client, fixed reference tools run on the host;
            # model-authored code (autonomous) stays isolated in Docker.
            runtime = HostPython if case["mode"] == "replay" else DockerPython
            docker = MockDocker(case) if self.mock else runtime(self.config)
            if case["mode"] == "autonomous":
                row["docker_image_id"] = docker.preflight()
            elif case["mode"] == "replay" and self.config["replay_tools"]:
                row["tool_runtime"] = docker.preflight()
            engine = ToolEngine(root, docker)
            adapter = MockAdapter(target, case) if self.mock else HTTPAdapter(target, self.config["request_timeout_s"])
            # Setup, container preflight, and model loading are excluded from task E2E.
            started = time.perf_counter()
            deadline = started + self.config["task_timeout_s"]
            def infer(history, schemas=None, cap=None):
                reply = self.call(adapter, history, schemas, cap or case["output_target"], deadline)
                replies.append(reply)
                turns.append({"input": json.loads(json.dumps(history)), "reply": asdict(reply),
                              "metrics": reply.metrics()})
                return reply

            if phase == "warmup":
                warm = case.get("warmup_messages") or messages
                reply = infer(warm, TOOL_SCHEMAS if case["mode"] == "autonomous" else None, 128)
                outcome = reply_status(reply)
                row["status"] = "warmup_completed" if outcome == "ok" else "warmup_" + outcome
            elif case["mode"] == "llm":
                reply = infer(messages)
                row["quality"] = check_llm(case, reply.text)
                row["status"] = reply_status(reply)
            elif case["mode"] == "replay":
                history = []
                row["status"] = "ok"
                for step in case["steps"]:
                    history.extend(step["messages"])
                    reply = infer(history)
                    if reply.finish not in {"stop", "completed"}:
                        row["status"] = "truncated"
                    elif not reply.text:
                        row["status"] = "empty_response"
                    if self.config["replay_tools"]:
                        for call in extract_reference_calls(step["reference"]):
                            result = engine.run(call, timeout=min(30, max(0.01, deadline - time.perf_counter())))
                            tool_trace.append({"source": "fixed_reference", "call": call, "result": result})
                            if not result["ok"]:
                                row["status"] = "tool_error"
                    # Reproduce fixed-context traffic; actual replies and tool
                    # outputs do not replace the official reference conversation.
                    history.append({"role": "assistant", "content": step["reference"]})
                row["quality"] = {"type": "replay_has_no_task_success_score", "pass": None}
            else:
                row["status"] = "max_turns"
                for _ in range(self.config["max_agent_turns"]):
                    reply = infer(messages, TOOL_SCHEMAS,
                                  min(target.get("max_output_tokens", 2048), case["output_target"]))
                    messages.append(assistant_message(reply, target["backend"]))
                    if reply.finish not in {"stop", "completed", "tool_calls"}:
                        row["status"] = "truncated"
                        break
                    if not reply.calls:
                        row["status"] = "ok" if reply.text else "empty_response"
                        break
                    for call in reply.calls:
                        if time.perf_counter() >= deadline:
                            raise TimeoutError("task_deadline_exceeded")
                        result = engine.run(call, timeout=min(30, deadline - time.perf_counter()))
                        tool_trace.append({"source": "model", "call": call, "result": result})
                        messages.append({"role": "tool", "tool_call_id": call["id"],
                                         "content": json.dumps(result, ensure_ascii=False)})
                task_end = time.perf_counter()
                validation_start = task_end
                row["quality"] = evaluate_agent(case, root, docker, tool_trace, 30)
                row["validation_duration_s"] = time.perf_counter() - validation_start
                row["e2e_s"] = task_end - started
        except Exception as error:
            # Do not echo arbitrary exception contents into public benchmark logs.
            row["status"] = ("budget_exhausted" if isinstance(error, BudgetExceeded) else
                             "timeout" if isinstance(error, TimeoutError) else "error")
            row["error_type"] = type(error).__name__
            if isinstance(error, RuntimeError):
                row["error_code"] = str(error)[:250]
            row["quality"]["pass"] = False if case["mode"] == "autonomous" else None
        row.setdefault("e2e_s", time.perf_counter() - started)
        if replies:
            last = replies[-1]
            row["finish_reason"] = last.finish
            if last.finish in {"length", "incomplete"} and not last.text and last.reasoning_seen:
                row["diagnostic"] = "reasoning_only_output_budget_exhausted"
            row["measurement_valid"] = (row["status"] == "ok" or
                (case["mode"] == "llm" and row["status"] == "truncated" and
                 bool(last.text or last.reasoning_seen)))
        else:
            row["measurement_valid"] = False
        row["llm_request_s"] = sum(r.duration_s for r in replies)
        row["tool_s"] = sum(t["result"].get("tool_duration_s", 0) for t in tool_trace)
        row["overhead_s"] = row["e2e_s"] - row["llm_request_s"] - row["tool_s"] if not self.mock else None
        row["turns"] = len(replies)
        row["tool_calls"] = len(tool_trace)
        row["failed_tool_calls"] = sum(not t["result"].get("ok", False) for t in tool_trace)
        row["usage"] = sum_usage(replies)
        row["estimated_usd"] = estimate_cost(row["usage"], target) if not self.mock else None
        row["resolved_model"] = replies[-1].resolved_model if replies else None
        row["resolved_provider"] = replies[-1].provider if replies else None
        row["turn_metrics"] = [r.metrics() for r in replies]
        row["output_target_native_tokens"] = case["output_target"]
        row["configured_output_budget"] = target.get("max_output_tokens")
        row["request_output_budget"] = replies[-1].request_output_limit if replies else None
        trace_path.write_text(json.dumps({"row": row, "turns": turns, "tools": tool_trace},
                                        ensure_ascii=False, indent=2), encoding="utf-8")
        self.save_row(row)
        print(f"{target['name']} | {case['category']} | {phase} | {row['status']}", flush=True)
        return row

    def run(self, targets, cases):
        manifest = json.loads((UPSTREAM / "manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["files"].items():
            actual = hashlib.sha256((UPSTREAM / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError("upstream_file_integrity_failure:" + relative)
        metadata = {"tool_version": __version__, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "synthetic": self.mock, "official_mlperf_result": False,
                    "host": {"os": platform.platform(), "python": platform.python_version(),
                             "processor": platform.processor()},
                    "config": self.config, "targets": targets,
                    "upstream": manifest, "cases": [{k: c[k] for k in
                        ("id", "sha256", "category", "mode", "origin", "context_length") if k in c} for c in cases],
                    "cache_policy": "Local cache_prompt false by default; cloud cache observed through usage, not disabled.",
                    "execution_order": [], "server_sessions": []}
        metadata["harness_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in sorted(Path(__file__).parent.glob("*.py"))}
        (self.run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        rng = random.Random(self.config["order_seed"])
        order = list(targets)
        rng.shuffle(order)
        # Batch by model and context: llama-server fixes n_ctx at startup.
        # Each capacity gets its own server, unloaded before the next starts.
        for target in order:
            self.target_requests = 0
            warmed = set()
            groups = {}
            managed = target["backend"] == "llamacpp" and (target.get("installation") or self.mock)
            for case in cases:
                size = resolve_context(target, case)[0] if managed else None
                groups.setdefault(size, []).append(case)
            batches = list(groups.items())
            rng.shuffle(batches)
            try:
                for size, batch in batches:
                    active = deepcopy(target)
                    if managed:
                        active["context_size"] = size
                        active["context_source"] = resolve_context(target, batch[0])[1]
                    manager = (LocalServer(active, self.run_dir) if active.get("installation") and not self.mock
                               else nullcontext())
                    session = None
                    if managed:
                        session = {"target": active["name"], "context_size": size,
                                   "context_source": active["context_source"],
                                   "case_ids": [c["id"] for c in batch],
                                   "synthetic": self.mock, "metadata": active.setdefault("metadata", {})}
                        metadata["server_sessions"].append(session)
                    first_row = len(self.rows)
                    try:
                        with manager:
                            for repeat in range(self.config["repeats"]):
                                shuffled = list(batch)
                                rng.shuffle(shuffled)
                                for case in shuffled:
                                    metadata["execution_order"].append([active["name"], case["id"], repeat])
                                    if case["id"] not in warmed:
                                        for w in range(self.config["warmups"]):
                                            self.run_case(active, case, repeat, "warmup", w)
                                        warmed.add(case["id"])
                                    self.run_case(active, case, repeat)
                                    if self.target_requests >= self.config["max_requests"]:
                                        raise BudgetExceeded("request_budget_exhausted")
                        if session is not None:
                            session["status"] = "completed"
                    except BudgetExceeded:
                        if session is not None:
                            session["status"] = "budget_exhausted"
                        raise
                    except Exception as error:
                        if session is not None:
                            session["status"] = "load_error"
                        # Only this capacity failed; retain other groups' results.
                        recorded = {(r["case_id"], r["repeat"]) for r in self.rows[first_row:]
                                    if r["phase"] == "measured"}
                        for repeat in range(self.config["repeats"]):
                            for case in batch:
                                if (case["id"], repeat) in recorded:
                                    continue
                                self.save_row({"target": active["name"], "requested_model": active["model"],
                                    "backend": active["backend"],
                                    "case_id": case["id"], "category": case["category"], "mode": case["mode"],
                                    "repeat": repeat, "phase": "measured", "synthetic": self.mock,
                                    "workload_context_length": case.get("context_length", DEFAULT_CONTEXT_LENGTH),
                                    "context_size": size, "context_source": active.get("context_source"),
                                    "status": "load_error", "error_code": str(error)[:250],
                                    "quality": {"type": "not_executed", "pass": False if case["mode"] == "autonomous" else None},
                                    "e2e_s": None, "turn_metrics": [], "usage": {}, "estimated_usd": None})
                    finally:
                        (self.run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            except BudgetExceeded:
                metadata["stopped_reason"] = "A target reached its per-target request budget; report is partial"
                metadata.setdefault("budget_exhausted_targets", []).append(target["name"])
            (self.run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        metadata["request_count"] = self.requests
        metadata["completed_rows"] = len(self.rows)
        (self.run_dir / "run.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return self.rows
