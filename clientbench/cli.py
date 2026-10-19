from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import urllib.parse

from .local import discover_installations, expand_local_targets, inventory, apply_local_options
from .nous import default_target, expand_targets as expand_nous_targets, fetch_models
from .report import write_report
from .runner import Runner
from .workloads import ROOT, load_cases


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def config_load(path):
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    defaults = {"repeats": 3, "warmups": 1, "order_seed": 20261006,
                "request_timeout_s": 180, "task_timeout_s": 600,
                "max_agent_turns": 12, "max_requests": 1000, "replay_tools": False,
                "docker_image": "python:3.12-slim"}
    config = {**defaults, **config}
    if not any(t.get("name") == "nous" for t in config["targets"]):
        config["targets"].append(default_target())
    for field in ("repeats", "max_agent_turns", "max_requests"):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(field + " must be a positive integer")
    if not isinstance(config["warmups"], int) or config["warmups"] < 0:
        raise ValueError("warmups must be nonnegative")
    for field in ("request_timeout_s", "task_timeout_s"):
        if not isinstance(config[field], (float, int)) or config[field] <= 0:
            raise ValueError(field + " must be positive")
    names = set()
    for target in config["targets"]:
        if target["name"] in names:
            raise ValueError("target names must be unique")
        names.add(target["name"])
        if not target["name"] or (not target["model"] and target.get("provider") != "nous"):
            raise ValueError("name and model must be nonempty")
        if target.get("max_output_tokens", 2048) < 1:
            raise ValueError("max_output_tokens must be positive")
        rates = target.get("price_per_million")
        if rates is not None:
            if not isinstance(rates, dict) or not {"input", "cached_input", "output"}.issubset(rates):
                raise ValueError("price_per_million requires input, cached_input, and output rates")
            if any(not isinstance(rates[k], (int, float)) or rates[k] < 0
                   for k in ("input", "cached_input", "output")):
                raise ValueError("token prices must be nonnegative numbers")
        if target["backend"] not in {"responses", "llamacpp", "chatcompletions"}:
            raise ValueError("unsupported backend")
        if target["backend"] == "llamacpp":
            target.setdefault("local_thinking", "off")
            if target["local_thinking"] not in {"off", "on", "auto"}:
                raise ValueError("local_thinking must be off, on, or auto")
        parsed = urllib.parse.urlsplit(target["base_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("base_url must be HTTP(S) without embedded credentials")
        if target["backend"] == "responses" and target.get("reasoning_effort", "low") not in {
            "low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Shared Sol/Luna profile requires low/medium/high/xhigh/max reasoning")
    return config


def resolve_targets(config, args):
    targets = [t for t in config["targets"] if
               (t["name"] in args.target if args.target else t.get("enabled", True))]
    if not targets:
        raise ValueError("no configured targets match --target")
    if args.target and set(args.target) - {t["name"] for t in targets}:
        raise ValueError("unknown configured target in --target")
    if args.cloud_model and not any(t.get("provider") == "nous" for t in targets):
        raise ValueError("--cloud-model requires --target nous")
    discovery = None
    expanded = []
    for target in targets:
        if target.get("provider") == "nous":
            # Expand first to reject a missing selection before any network call.
            chosen = expand_nous_targets(target, args.cloud_model)
            if not (args.mock or args.dry_run):
                catalog = fetch_models(target, config["request_timeout_s"])
                chosen = expand_nous_targets(target, args.cloud_model, catalog)
            expanded.extend(chosen)
        elif target["backend"] == "llamacpp" and target.get("discover_models", False):
            if args.mock:
                # Offline testing never inventories user drives or launches models.
                expanded.append({**target, "name": "local/MOCK-4B", "model": "MOCK-4B"})
                expanded.append({**target, "name": "local/MOCK-35B", "model": "MOCK-35B"})
                continue
            discovery = discover_installations(args.llama_root or target.get("llama_roots"),
                args.search_root or target.get("search_roots"), target.get("discovery_max_dirs", 20000))
            inv = inventory(discovery["installations"])
            discovery.update(inv)
            if args.model_filter:
                inv["models"] = [m for m in inv["models"] if args.model_filter.lower() in m["name"].lower()]
            if not inv["models"]:
                raise ValueError("No loadable GGUF candidates found. Use discover --llama-root PATH; check server and models folders.")
            expanded.extend(expand_local_targets(target, inv))
        else:
            expanded.append(target)
    return expanded, discovery


def main(argv=None):
    parser = argparse.ArgumentParser(description="ClientBench: portable MLPerf-inspired LLM/agent benchmark")
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover", help="Find llama.cpp and enumerate models without inference")
    discover.add_argument("--llama-root", action="append", help="Exact installation root; repeatable")
    discover.add_argument("--search-root", action="append", help="Search under a custom directory; repeatable")
    discover.add_argument("--out", help="Optional inventory JSON destination")
    listing = sub.add_parser("list", help="List pinned cases")
    listing.add_argument("--suite", choices=["all", "llm", "replay", "agent"], default="all")
    listing.add_argument("--extended", action="store_true")
    models = sub.add_parser("models", help="List Nous router model IDs without inference")
    models.add_argument("--config", default=str(ROOT / "config.example.json"))
    models.add_argument("--target", default="nous")
    models.add_argument("--filter", help="Substring in model ID or name")
    models.add_argument("--json", action="store_true", help="Print normalized catalog JSON")
    models.add_argument("--out", help="Save normalized catalog JSON")
    run = sub.add_parser("run")
    run.add_argument("--config", default=str(ROOT / "config.example.json"))
    run.add_argument("--suite", choices=["all", "llm", "replay", "agent"], default="all")
    run.add_argument("--extended", action="store_true", help="Include 8K LLM and Data Analyst replay")
    run.add_argument("--target", action="append", help="Configured target name; repeatable")
    run.add_argument("--cloud-model", action="append", help="Exact Nous model ID; repeat to compare models with one key")
    run.add_argument("--llama-root", action="append")
    run.add_argument("--search-root", action="append")
    run.add_argument("--model-filter", help="Optional substring filter, e.g. 35B")
    run.add_argument("--local-device", help="Exact llama.cpp device ID, or amd to select the sole AMD Radeon GPU")
    run.add_argument("--context-size", type=positive, help="Override local context size for all models")
    run.add_argument("--gpu-layers", type=nonnegative, help="Override local GPU layers; 0 for CPU weights")
    run.add_argument("--local-thinking", choices=["off", "on", "auto"],
                     help="Local baseline defaults to off; on/auto use max_output_tokens including reasoning")
    run.add_argument("--case-filter", help="Optional substring in case ID")
    run.add_argument("--repeats", type=positive)
    run.add_argument("--warmups", type=int)
    run.add_argument("--replay-tools", action="store_true")
    run.add_argument("--mock", action="store_true", help="Synthetic offline validation; no APIs or models")
    run.add_argument("--dry-run", action="store_true", help="Inventory and execution plan only")
    run.add_argument("--out", help="New output directory")
    report = sub.add_parser("report", help="Regenerate HTML and CSV from a saved run")
    report.add_argument("run_dir")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            result = discover_installations(args.llama_root, args.search_root)
            result.update(inventory(result["installations"]))
            content = json.dumps(result, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(content, encoding="utf-8")
            print(content)
            return
        if args.command == "list":
            for case in load_cases(args.suite, args.extended):
                print(f"{case['mode']:12} {case['category']:25} {case['id']}")
            return
        if args.command == "report":
            print(write_report(args.run_dir))
            return
        config = config_load(args.config)
        if args.command == "models":
            target = next((t for t in config["targets"] if t["name"] == args.target and t.get("provider") == "nous"), None)
            if target is None:
                raise ValueError("models requires a configured Nous provider target")
            catalog = fetch_models(target, config["request_timeout_s"])
            if args.filter:
                catalog = [m for m in catalog if args.filter.lower() in (m["id"] + " " + m["name"]).lower()]
            content = json.dumps(catalog, ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(content, encoding="utf-8")
            if args.json:
                print(content)
            else:
                for model in catalog:
                    tools = "yes" if model["tools_supported"] is True else "no" if model["tools_supported"] is False else "unknown"
                    print(f"{model['id']} | tools: {tools} | context: {model['context_length'] or 'unknown'}")
                print(f"{len(catalog)} text-model candidates. Catalog listing does not run inference.")
            return
        if args.local_thinking:
            for target in config["targets"]:
                if target["backend"] == "llamacpp":
                    target["local_thinking"] = args.local_thinking
        if args.repeats:
            config["repeats"] = args.repeats
        if args.warmups is not None:
            if args.warmups < 0:
                raise ValueError("warmups must be nonnegative")
            config["warmups"] = args.warmups
        if args.replay_tools:
            config["replay_tools"] = True
        targets, discovery = resolve_targets(config, args)
        apply_local_options(targets, args.local_device, args.context_size, args.gpu_layers, args.mock)
        if args.local_thinking:
            for target in targets:
                if target["backend"] == "llamacpp":
                    target["local_thinking"] = args.local_thinking
        cases = load_cases(args.suite, args.extended)
        if args.case_filter:
            cases = [case for case in cases if args.case_filter.lower() in case["id"].lower()]
        if not cases:
            raise ValueError("no matching cases")
        if any(c["mode"] == "autonomous" for c in cases):
            for target in targets:
                model = target.get("metadata", {}).get("catalog_model", {})
                if model.get("tools_supported") is False:
                    raise ValueError("Selected Nous model does not advertise tool calling: " + target["model"])
        estimate = len(targets) * sum((len(c.get("steps", [])) or
                                      (config["max_agent_turns"] if c["mode"] == "autonomous" else 1))
                                      * config["repeats"] + config["warmups"] for c in cases)
        plan = {"targets": [t["name"] for t in targets], "cases": len(cases),
                "repeats": config["repeats"], "warmups_per_case": config["warmups"],
                "worst_case_requests": estimate, "max_requests_per_target": config["max_requests"],
                "discovery": discovery, "synthetic": args.mock,
                "local_hardware": [{"target": t["name"], "context_size": t.get("context_size"),
                                    "gpu_layers": t.get("gpu_layers"), "server_args": t.get("server_args", []),
                                    "selected_device": t.get("metadata", {}).get("selected_device"),
                                    "gpu": t.get("metadata", {}).get("gpu")}
                                   for t in targets if t["backend"] == "llamacpp"],
                "execution": "Sequential models, one server at a time. No automatic retries."}
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return
        folder = args.out or str(ROOT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S-%fZ"))
        run_dir = Path(folder)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("--out must be new or empty; old results are never appended")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        runner = Runner(config, run_dir, args.mock)
        rows = runner.run(targets, cases)
        print("Report:", write_report(run_dir, rows))
    except (ValueError, OSError, RuntimeError) as error:
        print("ClientBench:", str(error), file=sys.stderr)
        raise SystemExit(2)
