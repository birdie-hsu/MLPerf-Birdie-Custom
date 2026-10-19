"""Uniform file tools plus Docker-only execution of model-authored Python."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid

TOOL_SCHEMAS = [
    {"name": "read", "description": "Read a UTF-8 file relative to the task workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                    "required": ["path"], "additionalProperties": False}, "strict": True},
    {"name": "write", "description": "Write a complete UTF-8 file in the task workspace.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                      "content": {"type": "string"}}, "required": ["path", "content"],
                    "additionalProperties": False}, "strict": True},
    {"name": "patch", "description": "Replace exactly one occurrence of old text with new text.",
     "parameters": {"type": "object", "properties": {"path": {"type": "string"},
                      "old": {"type": "string"}, "new": {"type": "string"}},
                    "required": ["path", "old", "new"], "additionalProperties": False}, "strict": True},
    {"name": "execute", "description": "Run Python in an isolated container, e.g. python analysis.py. No shell syntax.",
     "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                    "required": ["command"], "additionalProperties": False}, "strict": True},
]


def bounded_path(root, name, must_exist=False):
    if not isinstance(name, str) or not name or len(name) > 1000:
        raise ValueError("invalid_path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or ":" in name:
        raise ValueError("path_outside_workspace")
    base = Path(root).resolve()
    candidate = base / relative
    for part in [candidate, *candidate.parents]:
        if part == base:
            break
        if part.is_symlink() or part.is_junction():
            raise ValueError("symlink_not_allowed")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise ValueError("path_outside_workspace")
    if must_exist and not resolved.is_file():
        raise ValueError("file_not_found")
    return resolved


class DockerPython:
    def __init__(self, config):
        self.config = config
        self.image_id = None

    def preflight(self):
        if not shutil.which("docker"):
            raise RuntimeError("docker_not_found")
        check = subprocess.run(["docker", "image", "inspect", self.config["docker_image"],
                                "--format", "{{.Id}}"], capture_output=True, text=True, timeout=20,
                                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
        if check.returncode != 0:
            raise RuntimeError("docker_image_not_available; pull the configured image before running agents")
        self.image_id = check.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.image_id):
            raise RuntimeError("docker_image_identity_unavailable")
        return self.image_id

    def execute(self, root, command, timeout=30, tests=None):
        if not self.image_id:
            self.preflight()
        tokens = shlex.split(command, posix=True)
        if not tokens or tokens[0] not in {"python", "python3"}:
            raise ValueError("only_python_commands_are_supported")
        # Reject shell operators even though shell=False already prevents them.
        if any(t in {";", "&&", "||", "|", ">", "<"} for t in tokens):
            raise ValueError("shell_syntax_not_supported")
        name = "clientbench-" + uuid.uuid4().hex
        args = ["docker", "run", "--rm", "--name", name, "--pull=never", "--network=none",
                "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges",
                "--memory", self.config.get("docker_memory", "1g"), "--cpus",
                str(self.config.get("docker_cpus", "1")), "--pids-limit", "64",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--mount",
                f"type=bind,source={Path(root).resolve()},target=/workspace",
                "--workdir", "/workspace", "--env", "PYTHONDONTWRITEBYTECODE=1"]
        if tests:
            args.extend(["--mount", f"type=bind,source={Path(tests).resolve()},target=/eval,readonly"])
        args.extend([self.image_id, "python", *tokens[1:]])
        start = time.perf_counter()
        with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
            try:
                process = subprocess.run(args, stdout=stdout, stderr=stderr, timeout=timeout,
                                         creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15,
                               creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
                exit_code = -1
            stdout.seek(0)
            stderr.seek(0)
            return {"ok": exit_code == 0, "exit_code": exit_code,
                    "stdout": stdout.read(32000).decode("utf-8", errors="replace"),
                    "stderr": stderr.read(8000).decode("utf-8", errors="replace"),
                    "duration_s": time.perf_counter() - start}


class ToolEngine:
    def __init__(self, root, docker):
        self.root = Path(root).resolve()
        self.docker = docker

    def run(self, call, timeout=30):
        start = time.perf_counter()
        try:
            name = call["name"]
            args = call["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            schema = next(t for t in TOOL_SCHEMAS if t["name"] == name)["parameters"]
            if set(args) != set(schema["required"]) or any(not isinstance(v, str) for v in args.values()):
                raise ValueError("invalid_tool_arguments")
            if name == "read":
                path = bounded_path(self.root, args["path"], True)
                with path.open("r", encoding="utf-8-sig") as stream:
                    text = stream.read(32001)
                result = {"ok": True, "content": text[:32000], "truncated": len(text) > 32000}
            elif name in {"write", "patch"}:
                path = bounded_path(self.root, args["path"], name == "patch")
                if name == "patch":
                    if path.stat().st_size > 500000:
                        raise ValueError("file_too_large_to_patch")
                    content = path.read_text(encoding="utf-8")
                    if not args["old"] or content.count(args["old"]) != 1:
                        raise ValueError("patch_requires_one_exact_match")
                    content = content.replace(args["old"], args["new"], 1)
                else:
                    content = args["content"]
                if len(content.encode()) > 500000:
                    raise ValueError("write_too_large")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                result = {"ok": True, "path": args["path"]}
            elif name == "execute":
                result = self.docker.execute(self.root, args["command"], timeout=timeout)
            else:
                raise ValueError("unknown_tool")
        except (ValueError, TypeError, KeyError, StopIteration, OSError, RuntimeError) as error:
            result = {"ok": False, "error": str(error)[:500] if isinstance(error, (ValueError, RuntimeError))
                      else type(error).__name__}
        result["tool_duration_s"] = time.perf_counter() - start
        return result


def extract_reference_calls(text):
    """Parse balanced JSON fragments in the official agent reference responses."""
    decoder = json.JSONDecoder()
    calls, index = [], 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            item, count = decoder.raw_decode(text[start:])
            index = start + count
        except ValueError:
            index = start + 1
            continue
        if isinstance(item, dict) and item.get("type") == "tool_use":
            calls.append({"id": item.get("id", str(len(calls))), "name": item["name"],
                          "arguments": item["input"]})
    return calls
