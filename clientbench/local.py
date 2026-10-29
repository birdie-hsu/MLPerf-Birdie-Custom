"""Bounded llama.cpp discovery, GGUF inventory, and sequential server lifecycle."""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
import urllib.request

SERVER_NAMES = ("llama-server.exe", "llama-server")
SKIP_DIRS = {".git", "node_modules", "__pycache__", "windows", "program files",
             "program files (x86)", "$recycle.bin", "system volume information",
             "appdata", ".cache", ".codex", ".venv", "venv", "winSxS".lower()}
SHARD = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def default_search_roots():
    home = Path.home()
    roots = [(home, 3), (Path.cwd(), 5), (home / "Documents", 6),
             (home / "Desktop", 5), (home / "Downloads", 5)]
    # Root-level llama.cpp installations on other fixed drives are common.
    # Bound traversal depth; do not recursively sweep every user or system folder.
    if os.name == "nt":
        mask = ctypes.windll.kernel32.GetLogicalDrives()
        for i in range(26):
            drive = f"{chr(65 + i)}:\\"
            if mask & (1 << i) and ctypes.windll.kernel32.GetDriveTypeW(drive) == 3:
                roots.append((Path(drive), 3))
    else:
        roots.extend([(home, 3), (Path("/opt"), 3), (Path("/usr/local"), 3)])
    return roots


def find_servers(root):
    found = []
    for name in SERVER_NAMES:
        for rel in (name, "build/bin/" + name, "build/bin/Release/" + name,
                    "bin/" + name, "bin/Release/" + name):
            path = root / rel
            if path.is_file():
                found.append(path.resolve())
    return list(dict.fromkeys(found))


def discover_installations(explicit_roots=None, search_roots=None, max_dirs=20000):
    """An installation must contain models/ and a server executable.

    Explicit --llama-root wins. Symlink/reparse directories are not traversed.
    Returns inventory and search diagnostics instead of claiming full-disk coverage.
    """
    result, seen, warnings = [], {}, []
    examined = 0
    if explicit_roots:
        queue = [(Path(p).expanduser().resolve(), 0) for p in explicit_roots]
    elif search_roots:
        queue = [(Path(p).expanduser().resolve(), 8) for p in search_roots]
    else:
        queue = default_search_roots()
        command = shutil.which("llama-server") or shutil.which("llama-server.exe")
        if command:
            for ancestor in list(Path(command).resolve().parents)[:4]:
                if (ancestor / "models").is_dir():
                    queue.insert(0, (ancestor, 0))
    while queue and examined < max_dirs:
        root, depth = queue.pop()
        key = os.path.normcase(str(root))
        if seen.get(key, -1) >= depth or not root.is_dir():
            continue
        seen[key] = depth
        examined += 1
        models = root / "models"
        if models.is_dir():
            servers = find_servers(root)
            if servers and not any(i["root"] == str(root.resolve()) for i in result):
                result.append({"root": str(root.resolve()), "models": str(models.resolve()),
                               "server": str(servers[0]), "server_candidates": list(map(str, servers))})
            elif "llama" in root.name.lower() or explicit_roots:
                warnings.append(f"models exists but llama-server missing: {root}")
        if depth <= 0:
            continue
        try:
            with os.scandir(root) as entries:
                for entry in entries:
                    if entry.name.lower() in SKIP_DIRS or entry.name.startswith("."):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False) and not Path(entry.path).is_junction():
                            queue.append((Path(entry.path), depth - 1))
                    except OSError:
                        pass
        except OSError:
            warnings.append(f"cannot read directory: {root}")
    if queue:
        warnings.append(f"directory search capped at {max_dirs}; use --llama-root for exact discovery")
    return {"installations": sorted(result, key=lambda i: i["root"]),
            "examined_directories": examined, "warnings": warnings}


def inventory(installations):
    models, excluded = [], []
    for installation in installations:
        base = Path(installation["models"])
        for current, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = sorted(d for d in dirs if not (Path(current) / d).is_symlink()
                              and not (Path(current) / d).is_junction())
            for name in sorted(files):
                path = Path(current) / name
                if path.suffix.lower() != ".gguf" or path.is_symlink():
                    continue
                match = SHARD.match(name)
                if match and int(match[2]) != 1:
                    excluded.append({"path": str(path), "reason": "secondary_shard"})
                    continue
                # Projectors are GGUF assets, not standalone text models.
                if name.lower().startswith(("mmproj", "projector")):
                    excluded.append({"path": str(path), "reason": "multimodal_projector"})
                    continue
                parts = [path]
                if match:
                    parts = [path.with_name(f"{match[1]}-{i:05d}-of-{int(match[3]):05d}.gguf")
                             for i in range(1, int(match[3]) + 1)]
                missing = [str(p) for p in parts if not p.is_file()]
                relative = path.relative_to(base).as_posix()
                identity = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:10]
                models.append({**installation, "model_path": str(path.resolve()),
                               "parts": list(map(str, parts)), "missing_parts": missing,
                               "size_bytes": sum(p.stat().st_size for p in parts if p.is_file()),
                               "name": relative, "target_name": f"local/{relative}/{identity}"})
    return {"models": sorted(models, key=lambda m: (m["root"], m["name"])), "excluded": excluded}


def expand_local_targets(template, inv):
    expanded = []
    for model in inv["models"]:
        target = {**template, "name": model["target_name"], "model": model["target_name"],
                  "installation": model, "backend": "llamacpp"}
        target["metadata"] = {**template.get("metadata", {}),
                              "model_path": model["model_path"], "gguf_size_bytes": model["size_bytes"]}
        overrides = template.get("model_overrides", {})
        override = overrides.get(model["name"], {})
        allowed = {"context_size", "gpu_layers", "threads", "load_timeout_s", "server_args",
                   "max_output_tokens", "temperature", "seed", "extra_body", "local_thinking"}
        if set(override) - allowed:
            raise ValueError("model_overrides contains unsupported fields")
        target.update(override)
        if target.get("local_thinking", "off") not in {"off", "on", "auto"}:
            raise ValueError("model local_thinking must be off, on, or auto")
        expanded.append(target)
    return expanded


VENDORS = {"amd": lambda name: "AMD" in name and "RADEON" in name,
           "nvidia": lambda name: "NVIDIA" in name,
           "intel": lambda name: "INTEL" in name}


def select_device(server, requested):
    """Resolve a device against this binary; never guess a Vulkan index."""
    try:
        result = subprocess.run([server, "--list-devices"], capture_output=True,
                                text=True, errors="replace", timeout=30,
                                creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("llama-server device enumeration timed out") from error
    if result.returncode:
        raise RuntimeError("llama-server device enumeration failed: " + (result.stderr or result.stdout)[-2000:])
    output = result.stdout + "\n" + result.stderr
    devices = [{"id": match[1], "name": match[2].strip()}
               for line in output.splitlines()
               if (match := re.match(r"^\s*([\w.-]+\d+):\s*(.+)$", line))]
    vendor = VENDORS.get(requested.lower())
    matches = [d for d in devices if
               (vendor(d["name"].upper()) if vendor else d["id"].lower() == requested.lower())]
    available = "; ".join(f"{d['id']}: {d['name']}" for d in devices) or "none"
    if len(matches) != 1:
        raise ValueError(f"--local-device {requested}: expected one matching device, found {len(matches)}. "
                         f"Available devices: {available}. Use an exact device ID (e.g. CUDA0, Vulkan1, SYCL0) "
                         "when several GPUs match.")
    return {**matches[0], "enumeration": output}


def apply_local_options(targets, device=None, context_size=None, gpu_layers=None, mock=False):
    """Apply CLI overrides after per-model overrides and preserve selection evidence."""
    local = [t for t in targets if t["backend"] == "llamacpp"]
    if not local and any(v is not None for v in (device, context_size, gpu_layers)):
        raise ValueError("local hardware options require a llama.cpp target")
    selected = {}
    for target in local:
        if not mock and "installation" not in target and any(v is not None for v in (device, context_size, gpu_layers)):
            raise ValueError("local hardware options require a managed local server; use discover_models:true")
        if context_size is not None:
            target["context_size"] = context_size
        if gpu_layers is not None:
            target["gpu_layers"] = gpu_layers
        if not device:
            continue
        if mock:
            continue  # No user binaries or hardware probes in synthetic runs.
        server = target["installation"]["server"]
        if server not in selected:
            selected[server] = select_device(server, device)
        choice = selected[server]
        extra = target.get("server_args", [])
        controlled = {"-dev", "--device", "-sm", "--split-mode", "-fit", "--fit",
                      "-ngl", "--gpu-layers", "--n-gpu-layers"}
        if any(arg.split("=", 1)[0] in controlled for arg in extra):
            raise ValueError("--local-device conflicts with hardware flags in server_args; remove those flags")
        target["server_args"] = [*extra, "--device", choice["id"], "--split-mode", "none", "--fit", "off"]
        target["metadata"] = {**target.get("metadata", {}), "requested_device": device,
                              "selected_device": choice["id"], "gpu": choice["name"],
                              "device_enumeration": choice["enumeration"]}


class LocalServer:
    """Loads one GGUF, polls readiness, and stops the process on every exit path."""
    def __init__(self, target, run_dir):
        self.target = target
        self.run_dir = Path(run_dir)
        self.process = None
        self.log = None
        self.load_duration_s = None
        self.props = {}

    def __enter__(self):
        info = self.target["installation"]
        if info["missing_parts"]:
            raise RuntimeError("missing_gguf_shards")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        identifier = hashlib.sha256(self.target["name"].encode()).hexdigest()[:12]
        self.target["metadata"]["gguf_part_sha256"] = {
            Path(p).name: sha256(p) for p in info["parts"]}
        self.target["metadata"]["server_sha256"] = sha256(info["server"])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        args = [info["server"], "-m", info["model_path"], "--alias", self.target["model"],
                "--host", "127.0.0.1", "--port", str(port), "--jinja", "--no-context-shift", "-np", "1",
                "-c", str(self.target.get("context_size", 32768)),
                "-ngl", str(self.target.get("gpu_layers", 99)),
                "-t", str(self.target.get("threads", min(os.cpu_count() or 4, 8)))]
        extra = self.target.get("server_args", [])
        forbidden = {"-m", "--model", "--port", "--host", "--alias", "--api-key",
                     "--api-key-file", "--models-dir", "--models-preset", "--path", "--media-path"}
        if any(arg.split("=", 1)[0] in forbidden for arg in extra):
            raise ValueError("server_args cannot override model, host, port, or credentials")
        args.extend(extra)
        self.target["metadata"]["server_argv"] = args
        self.target["base_url"] = f"http://127.0.0.1:{port}/v1"
        self.target["metadata"]["server_log"] = f"servers/{identifier}.log"
        log_dir = self.run_dir / "servers"
        log_dir.mkdir(exist_ok=True)
        self.log = (log_dir / (identifier + ".log")).open("wb")
        start = time.perf_counter()
        try:
            self.process = subprocess.Popen(args, cwd=info["root"], stdout=self.log, stderr=self.log,
                                            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0))
            deadline = start + self.target.get("load_timeout_s", 300)
            while time.perf_counter() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama_server_exited_{self.process.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            self.load_duration_s = time.perf_counter() - start
                            try:
                                with urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5) as props:
                                    self.props = json.load(props)
                            except (OSError, ValueError):
                                pass
                            self.target["metadata"]["load_duration_s"] = self.load_duration_s
                            self.target["metadata"]["server_props"] = self.props
                            return self
                except OSError:
                    pass
                time.sleep(0.5)
            raise RuntimeError("llama_server_load_timeout")
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *_):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        if self.log:
            self.log.close()
