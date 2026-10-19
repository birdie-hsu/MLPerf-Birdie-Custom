"""Streaming adapters. No retries, no inferred token counts, no key logging."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import Any


class TransportError(RuntimeError):
    pass


def server_metrics(timings):
    """Use server timings directly; unavailable cloud timings stay missing."""
    prompt_ms = timings.get("prompt_ms")
    generation = timings.get("predicted_per_second")
    return {
        "prefill_s": prompt_ms / 1000 if prompt_ms is not None else None,
        "prefill_tps": timings.get("prompt_per_second"),
        "prefill_tokens": timings.get("prompt_n"),
        "cached_prompt_tokens": timings.get("cache_n"),
        "generation_tps": generation,
        # Keep the previous machine-readable field for existing analysis scripts.
        "server_decode_tps": generation,
    }


@dataclass
class Reply:
    text: str = ""
    reasoning_text: str = ""
    calls: list[dict] = field(default_factory=list)
    history_items: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    native_timings: dict = field(default_factory=dict)
    resolved_model: str | None = None
    response_id: str | None = None
    finish: str = "unknown"
    first_event_s: float | None = None
    first_text_s: float | None = None
    first_tool_s: float | None = None
    first_reasoning_s: float | None = None
    last_text_s: float | None = None
    duration_s: float = 0.0
    text_delta_count: int = 0
    reasoning_seen: bool = False
    synthetic: bool = False
    request_output_limit: int | None = None
    provider: str | None = None
    reasoning_details: list[dict] = field(default_factory=list)

    def metrics(self):
        # Stream deltas can contain several tokens. This is explicitly an estimate,
        # calculated only when completion token usage describes visible text alone.
        count = self.usage.get("output_tokens")
        reasons = self.usage.get("reasoning_tokens")
        visible = None
        if not self.calls and count is not None:
            if reasons is not None:
                visible = count - reasons
            elif not self.reasoning_seen:
                visible = count
        elapsed = None
        if self.first_text_s is not None and self.last_text_s is not None:
            elapsed = self.last_text_s - self.first_text_s
        tps = ((visible - 1) / elapsed if visible is not None and visible > 1
               and elapsed and self.text_delta_count > 1 else None)
        return {
            "first_event_s": self.first_event_s,
            "ttft_visible_s": self.first_text_s,
            "ttft_tool_s": self.first_tool_s,
            "ttft_reasoning_s": self.first_reasoning_s,
            "last_visible_s": self.last_text_s,
            "request_duration_s": self.duration_s,
            "visible_native_tokens": visible,
            "native_tps_estimate": tps,
            "visible_chars_per_s": (len(self.text) / self.duration_s
                                     if self.duration_s else None),
            "text_delta_count": self.text_delta_count,
            **server_metrics(self.native_timings),
            "request_output_limit": self.request_output_limit,
            "finish_reason": self.finish,
        }


def sse_events(lines):
    """Parse UTF-8 SSE including multiline data, CRLF, comments, and EOF."""
    data = []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if data:
                payload = "\n".join(data)
                data = []
                if payload == "[DONE]":
                    return
                yield json.loads(payload)
        elif line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
    if data:
        payload = "\n".join(data)
        if payload != "[DONE]":
            yield json.loads(payload)


def normalize_usage(raw, backend):
    if backend == "responses":
        return {
            "input_tokens": raw.get("input_tokens"),
            "output_tokens": raw.get("output_tokens"),
            "cached_input_tokens": (raw.get("input_tokens_details") or {}).get("cached_tokens"),
            "reasoning_tokens": (raw.get("output_tokens_details") or {}).get("reasoning_tokens"),
        }
    return {
        "input_tokens": raw.get("prompt_tokens"),
        "output_tokens": raw.get("completion_tokens"),
        "cached_input_tokens": (raw.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "reasoning_tokens": (raw.get("completion_tokens_details") or {}).get("reasoning_tokens"),
    }


def responses_input(messages):
    out = []
    for message in messages:
        if message.get("role") == "tool":
            out.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                        "output": message["content"]})
        elif message.get("_items") is not None:
            out.extend(message["_items"])
        else:
            out.append({"role": message["role"], "content": message["content"]})
    return out


def request_body(target, messages, tools, cap):
    backend = target["backend"]
    if backend == "responses":
        body = {"model": target["model"], "input": responses_input(messages),
                "stream": True, "store": False, "include": ["reasoning.encrypted_content"],
                "max_output_tokens": target.get("max_output_tokens", 8192),
                "reasoning": {"effort": target.get("reasoning_effort", "low")}}
        if tools:
            body.update(tools=[{"type": "function", **tool} for tool in tools],
                        parallel_tool_calls=False)
        # max_output_tokens includes invisible reasoning. Keep upstream visible
        # target separate; equal request limits do not imply equal visible length.
    else:
        clean = [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
        if backend == "chatcompletions":
            # Routers choose the upstream dialect. Do not leak llama.cpp-only
            # fields or sampling settings unsupported by reasoning models.
            body = {"model": target["model"], "messages": clean, "stream": True,
                    "stream_options": {"include_usage": True},
                    "max_tokens": max(cap, target.get("max_output_tokens", 8192))}
            if "temperature" in target:
                body["temperature"] = target["temperature"]
        elif backend == "llamacpp":
            thinking = target.get("local_thinking", "off")
            limit = cap if thinking == "off" else max(cap, target.get("max_output_tokens", 2048))
            body = {"model": target["model"], "messages": clean, "stream": True,
                    "stream_options": {"include_usage": True}, "max_tokens": limit,
                    "temperature": target.get("temperature", 0),
                    "seed": target.get("seed", 20261006)}
            if thinking != "auto":
                body["chat_template_kwargs"] = {"enable_thinking": thinking == "on"}
        else:
            raise ValueError("unsupported backend")
        if tools:
            body["tools"] = [{"type": "function", "function": tool} for tool in tools]
            if backend == "llamacpp":
                body["parallel_tool_calls"] = False
    extra = target.get("extra_body", {})
    if backend == "llamacpp" and "chat_template_kwargs" in extra:
        kwargs = extra["chat_template_kwargs"]
        if "enable_thinking" in kwargs:
            raise ValueError("Use local_thinking to configure enable_thinking")
        body["chat_template_kwargs"] = {**kwargs, **body.get("chat_template_kwargs", {})}
        extra = {k: v for k, v in extra.items() if k != "chat_template_kwargs"}
    if set(extra) & set(body):
        raise ValueError("extra_body cannot override benchmark-controlled fields")
    body.update(extra)
    return body


def consume(events, backend, clock):
    reply = Reply()
    call_parts = {}
    reasoning_parts = {}
    complete = False
    for event in events:
        now = clock()
        if reply.first_event_s is None:
            reply.first_event_s = now
        if event.get("error") or event.get("type") in {"error", "response.failed"}:
            raise TransportError("provider_stream_error")
        reply.resolved_model = event.get("model", reply.resolved_model)
        reply.response_id = event.get("id", reply.response_id)
        if isinstance(event.get("provider"), str):
            reply.provider = event["provider"]
        if backend == "responses":
            kind = event.get("type", "")
            if kind.startswith("response.reasoning"):
                reply.reasoning_seen = True
            if kind == "response.output_text.delta" and event.get("delta"):
                if reply.first_text_s is None:
                    reply.first_text_s = now
                reply.last_text_s = now
                reply.text += event["delta"]
                reply.text_delta_count += 1
            if kind == "response.function_call_arguments.delta" and event.get("delta"):
                if reply.first_tool_s is None:
                    reply.first_tool_s = now
            if kind in {"response.completed", "response.incomplete"}:
                response = event["response"]
                reply.resolved_model = response.get("model", reply.resolved_model)
                reply.response_id = response.get("id", reply.response_id)
                reply.usage = normalize_usage(response.get("usage") or {}, backend)
                reply.finish = response.get("status", "unknown")
                reply.history_items = response.get("output", [])
                for item in reply.history_items:
                    if item.get("type") == "function_call":
                        reply.calls.append({"id": item["call_id"], "name": item["name"],
                                            "arguments": item["arguments"]})
                complete = True
        else:
            if event.get("usage"):
                reply.usage = normalize_usage(event["usage"], backend)
            if event.get("timings"):
                reply.native_timings = event["timings"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                for detail in delta.get("reasoning_details") or []:
                    if not isinstance(detail, dict):
                        continue
                    reply.reasoning_seen = True
                    if reply.first_reasoning_s is None:
                        reply.first_reasoning_s = now
                    index = detail.get("index", len(reasoning_parts))
                    part = reasoning_parts.setdefault(index, {})
                    for key, value in detail.items():
                        if key in {"text", "data", "signature"} and isinstance(value, str):
                            part[key] = part.get(key, "") + value
                        else:
                            part[key] = value
                if delta.get("reasoning_content") or delta.get("reasoning"):
                    reply.reasoning_seen = True
                    if reply.first_reasoning_s is None:
                        reply.first_reasoning_s = now
                    reply.reasoning_text += delta.get("reasoning_content") or delta.get("reasoning")
                if delta.get("content"):
                    if reply.first_text_s is None:
                        reply.first_text_s = now
                    reply.last_text_s = now
                    reply.text += delta["content"]
                    reply.text_delta_count += 1
                for tc in delta.get("tool_calls", []):
                    if reply.first_tool_s is None:
                        reply.first_tool_s = now
                    part = call_parts.setdefault(tc.get("index", 0),
                                                 {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        part["id"] = tc["id"]
                    fn = tc.get("function", {})
                    part["name"] += fn.get("name", "")
                    part["arguments"] += fn.get("arguments", "")
                if choice.get("finish_reason"):
                    reply.finish = choice["finish_reason"]
                    complete = True
    if not complete:
        raise TransportError("stream_ended_without_completion")
    if backend != "responses":
        reply.calls = [call_parts[i] for i in sorted(call_parts)]
        reply.reasoning_details = list(reasoning_parts.values())
    reply.duration_s = clock()
    return reply


def request_headers(target, stream=True):
    headers = {"Content-Type": "application/json",
               "Accept": "text/event-stream" if stream else "application/json"}
    key_env = target.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key or not key.strip():
            raise TransportError("missing_api_key_environment_variable:" + key_env)
        headers["Authorization"] = "Bearer " + key.strip()
    return headers


class HTTPAdapter:
    def __init__(self, target, timeout):
        self.target = target
        self.timeout = timeout

    def call(self, messages, tools=None, cap=512, timeout=None):
        target = self.target
        body = request_body(target, messages, tools, cap)
        headers = request_headers(target)
        path = "/responses" if target["backend"] == "responses" else "/chat/completions"
        request = urllib.request.Request(target["base_url"].rstrip("/") + path,
                                         data=json.dumps(body).encode(), headers=headers)
        start = time.perf_counter()
        limit = min(self.timeout, timeout or self.timeout)
        def elapsed():
            value = time.perf_counter() - start
            if value > limit:
                raise TransportError("request_deadline_exceeded")
            return value
        try:
            with urllib.request.urlopen(request, timeout=limit) as response:
                reply = consume(sse_events(response), target["backend"], elapsed)
                reply.request_output_limit = body.get("max_output_tokens", body.get("max_tokens"))
                return reply
        except urllib.error.HTTPError as error:
            # Never log provider error bodies, which may echo secrets or prompts.
            raise TransportError(f"http_{error.code}") from None
        except urllib.error.URLError:
            raise TransportError("connection_error") from None


def assistant_message(reply, backend):
    if backend == "responses":
        return {"role": "assistant", "content": reply.text, "_items": reply.history_items}
    message: dict[str, Any] = {"role": "assistant", "content": reply.text or None}
    if reply.reasoning_text:
        message["reasoning_content"] = reply.reasoning_text
    if backend == "chatcompletions" and reply.reasoning_details:
        message["reasoning_details"] = reply.reasoning_details
    if reply.calls:
        message["tool_calls"] = [{"id": c["id"], "type": "function", "function": {
            "name": c["name"], "arguments": c["arguments"]}} for c in reply.calls]
    return message
