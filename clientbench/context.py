"""Official workload context lengths and explicit local capacity overrides."""

import re

DEFAULT_CONTEXT_LENGTH = 4096


def validate_context_size(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("context size must be a positive integer")
    return value


def workload_context_length(model_config):
    # Match upstream LlmConfig::GetContextLength: flat, legacy nested, default.
    if "context_length" in model_config:
        value = model_config["context_length"]
    else:
        value = model_config.get("model", {}).get("context_length", DEFAULT_CONTEXT_LENGTH)
    return validate_context_size(value)


def resolve_context(target, case):
    override = target.get("context_size")
    if override is not None:
        return validate_context_size(override), "override"
    value = case.get("context_length", DEFAULT_CONTEXT_LENGTH)
    return validate_context_size(value), "workload" if "context_length" in case else "default"


def validate_context_args(args):
    for arg in args:
        if (arg.split("=", 1)[0] in {"-c", "--ctx-size", "-np", "--parallel"}
                or re.match(r"^-(?:c|np)\d", arg)):
            raise ValueError("server_args cannot override context or parallelism; use context_size or --context-size")


def local_context_plan(target, cases, mock=False):
    validate_context_args(target.get("server_args", []))
    managed = bool(target.get("installation") or mock)
    entries = []
    for case in cases:
        size, source = resolve_context(target, case)
        entries.append({"case_id": case["id"],
                        "workload_context_length": case.get("context_length", DEFAULT_CONTEXT_LENGTH),
                        "context_size": size if managed else None,
                        "context_source": source if managed else "external_server"})
    return {"context_policy": ("external_server" if not managed else
                               "override" if target.get("context_size") is not None else "workload"),
            "case_contexts": entries}
