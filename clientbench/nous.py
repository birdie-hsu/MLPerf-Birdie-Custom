"""Nous API-key model discovery and explicit, opt-in router targets."""
from __future__ import annotations

import copy
import json
import urllib.error
import urllib.request

from .transport import TransportError, request_headers

NOUS_BASE_URL = "https://inference-api.nousresearch.com/v1"


def default_target():
    return {"name": "nous", "provider": "nous", "backend": "chatcompletions",
            "enabled": False, "base_url": NOUS_BASE_URL, "model": "",
            "api_key_env": "NOUS_API_KEY", "max_output_tokens": 8192,
            "price_per_million": None}


def fetch_models(target, timeout=30):
    request = urllib.request.Request(target["base_url"].rstrip("/") + "/models",
                                     headers=request_headers(target, stream=False))
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as error:
        raise TransportError(f"nous_catalog_http_{error.code}; check the API key and its model permissions") from None
    except urllib.error.URLError:
        raise TransportError("nous_catalog_connection_error") from None
    except (ValueError, UnicodeError):
        raise TransportError("nous_catalog_invalid_json") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise TransportError("nous_catalog_invalid_schema")
    result, seen = [], set()
    for model in payload["data"]:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str) or not model["id"].strip():
            continue
        if model["id"] in seen:
            continue
        seen.add(model["id"])
        architecture = model.get("architecture") or {}
        modalities = architecture.get("output_modalities") if isinstance(architecture, dict) else None
        # A minimal OpenAI catalog often omits capability metadata. Keep those
        # candidates, but never label their tool support as verified.
        if isinstance(modalities, list) and "text" not in modalities:
            continue
        if "embedding" in model["id"].lower() and not modalities:
            continue
        parameters = model.get("supported_parameters")
        supports_tools = ("tools" in parameters if isinstance(parameters, list) else None)
        result.append({"id": model["id"], "name": str(model.get("name") or model["id"]),
                       "context_length": model.get("context_length"),
                       "tools_supported": supports_tools,
                       "output_modalities": modalities,
                       "pricing": model.get("pricing")})
    return sorted(result, key=lambda item: item["id"])


def expand_targets(template, selected_models, catalog=None):
    selected = selected_models or ([template["model"]] if template.get("model") else [])
    if not selected:
        raise ValueError("Nous requires --cloud-model MODEL_ID. List IDs with: python bench.py models --target nous")
    if len(set(selected)) != len(selected):
        raise ValueError("Duplicate --cloud-model IDs would repeat the same benchmark")
    available = {model["id"]: model for model in catalog} if catalog is not None else None
    result = []
    for model_id in selected:
        if not isinstance(model_id, str) or not model_id.strip() or model_id != model_id.strip():
            raise ValueError("Cloud model ID must be nonempty without surrounding whitespace")
        if available is not None and model_id not in available:
            raise ValueError(f"Model ID not found in the Nous text-model catalog: {model_id}. Use bench.py models --target nous")
        target = copy.deepcopy(template)
        target.update(name=template["name"] + "/" + model_id, model=model_id)
        target.setdefault("metadata", {}).update(
            route="Nous router API", catalog_verified=available is not None)
        if available is not None:
            target["metadata"]["catalog_model"] = available[model_id]
        result.append(target)
    return result
