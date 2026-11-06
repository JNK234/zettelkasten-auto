"""Derive the JSON Schema of the API from the Pydantic models in ``models.py``.

The document is JSON Schema draft 2020-12. Its root accepts any request,
response or batch. ``$defs`` holds a request and response schema per method
(e.g. ``NotesPickRequest`` / ``NotesPickResponse``), plus ``Request`` and
``Response`` unions. ``x-methods`` and ``x-errors`` are annotations for
tooling and docs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.json_schema import models_json_schema

from vaultengine.rpc.models import ERROR_MESSAGES, METHODS, ErrorResponse, MethodSpec, RequestId

SCHEMA_PATH = Path(__file__).with_name("schema.json")
_REF = "#/$defs/{model}"


def _pascal(method: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in re.split(r"[._]", method))


def request_model(spec: MethodSpec) -> type[BaseModel]:
    params_required = any(field.is_required() for field in spec.params.model_fields.values())
    params_field = (spec.params, ...) if params_required else (spec.params, Field(default_factory=spec.params))
    return create_model(
        f"{_pascal(spec.name)}Request",
        __config__=ConfigDict(extra="forbid"),
        __doc__=f"{spec.summary} Omit \"id\" to send a notification (no response).",
        jsonrpc=(Literal["2.0"], ...),
        method=(Literal[spec.name], ...),  # type: ignore[valid-type]
        params=params_field,
        id=(RequestId, None),
    )


def response_model(spec: MethodSpec) -> type[BaseModel]:
    return create_model(
        f"{_pascal(spec.name)}Response",
        __config__=ConfigDict(extra="forbid"),
        __doc__=f"Successful response to {spec.name}.",
        jsonrpc=(Literal["2.0"], ...),
        id=(RequestId, ...),
        result=(spec.result, ...),
    )


def build_schema() -> dict[str, Any]:
    specs = list(METHODS.values())
    requests = [request_model(spec) for spec in specs]
    responses = [response_model(spec) for spec in specs]
    _, generated = models_json_schema(
        [(model, "validation") for model in [*requests, *responses, ErrorResponse]], ref_template=_REF
    )
    defs = generated["$defs"]
    defs["Request"] = {
        "description": "Any request this server accepts.",
        "anyOf": [{"$ref": _REF.format(model=m.__name__)} for m in requests],
    }
    defs["Response"] = {
        "description": "Any response this server sends: a method result or an error.",
        "anyOf": [{"$ref": _REF.format(model=m.__name__)} for m in [*responses, ErrorResponse]],
    }
    message = {"anyOf": [{"$ref": "#/$defs/Request"}, {"$ref": "#/$defs/Response"}]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "urn:vaultengine:rpc",
        "title": "vaultengine JSON-RPC 2.0 API",
        "description": (
            "JSON-RPC 2.0 over stdio: one UTF-8 JSON message per line, in both directions. "
            "Params are by-name objects. The root schema accepts a single message or a batch."
        ),
        "anyOf": [message, {"type": "array", "minItems": 1, "items": message}],
        "x-methods": {
            spec.name: {
                "summary": spec.summary,
                "params": {"$ref": _REF.format(model=spec.params.__name__)},
                "result": {"$ref": _REF.format(model=spec.result.__name__)},
                "request": {"$ref": _REF.format(model=req.__name__)},
                "response": {"$ref": _REF.format(model=resp.__name__)},
            }
            for spec, req, resp in zip(specs, requests, responses)
        },
        "x-errors": {str(code): message for code, message in ERROR_MESSAGES.items()},
        "$defs": dict(sorted(defs.items())),
    }


def schema_text() -> str:
    return json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"


def write_schema(path: Path = SCHEMA_PATH) -> Path:
    path.write_text(schema_text(), encoding="utf-8")
    return path
