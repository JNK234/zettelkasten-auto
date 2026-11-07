"""JSON-RPC 2.0 dispatcher and stdio transport.

Transport: newline-delimited JSON. Each request, notification or batch is one
UTF-8 line on stdin, and each response is one line on stdout. Notifications get
no response, and a batch of only notifications produces no line at all.
While serving, ``sys.stdout`` points at stderr so that a stray print cannot
corrupt the protocol stream. Diagnostics go to stderr.

Error handling follows the spec: -32700 for unparseable JSON, -32600 for
anything that is not a valid Request object (including an empty batch),
-32601 for an unknown method, -32602 when params fail validation, and -32603
for unexpected failures (including a result that does not match its own model).
Server errors: -32001 index not built, -32002 note not found, -32003 format not
configured (serve --format), -32004 write rejected (bad arguments, name taken;
nothing written), -32005 write failed (rolled back, or a file changed meanwhile).
A write that would add violations is not an error: its result has status "refused". A request's "id"
is echoed in its error response when it is a valid id, otherwise null.
"""

from __future__ import annotations

import json
import sys
import traceback
from typing import Any, BinaryIO, Callable

from pydantic import BaseModel, ValidationError

from vaultengine.rpc.models import (
    ERROR_MESSAGES,
    FORMAT_NOT_CONFIGURED,
    INDEX_NOT_BUILT,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    JSONRPC_VERSION,
    METHOD_NOT_FOUND,
    METHODS,
    NOTE_NOT_FOUND,
    PARSE_ERROR,
    WRITE_FAILED,
    WRITE_REJECTED,
    Request,
)
from vaultengine.service import (
    FormatNotConfigured, IndexNotBuilt, NoteNotFound, ServiceError, VaultService, WriteFailedError, WriteRejected,
)

_SERVICE_ERROR_CODES: dict[type[ServiceError], int] = {
    IndexNotBuilt: INDEX_NOT_BUILT,
    NoteNotFound: NOTE_NOT_FOUND,
    FormatNotConfigured: FORMAT_NOT_CONFIGURED,
    WriteRejected: WRITE_REJECTED,
    WriteFailedError: WRITE_FAILED,
}


def _reject_constant(name: str) -> None:
    raise ValueError(f"{name} is not valid JSON")


def _error(request_id: Any, code: int, message: str | None = None, data: Any = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message or ERROR_MESSAGES[code]}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": JSONRPC_VERSION, "id": request_id, "error": error}


def _validation_details(exc: ValidationError) -> list[dict[str, Any]]:
    return [{"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]} for err in exc.errors(include_url=False)]


def _salvage_id(item: dict[str, Any]) -> Any:
    value = item.get("id")
    valid = value is None or isinstance(value, str) or (isinstance(value, int) and not isinstance(value, bool))
    return value if valid else None


def encode(message: Any) -> str:
    """One-line JSON. U+2028/U+2029 are escaped so line-splitting clients stay safe."""
    text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


class RpcServer:
    def __init__(self, service: VaultService) -> None:
        self.service = service
        self._handlers: dict[str, Callable[[Any], dict[str, Any]]] = {
            "index.build": lambda p: service.build(full=p.full),
            "index.status": lambda p: service.status(),
            "vault.stats": lambda p: service.stats(top=p.top),
            "health.report": lambda p: service.health(limit=p.limit),
            "notes.pick": lambda p: service.pick(p.query, limit=p.limit),
            "note.get": lambda p: service.note(p.path),
            "vault.check": lambda p: service.check(changed=p.changed, limit=p.limit),
            "vault.fix": lambda p: service.fix(dry_run=p.dry_run, only=p.only),
            "vault.coverage": lambda p: service.coverage(limit=p.limit),
            "note.new": lambda p: service.new(
                p.type, p.title, folder=p.folder, name=p.name, values=p.values,
                relations=[(r.relation, r.note) for r in p.relations], sections=p.sections, dry_run=p.dry_run,
            ),
            "note.link": lambda p: service.link(p.source, p.target, p.relation, dry_run=p.dry_run),
            "note.set": lambda p: service.set(p.note, p.key, p.value, dry_run=p.dry_run),
            "note.ingest": lambda p: service.ingest(
                path=p.path, content=p.content, filename=p.filename, type=p.type, title=p.title,
                folder=p.folder, values=p.values, dry_run=p.dry_run,
            ),
        }
        if set(self._handlers) != set(METHODS):
            raise RuntimeError(f"handlers and METHODS disagree: {set(self._handlers) ^ set(METHODS)}")

    def handle_text(self, text: str) -> str | None:
        try:
            payload = json.loads(text, parse_constant=_reject_constant)
        except ValueError as exc:
            return encode(_error(None, PARSE_ERROR, data=str(exc)))
        response = self.handle_payload(payload)
        return None if response is None else encode(response)

    def handle_payload(self, payload: Any) -> Any:
        if isinstance(payload, list):
            if not payload:
                return _error(None, INVALID_REQUEST, data="empty batch")
            responses = [response for response in map(self._handle_one, payload) if response is not None]
            return responses or None
        return self._handle_one(payload)

    def _handle_one(self, item: Any) -> dict[str, Any] | None:
        if not isinstance(item, dict):
            return _error(None, INVALID_REQUEST, data="a request must be a JSON object")
        try:
            request = Request.model_validate(item)
        except ValidationError as exc:
            return _error(_salvage_id(item), INVALID_REQUEST, data=_validation_details(exc))

        response = self._dispatch(request)
        is_notification = "id" not in request.model_fields_set
        return None if is_notification else response

    def _dispatch(self, request: Request) -> dict[str, Any]:
        spec = METHODS.get(request.method)
        if spec is None:
            return _error(request.id, METHOD_NOT_FOUND, data={"method": request.method, "available": sorted(METHODS)})
        if isinstance(request.params, list):
            return _error(request.id, INVALID_PARAMS, data="params must be an object (by-name)")
        try:
            params = spec.params.model_validate(request.params or {})
        except ValidationError as exc:
            return _error(request.id, INVALID_PARAMS, data=_validation_details(exc))
        try:
            result = self._handlers[spec.name](params)
            validated: BaseModel = spec.result.model_validate(result)
        except ServiceError as exc:
            return _error(request.id, _SERVICE_ERROR_CODES.get(type(exc), INTERNAL_ERROR), exc.message, exc.data or None)
        except Exception as exc:  # noqa: BLE001 - reported to the client, logged to stderr
            traceback.print_exc(file=sys.stderr)
            return _error(request.id, INTERNAL_ERROR, data={"type": type(exc).__name__, "message": str(exc)})
        return {"jsonrpc": JSONRPC_VERSION, "id": request.id, "result": validated.model_dump(mode="json")}


def serve_stdio(service: VaultService, stdin: BinaryIO | None = None, stdout: BinaryIO | None = None) -> int:
    server = RpcServer(service)
    reader = stdin if stdin is not None else sys.stdin.buffer
    writer = stdout if stdout is not None else sys.stdout.buffer
    saved_stdout, sys.stdout = sys.stdout, sys.stderr
    shown_format = f", format {service.format_path}" if service.format_path else ""
    print(f"vaultengine: serving {service.root} (index {service.db_path}{shown_format}) on stdio", file=sys.stderr)
    try:
        for raw in reader:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                response: str | None = encode(_error(None, PARSE_ERROR, data=f"invalid UTF-8: {exc}"))
            else:
                text = text.rstrip("\r\n")  # the line terminator is framing, not payload
                if not text.strip():
                    continue
                response = server.handle_text(text)
            if response is not None:
                writer.write(response.encode("utf-8") + b"\n")
                writer.flush()
    finally:
        sys.stdout = saved_stdout
        service.close()
    return 0
