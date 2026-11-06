"""JSON-RPC 2.0 front-end for vaultengine (see server.py for the protocol)."""

from vaultengine.rpc.models import METHODS
from vaultengine.rpc.server import RpcServer, serve_stdio

__all__ = ["METHODS", "RpcServer", "serve_stdio"]
