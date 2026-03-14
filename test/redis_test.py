import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import src.api.dw_newron as dw_newron

HOST = os.getenv("REDIS_HOST", "localhost")
PORT = int(os.getenv("REDIS_PORT", "6380"))
PASSWORD = os.getenv("REDIS_PASSWORD", "")


def _reset_cached_client():
    """Force get_redis_client to rebuild the client with fresh env settings."""
    dw_newron._REDIS_CLIENT = None  # noqa: SLF001


def _set_env(monkeypatch=None):
    # Force the known-good values from the working snippet and include trailing
    # whitespace on the password to ensure it is stripped before connecting.
    set_env = monkeypatch.setenv if monkeypatch else os.environ.__setitem__
    clear_env = monkeypatch.delenv if monkeypatch else os.environ.pop

    clear_env("REDIS_URL", None)  # Prefer explicit host/port/password
    clear_env("REDIS_CONNECTION_STRING", None)

    set_env("REDIS_HOST", HOST)
    set_env("REDIS_PORT", str(PORT))
    set_env("REDIS_PASSWORD", f"{PASSWORD} ")
    set_env("REDIS_SSL", "true")
    set_env("REDIS_SOCKET_TIMEOUT", "10")
    set_env("REDIS_SOCKET_CONNECT_TIMEOUT", "10")


def test_redis_ping(monkeypatch=None):
    _set_env(monkeypatch)
    _reset_cached_client()

    client = dw_newron.get_redis_client()
    assert client is not None, "Redis client should be initialized"

    pong = client.ping()
    assert pong is True
    print("PING =>", pong)


if __name__ == "__main__":
    test_redis_ping()
