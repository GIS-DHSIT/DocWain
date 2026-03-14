"""
VLLMHealthMonitor - Background health monitoring for vLLM endpoint.

Thread-based periodic health check using urllib (no extra dependencies).
"""

import threading
import time
import urllib.request
import urllib.error
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class VLLMHealthMonitor:
    """Monitors vLLM endpoint health via background thread."""

    def __init__(self, endpoint: str, interval: float = 30.0, timeout: float = 5.0):
        """
        Args:
            endpoint: Base URL of the vLLM server (e.g. http://localhost:8001).
            interval: Seconds between health checks.
            timeout: HTTP request timeout in seconds.
        """
        # Normalise endpoint to base URL without path
        base = endpoint.rstrip("/")
        if "/v1/" in base:
            base = base.split("/v1/")[0]
        elif base.endswith("/v1"):
            base = base[:-3]
        self._health_url = f"{base}/health"

        self._interval = interval
        self._timeout = timeout
        self._healthy = False
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background health-check thread (idempotent)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name="vllm-health-monitor", daemon=True
        )
        self._thread.start()
        logger.info("vLLM health monitor started (endpoint=%s, interval=%ss)", self._health_url, self._interval)

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval + 2)
            self._thread = None
        logger.info("vLLM health monitor stopped")

    def is_healthy(self) -> bool:
        """Return the last-known health status (thread-safe)."""
        with self._lock:
            return self._healthy

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Background loop: check health, sleep, repeat."""
        # Immediate first check
        self._check()
        while not self._stop_event.is_set():
            self._stop_event.wait(self._interval)
            if not self._stop_event.is_set():
                self._check()

    def _check(self) -> None:
        try:
            req = urllib.request.Request(self._health_url, method="GET")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                healthy = resp.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
            healthy = False

        with self._lock:
            prev = self._healthy
            self._healthy = healthy

        if healthy != prev:
            if healthy:
                logger.info("vLLM endpoint is now HEALTHY")
            else:
                logger.warning("vLLM endpoint is now UNHEALTHY")
