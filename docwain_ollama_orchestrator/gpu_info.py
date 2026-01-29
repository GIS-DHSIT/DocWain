from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    available: bool
    total_mb: Optional[int]
    free_mb: Optional[int]
    method: str


def _try_pynvml() -> Optional[GPUInfo]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count <= 0:
            return None
        free_list = []
        total_list = []
        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_list.append(int(info.total / (1024 * 1024)))
            free_list.append(int(info.free / (1024 * 1024)))
        return GPUInfo(
            available=True,
            total_mb=max(total_list) if total_list else None,
            free_mb=max(free_list) if free_list else None,
            method="pynvml",
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("pynvml GPU query failed: %s", exc)
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _try_nvidia_smi() -> Optional[GPUInfo]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=2,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("nvidia-smi query failed: %s", exc)
        return None
    totals = []
    frees = []
    for line in output.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            totals.append(int(parts[0]))
            frees.append(int(parts[1]))
        except ValueError:
            continue
    if not totals:
        return None
    return GPUInfo(
        available=True,
        total_mb=max(totals),
        free_mb=max(frees) if frees else None,
        method="nvidia-smi",
    )


def get_gpu_info() -> GPUInfo:
    info = _try_pynvml()
    if info:
        return info
    info = _try_nvidia_smi()
    if info:
        return info
    return GPUInfo(available=False, total_mb=None, free_mb=None, method="unavailable")
