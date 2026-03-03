from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Optional

import ray

from docwain_ollama_orchestrator.config import AppConfig
from docwain_ollama_orchestrator.ray_service import get_or_create_actor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ollama multi-model session bootstrapper")
    sub = parser.add_subparsers(dest="command")
    run_parser = sub.add_parser("run", help="Run the registry service")
    run_parser.add_argument("--warmup", dest="warmup", action="store_true")
    run_parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    run_parser.set_defaults(warmup=None)
    run_parser.add_argument("--refresh-interval", type=int, default=None)
    run_parser.add_argument("--print-registry", action="store_true")
    return parser.parse_args()


def _build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()
    if args.refresh_interval is not None:
        config = AppConfig(**{**config.__dict__, "refresh_interval_seconds": args.refresh_interval})
    if args.warmup is not None:
        config = AppConfig(**{**config.__dict__, "warmup_enabled": args.warmup})
    return config


def run() -> None:
    args = _parse_args()
    if args.command is None:
        args.command = "run"
    if args.command != "run":
        raise SystemExit("Unknown command")

    config = _build_config(args)
    actor = get_or_create_actor(config)

    if args.print_registry:
        registry = ray.get(actor.get_registry.remote())
        print(json.dumps(registry, indent=2))

    logger.info("Ollama registry service started. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(5)
    except KeyboardInterrupt:
        logger.info("Shutting down service.")


if __name__ == "__main__":
    run()
