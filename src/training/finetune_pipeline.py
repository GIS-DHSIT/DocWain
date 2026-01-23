import argparse
import json
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List

from qdrant_client import QdrantClient

from .dataset_preflight import ProfileStats, gate_training, save_profile_stats
from .dataset_writer import DatasetWriter
from .embedding_aware_pairgen import Chunk, LineFrequencyCleaner, build_pairs_for_profile
from .finetune_runner import FinetuneConfig, FinetuneRunner
from .qdrant_retriever import QdrantRetriever
from .qdrant_schema_probe import run_probe


def configure_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    return log_path


def build_chunks(retriever: QdrantRetriever, mode: str, profile: str = None):
    if mode == "per-profile":
        return list(retriever.retrieve_profile(profile))
    return list(retriever.retrieve_all())


def main():
    parser = argparse.ArgumentParser(description="DocWain finetune pipeline (schema-aware).")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--mode", choices=["per-profile", "merged"], default="per-profile")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--base-model", default="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--qdrant-api-key", default=os.getenv("QDRANT_API_KEY"))
    parser.add_argument("--use-ollama", action="store_true", default=False)
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--max-points", type=int, default=None)
    args = parser.parse_args()

    run_id = args.run_name or f"run-{int(time.time())}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(run_dir)
    logging.info("Starting pipeline for collection %s | mode=%s", args.collection, args.mode)

    schema = run_probe(
        collection=args.collection,
        run_dir=run_dir,
        qdrant_url=args.qdrant_url,
        qdrant_api_key=args.qdrant_api_key,
    )

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=120)
    retriever = QdrantRetriever(
        client=client,
        collection=args.collection,
        text_path=schema.text_field.path,
        profile_path=schema.profile_field.path,
        profile_type=schema.profile_type,
        with_vectors=True,
        max_points=args.max_points,
    )
    profiles = [None]
    if args.mode == "per-profile":
        profiles = retriever.discover_profiles()
        logging.info("Discovered %d profile_ids", len(profiles))

    profile_stats: List[ProfileStats] = []
    artifacts = {"train_jsonl": [], "manifest": str(run_dir / "finetune_manifest.json"), "schema": str(run_dir / "schema.json")}

    for profile in profiles:
        pid = profile if profile is not None else "merged"
        stats = ProfileStats(profile_id=pid)
        cleaner = LineFrequencyCleaner()
        raw_chunks = build_chunks(retriever, args.mode, profile)
        processed_chunks: List[Chunk] = []
        empty_with_vec = []
        for ch in raw_chunks:
            stats.chunks_total += 1
            if ch.text.strip():
                stats.chunks_with_text += 1
            else:
                if ch.vector:
                    empty_with_vec.append(ch.point_id)
                continue
            stats.tokens.append(len(ch.text.split()))
            if ch.payload.get("document_id"):
                stats.unique_docs.add(str(ch.payload.get("document_id")))
            if ch.payload.get("source_file"):
                stats.unique_sources.add(str(ch.payload.get("source_file")))
            processed_chunks.append(
                Chunk(
                    profile_id=pid,
                    text=ch.text,
                    embedding=ch.vector,
                    metadata={
                        "doc_id": ch.doc_id,
                        "document_id": ch.doc_id,
                        "source_file": ch.source_file,
                        "chunk_id": ch.point_id,
                        "chunk_index": ch.chunk_index,
                        **(ch.payload or {}),
                    },
                )
            )
        if empty_with_vec:
            logging.warning(
                "Found %d points with embeddings but no text for profile=%s: sample ids=%s",
                len(empty_with_vec),
                pid,
                empty_with_vec[:5],
            )

        pairs, drop_counters = build_pairs_for_profile(
            processed_chunks,
            cleaner=cleaner,
            min_bundle_size=3,
            max_bundle_size=6,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model,
        )
        stats.drop_reasons.empty_text += drop_counters.get("empty_text", 0)
        stats.drop_reasons.after_cleanup += drop_counters.get("after_cleanup", 0)
        stats.bundles_created = len(pairs)
        stats.pairs_created = len(pairs)
        dataset_path = run_dir / "datasets" / f"{pid}.jsonl"
        writer = DatasetWriter(dataset_path)
        writer.write_examples(pairs)
        writer.close()
        artifacts["train_jsonl"].append(str(dataset_path))
        stats.diagnosis = {
            "detected_text_field": ".".join(schema.text_field.path),
            "profile_field": ".".join(schema.profile_field.path),
            "profile_type": schema.profile_type,
            "observed_keys": schema.observed_keys,
            "sample_payloads": schema.sample_payloads,
            "filter_used": {
                "path": ".".join(schema.profile_field.path),
                "value": profile,
            },
        }
        profile_stats.append(stats)

    save_profile_stats(run_dir, profile_stats)
    readiness = gate_training(profile_stats, min_pairs=1)
    finetune_results: Dict[str, Dict] = {}
    ready_profiles = [pid for pid, r in readiness.items() if r["status"] == "ready"]
    if ready_profiles:
        runner = FinetuneRunner(
            FinetuneConfig(
                base_model=args.base_model,
                min_pairs=1,
            )
        )
        for pid in ready_profiles:
            ds_path = run_dir / "datasets" / f"{pid}.jsonl"
            finetune_results[pid] = runner.run(ds_path, run_dir / f"finetune_{pid}")
            if finetune_results[pid].get("status") == "success":
                readiness[pid] = {"status": "trained", "adapter_dir": finetune_results[pid].get("adapter_dir")}
            else:
                readiness[pid] = {"status": "failed_training", "reason": finetune_results[pid].get("reason")}
    else:
        logging.warning("No profiles ready for training; skipping finetune.")

    failures = [
        {"profile_id": pid, **info}
        for pid, info in readiness.items()
        if info.get("status") not in {"ready", "trained"}
    ]

    status = {
        "status": "partial" if failures and ready_profiles else "failed" if not ready_profiles else "success",
        "collection_name": args.collection,
        "training_run_id": run_id,
        "run_name": args.run_name,
        "base_model": args.base_model,
        "config_hash": str(uuid.uuid4())[:8],
        "profiles": [
            {
                **s.as_json(),
                "status": readiness[s.profile_id]["status"],
                "diagnosis": readiness[s.profile_id].get("diagnosis", s.diagnosis),
            }
            for s in profile_stats
        ],
        "failures": failures,
        "artifacts": artifacts,
        "finetune_results": finetune_results,
    }
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
