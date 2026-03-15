"""DocWain Celery application — task queue for document processing pipeline."""

import os
from celery import Celery

app = Celery("docwain")

app.config_from_object({
    "broker_url": os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    "result_backend": os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1"),

    "task_serializer": "json",
    "result_serializer": "json",
    "accept_content": ["json"],
    "timezone": "UTC",
    "enable_utc": True,

    # Queue definitions
    "task_queues": {
        "extraction_queue": {"exchange": "extraction", "routing_key": "extraction"},
        "screening_queue": {"exchange": "screening", "routing_key": "screening"},
        "kg_queue": {"exchange": "kg", "routing_key": "kg"},
        "embedding_queue": {"exchange": "embedding", "routing_key": "embedding"},
        "backfill_queue": {"exchange": "backfill", "routing_key": "backfill"},
    },

    "task_routes": {
        "src.tasks.extraction.extract_document": {"queue": "extraction_queue"},
        "src.tasks.screening.screen_document": {"queue": "screening_queue"},
        "src.tasks.kg.build_knowledge_graph": {"queue": "kg_queue"},
        "src.tasks.embedding.embed_document": {"queue": "embedding_queue"},
        "src.tasks.backfill.backfill_kg_refs": {"queue": "backfill_queue"},
    },

    # Reliability
    "task_acks_late": True,
    "task_reject_on_worker_lost": True,
    "task_time_limit": 1800,
    "task_soft_time_limit": 1500,

    # Retry
    "task_default_retry_delay": 60,
    "task_max_retries": 3,

    # Monitoring
    "worker_send_task_events": True,
    "task_send_sent_event": True,
})

app.autodiscover_tasks(["src.tasks.extraction", "src.tasks.screening",
                         "src.tasks.kg", "src.tasks.embedding",
                         "src.tasks.backfill"])
