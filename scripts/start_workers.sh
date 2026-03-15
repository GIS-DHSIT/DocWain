#!/bin/bash
# Start all DocWain Celery workers

echo "Starting DocWain Celery workers..."

celery -A src.celery_app worker -Q extraction_queue -c 2 --hostname=extract@%h --loglevel=info &
celery -A src.celery_app worker -Q screening_queue -c 4 --hostname=screen@%h --loglevel=info &
celery -A src.celery_app worker -Q kg_queue -c 4 --hostname=kg@%h --loglevel=info &
celery -A src.celery_app worker -Q embedding_queue -c 2 --hostname=embed@%h --loglevel=info &
celery -A src.celery_app worker -Q backfill_queue -c 4 --hostname=backfill@%h --loglevel=info &

echo "Starting Flower monitoring on port 5555..."
celery -A src.celery_app flower --port=5555 &

echo "All workers started."
wait
