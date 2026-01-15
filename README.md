## DocWain Microsoft Teams Integration

- POST `(/teams/messages)` is the FastAPI entrypoint for all Teams activities. It accepts the raw Teams activity JSON payload and returns a Bot Framework-style message response.
- File uploads from Teams are stored in `TEAMS_UPLOAD_DIR` (defaults to `/tmp`) while they are processed and embedded into Qdrant collections scoped by the Teams conversation id.
- Configure `TEAMS_SHARED_SECRET` (optional) to validate calls from your bot. Provide the header when invoking the endpoint from your bot service.
- To persist uploaded files, set `TEAMS_BLOB_CONNECTION_STRING` and `TEAMS_BLOB_CONTAINER` (default: `local-uploads`). `TEAMS_BLOB_PATH_PREFIX` can be used to namespace blobs (default: `teams`). Files are uploaded after processing and also cleaned up from `TEAMS_UPLOAD_DIR`.

### Deployment and Teams manifest updates
- Expose the `/teams/messages` route on your deployment host (e.g., `https://<api-host>/teams/messages`).
- In the Teams bot manifest, set the messaging endpoint to the URL above and enable file uploads/attachments so Teams sends `file.download.info` payloads.
- Restart or redeploy the API after updating environment variables, then re-upload the manifest to your Teams tenant so the new endpoint is active.

### Redis cache setup
- Redis backs chat history, feedback, and answer caching; configure it with `REDIS_URL` (`rediss://:<key>@<host>:6380/0`) or `REDIS_CONNECTION_STRING` (Azure style `host:port,password=...,ssl=True`), or explicitly set `REDIS_HOST`, `REDIS_PASSWORD`, and `REDIS_DB` (TLS on port 6380).
- Local development: `docker-compose up --build` now starts a Redis 7 container and wires the app to it with no auth and SSL disabled. Verify it is running with `docker compose exec redis redis-cli ping`.
- If you point at your own Redis instance, update the env vars above and restart the service so the client reinitializes with the new settings.

### Fine-tuning with Unsloth (LLama 3.2) from a Qdrant collection

1) Install deps: `pip install -r requirements.txt` (includes `unsloth`, `trl`, `datasets`, `transformers`, `bitsandbytes`).
2) Kick off a run using only the Qdrant collection name. The API discovers profile ids in that collection, samples chunks, generates QA pairs, and starts Unsloth finetunes for each profile (or a filtered subset):
```bash
curl -X POST http://localhost:8000/finetune -H "Content-Type: application/json" -d '{
  "collection_name": "default",
  "profile_ids": ["finance"],
  "base_model": "llama3.2",
  "max_points": 120,
  "questions_per_chunk": 2,
  "generation_model": "llama3.2",
  "learning_rate": 0.0002,
  "num_epochs": 1,
  "max_steps": 200,
  "batch_size": 4,
  "gradient_accumulation": 2,
  "lora_r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05
}'
```
   - `collection_name` is required; omit `profile_id`/`profile_ids` to fine-tune every profile discovered in that collection.
   - Synthetic QA is written under `finetune_artifacts/<profile>/auto_datasets/`; you can still append curated `training_examples` in the request to blend with the generated data.
3) Monitor: `curl http://localhost:8000/finetune/<job_id>` to check status.
4) Use it: finetuned models auto-register to the model list (`GET /models`) and are served for their profile. Queries to `/ask` with that profile will use the finetuned model; override by passing a different `model_name` if needed.
5) Improve quality:
   - Raise `questions_per_chunk` or `max_points` for more coverage.
   - Swap `generation_model` to a stronger generator if available.
   - Increase `num_epochs`/`max_steps` and adjust `learning_rate`, `batch_size`, and `gradient_accumulation` for tougher domains.
   - Tune LoRA knobs: `lora_r`, `lora_alpha`, `lora_dropout`.
