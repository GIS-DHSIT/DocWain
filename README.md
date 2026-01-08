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

### Fine-tuning with Unsloth (LLama 3.2)

1) Install deps: `pip install -r requirements.txt` (includes `unsloth`, `trl`, `datasets`, `transformers`, `bitsandbytes`).
2) Prepare data: supply `training_examples` inline or point `dataset_path` to a JSON/JSONL file with `instruction`/`output` (and optional `input`). Set `include_actual_data=true` to auto-pull profile documents (default is false).
3) Trigger run:
```bash
curl -X POST http://localhost:8000/finetune -H "Content-Type: application/json" -d '{
  "profile_id": "finance",
  "base_model": "llama3.2",
  "learning_rate": 0.0002,
  "num_epochs": 1,
  "max_steps": 200,
  "batch_size": 4,
  "gradient_accumulation": 2,
  "lora_r": 16,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "include_actual_data": false,
  "training_examples": [
    {"instruction": "Summarize quarterly report risks", "output": "Highlight credit, liquidity, and market risks."}
  ]
}'
```
4) Monitor: `curl http://localhost:8000/finetune/<job_id>` to check status.
5) Use it: finetuned models auto-register to the model list (`GET /models`) and are served for their profile. Queries to `/ask` with that profile will use the finetuned model; override by passing a different `model_name` if needed.
6) Options to refine quality:
   - Increase `num_epochs` or `max_steps` for more training.
   - Adjust `learning_rate`, `batch_size`, and `gradient_accumulation` for stability.
   - Tune LoRA knobs: `lora_r`, `lora_alpha`, `lora_dropout`.
   - Add richer instructions and domain outputs in `training_examples` or expand the dataset file.

### Automated finetuning from Qdrant context

1) Generate synthetic QA from existing vector chunks and kick off finetune in one call:
```bash
curl -X POST http://localhost:8000/finetune/auto -H "Content-Type: application/json" -d '{
  "profile_id": "finance",
  "subscription_id": "default",
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
2) The API samples Qdrant chunks for the profile, generates questions/answers per chunk, writes a JSONL dataset under `finetune_artifacts/<profile>/auto_datasets/`, and starts Unsloth finetuning with that dataset (no extra data pulled by default).
3) Track progress: `curl http://localhost:8000/finetune/<job_id>`.
4) Discover models: `curl http://localhost:8000/models` to see finetuned entries served per profile.
5) Improve quality:
   - Raise `questions_per_chunk` or `max_points` for more coverage.
   - Swap `generation_model` to a stronger generator if available.
   - Append curated `training_examples` to the auto request for targeted behaviors.
   - Increase `num_epochs`/`max_steps` and adjust `learning_rate` for tougher domains.
