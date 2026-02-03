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

### Dialogue intelligence (persona + sentiment)
Enable DocWain's conversation intelligence layer and tune routing sensitivity with:
```bash
export DOCWAIN_PERSONA_ENABLED=true
export DOCWAIN_SENTIMENT_ENABLED=true
export DOCWAIN_SMALLTALK_MODEL=llama3.2   # optional: model used for intent/sentiment fallback
export DOCWAIN_INTENT_THRESHOLD=0.65
```

## Embedding Pickles from Blob Storage

How to run:
- Set `AZURE_STORAGE_CONNECTION_STRING` (raw or base64-encoded). `AZURE_BLOB_CONNECTION_STRING` is ignored.
- Set `DOCWAIN_BLOB_CONTAINER=document-content` or `AZURE_BLOB_CONTAINER_NAME=document-content`, and optionally `DOCWAIN_BLOB_PREFIX` to scope blobs (e.g. `pickles/`).
- Chat history uses `AZURE_CHAT_CONTAINER_NAME` (defaults to `chat-history`); document pickles use `DOCWAIN_BLOB_CONTAINER`/`AZURE_BLOB_CONTAINER_NAME`.
- Start the API: `uvicorn src.main:app --reload`

Example embed call (processes available pickle blobs):
```bash
curl -X POST http://localhost:8000/api/documents/embed \\
  -H \"Content-Type: application/json\" \\
  -d '{\"max_blobs\": 5}'
```

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

## Tools Framework (API)

- All tools live under `/api/tools/*` with a standard response shape and correlation id. Use `/api/tools/run` for generic invocation:  
  `curl -X POST http://localhost:8000/api/tools/run -H "Content-Type: application/json" -d '{"tool_name":"translator","input":{"text":"hello","target_lang":"fr"}}'`
- Speech-to-Text: `curl -X POST http://localhost:8000/api/tools/stt/transcribe -F "audio_file=@sample.wav"` (returns transcript + segments; requires Whisper installed).
- Text-to-Speech: `curl -X POST http://localhost:8000/api/tools/tts/speak -H "Content-Type: application/json" -d '{"text":"hello docwain"}'` (streams WAV audio).
- Translator: `curl -X POST http://localhost:8000/api/tools/translator/translate -H "Content-Type: application/json" -d '{"text":"hola","target_lang":"en"}'`
- Creator/Tutor/Email draft examples:  
  `curl -X POST http://localhost:8000/api/tools/creator/generate -H "Content-Type: application/json" -d '{"content_type":"summary","text":"Safety checklist"}'`  
  `curl -X POST http://localhost:8000/api/tools/tutor/lesson -H "Content-Type: application/json" -d '{"topic":"network security","learning_level":"beginner"}'`  
  `curl -X POST http://localhost:8000/api/tools/email/draft -H "Content-Type: application/json" -d '{"intent":"follow up","recipient_role":"customer","tone":"concise"}'`
- Connectors and analysis: `/api/tools/db/query`, `/api/tools/code/docs`, `/api/tools/web/analyze`, `/api/tools/jira_confluence/summarize`, `/api/tools/resumes/analyze`, `/api/tools/medical/summarize`, `/api/tools/lawhere/analyze`.

## Profile-Isolated Document Understanding (Ollama)

DocWain now supports profile-isolated ingestion and a 3-stage Document Understanding pipeline:
1) Document Identification (doc name/type/properties)
2) Content Identification (sections, tables, images)
3) Content Understanding (summaries, entities, intent tags)

### Updated file tree
```
src/
  api/
    document_understanding_service.py
    profile_documents_api.py
    profiles_api.py
  doc_understanding/
    __init__.py
    content_map.py
    identify.py
    understand.py
  profiles/
    profile_store.py
  retrieval/
    intent_router.py
    profile_query.py
```

### API contracts & examples

#### Create profile
```
POST /api/profiles
{
  "subscription_id": "sub-123",
  "profile_name": "Finance",
  "profile_id": "optional-uuid"
}
```

#### Upload document under profile (extract + understand + embed)
```
POST /api/profiles/{profile_id}/documents/upload
FormData:
  subscription_id: sub-123
  profile_name: Finance
  file: @invoice.pdf
```

#### Run understanding for an existing document
```
POST /api/profiles/{profile_id}/documents/{document_id}/understand
{
  "subscription_id": "sub-123",
  "profile_name": "Finance",
  "model_name": "llama3.2",
  "embed_after": true
}
```

#### Query with explicit profile_id
```
POST /api/profiles/{profile_id}/query
{
  "subscription_id": "sub-123",
  "query": "Summarize the latest invoice totals",
  "model_name": "llama3.2",
  "top_k": 6
}
```

### Migration guide (existing data)
1) Backfill `profile_name` for all documents in MongoDB (copy from profiles or set a default).
2) Backfill `document_type` (run `/profiles/{profile_id}/documents/{document_id}/understand` or batch understand).
3) Re-embed documents so Qdrant payloads include `profile_name`, `document_type`, and `chunk_kind`.
4) Validate retrieval filters: ensure all profile queries include `subscription_id` + `profile_id` filters.

### Running tests
```bash
pytest src/tests
```
