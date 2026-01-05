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
- Redis backs chat history, feedback, and answer caching; configure it with `REDIS_URL` (`rediss://default:<key>@<host>:6380/0`) or `REDIS_CONNECTION_STRING` (Azure style `host:port,password=...,ssl=True`), or explicitly set `REDIS_HOST`, `REDIS_PORT`, `REDIS_USERNAME`, `REDIS_PASSWORD`, `REDIS_SSL`, and `REDIS_DB`.
- Local development: `docker-compose up --build` now starts a Redis 7 container and wires the app to it with no auth and SSL disabled. Verify it is running with `docker compose exec redis redis-cli ping`.
- If you point at your own Redis instance, update the env vars above and restart the service so the client reinitializes with the new settings.
