## DocWain Microsoft Teams Integration

- POST `(/teams/messages)` is the FastAPI entrypoint for all Teams activities. It accepts the raw Teams activity JSON payload and returns a Bot Framework-style message response.
- File uploads from Teams are stored in `TEAMS_UPLOAD_DIR` (defaults to `/tmp`) while they are processed and embedded into Qdrant collections scoped by the Teams conversation id.
- Configure `TEAMS_SHARED_SECRET` (optional) to validate calls from your bot. Provide the header when invoking the endpoint from your bot service.
- To persist uploaded files, set `TEAMS_BLOB_CONNECTION_STRING` and `TEAMS_BLOB_CONTAINER` (default: `local-uploads`). `TEAMS_BLOB_PATH_PREFIX` can be used to namespace blobs (default: `teams`). Files are uploaded after processing and also cleaned up from `TEAMS_UPLOAD_DIR`.

### Deployment and Teams manifest updates
- Expose the `/teams/messages` route on your deployment host (e.g., `https://<api-host>/teams/messages`).
- In the Teams bot manifest, set the messaging endpoint to the URL above and enable file uploads/attachments so Teams sends `file.download.info` payloads.
- Restart or redeploy the API after updating environment variables, then re-upload the manifest to your Teams tenant so the new endpoint is active.
