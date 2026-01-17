# DocWain Teams App

This folder contains the manifest scaffold to run DocWain as a Microsoft Teams personal bot. Replace the placeholders and package the app for upload.

## Configure the bot endpoint

- Deploy the API where Teams can reach it (e.g., `https://your-hostname`).
- Set the Teams messaging endpoint to `https://your-hostname/teams/messages`.
- Use Azure Bot Service credentials (Bot Framework JWT validation):
  - Set `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD`/`MICROSOFT_APP_PWD`.
  - Teams will POST with a bearer JWT; the Bot Framework adapter validates it automatically.
- (Optional) Shared-secret fallback for local testing:
  - Set `TEAMS_SHARED_SECRET=<long-random-string>` and send it via `Authorization: Bearer <secret>` or `x-teams-shared-secret`.
- Optional request signatures:
  - Set `TEAMS_SIGNATURE_ENABLED=true` to validate `x-teams-signature` as `sha256=<hex>` of the raw body using `TEAMS_SHARED_SECRET`.
- Optional defaults (override as needed):
  - `TEAMS_DEFAULT_PROFILE`, `TEAMS_DEFAULT_SUBSCRIPTION`
  - `TEAMS_DEFAULT_MODEL`, `TEAMS_DEFAULT_PERSONA`
  - `TEAMS_SESSION_AS_SUBSCRIPTION` (use the Teams conversation id as subscription id)
  - `TEAMS_PROFILE_PER_USER` (use the Teams user id as profile id)
  - `TEAMS_MAX_ATTACHMENT_MB` (attachment size limit, default 50)
  - `TEAMS_HTTP_TIMEOUT_SEC` (download timeout, default 20)
  - `TEAMS_HTTP_RETRIES` (download retries, default 2)
  - `TEAMS_BOT_ACCESS_TOKEN` (optional connector token fallback for secure downloads)
  - `TEAMS_BLOB_CONNECTION_STRING`, `TEAMS_BLOB_CONTAINER`, `TEAMS_BLOB_PATH_PREFIX` (optional blob upload)
  - `DOCWAIN_WEB_URL` or `TEAMS_WEB_APP_URL` for Adaptive Card links

## Teams experience

- Uploading files automatically downloads them with a connector token, extracts content, and trains embeddings in the Teams-scoped Qdrant collection (`subscription_id` derived from conversation id when enabled).
- After ingestion, the bot returns Adaptive Cards with tool shortcuts (summaries, field extraction, list docs, model/persona switch, open web).
- Adaptive Card submits route to the tool handlers; responses are also Adaptive Cards.
- Manual “train/index” commands are not needed in Teams; ingestion happens on upload.

## Troubleshooting uploads

- If a file upload fails immediately, confirm the attachment uses the Teams file download info payload.
- Inline images require Bot Framework access for `/v3/attachments` URLs. Set `MICROSOFT_APP_ID/PASSWORD` (preferred) or `TEAMS_BOT_ACCESS_TOKEN`, or upload the image as a file attachment.
- Large files may be rejected if they exceed `TEAMS_MAX_ATTACHMENT_MB`.

## Azure Bot Service quick setup

1. Create an Azure Bot resource (Multi Tenant) and note the `Microsoft App ID`.
2. Create a client secret for the bot app registration; set `MICROSOFT_APP_PASSWORD` (or `MICROSOFT_APP_PWD`) in DocWain.
3. Set the messaging endpoint to `https://<your-host>/api/teams/messages` and enable the Teams channel.
4. Package and upload `src/teams/manifest.json` (with updated IDs and icons) via the Teams Developer Portal.
5. Verify uploads: attach a file in the Teams chat, wait for the upload success card, then run a tool action (summarize/extract/list) without sending a manual train command.

## Update the manifest

Edit `teams/manifest.json`:

- Replace all `00000000-...` with your Azure Bot resource ID (App ID).
- Replace `YOUR_HOSTNAME` with the public host the API is served from.
- Update `packageName`, `name`, and descriptions if desired.

## Add icons (required by Teams)

Place two PNG files in this folder before zipping:

- `color.png` (192x192)
- `outline.png` (32x32, transparent background)

## Package the app

Zip the manifest and icons from inside the `teams` folder:

```bash
cd teams
zip docwain-teams.zip manifest.json color.png outline.png
```

Upload `docwain-teams.zip` via Teams App Studio (or the Teams Developer Portal), install for personal scope, and test by sending a message; the bot will route it to `/teams/messages` and return DocWain answers with sources.
