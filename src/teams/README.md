# DocWain Teams App

This folder contains the manifest scaffold to run DocWain as a Microsoft Teams personal bot. Replace the placeholders and package the app for upload.

## Configure the bot endpoint

- Deploy the API where Teams can reach it (e.g., `https://your-hostname`).
- Set the Teams messaging endpoint to `https://your-hostname/teams/messages`.
- Provide a shared secret to secure requests:
  - Set `TEAMS_SHARED_SECRET=<long-random-string>` in the DocWain API environment.
  - In Teams (or your bot registration), send the same value in the `Authorization: Bearer <secret>` or `x-teams-shared-secret` header.
- Optional request signatures:
  - Set `TEAMS_SIGNATURE_ENABLED=true` to validate `x-teams-signature` as `sha256=<hex>` of the raw body using `TEAMS_SHARED_SECRET`.
- Optional defaults (override as needed):
  - `TEAMS_DEFAULT_PROFILE` (profile/collection ID to query)
  - `TEAMS_DEFAULT_SUBSCRIPTION`
  - `TEAMS_DEFAULT_MODEL` (e.g., `llama3.2`)
  - `TEAMS_DEFAULT_PERSONA`
  - `TEAMS_SESSION_AS_SUBSCRIPTION` (use the Teams conversation id as subscription id)
  - `TEAMS_PROFILE_PER_USER` (use the Teams user id as profile id)
  - `TEAMS_MAX_ATTACHMENT_MB` (attachment size limit, default 50)
  - `TEAMS_HTTP_TIMEOUT_SEC` (download timeout, default 20)
  - `TEAMS_HTTP_RETRIES` (download retries, default 2)
  - `TEAMS_BOT_ACCESS_TOKEN` (optional Bot Framework token for inline image downloads)
  - `TEAMS_BLOB_CONNECTION_STRING`, `TEAMS_BLOB_CONTAINER`, `TEAMS_BLOB_PATH_PREFIX` (optional blob upload)

## Troubleshooting uploads

- If a file upload fails immediately, confirm the attachment uses the Teams file download info payload.
- Inline images require Bot Framework access for `/v3/attachments` URLs. Set `TEAMS_BOT_ACCESS_TOKEN` or upload the image as a file attachment.
- Large files may be rejected if they exceed `TEAMS_MAX_ATTACHMENT_MB`.

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
