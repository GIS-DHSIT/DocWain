# DocWain Teams App

This folder contains the manifest scaffold to run DocWain as a Microsoft Teams personal bot. Replace the placeholders and package the app for upload.

## Configure the bot endpoint

- Deploy the API where Teams can reach it (e.g., `https://your-hostname`).
- Set the Teams messaging endpoint to `https://your-hostname/teams/messages`.
- Provide a shared secret to secure requests:
  - Set `TEAMS_SHARED_SECRET=<long-random-string>` in the DocWain API environment.
  - In Teams (or your bot registration), send the same value in the `Authorization: Bearer <secret>` or `x-teams-shared-secret` header.
- Optional defaults (override as needed):
  - `TEAMS_DEFAULT_PROFILE` (profile/collection ID to query)
  - `TEAMS_DEFAULT_SUBSCRIPTION`
  - `TEAMS_DEFAULT_MODEL` (e.g., `llama3.2`)
  - `TEAMS_DEFAULT_PERSONA`

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
