# DocWain Teams App

Deploy DocWain as a Microsoft Teams bot for document intelligence directly in your Teams conversations.

## Architecture

```
Teams Client → Azure Bot Service → POST /api/teams/messages → DocWainTeamsBot
                                                                    ↓
                                                              TeamsChatService → RAG Pipeline
                                                                    ↓
                                                              Adaptive Card Response
```

Two adapter paths:
- **Bot Framework SDK** (preferred): Full `TeamsActivityHandler` with typing indicators, file consent, and Adaptive Cards
- **Legacy HTTP adapter**: Stateless REST fallback when `botbuilder` is not installed

## Azure Bot Resource Setup

1. **Create an Azure Bot resource** (Single-Tenant) in the Azure Portal
2. Note the **Microsoft App ID** (a GUID) and the **Tenant ID**
3. Under **Certificates & Secrets**, create a new client secret
4. Under **Channels**, enable the **Microsoft Teams** channel
5. Set the **Messaging endpoint** to `https://<your-host>/api/teams/messages`

## Environment Variables

Set these in your `.env` file:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MICROSOFT_APP_ID` | Yes | — | Azure Bot App ID (GUID) |
| `MICROSOFT_APP_PASSWORD` | Yes | — | Azure Bot client secret |
| `MICROSOFT_APP_TENANT_ID` | Yes (Single-Tenant) | — | Azure AD / Entra ID tenant ID (GUID) |
| `MICROSOFT_APP_TYPE` | No | `SingleTenant` | App type: `SingleTenant`, `MultiTenant`, or `UserAssignedMSI` |
| `TEAMS_SHARED_SECRET` | No | — | Shared-secret fallback for local dev |
| `TEAMS_SIGNATURE_ENABLED` | No | `false` | Enable HMAC request signatures |
| `TEAMS_DEFAULT_MODEL` | No | `llama3.2` | Default LLM model |
| `TEAMS_DEFAULT_PERSONA` | No | `Document Assistant` | Default persona |
| `TEAMS_SESSION_AS_SUBSCRIPTION` | No | `true` | Use conversation ID as subscription scope |
| `TEAMS_PROFILE_PER_USER` | No | `true` | Use Teams user ID as profile scope |
| `TEAMS_MAX_ATTACHMENT_MB` | No | `50` | Max file upload size in MB |
| `TEAMS_HTTP_TIMEOUT_SEC` | No | `20` | File download timeout |
| `TEAMS_HTTP_RETRIES` | No | `2` | File download retry count |
| `TEAMS_BOT_ACCESS_TOKEN` | No | — | Fallback token for secured downloads |
| `TEAMS_DIAG_MODE` | No | `false` | Enable diagnostic logging |
| `DOCWAIN_WEB_URL` | No | `https://www.docwain.ai` | Web app URL for card links |

## Single-Tenant Authentication

DocWain's Azure Bot is registered as **Single Tenant** (`msaAppType: SingleTenant`).
This means the bot only accepts tokens issued by the configured Entra ID tenant.

**Required environment variables for single-tenant:**
- `MICROSOFT_APP_ID` — the bot's App Registration client ID
- `MICROSOFT_APP_PASSWORD` — the App Registration client secret
- `MICROSOFT_APP_TENANT_ID` — your Entra ID tenant GUID (e.g., `13a1a520-d90b-4bfe-ada0-2be3d1f3c582`)

The adapter passes `channel_auth_tenant` to `BotFrameworkAdapterSettings`, which restricts
token validation to `https://login.microsoftonline.com/<tenant_id>` instead of the
multi-tenant common endpoint. Without this, all incoming Teams activities will fail with `401`.

To verify your configuration:
```bash
curl https://<your-host>/api/admin/teams/status
```

The response should show `"app_type": "SingleTenant"` and `"tenant_id_configured": true`.

## Manifest Update

Edit `teams-app/manifest.json`:

1. Replace the `botId` in the `bots` array with your **Microsoft App ID**
2. Update `validDomains` with your API hostname
3. Update `webApplicationInfo.id` with the same App ID
4. Ensure `color.png` (192x192) and `outline.png` (32x32) are in `teams-app/`

## Packaging

Use the packaging script to create the upload-ready ZIP:

```bash
bash scripts/package_teams_app.sh
```

This produces `dist/docwain-teams.zip` containing `manifest.json`, `color.png`, and `outline.png`.

Upload the ZIP via the **Teams Developer Portal** or **Teams Admin Center**.

## Teams Experience

### Document Upload
- Attach files (PDF, Word, images) directly in chat
- DocWain downloads, extracts, and embeds automatically
- A success card appears with tool shortcuts

### Asking Questions
- Type natural language questions about uploaded documents
- Responses come as Adaptive Cards with collapsible sources
- Conversation history is preserved for context-aware follow-ups

### Tool Actions
- **Summarize recent** — summarizes latest uploads
- **Extract fields** — invoice/contract field extraction presets
- **Generate content** — document-grounded content generation
- **List docs** — shows recent uploads with metadata
- **Preferences** — interactive model/persona selection

### Preferences
The Preferences card lets users choose their model and persona via dropdown menus. Settings persist across messages in the same conversation.

## Testing Checklist

1. Send a text message — bot responds with an answer card
2. Upload a PDF — bot confirms ingestion with a success card
3. Click "Summarize recent" — bot returns a summary of uploaded docs
4. Click "Preferences" — model/persona dropdowns appear
5. Type "help" — bot shows the help card

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Bot doesn't respond | Endpoint not reachable | Check messaging endpoint URL and firewall |
| `401 Unauthorized` on send | App ID/secret mismatch | Verify `MICROSOFT_APP_ID` and `MICROSOFT_APP_PASSWORD` |
| `401 Unauthorized` on receive | Missing tenant ID (Single-Tenant) | Set `MICROSOFT_APP_TENANT_ID` to your Entra ID tenant GUID |
| `403 Forbidden` outbound | Service URL not trusted | DocWain auto-trusts; check Azure Bot channel config |
| File upload fails | Missing download URL | Ensure the file was sent via Teams attachment picker |
| Inline images rejected | No connector token | Set `MICROSOFT_APP_ID`/`PASSWORD` or `TEAMS_BOT_ACCESS_TOKEN` |
| Empty responses | No documents embedded | Upload and wait for success card before asking questions |

## Files

| File | Purpose |
|------|---------|
| `src/teams/bot_app.py` | Bot Framework ActivityHandler |
| `src/teams/adapter.py` | Legacy HTTP adapter |
| `src/teams/logic.py` | Chat orchestration + context building |
| `src/teams/state.py` | Redis/in-memory state store |
| `src/teams/attachments.py` | File download + ingestion |
| `src/teams/tools.py` | Adaptive Card action routing |
| `src/teams/cards/` | Adaptive Card JSON templates |
| `teams-app/manifest.json` | Current Teams manifest |
| `scripts/package_teams_app.sh` | Packaging script |
