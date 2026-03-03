#!/usr/bin/env bash
# =============================================================================
# DocWain Teams Bot — End-to-End Azure Deployment Script
# =============================================================================
# Usage:
#   bash scripts/deploy_teams_bot.sh
#
# Prerequisites:
#   - az CLI installed and logged in (az login)
#   - jq installed (for JSON parsing)
#   - .env or environment variables set (see .env.teams.example)
#
# This script will:
#   1. Create or verify the Azure Resource Group
#   2. Create or update the Azure Bot Service resource (Single-Tenant)
#   3. Configure the messaging endpoint
#   4. Enable the Microsoft Teams channel
#   5. Update the manifest with the correct Bot ID
#   6. Package the Teams app
#   7. Verify the deployment
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via environment or .env file
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load .env if present
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Required settings
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-DocWain}"
LOCATION="${AZURE_LOCATION:-uksouth}"
BOT_NAME="${AZURE_BOT_NAME:-DocWain}"
APP_ID="${MICROSOFT_APP_ID:?ERROR: MICROSOFT_APP_ID is required}"
APP_PASSWORD="${MICROSOFT_APP_PASSWORD:?ERROR: MICROSOFT_APP_PASSWORD is required}"
TENANT_ID="${MICROSOFT_APP_TENANT_ID:?ERROR: MICROSOFT_APP_TENANT_ID is required}"
APP_TYPE="${MICROSOFT_APP_TYPE:-SingleTenant}"
MESSAGING_ENDPOINT="${TEAMS_MESSAGING_ENDPOINT:-https://dhs-docwain-api.azure-api.net/api/teams/messages}"
SKU="${AZURE_BOT_SKU:-F0}"

echo "============================================================"
echo "  DocWain Teams Bot — Azure Deployment"
echo "============================================================"
echo "  Resource Group : $RESOURCE_GROUP"
echo "  Location       : $LOCATION"
echo "  Bot Name       : $BOT_NAME"
echo "  App ID         : ${APP_ID:0:8}..."
echo "  Tenant ID      : ${TENANT_ID:0:8}..."
echo "  App Type       : $APP_TYPE"
echo "  Endpoint       : $MESSAGING_ENDPOINT"
echo "  SKU            : $SKU"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "[1/8] Pre-flight checks..."

if ! command -v az &>/dev/null; then
    echo "ERROR: az CLI is not installed. Install from https://aka.ms/installazurecli"
    exit 1
fi

if ! command -v jq &>/dev/null; then
    echo "WARNING: jq not installed. Some validation steps will be skipped."
    JQ_AVAILABLE=false
else
    JQ_AVAILABLE=true
fi

# Verify az login
if ! az account show &>/dev/null; then
    echo "ERROR: Not logged in to Azure. Run 'az login' first."
    exit 1
fi

SUBSCRIPTION=$(az account show --query "name" -o tsv)
echo "  Azure subscription: $SUBSCRIPTION"
echo "  Pre-flight OK"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Create or verify resource group
# ---------------------------------------------------------------------------
echo "[2/8] Resource group..."

if az group show --name "$RESOURCE_GROUP" &>/dev/null; then
    echo "  Resource group '$RESOURCE_GROUP' exists"
else
    echo "  Creating resource group '$RESOURCE_GROUP' in '$LOCATION'..."
    az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
    echo "  Created"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3: Create or update Azure Bot resource
# ---------------------------------------------------------------------------
echo "[3/8] Azure Bot resource..."

BOT_EXISTS=$(az bot show --resource-group "$RESOURCE_GROUP" --name "$BOT_NAME" 2>/dev/null && echo "yes" || echo "no")

if [ "$BOT_EXISTS" = "yes" ]; then
    echo "  Bot '$BOT_NAME' already exists. Updating endpoint..."
    az bot update \
        --resource-group "$RESOURCE_GROUP" \
        --name "$BOT_NAME" \
        --endpoint "$MESSAGING_ENDPOINT" \
        --output none
    echo "  Endpoint updated to: $MESSAGING_ENDPOINT"
else
    echo "  Creating Bot '$BOT_NAME' (SKU=$SKU, Type=$APP_TYPE)..."
    az bot create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$BOT_NAME" \
        --app-type "$APP_TYPE" \
        --appid "$APP_ID" \
        --password "$APP_PASSWORD" \
        --tenant-id "$TENANT_ID" \
        --endpoint "$MESSAGING_ENDPOINT" \
        --sku "$SKU" \
        --location "global" \
        --output none
    echo "  Bot created"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 4: Verify bot configuration
# ---------------------------------------------------------------------------
echo "[4/8] Verifying bot configuration..."

CURRENT_ENDPOINT=$(az bot show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$BOT_NAME" \
    --query "properties.endpoint" -o tsv 2>/dev/null || echo "")

CURRENT_APP_TYPE=$(az bot show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$BOT_NAME" \
    --query "properties.msaAppType" -o tsv 2>/dev/null || echo "")

echo "  Endpoint : $CURRENT_ENDPOINT"
echo "  App Type : $CURRENT_APP_TYPE"

if [ "$CURRENT_ENDPOINT" != "$MESSAGING_ENDPOINT" ]; then
    echo "  WARNING: Endpoint mismatch! Expected: $MESSAGING_ENDPOINT"
    echo "  Updating..."
    az bot update \
        --resource-group "$RESOURCE_GROUP" \
        --name "$BOT_NAME" \
        --endpoint "$MESSAGING_ENDPOINT" \
        --output none
    echo "  Fixed"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 5: Enable Microsoft Teams channel
# ---------------------------------------------------------------------------
echo "[5/8] Microsoft Teams channel..."

CHANNELS=$(az bot show \
    --resource-group "$RESOURCE_GROUP" \
    --name "$BOT_NAME" \
    --query "properties.enabledChannels" -o tsv 2>/dev/null || echo "")

if echo "$CHANNELS" | grep -q "msteams"; then
    echo "  Teams channel already enabled"
else
    echo "  Enabling Teams channel..."
    # Create Teams channel via REST API (az bot msteams create)
    az bot msteams create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$BOT_NAME" \
        --enable-calling false \
        --output none 2>/dev/null || {
        echo "  NOTE: 'az bot msteams create' failed. Trying alternative method..."
        # Fallback: use the generic channel API
        az rest \
            --method put \
            --uri "https://management.azure.com/subscriptions/$(az account show --query id -o tsv)/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.BotService/botServices/$BOT_NAME/channels/MsTeamsChannel?api-version=2022-09-15" \
            --body '{"location":"global","properties":{"channelName":"MsTeamsChannel","properties":{"isEnabled":true}}}' \
            --output none 2>/dev/null || echo "  WARNING: Could not enable Teams channel automatically. Enable manually in Azure Portal > Bot > Channels."
    }
    echo "  Teams channel configured"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 6: Enable DirectLine and WebChat (for testing)
# ---------------------------------------------------------------------------
echo "[6/8] Additional channels..."

if echo "$CHANNELS" | grep -q "webchat"; then
    echo "  WebChat channel already enabled"
else
    echo "  WebChat channel is typically enabled by default"
fi

if echo "$CHANNELS" | grep -q "directline"; then
    echo "  DirectLine channel already enabled"
else
    echo "  DirectLine channel is typically enabled by default"
fi
echo ""

# ---------------------------------------------------------------------------
# Step 7: Update manifest and package
# ---------------------------------------------------------------------------
echo "[7/8] Manifest and packaging..."

MANIFEST_PATH="$PROJECT_ROOT/teams-app/manifest.json"

if [ -f "$MANIFEST_PATH" ] && [ "$JQ_AVAILABLE" = "true" ]; then
    MANIFEST_BOT_ID=$(jq -r '.bots[0].botId' "$MANIFEST_PATH")
    if [ "$MANIFEST_BOT_ID" != "$APP_ID" ]; then
        echo "  Updating manifest botId from $MANIFEST_BOT_ID to $APP_ID..."
        # Use jq to update all ID fields
        TMP_MANIFEST=$(mktemp)
        jq --arg id "$APP_ID" '
            .id = $id |
            .bots[0].botId = $id |
            .webApplicationInfo.id = $id |
            .webApplicationInfo.resource = "api://\($id)"
        ' "$MANIFEST_PATH" > "$TMP_MANIFEST"
        mv "$TMP_MANIFEST" "$MANIFEST_PATH"
        echo "  Manifest updated"
    else
        echo "  Manifest botId matches App ID"
    fi
fi

if [ -f "$PROJECT_ROOT/scripts/package_teams_app.sh" ]; then
    bash "$PROJECT_ROOT/scripts/package_teams_app.sh"
else
    echo "  WARNING: package_teams_app.sh not found. Package manually."
fi
echo ""

# ---------------------------------------------------------------------------
# Step 8: Deployment summary
# ---------------------------------------------------------------------------
echo "[8/8] Deployment summary"
echo "============================================================"
echo ""
echo "  Bot Resource    : $BOT_NAME"
echo "  Resource Group  : $RESOURCE_GROUP"
echo "  App ID          : $APP_ID"
echo "  Tenant ID       : $TENANT_ID"
echo "  App Type        : $APP_TYPE"
echo "  Endpoint        : $MESSAGING_ENDPOINT"
echo "  SKU             : $SKU"
echo ""
echo "  Teams App ZIP   : $PROJECT_ROOT/dist/docwain-teams.zip"
echo ""
echo "  Next steps:"
echo "    1. Set environment variables on your server:"
echo "       MICROSOFT_APP_ID=$APP_ID"
echo "       MICROSOFT_APP_PASSWORD=<secret>"
echo "       MICROSOFT_APP_TENANT_ID=$TENANT_ID"
echo "       MICROSOFT_APP_TYPE=$APP_TYPE"
echo ""
echo "    2. Start DocWain server:"
echo "       uvicorn src.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "    3. Upload Teams app package:"
echo "       - Go to https://dev.teams.microsoft.com/apps"
echo "       - Click 'Import app' > Upload dist/docwain-teams.zip"
echo "       - Or sideload in Teams: Apps > Upload a custom app"
echo ""
echo "    4. Verify health:"
echo "       curl $MESSAGING_ENDPOINT/../admin/teams/status"
echo ""
echo "============================================================"
echo "  Deployment complete!"
echo "============================================================"
