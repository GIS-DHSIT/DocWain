"""
DocWain Teams Bot — Azure SDK Deployment Module

Programmatic management of Azure Bot Service resources for the DocWain Teams bot.
Uses azure-mgmt-botservice + azure-identity for authentication and resource management.

Usage:
    from src.teams.azure_deploy import TeamsBotDeployer

    deployer = TeamsBotDeployer()
    deployer.deploy()           # Full end-to-end deployment
    deployer.verify()           # Health check
    deployer.enable_teams()     # Enable Teams channel only

CLI Usage:
    python -m src.teams.azure_deploy deploy
    python -m src.teams.azure_deploy verify
    python -m src.teams.azure_deploy status

Requirements:
    pip install azure-mgmt-botservice azure-mgmt-resource azure-identity
"""

from __future__ import annotations

import json
import logging

from src.utils.logging_utils import get_logger
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BotDeployConfig:
    """Configuration for Azure Bot deployment."""

    subscription_id: str = ""
    resource_group: str = "DocWain"
    location: str = "global"
    bot_name: str = "DocWain"
    app_id: str = ""
    app_password: str = ""
    tenant_id: str = ""
    app_type: str = "SingleTenant"  # SingleTenant | MultiTenant | UserAssignedMSI
    messaging_endpoint: str = "https://dhs-docwain-api.azure-api.net/api/teams/messages"
    sku: str = "F0"
    display_name: str = "DocWain AI Assistant"
    description: str = "DocWain document intelligence bot for Microsoft Teams."

    @classmethod
    def from_env(cls) -> BotDeployConfig:
        """Load configuration from environment variables."""
        return cls(
            subscription_id=os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            resource_group=os.getenv("AZURE_RESOURCE_GROUP", "DocWain"),
            location=os.getenv("AZURE_LOCATION", "global"),
            bot_name=os.getenv("AZURE_BOT_NAME", "DocWain"),
            app_id=os.getenv("MICROSOFT_APP_ID", ""),
            app_password=os.getenv("MICROSOFT_APP_PASSWORD", ""),
            tenant_id=os.getenv("MICROSOFT_APP_TENANT_ID", ""),
            app_type=os.getenv("MICROSOFT_APP_TYPE", "SingleTenant"),
            messaging_endpoint=os.getenv(
                "TEAMS_MESSAGING_ENDPOINT",
                "https://dhs-docwain-api.azure-api.net/api/teams/messages",
            ),
            sku=os.getenv("AZURE_BOT_SKU", "F0"),
        )

    def validate(self) -> None:
        """Raise ValueError if required fields are missing."""
        missing = []
        if not self.app_id:
            missing.append("MICROSOFT_APP_ID")
        if not self.app_password:
            missing.append("MICROSOFT_APP_PASSWORD")
        if self.app_type.lower() == "singletenant" and not self.tenant_id:
            missing.append("MICROSOFT_APP_TENANT_ID")
        if missing:
            raise ValueError(f"Missing required config: {', '.join(missing)}")

# ---------------------------------------------------------------------------
# Deployer
# ---------------------------------------------------------------------------

class TeamsBotDeployer:
    """End-to-end Azure Bot Service deployer using Azure SDK."""

    def __init__(self, config: Optional[BotDeployConfig] = None):
        self.config = config or BotDeployConfig.from_env()
        self._bot_client = None
        self._resource_client = None

    # -- SDK client lazy init --------------------------------------------------

    def _get_credential(self):
        """Get Azure credential (DefaultAzureCredential supports az login, managed identity, etc.)."""
        try:
            from azure.identity import DefaultAzureCredential
            return DefaultAzureCredential()
        except ImportError:
            raise ImportError(
                "azure-identity is required. Install with: pip install azure-identity"
            )

    def _get_bot_client(self):
        """Get or create Azure Bot Service management client."""
        if self._bot_client is None:
            try:
                from azure.mgmt.botservice import AzureBotService
            except ImportError:
                raise ImportError(
                    "azure-mgmt-botservice is required. Install with: pip install azure-mgmt-botservice"
                )
            credential = self._get_credential()
            sub_id = self.config.subscription_id or self._resolve_subscription_id()
            self._bot_client = AzureBotService(credential, sub_id)
        return self._bot_client

    def _get_resource_client(self):
        """Get or create Azure Resource management client."""
        if self._resource_client is None:
            try:
                from azure.mgmt.resource import ResourceManagementClient
            except ImportError:
                raise ImportError(
                    "azure-mgmt-resource is required. Install with: pip install azure-mgmt-resource"
                )
            credential = self._get_credential()
            sub_id = self.config.subscription_id or self._resolve_subscription_id()
            self._resource_client = ResourceManagementClient(credential, sub_id)
        return self._resource_client

    def _resolve_subscription_id(self) -> str:
        """Resolve subscription ID from az CLI if not configured."""
        try:
            result = subprocess.run(
                ["az", "account", "show", "--query", "id", "-o", "tsv"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                sub_id = result.stdout.strip()
                self.config.subscription_id = sub_id
                return sub_id
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        raise ValueError(
            "Cannot determine Azure subscription ID. Set AZURE_SUBSCRIPTION_ID "
            "or run 'az login'."
        )

    # -- Resource Group --------------------------------------------------------

    def ensure_resource_group(self) -> Dict[str, Any]:
        """Create or verify the resource group."""
        client = self._get_resource_client()
        logger.info("Ensuring resource group '%s' in '%s'...", self.config.resource_group, self.config.location)

        rg = client.resource_groups.create_or_update(
            self.config.resource_group,
            {"location": self.config.location if self.config.location != "global" else "uksouth"},
        )
        logger.info("Resource group ready: %s", rg.name)
        return {"name": rg.name, "location": rg.location, "provisioning_state": rg.properties.provisioning_state}

    # -- Bot Resource ----------------------------------------------------------

    def create_or_update_bot(self) -> Dict[str, Any]:
        """Create or update the Azure Bot Service resource."""
        from azure.mgmt.botservice.models import (
            Bot,
            BotProperties,
            Sku,
        )

        self.config.validate()
        client = self._get_bot_client()

        logger.info("Creating/updating bot '%s'...", self.config.bot_name)

        bot_properties = BotProperties(
            display_name=self.config.display_name,
            description=self.config.description,
            endpoint=self.config.messaging_endpoint,
            msa_app_id=self.config.app_id,
            msa_app_type=self.config.app_type,
            msa_app_tenant_id=self.config.tenant_id if self.config.app_type.lower() == "singletenant" else None,
        )

        bot = Bot(
            location="global",
            sku=Sku(name=self.config.sku),
            kind="azurebot",
            properties=bot_properties,
        )

        result = client.bots.create(
            resource_group_name=self.config.resource_group,
            resource_name=self.config.bot_name,
            parameters=bot,
        )

        logger.info("Bot '%s' ready | endpoint=%s", result.name, result.properties.endpoint)
        return {
            "name": result.name,
            "endpoint": result.properties.endpoint,
            "app_id": result.properties.msa_app_id,
            "app_type": result.properties.msa_app_type,
            "provisioning_state": result.properties.provisioning_state,
        }

    # -- Channels --------------------------------------------------------------

    def enable_teams_channel(self) -> Dict[str, Any]:
        """Enable the Microsoft Teams channel on the bot."""
        from azure.mgmt.botservice.models import (
            BotChannel,
            MsTeamsChannel,
            MsTeamsChannelProperties,
        )

        client = self._get_bot_client()
        logger.info("Enabling Teams channel on '%s'...", self.config.bot_name)

        channel = BotChannel(
            location="global",
            properties=MsTeamsChannel(
                properties=MsTeamsChannelProperties(
                    is_enabled=True,
                    enable_calling=False,
                ),
            ),
        )

        result = client.channels.create(
            resource_group_name=self.config.resource_group,
            resource_name=self.config.bot_name,
            channel_name="MsTeamsChannel",
            parameters=channel,
        )

        logger.info("Teams channel enabled")
        return {"channel": "MsTeamsChannel", "provisioning_state": result.properties.provisioning_state}

    # -- Endpoint Update -------------------------------------------------------

    def update_endpoint(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Update the bot's messaging endpoint."""
        client = self._get_bot_client()
        target_endpoint = endpoint or self.config.messaging_endpoint

        logger.info("Updating endpoint to '%s'...", target_endpoint)

        # Get current bot, update endpoint
        current = client.bots.get(
            resource_group_name=self.config.resource_group,
            resource_name=self.config.bot_name,
        )
        current.properties.endpoint = target_endpoint

        result = client.bots.update(
            resource_group_name=self.config.resource_group,
            resource_name=self.config.bot_name,
            properties=current.properties,
        )

        logger.info("Endpoint updated: %s", result.properties.endpoint)
        return {"endpoint": result.properties.endpoint}

    # -- Status ----------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Get the current bot resource status."""
        client = self._get_bot_client()

        try:
            bot = client.bots.get(
                resource_group_name=self.config.resource_group,
                resource_name=self.config.bot_name,
            )
        except Exception as exc:
            return {"exists": False, "error": str(exc)}

        # List channels
        channels = []
        try:
            channel_list = client.channels.list_by_resource_group(
                resource_group_name=self.config.resource_group,
                resource_name=self.config.bot_name,
            )
            for ch in channel_list:
                channels.append(ch.name)
        except Exception:
            pass

        return {
            "exists": True,
            "name": bot.name,
            "endpoint": bot.properties.endpoint,
            "app_id": bot.properties.msa_app_id,
            "app_type": bot.properties.msa_app_type,
            "tenant_id": bot.properties.msa_app_tenant_id or "",
            "sku": bot.sku.name if bot.sku else "unknown",
            "provisioning_state": bot.properties.provisioning_state,
            "channels": channels,
            "teams_enabled": "MsTeamsChannel" in channels,
        }

    # -- Manifest Sync ---------------------------------------------------------

    def sync_manifest(self) -> Dict[str, Any]:
        """Update the manifest.json with the correct App ID and repackage."""
        manifest_path = Path(__file__).parent.parent.parent / "teams-app" / "manifest.json"
        if not manifest_path.exists():
            return {"error": "manifest.json not found"}

        data = json.loads(manifest_path.read_text())
        updated = False

        if data.get("id") != self.config.app_id:
            data["id"] = self.config.app_id
            updated = True
        if data.get("bots", [{}])[0].get("botId") != self.config.app_id:
            data["bots"][0]["botId"] = self.config.app_id
            updated = True
        if data.get("webApplicationInfo", {}).get("id") != self.config.app_id:
            data["webApplicationInfo"]["id"] = self.config.app_id
            data["webApplicationInfo"]["resource"] = f"api://{self.config.app_id}"
            updated = True

        if updated:
            manifest_path.write_text(json.dumps(data, indent=2) + "\n")
            logger.info("Manifest updated with App ID %s", self.config.app_id)
        else:
            logger.info("Manifest already up to date")

        return {"path": str(manifest_path), "updated": updated, "app_id": self.config.app_id}

    # -- Full Deploy -----------------------------------------------------------

    def deploy(self) -> Dict[str, Any]:
        """Full end-to-end deployment."""
        self.config.validate()
        results: Dict[str, Any] = {}

        logger.info("=" * 60)
        logger.info("DocWain Teams Bot — Full Deployment")
        logger.info("=" * 60)

        # 1. Resource group
        logger.info("[1/5] Resource group...")
        results["resource_group"] = self.ensure_resource_group()

        # 2. Bot resource
        logger.info("[2/5] Bot resource...")
        results["bot"] = self.create_or_update_bot()

        # 3. Teams channel
        logger.info("[3/5] Teams channel...")
        try:
            results["teams_channel"] = self.enable_teams_channel()
        except Exception as exc:
            logger.warning("Teams channel setup failed (may already exist): %s", exc)
            results["teams_channel"] = {"status": "skipped", "reason": str(exc)}

        # 4. Sync manifest
        logger.info("[4/5] Manifest sync...")
        results["manifest"] = self.sync_manifest()

        # 5. Package
        logger.info("[5/5] Packaging...")
        package_script = Path(__file__).parent.parent.parent / "scripts" / "package_teams_app.sh"
        if package_script.exists():
            try:
                subprocess.run(["bash", str(package_script)], check=True, capture_output=True, timeout=30)
                results["package"] = {"status": "ok", "path": "dist/docwain-teams.zip"}
            except subprocess.CalledProcessError as exc:
                results["package"] = {"status": "error", "stderr": exc.stderr.decode()[:200]}
        else:
            results["package"] = {"status": "skipped", "reason": "package script not found"}

        logger.info("=" * 60)
        logger.info("Deployment complete!")
        logger.info("=" * 60)

        return results

    def verify(self) -> Dict[str, Any]:
        """Verify the deployment is healthy."""
        status = self.get_status()

        checks = {
            "bot_exists": status.get("exists", False),
            "endpoint_correct": status.get("endpoint") == self.config.messaging_endpoint,
            "app_type_correct": (status.get("app_type") or "").lower() == self.config.app_type.lower(),
            "teams_enabled": status.get("teams_enabled", False),
        }
        checks["all_pass"] = all(checks.values())

        if not checks["all_pass"]:
            logger.warning("Verification failed: %s", {k: v for k, v in checks.items() if not v})
        else:
            logger.info("All verification checks passed")

        return {**status, "checks": checks}

# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for deployment operations."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m src.teams.azure_deploy <command>")
        print("")
        print("Commands:")
        print("  deploy     Full end-to-end deployment")
        print("  status     Show current bot status")
        print("  verify     Run verification checks")
        print("  teams      Enable Teams channel only")
        print("  endpoint   Update messaging endpoint only")
        print("  manifest   Sync manifest with App ID")
        print("  package    Package Teams app ZIP")
        sys.exit(1)

    command = sys.argv[1].lower()
    deployer = TeamsBotDeployer()

    if command == "deploy":
        result = deployer.deploy()
        print(json.dumps(result, indent=2, default=str))

    elif command == "status":
        result = deployer.get_status()
        print(json.dumps(result, indent=2, default=str))

    elif command == "verify":
        result = deployer.verify()
        print(json.dumps(result, indent=2, default=str))
        if not result.get("checks", {}).get("all_pass"):
            sys.exit(1)

    elif command == "teams":
        result = deployer.enable_teams_channel()
        print(json.dumps(result, indent=2, default=str))

    elif command == "endpoint":
        endpoint = sys.argv[2] if len(sys.argv) > 2 else None
        result = deployer.update_endpoint(endpoint)
        print(json.dumps(result, indent=2, default=str))

    elif command == "manifest":
        result = deployer.sync_manifest()
        print(json.dumps(result, indent=2, default=str))

    elif command == "package":
        package_script = Path(__file__).parent.parent.parent / "scripts" / "package_teams_app.sh"
        subprocess.run(["bash", str(package_script)], check=True)

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
