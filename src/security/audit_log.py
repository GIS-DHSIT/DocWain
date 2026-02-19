import logging
from typing import Optional, Dict

logger = logging.getLogger("security")

def log_event(event: str, meta: Optional[Dict] = None):
    logger.warning(f"SECURITY_EVENT={event} META={meta or {}}")
