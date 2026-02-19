import os
from fastapi import HTTPException, Request

def require_api_key(request: Request):
    expected = os.getenv("DEV_API_KEY")
    if not expected:
        return
    if request.headers.get("x-api-key") != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")