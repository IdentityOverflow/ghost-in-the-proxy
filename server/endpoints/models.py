"""Models listing endpoint."""
import time

from fastapi import HTTPException

from ..routing.router import resolve_provider_and_model
from ..config import settings


async def list_models():
    """List available models, filtered by MODEL_MAP if configured."""
    provider, _ = resolve_provider_and_model(None)
    if not provider:
        raise HTTPException(500, "No provider configured")

    # If MODEL_MAP is configured, return only the mapped alias names as available models
    if settings.model_map:
        created = int(time.time())
        return {
            "object": "list",
            "data": [
                {
                    "id": alias,
                    "object": "model",
                    "created": created,
                    "owned_by": "system"
                }
                for alias in settings.model_map.keys()
            ]
        }

    # If MODEL_MAP is empty, return all models from the default provider
    return await provider.list_models()
