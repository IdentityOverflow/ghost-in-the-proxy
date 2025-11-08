"""Health check endpoint."""


async def root():
    """Basic health check endpoint."""
    return {
        "ok": True,
        "name": "Project-2501",
        "version": "0.3.0"
    }
