from pydantic import BaseModel
import os, json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseModel):
    model_config = {"protected_namespaces": ()}

    default_provider: str = os.getenv("DEFAULT_PROVIDER", "ollama")
    model_map: dict = json.loads(os.getenv("MODEL_MAP", "{}"))


    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")


    openrouter_base_url: str | None = os.getenv("OPENROUTER_BASE_URL")
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")


    azure_openai_base_url: str | None = os.getenv("AZURE_OPENAI_BASE_URL")
    azure_openai_api_key: str | None = os.getenv("AZURE_OPENAI_API_KEY")
    azure_openai_api_version: str | None = os.getenv("AZURE_OPENAI_API_VERSION")


    lmstudio_base_url: str | None = os.getenv("LMSTUDIO_BASE_URL")
    ollama_base_url: str | None = os.getenv("OLLAMA_BASE_URL")

    # When set, replaces the client's system prompt for every request.
    # Unset (default) = faithful passthrough of the client's own prompt.
    system_prompt_override: str | None = os.getenv("SYSTEM_PROMPT_OVERRIDE")


settings = Settings()

# Boot summary without secrets: never print API keys.
print(
    "Loaded config:",
    {
        "default_provider": settings.default_provider,
        "model_map": settings.model_map,
        "providers_configured": [
            name
            for name, url in {
                "openai": settings.openai_base_url,
                "openrouter": settings.openrouter_base_url,
                "azure": settings.azure_openai_base_url,
                "lmstudio": settings.lmstudio_base_url,
                "ollama": settings.ollama_base_url,
            }.items()
            if url
        ],
        "system_prompt_override": bool(settings.system_prompt_override),
    },
)