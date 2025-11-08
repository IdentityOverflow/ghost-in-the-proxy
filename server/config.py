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


settings = Settings()

print("Loaded config:", settings.dict())