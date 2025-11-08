from ..config import settings
from ..providers.openai_like import OpenAILikeProvider


PROVIDERS = {}


if settings.openai_base_url:
    PROVIDERS["openai"] = OpenAILikeProvider("openai", settings.openai_base_url, settings.openai_api_key)
if settings.openrouter_base_url and settings.openrouter_api_key:
    PROVIDERS["openrouter"] = OpenAILikeProvider("openrouter", settings.openrouter_base_url, settings.openrouter_api_key,
        extra_headers={"HTTP-Referer": "http://localhost", "X-Title": "AI Passthrough"})
if settings.azure_openai_base_url and settings.azure_openai_api_key:
    PROVIDERS["azure"] = OpenAILikeProvider("azure", settings.azure_openai_base_url, settings.azure_openai_api_key,
        extra_headers={"api-version": settings.azure_openai_api_version} if settings.azure_openai_api_version else {})
if settings.lmstudio_base_url:
    PROVIDERS["lmstudio"] = OpenAILikeProvider("lmstudio", settings.lmstudio_base_url)
if settings.ollama_base_url:
    PROVIDERS["ollama"] = OpenAILikeProvider("ollama", settings.ollama_base_url)




def resolve_provider_and_model(incoming_model: str | None):
    if incoming_model and incoming_model in settings.model_map:
        target = settings.model_map[incoming_model]
        prov, _, mapped_model = target.partition(":")
        return PROVIDERS.get(prov), (mapped_model or incoming_model)
    return PROVIDERS.get(settings.default_provider), incoming_model