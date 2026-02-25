"""Provider factory — returns the appropriate AIProvider based on environment configuration."""

import logging
import os

from .base import AIProvider

logger = logging.getLogger(__name__)

_provider: AIProvider | None = None


def get_provider() -> AIProvider:
    """Return a singleton AI provider based on environment variables.

    The provider is created and initialized once, then reused for all requests.

    - If ``USE_OPENAI=true`` and ``OPENAI_API_KEY`` is set → :class:`OpenAIProvider`
    - If ``USE_OPENAI=true`` but no API key → log warning, fall back to :class:`LocalInsightFaceProvider`
    - Otherwise → :class:`LocalInsightFaceProvider`
    """
    global _provider
    if _provider is not None:
        return _provider

    use_openai = os.environ.get("USE_OPENAI", "").lower() == "true"

    if use_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            from .openai_provider import OpenAIProvider

            logger.info("Using OpenAIProvider (USE_OPENAI=true, API key present).")
            _provider = OpenAIProvider()
            return _provider
        else:
            logger.warning(
                "USE_OPENAI is true but OPENAI_API_KEY is not set. "
                "Falling back to LocalInsightFaceProvider."
            )

    from .insightface_provider import LocalInsightFaceProvider

    logger.info("Using LocalInsightFaceProvider. Loading models...")
    provider = LocalInsightFaceProvider()
    provider.load_models()
    _provider = provider
    return _provider
