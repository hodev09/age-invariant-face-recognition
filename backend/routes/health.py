"""Health check route."""

from fastapi import APIRouter

from ai_providers.factory import get_provider
from models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status and whether AI models are loaded."""
    try:
        provider = get_provider()
        model_loaded = await provider.is_loaded()
    except Exception:
        model_loaded = False

    return HealthResponse(status="ok", model_loaded=model_loaded)
