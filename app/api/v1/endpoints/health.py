from fastapi import APIRouter
from app.core.config import get_api_keys
from app.schemas.compliance import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check():
    """
    Check backend health status and number of active Gemini API keys configured.
    """
    keys = get_api_keys()
    return HealthResponse(status="ok", keys_configured=len(keys))
