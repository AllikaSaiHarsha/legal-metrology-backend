import logging
from fastapi import APIRouter, File, UploadFile, Request, HTTPException
from app.core.config import get_api_keys
from app.services.gemini_service import GeminiVisionService
from app.services.image_service import ImageService
from app.schemas.compliance import ComplianceAnalysisResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/analyze",
    response_model=ComplianceAnalysisResponse,
    summary="Analyze Product Packaging for Rule 6 Compliance"
)
async def analyze_image(request: Request, file: UploadFile = File(...)):
    """
    Accepts an uploaded package label image (JPEG/PNG/WEBP), saves it, runs Gemini Multimodal
    inspection across statutory declarations, and returns normalized detection boxes.
    """
    api_keys = get_api_keys()
    if not api_keys:
        raise HTTPException(
            status_code=500,
            detail="Gemini API Key missing in .env (set GEMINI_API_KEY or GEMINI_API_KEYS)"
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        filename, _ = ImageService.save_upload(contents, original_filename=file.filename)
        real_width, real_height = ImageService.get_image_dimensions(contents)

        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/uploads/{filename}"

        # Run Multimodal inspection
        raw_data = GeminiVisionService.analyze_label(contents, api_keys)

        # Sanitize bounding boxes and statutory declarations
        sanitized_detections = ImageService.sanitize_detections(raw_data.get("detections", []))

        return ComplianceAnalysisResponse(
            product_name=raw_data.get("product_name") or "",
            manufacturer=raw_data.get("manufacturer") or "",
            detections=sanitized_detections,
            filename=filename,
            original_width=1000,
            original_height=1000,
            real_width=real_width,
            real_height=real_height,
            image_url=image_url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))
