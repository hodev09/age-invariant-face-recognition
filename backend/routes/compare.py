"""POST /compare-faces route for face comparison."""

import logging

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ai_providers.factory import get_provider
from models.schemas import ComparisonResponse, ErrorResponse, RejectionResponse
from services.pipeline import compare_faces
from utils.validator import validate_upload

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/compare-faces",
    responses={
        200: {"model": ComparisonResponse},
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def compare_faces_route(
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
):
    """Compare two face images and return similarity results."""

    # Validate uploads
    try:
        image1_bytes = await image1.read()
        validate_upload(image1.filename or "", image1_bytes)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )

    try:
        image2_bytes = await image2.read()
        validate_upload(image2.filename or "", image2_bytes)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )

    # Get AI provider
    provider = get_provider()

    # Check if models are loaded
    if not await provider.is_loaded():
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="AI models are not loaded. Please try again later."
            ).model_dump(),
        )

    # Run pipeline
    try:
        result = await compare_faces(image1_bytes, image2_bytes, provider)
    except ValueError as exc:
        error_msg = str(exc)
        # Face detection / multi-face errors → 422
        if "No face detected" in error_msg or "Multiple faces detected" in error_msg:
            return JSONResponse(
                status_code=422,
                content=ErrorResponse(error=error_msg).model_dump(),
            )
        # Invalid image data → 400
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(error=error_msg).model_dump(),
        )
    except RuntimeError as exc:
        # Model not loaded at runtime
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(error=str(exc)).model_dump(),
        )
    except Exception as exc:
        logger.exception("Unexpected error during face comparison")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(error="Internal server error").model_dump(),
        )

    # Return appropriate response based on result
    if result.result == "rejected":
        return RejectionResponse(
            age1=result.age1,
            age2=result.age2,
            age_group1=result.age_group1,
            age_group2=result.age_group2,
            result=result.result,
            message=result.message,
        )

    return ComparisonResponse(
        age1=result.age1,
        age2=result.age2,
        age_group1=result.age_group1,
        age_group2=result.age_group2,
        similarity_score=result.similarity_score,
        confidence=result.confidence,
        result=result.result,
        message=result.message,
    )
