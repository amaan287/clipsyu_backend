"""API routes for guide generation."""
from fastapi import APIRouter, HTTPException
from src.api.schemas import RecipeRequest, Guide
from src.services.guide_generator_service import GuideGeneratorService

router = APIRouter()

# Initialize service (could be injected via dependency injection in production)
guide_service = GuideGeneratorService()


@router.post("/generate", response_model=Guide)
def generate_guide(payload: RecipeRequest):
    """
    Generate a guide from a video URL.
    
    Args:
        payload: Request payload containing video URL
        
    Returns:
        Guide object containing extracted information
        
    Raises:
        HTTPException: If guide generation fails
    """
    try:
        guide = guide_service.generate_guide(payload.url)
        return guide
    except ValueError as e:
        # Validation errors (e.g., unsupported platform)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Internal errors
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health_check():
    """
    Health check endpoint.
    
    Returns:
        Status message
    """
    return {"status": "healthy", "message": "Guide generation service is running"}
