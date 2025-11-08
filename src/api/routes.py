from fastapi import APIRouter, HTTPException
from src.api.schemas import RecipeRequest, Guide
from src.service.recipe_generator import generate_recipe_from_url

router = APIRouter()

@router.post("/generate", response_model=Guide)
def generate_recipe(payload: RecipeRequest):
    try:
        guide = generate_recipe_from_url(payload.url)
        return guide
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
