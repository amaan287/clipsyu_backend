"""
Recipe Extractor API Server

A FastAPI application for extracting recipes from YouTube and Instagram videos.
Provides authentication, recipe extraction, and recipe management endpoints.
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import sys
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import jwt
import json

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from dal import RecipeExtractionRequest, RecipeExtractionResponse, RefreshTokenRequest
from controller.api_controller import RecipeExtractionController
from controller.auth_controller import AuthController, AuthRequest
from config.config import JWT_SECRET, JWT_ALGORITHM
from middlewares.authentication_middleware import verify_access_token





class PrettyJSONResponse(JSONResponse):
    """Custom JSON response with pretty formatting"""
    
    def render(self, content) -> bytes:
        return json.dumps(
            content, 
            indent=2, 
            ensure_ascii=False
        ).encode('utf-8')


# ==================== Application Setup ====================

app = FastAPI(
    title="Recipe Extractor API",
    version="1.0.0",
    description="Extract recipes from YouTube and Instagram videos",
    default_response_class=PrettyJSONResponse
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
)

# Initialize controllers
recipe_controller = RecipeExtractionController()
auth_controller = AuthController()




# ==================== Health Check Routes ====================

@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Recipe Extractor API is running",
        "version": "1.0.0"
    }


@app.head("/", tags=["Health"])
async def health_check_head():
    """Health check HEAD endpoint"""
    return {}


# ==================== Authentication Routes ====================

@app.post("/api/v1/auth/login", tags=["Authentication"])
async def login(request: AuthRequest):
    """
    Authenticate user with email and name
    Creates new user if doesn't exist, updates existing user otherwise
    Returns access and refresh tokens
    """
    return auth_controller.authenticate_user(request)


@app.post("/api/v1/auth/refresh", tags=["Authentication"])
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token
    Returns new access token if refresh token is valid
    """
    return auth_controller.refresh_access_token(request.refresh_token)


@app.get("/api/v1/users/{user_id}", tags=["Users"])
async def get_user(user_id: str):
    """Get user information by user ID"""
    return auth_controller.get_user_by_id(user_id)


# ==================== Recipe Routes ====================

@app.post(
    "/api/v1/recipes",
    response_model=RecipeExtractionResponse,
    tags=["Recipes"]
)
async def create_recipe(
    request: RecipeExtractionRequest,
    user_id: str = Depends(verify_access_token)
):
    """
    Extract recipe from YouTube or Instagram video URL
    Requires valid access token
    """
    response = await recipe_controller.extract_recipe_from_youtube(
        request,
        user_id=user_id
    )
    
    if not response.success:
        raise HTTPException(
            status_code=400,
            detail=response.error or response.message
        )
    
    return response


@app.get(
    "/api/v1/recipes/{recipe_id}",
    response_model=RecipeExtractionResponse,
    tags=["Recipes"]
)
async def get_recipe(recipe_id: str):
    """Get recipe by ID"""
    response = recipe_controller.get_recipe_by_id(recipe_id)
    
    if not response.success:
        raise HTTPException(
            status_code=404,
            detail=response.error or response.message
        )
    
    return response


@app.get(
    "/api/v1/recipes",
    response_model=RecipeExtractionResponse,
    tags=["Recipes"]
)
async def list_recipes():
    """Get all recipes"""
    return recipe_controller.get_all_recipes()


@app.get(
    "/api/v1/users/{user_id}/recipes",
    response_model=RecipeExtractionResponse,
    tags=["Recipes"]
)
async def get_user_recipes(
    user_id: str,
    token_user_id: str = Depends(verify_access_token)
):
    """
    Get all recipes for a specific user
    Requires valid access token
    """
    response = recipe_controller.get_recipes_by_user_id(user_id)
    
    if not response.success:
        raise HTTPException(
            status_code=400,
            detail=response.error or response.message
        )
    
    return response


# ==================== Application Entry Point ====================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        reload=True
    )