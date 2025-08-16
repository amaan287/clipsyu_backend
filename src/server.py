from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path
from pydantic import BaseModel
from typing import Optional
import jwt

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))
from dal import RecipeExtractionRequest, RecipeExtractionResponse
from controller.api_controller import RecipeExtractionController
from controller.auth_controller import AuthController, AuthRequest
from config.config import JWT_SECRET, JWT_ALGORITHM

class RefreshTokenRequest(BaseModel):
    refresh_token: str

app = FastAPI(title="Recipe Extractor API", version="1.0.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "PUT", "DELETE", "OPTIONS", "PATCH", "HEAD"],
    allow_headers=["*"],
)

controller = RecipeExtractionController()
auth_controller = AuthController()

@app.post("/auth/login")
async def login(request: AuthRequest):
    return auth_controller.authenticate_user(request)


@app.post("/auth/refresh")
async def refresh_token(request: RefreshTokenRequest):
    return auth_controller.refresh_access_token(request.refresh_token)

@app.get("/auth/user/{user_id}")
async def get_user(user_id: str):
    return auth_controller.get_user_by_id(user_id)
    
@app.post("/extract-recipe", response_model=RecipeExtractionResponse)
async def extract_recipe(request: RecipeExtractionRequest, Authorization: Optional[str] = Header(default=None)):
    # Require Bearer access token
    if not Authorization or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = Authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid access token")

    # Add await here
    response = await controller.extract_recipe_from_youtube(request, user_id=user_id)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error or response.message)
    return response
@app.get("/recipe/{recipe_id}", response_model=RecipeExtractionResponse)
async def get_recipe(recipe_id: str):
    response = controller.get_recipe_by_id(recipe_id)
    if not response.success:
        raise HTTPException(status_code=404, detail=response.error or response.message)
    return response

@app.get("/recipes", response_model=RecipeExtractionResponse)
async def get_all_recipes():
    response = controller.get_all_recipes()
    return response

@app.get("/recipes/user/{user_id}", response_model=RecipeExtractionResponse)
async def get_user_recipes(user_id: str, Authorization: Optional[str] = Header(default=None)):
    # Require Bearer access token
    if not Authorization or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = Authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        if payload.get("type") != "access":
            raise HTTPException(status_code=401, detail="Invalid token type")
        token_user_id = payload.get("user_id")
        if not token_user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Access token has expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid access token")

    response = controller.get_recipes_by_user_id(user_id)
    if not response.success:
        raise HTTPException(status_code=400, detail=response.error or response.message)
    return response

@app.get("/")
async def root():
    return {"message": "Recipe Extractor API is running!"}

@app.head("/")
async def root():
    return {"message": "Recipe Extractor API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)