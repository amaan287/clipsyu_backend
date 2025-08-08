from fastapi import Header,FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path
import datetime
import httpx
from pydantic import BaseModel
import time
import requests

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))
from service.jwtService import verify_google_token,verify_jwt_token,create_jwt_token
from dal import RecipeExtractionRequest, RecipeExtractionResponse,UserResponse,GoogleAuthRequest,users_collection,User,GoogleCodeAuthRequest
from api_controller import RecipeExtractionController
from config.config import AUTH_CLIENT_ID,AUTH_ANDROID_CLIENT_ID,AUTH_IOS_CLIENT_ID,GOOGLE_CLIENT_SECRET
import jwt


app = FastAPI(title="Recipe Extractor API", version="1.0.0")

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller = RecipeExtractionController()

# Add new Pydantic model for OAuth callback
class GoogleOAuthCallbackRequest(BaseModel):
    code: str
    redirect_uri: str

class GoogleTokenResponse(BaseModel):
    id_token: str


async def getUserInfo(token:str):
    response = await requests.get("https://www.googleapis.com/userinfo/v2/me",{"Authorization":f"Bearer {token}"})
    print(response)

@app.post("/auth/google", response_model=UserResponse)
async def google_auth(auth_request: GoogleAuthRequest):
    """Authenticate user with Google ID token, store if new, return JWT and refresh tokens"""
    
    try:
        # Verify Google token against all possible client IDs
        google_user_info = await getUserInfo(GoogleAuthRequest.token)
        
    except Exception as e:
        # Enhanced error logging
        try:
            decoded = jwt.decode(auth_request.id_token, options={"verify_signature": False})
            print(f"❌ DEBUG: Token details - aud: {decoded.get('aud')}, iss: {decoded.get('iss')}")
        except Exception as decode_error:
            print(f"❌ DEBUG: Could not decode token: {decode_error}")
        
        raise HTTPException(status_code=400, detail=f"Google token verification failed: {str(e)}")
    
    # Rest of your existing code remains the same...
    google_id = google_user_info["user_id"]
    email = google_user_info["email"]
    name = google_user_info.get("name", "")
    picture = google_user_info.get("picture", "")

    # Try to find the user in MongoDB
    existing_user = users_collection.find_one({"google_id": google_id})

    if not existing_user:
        # User doesn't exist — insert new one
        new_user = User(
            google_id=google_id,
            email=email,
            name=name,
            picture=picture,
        )
        # ✅ FIX: Get the inserted ID properly
        result = users_collection.insert_one(new_user.dict())
        user_id = str(result.inserted_id)
    else:
        # Update fields in case they've changed
        users_collection.update_one(
            {"google_id": google_id},
            {"$set": {"name": name, "picture": picture}}
        )
        user_id = str(existing_user['_id'])
    
    # Generate tokens
    payload = {
        "user_id": google_id,
        "email": email,
    }

    jwt_token = create_jwt_token(payload)
    refresh_token = create_jwt_token(payload, expires_delta=datetime.timedelta(days=30))

    return UserResponse(
        id=user_id,
        google_id=google_id,
        email=email,
        name=name,
        picture=picture,
        jwt_token=jwt_token,
        refresh_token=refresh_token,
    )


@app.post("/extract-recipe", response_model=RecipeExtractionResponse)
async def extract_recipe(request: RecipeExtractionRequest):
    response = controller.extract_recipe_from_youtube(request)
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

@app.get("/")
async def root():
    return {"message": "Recipe Extractor API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, reload=True)