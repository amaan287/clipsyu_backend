from fastapi import Header,FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import sys
from pathlib import Path
import datetime
import httpx
from pydantic import BaseModel
# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))
from service.jwtService import verify_google_token,verify_jwt_token,create_jwt_token
from dal import RecipeExtractionRequest, RecipeExtractionResponse,UserResponse,GoogleAuthRequest,users_collection,User
from api_controller import RecipeExtractionController
from config.config import AUTH_CLIENT_ID,GOOGLE_CLIENT_SECRET
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

# Your Google OAuth client credentials
GOOGLE_CLIENT_ID = AUTH_CLIENT_ID


@app.post("/auth/google", response_model=UserResponse)
async def google_auth(auth_request: GoogleAuthRequest):
    """Authenticate user with Google ID token, store if new, return JWT and refresh tokens"""
    
    
    try:
        # Verify Google token and extract user info
        google_user_info = await verify_google_token(auth_request.id_token)
        
    except Exception as e:
        # Let's also decode the token to see its audience
        try:
            # Decode without verification to see the token contents
            decoded = jwt.decode(auth_request.id_token, options={"verify_signature": False})
        except Exception as decode_error:
            print(f"❌ DEBUG: Could not decode token: {decode_error}")
        
        raise HTTPException(status_code=400, detail=f"Google token verification failed: {str(e)}")
    
    # Rest of your existing code...
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
        users_collection.insert_one(new_user.dict())
        user_data = new_user
    else:
        # Update fields in case they've changed
        users_collection.update_one(
        {"google_id": google_id},
        {"$set": {"name": name, "picture": picture ,"id":str(existing_user['_id'])}}
    )
    
    # Generate tokens
    payload = {
        "user_id": google_id,
        "email": email,
    }

    jwt_token = create_jwt_token(payload)
    refresh_token = create_jwt_token(payload, expires_delta=datetime.timedelta(days=30))

    return UserResponse(
        id=str(existing_user['_id']),
        google_id  = google_id,
        email=email,
        name=name,
        picture=picture,
        jwt_token=jwt_token,
        refresh_token=refresh_token,
    )

@app.get("/auth/me")
async def get_current_user(Authorization: str = Header(None)):
    """Get current user info from JWT token"""
    print( f"header: {Authorization}" )
    if not Authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        # Extract token from "Bearer <token>"
        token = Authorization.split(" ")[1]
        print( f"token:  {token}" )
        
        payload = verify_jwt_token(token)
        print(f"payload {payload}")
        # Get user from database
        user =  users_collection.find_one({"google_id": payload["user_id"]})
        print(user["id"])
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return UserResponse(
        id=user["id"],
        google_id  = user["google_id"],
        email=user["email"],
        name=user["name"],
        picture=user["picture"],
    )
    except IndexError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

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
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)