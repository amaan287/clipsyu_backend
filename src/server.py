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

# Your Google OAuth client credentials

GOOGLE_CLIENT_IDS = {
    "web":AUTH_CLIENT_ID ,
    "android": AUTH_ANDROID_CLIENT_ID, 
    "ios": AUTH_IOS_CLIENT_ID
}

# You can also store them as separate environment variables
# GOOGLE_WEB_CLIENT_ID = os.getenv("GOOGLE_WEB_CLIENT_ID")
# GOOGLE_ANDROID_CLIENT_ID = os.getenv("GOOGLE_ANDROID_CLIENT_ID") 
# GOOGLE_IOS_CLIENT_ID = os.getenv("GOOGLE_IOS_CLIENT_ID")

async def verify_google_token_multiplatform(id_token: str):
    """Verify Google token against all possible client IDs"""
    
    # First, decode token without verification to see the audience
    try:
        decoded = jwt.decode(id_token, options={"verify_signature": False})
        token_audience = decoded.get("aud")
        print(f"🔍 DEBUG: Token audience: {token_audience}")
    except Exception as e:
        print(f"❌ DEBUG: Could not decode token: {e}")
        raise Exception("Invalid token format")
    
    # Try to verify with each client ID
    verification_errors = []
    
    for platform, client_id in GOOGLE_CLIENT_IDS.items():
        try:
            print(f"🔄 DEBUG: Trying verification with {platform} client ID: {client_id}")
            
            # Your existing verify_google_token function logic here
            # but with the specific client_id parameter
            google_user_info = await verify_google_token_with_client_id(id_token, client_id)
            
            print(f"✅ DEBUG: Successfully verified with {platform} client ID")
            return google_user_info
            
        except Exception as e:
            error_msg = f"{platform}: {str(e)}"
            verification_errors.append(error_msg)
            print(f"❌ DEBUG: Verification failed for {platform}: {e}")
            continue
    
    # If we get here, all verifications failed
    raise Exception(f"Token verification failed for all platforms. Errors: {'; '.join(verification_errors)}")


async def exchange_google_code_for_tokens(code: str, client_id: str, redirect_uri: str):
    """Exchange authorization code for Google tokens"""
    
    # Your Google client secret - make sure this is in your environment variables
    client_secret = GOOGLE_CLIENT_SECRET  # Move this to env var
    
    token_url = "https://oauth2.googleapis.com/token"
    
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(token_url, data=data)
            response.raise_for_status()
            
            tokens = response.json()
            print(f"🔍 DEBUG: Token exchange successful: {tokens.keys()}")
            
            return tokens  # Contains access_token, id_token, refresh_token, etc.
            
        except httpx.HTTPStatusError as e:
            print(f"❌ DEBUG: Token exchange failed: {e.response.status_code} - {e.response.text}")
            raise Exception(f"Token exchange failed: {e.response.text}")
        except Exception as e:
            print(f"❌ DEBUG: Token exchange error: {e}")
            raise Exception(f"Token exchange error: {str(e)}")

# Add this new endpoint for authorization code flow
@app.post("/auth/google/code", response_model=UserResponse)
async def google_auth_code(auth_request: GoogleCodeAuthRequest):
    """Authenticate user with Google authorization code"""
    
    try:
        # Step 1: Exchange authorization code for tokens
        tokens = await exchange_google_code_for_tokens(
            auth_request.code, 
            auth_request.client_id, 
            auth_request.redirect_uri
        )
        
        # Step 2: Extract and verify the ID token
        id_token = tokens.get("id_token")
        if not id_token:
            raise Exception("No id_token received from Google")
        
        # Step 3: Verify the ID token (reuse your existing function)
        google_user_info = await verify_google_token_multiplatform(id_token)
        
    except Exception as e:
        print(f"❌ DEBUG: Google auth code flow failed: {e}")
        raise HTTPException(status_code=400, detail=f"Google authentication failed: {str(e)}")
    
    # Rest of the code is the same as your existing endpoint
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
        user_id = str(new_user.dict().get('_id'))
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


@app.post("/auth/google", response_model=UserResponse)
async def google_auth(auth_request: GoogleAuthRequest):
    """Authenticate user with Google ID token, store if new, return JWT and refresh tokens"""
    
    try:
        # Verify Google token against all possible client IDs
        google_user_info = await verify_google_token_multiplatform(auth_request.id_token)
        
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
        users_collection.insert_one(new_user.dict())
        user_data = new_user
        user_id = str(new_user.dict().get('_id'))  # Get the new user's ID
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
    uvicorn.run(app, host="localhost", port=8000, reload=True)