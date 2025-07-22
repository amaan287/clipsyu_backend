import datetime 
import jwt
from fastapi import FastAPI, HTTPException, Depends
from google.oauth2 import id_token
from google.auth.transport import requests
from config.config import JWT_ALGORITHM,JWT_SECRET,AUTH_CLIENT_ID

def create_jwt_token(user_data: dict, expires_delta: datetime.timedelta = datetime.timedelta(days=30)) -> str:
    """Create JWT token for user"""
    print(f"user_data: {user_data}")
    payload = {
        "user_id": user_data["user_id"],  # 'sub' is the stable user ID provided by Google
        "email": user_data["email"],
        "exp": datetime.datetime.utcnow() + expires_delta
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
def create_refresh_token(user_data: dict, expires_delta: datetime.timedelta = datetime.timedelta(days=30))->str:
    """Create JWT token for user"""
    print(f"user_data: {user_data}")
    payload = {
        "user_id": user_data["user_id"],  # 'sub' is the stable user ID provided by Google
        "email": user_data["email"],
        "exp": datetime.datetime.utcnow() + expires_delta
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)       
def verify_jwt_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

async def verify_google_token(token: str) -> dict:
    """Verify Google ID token"""
    try:
        # Verify the token
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), AUTH_CLIENT_ID
        )
        
        # Check if token is for our app
        if idinfo['aud'] != AUTH_CLIENT_ID:
            raise ValueError('Wrong audience.')
        
        print(f"idinfo: {idinfo}")
        return {
            "user_id": idinfo["sub"],
            
            "email": idinfo["email"],
            "name": idinfo.get("name", ""),
            "picture": idinfo.get("picture", "")
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token: {str(e)}")
