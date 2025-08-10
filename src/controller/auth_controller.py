import sys
import os
from pathlib import Path
from bson import ObjectId
from datetime import datetime, timedelta
from typing import Optional
import jwt
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr
import re
from config.config import JWT_SECRET, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES, REFRESH_TOKEN_EXPIRE_DAYS
from dal import users_collection

# Load environment variables
load_dotenv()

# You'll need to create these in your dal.py or wherever you define your models
class AuthRequest(BaseModel):
    email: EmailStr
    name: str
    profile_pic: Optional[str] = None

class AuthResponse(BaseModel):
    success: bool
    message: str
    user_id: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    user_data: Optional[dict] = None
    error: Optional[str] = None

# Assuming you have a users_collection similar to recipe_collection
# You'll need to add this to your dal.py
# from dal import users_collection

def serialize_user_data(user_data):
    """Convert MongoDB objects to JSON-serializable format"""
    if isinstance(user_data, dict):
        serialized = {}
        for key, value in user_data.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = serialize_user_data(value)
            elif isinstance(value, list):
                serialized[key] = [serialize_user_data(item) if isinstance(item, dict) else item for item in value]
            else:
                serialized[key] = value
        return serialized
    return user_data

class AuthController:
    def __init__(self):
        self.jwt_secret_key = JWT_SECRET
        self.jwt_algorithm = JWT_ALGORITHM
        self.access_token_expire_minutes = ACCESS_TOKEN_EXPIRE_MINUTES
        self.refresh_token_expire_days = REFRESH_TOKEN_EXPIRE_DAYS
        
        # Validate required environment variables
        if not self.jwt_secret_key:
            raise ValueError("JWT_SECRET_KEY environment variable is required")
    
    def _ensure_user_timestamps(self, users_collection, user_doc: dict) -> dict:
        """Ensure `created_at` and `updated_at` exist on the user document; persist if missing."""
        if not user_doc:
            return user_doc
        updates = {}
        if not user_doc.get("created_at"):
            updates["created_at"] = datetime.now()
        if not user_doc.get("updated_at"):
            updates["updated_at"] = datetime.now()
        if updates:
            try:
                users_collection.update_one({"_id": user_doc["_id"]}, {"$set": updates})
                user_doc = users_collection.find_one({"_id": user_doc["_id"]})
            except Exception as _:
                # Best-effort; if update fails, still return with synthesized fields
                user_doc = {**user_doc, **updates}
        return user_doc
    
    def generate_tokens(self, user_id: str, email: str) -> tuple[str, str]:
        """
        Generate access and refresh tokens for user
        """
        # Access token payload
        access_payload = {
            "user_id": user_id,
            "email": email,
            "type": "access",
            "exp": datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
            "iat": datetime.utcnow()
        }
        
        # Refresh token payload
        refresh_payload = {
            "user_id": user_id,
            "email": email,
            "type": "refresh",
            "exp": datetime.utcnow() + timedelta(days=self.refresh_token_expire_days),
            "iat": datetime.utcnow()
        }
        
        access_token = jwt.encode(access_payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        refresh_token = jwt.encode(refresh_payload, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        
        return access_token, refresh_token
    
    def authenticate_user(self, request: AuthRequest) -> AuthResponse:
        """
        Authenticate user - login or register if new user
        """
        try:
            print(f"Starting authentication for email: {request.email}")
            
            # Step 1: Check if user exists in database
            print("Checking if user exists...")
            from dal import users_collection  # Provided by src/dal.py
            existing_user = users_collection.find_one({"email": request.email})
            
            if existing_user:
                # Step 2a: User exists - update profile info and return tokens
                print(f"User exists with ID: {existing_user['_id']}")
                
                # Update user info in case name or profile pic changed
                update_data = {
                    "name": request.name,
                    "updated_at": datetime.now()
                }
                
                if request.profile_pic:
                    update_data["profile_pic"] = request.profile_pic
                
                try:
                    users_collection.update_one(
                        {"_id": existing_user["_id"]},
                        {"$set": update_data}
                    )
                    print("User profile updated successfully")
                except Exception as update_error:
                    print(f"Warning: Could not update user profile: {update_error}")
                
                # Get updated user data
                updated_user = users_collection.find_one({"_id": existing_user["_id"]})
                updated_user = self._ensure_user_timestamps(users_collection, updated_user)
                user_id = str(updated_user["_id"])
                
            else:
                # Step 2b: New user - create account
                print("Creating new user account...")
                
                user_data = {
                    "email": request.email,
                    "name": request.name,
                    "profile_pic": request.profile_pic,
                    "created_at": datetime.now(),
                    "updated_at": datetime.now()
                }
                
                try:
                    result = users_collection.insert_one(user_data)
                    user_id = str(result.inserted_id)
                    print(f"New user created with ID: {user_id}")
                    
                    # Get the created user data
                    updated_user = users_collection.find_one({"_id": result.inserted_id})
                    updated_user = self._ensure_user_timestamps(users_collection, updated_user)
                    
                except Exception as db_error:
                    print(f"Error: Could not create user in MongoDB: {db_error}")
                    return AuthResponse(
                        success=False,
                        message="Failed to create user account",
                        error=str(db_error)
                    )
            
            # Step 3: Generate JWT tokens
            print("Generating JWT tokens...")
            access_token, refresh_token = self.generate_tokens(user_id, request.email)
            
            # Step 4: Prepare user data for response (exclude sensitive info)
            user_response_data = {
                "id": user_id,
                "email": updated_user.get("email"),
                "name": updated_user.get("name"),
                "profile_pic": updated_user.get("profile_pic"),
                "created_at": updated_user.get("created_at", datetime.now()),
                "updated_at": updated_user.get("updated_at", datetime.now())
            }
            
            # Step 5: Serialize user data for JSON response
            serialized_user_data = serialize_user_data(user_response_data)
            
            return AuthResponse(
                success=True,
                message="Authentication successful",
                user_id=user_id,
                access_token=access_token,
                refresh_token=refresh_token,
                user_data=serialized_user_data
            )
            
        except Exception as e:
            print(f"Error during authentication: {e}")
            return AuthResponse(
                success=False,
                message="Authentication failed",
                error=str(e)
            )
    
    def refresh_access_token(self, refresh_token: str) -> AuthResponse:
        """
        Generate new access token using refresh token
        """
        try:
            print("Refreshing access token...")
            
            # Step 1: Verify refresh token
            try:
                payload = jwt.decode(refresh_token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
                
                # Check if token is refresh type
                if payload.get("type") != "refresh":
                    raise jwt.InvalidTokenError("Invalid token type")
                
                user_id = payload.get("user_id")
                email = payload.get("email")
                
            except jwt.ExpiredSignatureError:
                return AuthResponse(
                    success=False,
                    message="Refresh token has expired",
                    error="Token expired"
                )
            except jwt.InvalidTokenError as e:
                return AuthResponse(
                    success=False,
                    message="Invalid refresh token",
                    error=str(e)
                )
            
            # Step 2: Verify user still exists
            from dal import users_collection
            user = users_collection.find_one({"_id": ObjectId(user_id)})
            
            if not user:
                return AuthResponse(
                    success=False,
                    message="User not found",
                    error="User account no longer exists"
                )
            
            # Step 3: Generate new access token
            new_access_token, _ = self.generate_tokens(user_id, email)
            
            # Step 4: Prepare user data for response (exclude sensitive info)
            user_response_data = {
                "id": str(user["_id"]),
                "email": user.get("email"),
                "name": user.get("name"),
                "profile_pic": user.get("profile_pic"),
                "created_at": user.get("created_at", datetime.now()),
                "updated_at": user.get("updated_at", datetime.now())
            }
            
            # Step 5: Serialize user data for JSON response
            serialized_user_data = serialize_user_data(user_response_data)
            
            return AuthResponse(
                success=True,
                message="Access token refreshed successfully",
                user_id=user_id,
                access_token=new_access_token,
                user_data=serialized_user_data,
                refresh_token=refresh_token  # Keep the same refresh token
            )
            
        except Exception as e:
            print(f"Error refreshing token: {e}")
            return AuthResponse(
                success=False,
                message="Failed to refresh token",
                error=str(e)
            )
    
    def get_user_by_id(self, user_id: str) -> AuthResponse:
        """
        Get user by ID from MongoDB
        """
        try:
            from dal import users_collection
            user = users_collection.find_one({"_id": ObjectId(user_id)})
            
            if user:
                user = self._ensure_user_timestamps(users_collection, user)
                # Prepare user data (exclude sensitive info)
                user_response_data = {
                    "id": str(user["_id"]),
                    "email": user.get("email"),
                    "name": user.get("name"),
                    "profile_pic": user.get("profile_pic"),
                    "created_at": user.get("created_at", datetime.now()),
                    "updated_at": user.get("updated_at", datetime.now())
                }
                
                # Serialize the user data
                serialized_user_data = serialize_user_data(user_response_data)
                
                return AuthResponse(
                    success=True,
                    message="User retrieved successfully",
                    user_id=user_id,
                    user_data=serialized_user_data
                )
            else:
                return AuthResponse(
                    success=False,
                    message="User not found",
                    error="User with the specified ID was not found"
                )
                
        except Exception as e:
            print(f"Error retrieving user: {e}")
            return AuthResponse(
                success=False,
                message="Failed to retrieve user",
                error=str(e)
            )