import sys
import os
from pathlib import Path
from bson import ObjectId
from datetime import datetime
from dal import RecipeExtractionRequest, RecipeExtractionResponse, recipe_collection
from agents.agent import extract_recipe_with_ai, download_youtube_video, get_first_comment_from_video, transcribe_video_with_gemini, extract_text_from_video_frames
from agents.instagram_agent import download_instagram_video, get_instagram_comments_info
import os
from dotenv import load_dotenv
from pydantic import BaseModel
import re
import asyncio
# Load environment variables
load_dotenv()

def serialize_recipe_data(recipe_data):
    """Convert MongoDB objects to JSON-serializable format"""
    if isinstance(recipe_data, dict):
        serialized = {}
        for key, value in recipe_data.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, datetime):
                serialized[key] = value.isoformat()
            elif isinstance(value, dict):
                serialized[key] = serialize_recipe_data(value)
            elif isinstance(value, list):
                serialized[key] = [serialize_recipe_data(item) if isinstance(item, dict) else item for item in value]
            else:
                serialized[key] = value
        return serialized
    return recipe_data

class RecipeExtractionController:
    def __init__(self):
        self.google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_APIKEY")
        
        # Validate required environment variables
        if not self.google_cloud_api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY environment variable is required")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_APIKEY environment variable is required")
    
    async def extract_recipe_from_youtube(self, request: RecipeExtractionRequest, user_id: str = None) -> RecipeExtractionResponse:
        """
        Extract recipe from YouTube video URL
        """
        try:
            print(f"Starting recipe extraction for URL: {request.url}")
            
            # Step 1: Download the video
            print("Downloading video...")
            if "youtube.com/" in request.url or "youtu.be/" in request.url:
                video_file, metadata = await download_youtube_video(request.url)
            elif "instagram.com/" in request.url or "instagr.am/" in request.url:
                video_file, metadata = await download_instagram_video(request.url)
            else:
                raise ValueError("Unsupported URL. Only YouTube and Instagram URLs are supported.")
            
            # Step 2: Get first comment
            print("Fetching first comment...")
            comment_info = None
            if request.url.startswith("https://www.youtube.com/"):
                comment_info = get_first_comment_from_video(self.google_cloud_api_key, request.url)
            elif request.url.startswith("https://www.instagram.com/"):
                comment_info = get_instagram_comments_info(request.url)

            # Step 3: Analyze video with Gemini
            print("Analyzing video with Gemini...")
            video_analysis = transcribe_video_with_gemini(video_file)
            
            # Step 4: Extract text from video frames using OCR
            print("Extracting text from video frames...")
            ocr_text = extract_text_from_video_frames(video_file, video_analysis)
            
            # Step 5: Extract recipe using AI
            print("Extracting recipe with AI...")
            recipe_data = extract_recipe_with_ai(metadata, comment_info, video_analysis, ocr_text)
            
            # Step 6: Add URL to recipe data
            if request.url.startswith("https://www.youtube.com/"):
                recipe_data['youtube_url'] = request.url
            elif request.url.startswith("https://www.instagram.com/"):
                recipe_data['instagram_url'] = request.url
            
            # Step 7: Save to MongoDB
            print("Saving recipe to MongoDB...")
            try:
                # Add created_at and updated_at if not present
                if 'created_at' not in recipe_data:
                    recipe_data['created_at'] = datetime.now()
                if 'updated_at' not in recipe_data:
                    recipe_data['updated_at'] = datetime.now()

                # Attach user info if provided
                if user_id:
                    recipe_data['user_id'] = user_id
                
                # Convert datetime string to datetime object if needed
                if isinstance(recipe_data.get('savedDate'), str):
                    recipe_data['savedDate'] = datetime.fromisoformat(recipe_data['savedDate'].replace('Z', '+00:00'))
                
                result = recipe_collection.insert_one(recipe_data)
                recipe_id = str(result.inserted_id)
                print(f"Recipe saved with ID: {recipe_id}")
            except Exception as db_error:
                print(f"Warning: Could not save to MongoDB: {db_error}")
                recipe_id = None
            
            # Step 8: Clean up downloaded video file
            try:
                os.remove(video_file)
                print(f"Cleaned up video file: {video_file}")
            except Exception as e:
                print(f"Warning: Could not delete video file {video_file}: {e}")
            
            # Step 9: Serialize recipe data for JSON response
            serialized_recipe_data = serialize_recipe_data(recipe_data)
            
            return RecipeExtractionResponse(
                success=True,
                message="Recipe extracted successfully" + (f" and saved with ID: {recipe_id}" if recipe_id else " (MongoDB save failed)"),
                recipe_id=recipe_id,
                recipe_data=serialized_recipe_data
            )
            
        except Exception as e:
            print(f"Error during recipe extraction: {e}")
            return RecipeExtractionResponse(
                success=False,
                message="Failed to extract recipe",
                error=str(e)
            )
    
    def get_recipe_by_id(self, recipe_id: str) -> RecipeExtractionResponse:
        """
        Get recipe by ID from MongoDB
        """
        try:
            recipe = recipe_collection.find_one({"_id": ObjectId(recipe_id)})
            
            if recipe:
                # Convert ObjectId to string for JSON serialization
                recipe['_id'] = str(recipe['_id'])
                # Convert datetime objects to strings
                for key, value in recipe.items():
                    if isinstance(value, datetime):
                        recipe[key] = value.isoformat()
                
                # Serialize the recipe data
                serialized_recipe_data = serialize_recipe_data(recipe)
                
                return RecipeExtractionResponse(
                    success=True,
                    message="Recipe retrieved successfully",
                    recipe_id=recipe_id,
                    recipe_data=serialized_recipe_data
                )
            else:
                return RecipeExtractionResponse(
                    success=False,
                    message="Recipe not found",
                    error="Recipe with the specified ID was not found"
                )
                
        except Exception as e:
            print(f"Error retrieving recipe: {e}")
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipe",
                error=str(e)
            )
    
    def get_all_recipes(self) -> RecipeExtractionResponse: 
        """
        Get all recipes from MongoDB
        """
        try:
            recipes = list(recipe_collection.find())
            
            # Convert ObjectId and datetime objects for serialization
            for recipe in recipes:
                recipe['_id'] = str(recipe['_id'])
                for key, value in recipe.items():
                    if isinstance(value, datetime):
                        recipe[key] = value.isoformat()
            
            # Serialize all recipes
            serialized_recipes = [serialize_recipe_data(recipe) for recipe in recipes]
            
            return RecipeExtractionResponse(
                success=True,
                message=f"Retrieved {len(recipes)} recipes successfully",
                recipe_data={"recipes": serialized_recipes}
            )
                
        except Exception as e:
            print(f"Error retrieving recipes: {e}")
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipes",
                error=str(e)
            )
    
    def get_recipes_by_user_id(self, user_id: str) -> RecipeExtractionResponse:
        """
        Get all recipes for a specific user from MongoDB
        """
        try:
            recipes = list(recipe_collection.find({"user_id": user_id}))
            
            # Convert ObjectId and datetime objects for serialization
            for recipe in recipes:
                recipe['_id'] = str(recipe['_id'])
                for key, value in recipe.items():
                    if isinstance(value, datetime):
                        recipe[key] = value.isoformat()
            
            # Serialize all recipes
            serialized_recipes = [serialize_recipe_data(recipe) for recipe in recipes]
            
            return RecipeExtractionResponse(
                success=True,
                message=f"Retrieved {len(recipes)} recipes for user {user_id} successfully",
                recipe_data={"recipes": serialized_recipes}
            )
                
        except Exception as e:
            print(f"Error retrieving recipes for user {user_id}: {e}")
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipes for user",
                error=str(e)
            )