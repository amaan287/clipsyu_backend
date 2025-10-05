"""
Recipe Extraction Controller Module

This module handles recipe extraction from YouTube and Instagram videos.
It coordinates video downloading, analysis, OCR, and AI-powered recipe extraction.
"""

import os
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from bson import ObjectId
from dotenv import load_dotenv

from dal import RecipeExtractionRequest, RecipeExtractionResponse, recipe_collection
from agents.agent import (
    extract_recipe_with_ai,
    download_youtube_video,
    get_first_comment_from_video,
    transcribe_video_with_gemini,
    extract_text_from_video_frames
)
from agents.instagram_agent import download_instagram_video, get_instagram_comments_info

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecipeSerializer:
    """Handles serialization of recipe data for JSON responses"""
    
    @staticmethod
    def serialize(recipe_data: Any) -> Any:
        """
        Convert MongoDB objects to JSON-serializable format
        
        Args:
            recipe_data: Data to serialize (dict, list, or primitive)
            
        Returns:
            JSON-serializable version of the data
        """
        if isinstance(recipe_data, dict):
            return {
                key: RecipeSerializer.serialize(value)
                for key, value in recipe_data.items()
            }
        
        if isinstance(recipe_data, list):
            return [
                RecipeSerializer.serialize(item) 
                for item in recipe_data
            ]
        
        if isinstance(recipe_data, ObjectId):
            return str(recipe_data)
        
        if isinstance(recipe_data, datetime):
            return recipe_data.isoformat()
        
        return recipe_data

class RecipeExtractionController:
    """
    Controller for extracting recipes from video URLs
    
    Supports YouTube and Instagram video sources.
    Coordinates downloading, transcription, OCR, and AI-based recipe extraction.
    """
    
    # URL patterns for supported platforms
    YOUTUBE_PATTERNS = ("youtube.com/", "youtu.be/")
    INSTAGRAM_PATTERNS = ("instagram.com/", "instagr.am/")
    
    def __init__(self):
        """Initialize controller and validate required API keys"""
        self.google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_APIKEY")
        
        self._validate_environment_variables()
    
    def _validate_environment_variables(self) -> None:
        """Validate that all required environment variables are set"""
        if not self.google_cloud_api_key:
            raise ValueError("GOOGLE_CLOUD_API_KEY environment variable is required")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_APIKEY environment variable is required")
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if URL is from YouTube"""
        return any(pattern in url for pattern in self.YOUTUBE_PATTERNS)
    
    def _is_instagram_url(self, url: str) -> bool:
        """Check if URL is from Instagram"""
        return any(pattern in url for pattern in self.INSTAGRAM_PATTERNS)
    
    async def _download_video(self, url: str) -> tuple[Optional[str], Optional[Dict]]:
        """
        Download video from supported platforms
        
        Args:
            url: Video URL
            
        Returns:
            Tuple of (video_file_path, metadata)
            
        Raises:
            ValueError: If URL is from unsupported platform
        """
        logger.info(f"Downloading video from URL: {url}")
        
        if self._is_youtube_url(url):
            return await download_youtube_video(url)
        elif self._is_instagram_url(url):
            return await download_instagram_video(url)
        else:
            raise ValueError(
                "Unsupported URL. Only YouTube and Instagram URLs are supported."
            )
    
    def _fetch_comment_info(self, url: str) -> Optional[Dict]:
        """
        Fetch comment information from video
        
        Args:
            url: Video URL
            
        Returns:
            Comment information dict or None
        """
        logger.info("Fetching comment information...")
        
        if "youtube.com" in url:
            return get_first_comment_from_video(self.google_cloud_api_key, url)
        elif "instagram.com" in url:
            return get_instagram_comments_info(url)
        
        return None
    
    def _add_platform_url(self, recipe_data: Dict, url: str) -> None:
        """Add platform-specific URL to recipe data"""
        if "youtube.com" in url or "youtu.be" in url:
            recipe_data['youtube_url'] = url
        elif "instagram.com" in url:
            recipe_data['instagram_url'] = url
    
    def _prepare_recipe_for_db(
        self, 
        recipe_data: Dict, 
        user_id: Optional[str]
    ) -> None:
        """
        Prepare recipe data for database insertion
        
        Args:
            recipe_data: Recipe data dictionary (modified in place)
            user_id: Optional user ID to associate with recipe
        """
        # Add timestamps if not present
        if 'created_at' not in recipe_data:
            recipe_data['created_at'] = datetime.now()
        if 'updated_at' not in recipe_data:
            recipe_data['updated_at'] = datetime.now()
        
        # Attach user info if provided
        if user_id:
            recipe_data['user_id'] = user_id
        
        # Convert datetime string to datetime object if needed
        saved_date = recipe_data.get('savedDate')
        if isinstance(saved_date, str):
            recipe_data['savedDate'] = datetime.fromisoformat(
                saved_date.replace('Z', '+00:00')
            )
    
    def _save_recipe_to_db(self, recipe_data: Dict) -> Optional[str]:
        """
        Save recipe to MongoDB
        
        Args:
            recipe_data: Recipe data to save
            
        Returns:
            Recipe ID if successful, None otherwise
        """
        try:
            logger.info("Saving recipe to MongoDB...")
            result = recipe_collection.insert_one(recipe_data)
            recipe_id = str(result.inserted_id)
            logger.info(f"Recipe saved with ID: {recipe_id}")
            return recipe_id
        except Exception as db_error:
            logger.error(f"Could not save to MongoDB: {db_error}")
            return None
    
    def _cleanup_video_file(self, video_file: str) -> None:
        """
        Remove downloaded video file
        
        Args:
            video_file: Path to video file to delete
        """
        try:
            if video_file and os.path.exists(video_file):
                os.remove(video_file)
                logger.info(f"Cleaned up video file: {video_file}")
        except Exception as e:
            logger.warning(f"Could not delete video file {video_file}: {e}")
    
    async def extract_recipe_from_youtube(
        self, 
        request: RecipeExtractionRequest, 
        user_id: Optional[str] = None
    ) -> RecipeExtractionResponse:
        """
        Extract recipe from YouTube or Instagram video URL
        
        Args:
            request: Recipe extraction request containing video URL
            user_id: Optional user ID to associate with the recipe
            
        Returns:
            Response containing extracted recipe data and status
        """
        video_file = None
        
        try:
            logger.info(f"Starting recipe extraction for URL: {request.url}")
            
            # Download the video
            video_file, metadata = await self._download_video(request.url)
            
            if not video_file or not metadata:
                raise ValueError("Failed to download video or retrieve metadata")
            
            # Fetch comment information
            comment_info = self._fetch_comment_info(request.url)
            
            # Analyze video with Gemini
            logger.info("Analyzing video with Gemini...")
            video_analysis = transcribe_video_with_gemini(video_file)
            
            # Extract text from video frames using OCR
            logger.info("Extracting text from video frames...")
            ocr_text = extract_text_from_video_frames(video_file, video_analysis)
            
            # Extract recipe using AI
            logger.info("Extracting recipe with AI...")
            recipe_data = extract_recipe_with_ai(
                metadata, 
                comment_info, 
                video_analysis, 
                ocr_text
            )
            
            # Add platform-specific URL
            self._add_platform_url(recipe_data, request.url)
            
            # Prepare and save to database
            self._prepare_recipe_for_db(recipe_data, user_id)
            recipe_id = self._save_recipe_to_db(recipe_data)
            
            # Serialize for JSON response
            serialized_recipe_data = RecipeSerializer.serialize(recipe_data)
            
            success_message = "Recipe extracted successfully"
            if recipe_id:
                success_message += f" and saved with ID: {recipe_id}"
            else:
                success_message += " (MongoDB save failed)"
            
            return RecipeExtractionResponse(
                success=True,
                message=success_message,
                recipe_id=recipe_id,
                recipe_data=serialized_recipe_data
            )
            
        except Exception as e:
            logger.error(f"Error during recipe extraction: {e}", exc_info=True)
            return RecipeExtractionResponse(
                success=False,
                message="Failed to extract recipe",
                error=str(e)
            )
        
        finally:
            # Always cleanup video file
            if video_file:
                self._cleanup_video_file(video_file)
    
    def get_recipe_by_id(self, recipe_id: str) -> RecipeExtractionResponse:
        """
        Get recipe by ID from MongoDB
        
        Args:
            recipe_id: MongoDB ObjectId as string
            
        Returns:
            Response containing recipe data or error
        """
        try:
            recipe = recipe_collection.find_one({"_id": ObjectId(recipe_id)})
            
            if not recipe:
                return RecipeExtractionResponse(
                    success=False,
                    message="Recipe not found",
                    error="Recipe with the specified ID was not found"
                )
            
            # Serialize the recipe data
            serialized_recipe_data = RecipeSerializer.serialize(recipe)
            
            return RecipeExtractionResponse(
                success=True,
                message="Recipe retrieved successfully",
                recipe_id=recipe_id,
                recipe_data=serialized_recipe_data
            )
                
        except Exception as e:
            logger.error(f"Error retrieving recipe {recipe_id}: {e}", exc_info=True)
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipe",
                error=str(e)
            )
    
    def get_all_recipes(self) -> RecipeExtractionResponse: 
        """
        Get all recipes from MongoDB
        
        Returns:
            Response containing all recipes or error
        """
        try:
            recipes = list(recipe_collection.find())
            
            # Serialize all recipes
            serialized_recipes = [
                RecipeSerializer.serialize(recipe) 
                for recipe in recipes
            ]
            
            return RecipeExtractionResponse(
                success=True,
                message=f"Retrieved {len(recipes)} recipes successfully",
                recipe_data={"recipes": serialized_recipes}
            )
                
        except Exception as e:
            logger.error(f"Error retrieving all recipes: {e}", exc_info=True)
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipes",
                error=str(e)
            )
    
    def get_recipes_by_user_id(self, user_id: str) -> RecipeExtractionResponse:
        """
        Get all recipes for a specific user from MongoDB
        
        Args:
            user_id: User ID to filter recipes
            
        Returns:
            Response containing user's recipes or error
        """
        try:
            recipes = list(recipe_collection.find({"user_id": user_id}))
            
            # Serialize all recipes
            serialized_recipes = [
                RecipeSerializer.serialize(recipe) 
                for recipe in recipes
            ]
            
            return RecipeExtractionResponse(
                success=True,
                message=f"Retrieved {len(recipes)} recipes for user {user_id} successfully",
                recipe_data={"recipes": serialized_recipes}
            )
                
        except Exception as e:
            logger.error(
                f"Error retrieving recipes for user {user_id}: {e}", 
                exc_info=True
            )
            return RecipeExtractionResponse(
                success=False,
                message="Failed to retrieve recipes for user",
                error=str(e)
            )