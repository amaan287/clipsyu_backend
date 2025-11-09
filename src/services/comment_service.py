"""Comment extraction services."""
from datetime import datetime
from typing import Dict, Optional
from googleapiclient.discovery import build

from src.interfaces import ICommentExtractor
from src.config import Config
from src.utils.url_parser import URLParser


class YouTubeCommentExtractor(ICommentExtractor):
    """Service for extracting YouTube comments."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize YouTube comment extractor.
        
        Args:
            api_key: YouTube API key (optional, uses config if not provided)
        """
        self.api_key = api_key or Config.YOUTUBE_API_KEY
    
    def get_comments(self, url: str) -> Dict:
        """
        Extract first comment from YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary containing comment information
        """
        video_id = URLParser.extract_youtube_video_id(url)
        
        if not video_id:
            return {"error": "Could not extract Video ID from URL."}
        
        try:
            youtube = build('youtube', 'v3', developerKey=self.api_key)
            
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=1,
                order="time"
            )
            response = request.execute()
            
            if not response.get("items"):
                return {"message": "No comments found on this video."}
            
            first_comment = response["items"][0]
            snippet = first_comment["snippet"]["topLevelComment"]["snippet"]
            
            return {
                "author": snippet["authorDisplayName"],
                "text": snippet["textDisplay"],
                "likes": snippet["likeCount"],
                "published": snippet["publishedAt"]
            }
            
        except Exception as e:
            return {"error": f"An error occurred: {e}"}


class InstagramCommentExtractor(ICommentExtractor):
    """
    Service for extracting Instagram comments.
    
    Note: Instagram's API is restricted, this is a placeholder implementation.
    """
    
    def get_comments(self, url: str) -> Dict:
        """
        Extract comment information from Instagram URL.
        
        Note: Instagram comments require special API access or web scraping.
        
        Args:
            url: Instagram URL
            
        Returns:
            Dictionary containing comment information or note about limitations
        """
        post_id, post_type = URLParser.extract_instagram_info(url)
        
        if post_id:
            return {
                "post_id": post_id,
                "post_type": post_type,
                "note": "Instagram comments require special API access or web scraping",
                "url": url,
                "extracted_at": datetime.now().isoformat()
            }
        else:
            return {
                "error": "Could not extract post ID from Instagram URL",
                "url": url,
                "extracted_at": datetime.now().isoformat()
            }

