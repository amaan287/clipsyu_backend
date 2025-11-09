"""Utility functions for URL parsing and extraction."""
import re
from typing import Optional, Tuple


class URLParser:
    """URL parsing utilities."""
    
    @staticmethod
    def extract_youtube_video_id(url: str) -> Optional[str]:
        """
        Extract YouTube video ID from URL.
        
        Args:
            url: YouTube URL
            
        Returns:
            Video ID or None if not found
        """
        regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})"
        match = re.search(regex, url)
        return match.group(1) if match else None
    
    @staticmethod
    def extract_instagram_info(url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract Instagram post ID and type from URL.
        
        Args:
            url: Instagram URL
            
        Returns:
            Tuple of (post_id, post_type)
        """
        url = url.strip()
        
        patterns = [
            (r'instagram\.com/p/([A-Za-z0-9_-]+)', 'post'),
            (r'instagram\.com/reel/([A-Za-z0-9_-]+)', 'reel'),
            (r'instagram\.com/tv/([A-Za-z0-9_-]+)', 'tv'),
            (r'instagram\.com/stories/([^/]+)/([0-9]+)', 'story'),
        ]
        
        for pattern, post_type in patterns:
            match = re.search(pattern, url)
            if match:
                post_id = match.group(2) if post_type == 'story' else match.group(1)
                return post_id, post_type
        
        return None, None
    
    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Check if URL is from YouTube."""
        from src.config import Constants
        return any(domain in url for domain in Constants.YOUTUBE_DOMAINS)
    
    @staticmethod
    def is_instagram_url(url: str) -> bool:
        """Check if URL is from Instagram."""
        from src.config import Constants
        return any(domain in url for domain in Constants.INSTAGRAM_DOMAINS)


class TextCleaner:
    """Text cleaning utilities."""
    
    @staticmethod
    def clean_json_response(content: str) -> str:
        """
        Remove code block markers from JSON response.
        
        Args:
            content: Raw JSON content with possible markdown code blocks
            
        Returns:
            Cleaned JSON string
        """
        text = content.strip()
        if text.startswith('```json'):
            text = text[7:]
        if text.endswith('```'):
            text = text[:-3]
        elif text.startswith('```'):
            text = text[3:]
            if text.endswith('```'):
                text = text[:-3]
        return text.strip()
    
    @staticmethod
    def clean_extracted_text(text: str) -> str:
        """
        Clean extracted text from OCR or transcription.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        return text.strip().replace('\n', ' ').replace('\r', ' ')

