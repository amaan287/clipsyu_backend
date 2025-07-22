import yt_dlp
import re
from dotenv import load_dotenv
import os
import google.generativeai as genai
import json
from pathlib import Path
from datetime import datetime
import time
import cv2
import pytesseract
from PIL import Image
import numpy as np
import requests
from urllib.parse import urlparse

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_APIKEY")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

def extract_instagram_url_info(url):
    """Extract Instagram post/reel ID and type from URL"""
    # Clean the URL
    url = url.strip()
    
    # Instagram URL patterns
    patterns = [
        r'instagram\.com/p/([A-Za-z0-9_-]+)',      # Regular posts
        r'instagram\.com/reel/([A-Za-z0-9_-]+)',  # Reels
        r'instagram\.com/tv/([A-Za-z0-9_-]+)',    # IGTV
        r'instagram\.com/stories/([^/]+)/([0-9]+)', # Stories
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            if 'stories' in pattern:
                return match.group(2), 'story'
            else:
                return match.group(1), 'post'
    
    return None, None

def download_instagram_video(url: str, cookies: str = "cookies.txt", output_path: str = "./downloads"):
    """Download Instagram video using yt-dlp"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Clean title for filename
    def clean_filename(title):
        # Remove or replace problematic characters
        cleaned = re.sub(r'[<>:"/\\|?*#]', '', title)
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
        return cleaned.strip()
    
    try:
        # First, get video info without downloading
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            # Extract metadata
            original_title = info.get('title', 'instagram_video')
            safe_title = clean_filename(original_title)
            
            # If title is too generic, use uploader and timestamp
            if not safe_title or safe_title == 'instagram_video':
                uploader = info.get('uploader', 'unknown')
                timestamp = info.get('timestamp', int(time.time()))
                safe_title = f"{uploader}_{timestamp}"
                safe_title = clean_filename(safe_title)
        
        # Set up download options
        ydl_opts = {
            'outtmpl': f'{output_path}/{safe_title}.%(ext)s',
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'cookies': cookies if os.path.exists(cookies) else None,
            'quiet': False,
            'no_warnings': False,
            'extract_flat': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
        
        # Remove cookies if file doesn't exist
        if not os.path.exists(cookies):
            ydl_opts.pop('cookies', None)
            print(f"Warning: Cookies file '{cookies}' not found. Proceeding without cookies.")
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Downloading Instagram video: {url}")
            ydl.download([url])
            info = ydl.extract_info(url, download=False)
            
            # Extract comprehensive metadata
            metadata = {
                "title": info.get("title", "Instagram Video"),
                "description": info.get("description", ""),
                "uploader": info.get("uploader", ""),
                "uploader_id": info.get("uploader_id", ""),
                "uploader_url": info.get("uploader_url", ""),
                "upload_date": info.get("upload_date", ""),
                "timestamp": info.get("timestamp", ""),
                "duration": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "comment_count": info.get("comment_count", 0),
                "webpage_url": info.get("webpage_url", url),
                "extractor": info.get("extractor", "instagram"),
                "format": info.get("format", ""),
                "width": info.get("width", 0),
                "height": info.get("height", 0),
                "fps": info.get("fps", 0),
                "age_limit": info.get("age_limit", 0),
            }
            
            print("--- Instagram Video Metadata ---")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            # Determine the actual downloaded file path
            downloaded_file = f"{output_path}/{safe_title}.mp4"
            
            # Check if file exists, if not, try to find it
            if not os.path.exists(downloaded_file):
                # Look for any video files in the directory that might match
                for file in os.listdir(output_path):
                    if file.endswith(('.mp4', '.mkv', '.webm')) and safe_title[:20] in file:
                        downloaded_file = f"{output_path}/{file}"
                        break
            
            # Verify the file exists
            if not os.path.exists(downloaded_file):
                # List all files in the directory to help debug
                files = os.listdir(output_path)
                print(f"Available files in {output_path}: {files}")
                raise FileNotFoundError(f"Downloaded video file not found. Expected: {downloaded_file}")
            
            print(f"Instagram video downloaded to: {downloaded_file}")
            return downloaded_file, metadata
            
    except Exception as e:
        print(f"Error downloading Instagram video: {e}")
        raise

def get_instagram_comments_info(url):
    """
    Extract basic comment information from Instagram URL
    Note: This is a placeholder since Instagram doesn't provide public API for comments
    """
    try:
        # This is a simplified approach - Instagram's API is restricted
        # In a real implementation, you might need to use Instagram's official API
        # or implement web scraping (which has its own legal/ethical considerations)
        
        post_id, post_type = extract_instagram_url_info(url)
        
        if post_id:
            comment_info = {
                "post_id": post_id,
                "post_type": post_type,
                "note": "Instagram comments require special API access or web scraping",
                "url": url,
                "extracted_at": datetime.now().isoformat()
            }
            
            print(f"Instagram post ID: {post_id}")
            print(f"Post type: {post_type}")
            print("Note: Comment extraction requires Instagram API access")
            
            return comment_info
        else:
            return {
                "error": "Could not extract post ID from Instagram URL",
                "url": url,
                "extracted_at": datetime.now().isoformat()
            }
            
    except Exception as e:
        return {
            "error": f"Error extracting Instagram info: {str(e)}",
            "url": url,
            "extracted_at": datetime.now().isoformat()
        }

