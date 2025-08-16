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
import subprocess

from typing import Optional, Dict, Tuple
from playwright.async_api import async_playwright
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

async def download_instagram_video(url: str, cookies: str = "cookies.txt", output_path: str = "./downloads") -> Tuple[Optional[str], Optional[Dict]]:
    """Download Instagram video using yt-dlp CLI with Playwright pre-navigation and extract metadata"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Check if cookies file exists
    if not os.path.exists(cookies):
        print(f"Warning: Cookies file '{cookies}' not found. Proceeding without cookies.")
        use_cookies = False
    else:
        use_cookies = True

    # Pre-navigate to URL using Playwright (async)
    print(f"Pre-navigating to Instagram URL with Playwright: {url}")
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox", 
                    "--disable-setuid-sandbox",
                    "--disable-blink-features=AutomationControlled",
                    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ]
            )
            page = await browser.new_page()
            
            # Set additional headers
            await page.set_extra_http_headers({
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Sec-Fetch-Dest': 'document',
                'Sec-Fetch-Mode': 'navigate',
                'Sec-Fetch-Site': 'none'
            })
            
            # Load cookies if available
            if use_cookies and os.path.exists(cookies):
                try:
                    with open(cookies, 'r') as f:
                        if cookies.endswith('.json'):
                            cookies_data = json.load(f)
                            await page.context.add_cookies(cookies_data)
                        else:
                            print("Note: Netscape cookies format detected, will be used by yt-dlp directly")
                except Exception as e:
                    print(f"Warning: Could not load cookies for Playwright: {e}")
            
            # Navigate to the Instagram URL
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                page_title = await page.title()
                print(f"Successfully navigated to: {page_title}")
                await page.wait_for_timeout(3000)  # Wait a bit longer for Instagram
            except Exception as nav_error:
                print(f"Navigation error: {nav_error}")
            
            await browser.close()
            
    except Exception as e:
        print(f"Warning: Playwright navigation failed: {e}")
        print("Proceeding with direct yt-dlp download...")

    # Command to get metadata in JSON
    metadata_cmd = ["yt-dlp", "--dump-json", url]
    if use_cookies:
        metadata_cmd.extend(["--cookies", cookies])
    else:
        metadata_cmd.extend(["--cookies-from-browser", "chrome"])
    
    # Add Instagram-specific options
    metadata_cmd.extend([
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ])
    
    print(f"Fetching metadata for: {url}")
    try:
        result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode != 0:
            print(f"Error fetching metadata: {result.stderr}")
            # Try with Firefox cookies if Chrome failed
            if not use_cookies:
                # Replace chrome with firefox in the command
                for i, arg in enumerate(metadata_cmd):
                    if arg == "chrome":
                        metadata_cmd[i] = "firefox"
                        break
                result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    print("Failed with Firefox cookies too. Trying without cookies...")
                    # Remove cookies arguments and try again
                    metadata_cmd = ["yt-dlp", "--dump-json", url, 
                                  "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"]
                    result = subprocess.run(metadata_cmd, capture_output=True, text=True, timeout=60)
                    if result.returncode != 0:
                        return None, None
            else:
                return None, None
            
        if not result.stdout.strip():
            print("Error: No metadata returned from yt-dlp")
            return None, None
            
        # Parse JSON metadata
        info = json.loads(result.stdout)
        
    except subprocess.TimeoutExpired:
        print("Error: Metadata fetch timed out")
        return None, None
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}")
        print(f"Raw output: {result.stdout[:200]}...")
        print(f"Error output: {result.stderr}")
        return None, None
    except Exception as e:
        print(f"Unexpected error during metadata fetch: {e}")
        return None, None

    # Extract Instagram-specific metadata safely
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
        "id": info.get("id", ""),
    }

    print("--- Instagram Video Metadata ---")
    for key, value in metadata.items():
        print(f"{key}: {value}")

    # Clean filename for download
    safe_title = clean_filename(metadata["title"])
    if not safe_title or safe_title.lower() in ['instagram video', '']:
        # Use uploader and timestamp if title is generic
        uploader = clean_filename(metadata.get("uploader", "unknown"))
        timestamp = metadata.get("timestamp") or int(time.time())
        safe_title = f"{uploader}_{timestamp}"
    
    safe_title = safe_title[:100]  # Limit length to avoid filesystem issues
    
    # Command to download video
    download_cmd = [
        "yt-dlp",
        "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
        "-o", f"{output_path}/%(title).100s.%(ext)s",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        url
    ]
    
    if use_cookies:
        download_cmd.extend(["--cookies", cookies])
    else:
        download_cmd.extend(["--cookies-from-browser", "chrome"])

    print(f"Downloading Instagram video: {url}")
    try:
        download_result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
        
        if download_result.returncode != 0:
            print(f"Error downloading video: {download_result.stderr}")
            # Try with Firefox cookies
            if not use_cookies:
                # Replace chrome with firefox in the command
                for i, arg in enumerate(download_cmd):
                    if arg == "chrome":
                        download_cmd[i] = "firefox"
                        break
                download_result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
                if download_result.returncode != 0:
                    print("Failed with Firefox cookies too. Trying without cookies...")
                    # Remove cookies arguments and try again
                    download_cmd = [
                        "yt-dlp",
                        "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
                        "-o", f"{output_path}/%(title).100s.%(ext)s",
                        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        url
                    ]
                    download_result = subprocess.run(download_cmd, capture_output=True, text=True, timeout=300)
                    if download_result.returncode != 0:
                        return None, metadata
            else:
                return None, metadata
            
        print("Download completed successfully!")
        
        # Find the downloaded file
        downloaded_file = find_downloaded_file(output_path, safe_title, metadata["id"])
        
        return downloaded_file, metadata
        
    except subprocess.TimeoutExpired:
        print("Error: Video download timed out")
        return None, metadata
    except Exception as e:
        print(f"Unexpected error during download: {e}")
        return None, metadata


def find_downloaded_file(output_path: str, safe_title: str, video_id: str) -> Optional[str]:
    """Find the downloaded file in the output directory"""
    
    # Common video extensions
    extensions = ['.mp4', '.webm', '.mkv', '.avi']
    
    # Try to find file by title
    for ext in extensions:
        potential_file = os.path.join(output_path, f"{safe_title}{ext}")
        if os.path.exists(potential_file):
            return potential_file
    
    # If not found by title, search all files in directory
    if os.path.exists(output_path):
        for filename in os.listdir(output_path):
            if (video_id and video_id in filename) or safe_title in filename:
                full_path = os.path.join(output_path, filename)
                if os.path.isfile(full_path) and any(filename.endswith(ext) for ext in extensions):
                    return full_path
    
    print(f"Warning: Could not locate downloaded file in {output_path}")
    return None


def clean_filename(title: str) -> str:
    """Clean title for filename"""
    if not title:
        return ""
    # Remove or replace problematic characters
    cleaned = re.sub(r'[<>:"/\\|?*#]', '', title)
    cleaned = re.sub(r'[^\w\s-]', '', cleaned)
    return cleaned.strip()


# Usage example (synchronous wrapper if needed)
    """Synchronous wrapper for the async Instagram downloader"""
    import asyncio
    
    try:
        # Try to get the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use run()
            print("Warning: Running in async context. Please use the async version directly.")
            return None, None
        else:
            return loop.run_until_complete(download_instagram_video(url, cookies, output_path))
    except RuntimeError:
        # No event loop exists, create a new one
        return asyncio.run(download_instagram_video(url, cookies, output_path))
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

