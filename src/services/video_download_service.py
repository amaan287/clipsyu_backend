"""Video download services implementation."""
import aiohttp
import yt_dlp
import json
import os
from typing import Dict, Optional, Tuple
from playwright.async_api import async_playwright

from src.interfaces import IVideoDownloader, IVideoMetadataExtractor


class HTTPVideoDownloader(IVideoDownloader):
    """Service for downloading videos via HTTP."""
    
    async def download_video(self, url: str) -> Tuple[Optional[bytes], Optional[Dict]]:
        """
        Download video from direct URL.
        
        Args:
            url: Direct video URL
            
        Returns:
            Tuple of (video_bytes, metadata_dict)
        """
        print(f"📥 Downloading video from: {url}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    print(f"📊 Response status: {response.status}")
                    
                    if response.status != 200:
                        raise Exception(f"Failed to download video: {response.status}")
                    
                    content = await response.read()
                    print(f"📦 Downloaded {len(content)} bytes")
                    
                    return content, None
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None, None


class YTDLPMetadataExtractor(IVideoMetadataExtractor):
    """Service for extracting video metadata using yt-dlp."""
    
    def __init__(self, cookies_file: Optional[str] = None):
        """
        Initialize metadata extractor.
        
        Args:
            cookies_file: Optional path to cookies file
        """
        self.cookies_file = cookies_file
    
    def extract_metadata(
        self, 
        url: str, 
        cookies_file: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Extract video metadata and direct URL using yt-dlp.
        
        Args:
            url: Video URL
            cookies_file: Optional cookies file path
            
        Returns:
            Tuple of (direct_video_url, metadata_dict)
        """
        cookies = cookies_file or self.cookies_file
        
        ydl_opts = {
            'skip_download': True,
            'quiet': True,
            'no_warnings': False,
            'format': 'best[ext=mp4]/best',  # More flexible format selection
            'nocheckcertificate': True,
            'age_limit': None,
            'no_color': True,
            # These options help with YouTube's recent changes
            'extractor_args': {
                'youtube': {
                    'player_client': ['android', 'web'],  # Try multiple clients
                    'skip': ['dash', 'hls']  # Skip problematic formats
                }
            }
        }
        
        if cookies and os.path.exists(cookies):
            ydl_opts['cookiefile'] = cookies
            print(f"Using cookies file: {cookies}")
        else:
            print(f"No cookies file found, proceeding without cookies")
        
        try:
            print(f"Extracting video info using yt-dlp...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if not info:
                    print("❌ yt-dlp returned no info")
                    return None, None
                
                print(f"✅ Info extracted by yt-dlp")
                
                metadata = self._extract_metadata_from_info(info, url)
                video_url = self._find_best_video_url(info)
                
                self._log_metadata(metadata)
                
                if not video_url:
                    print(f"❌ Could not find suitable video URL in formats")
                    print(f"   Available formats: {len(info.get('formats', []))}")
                
                return video_url, metadata
        except Exception as e:
            print(f"❌ Error extracting metadata with yt-dlp: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    @staticmethod
    def _extract_metadata_from_info(info: Dict, url: str) -> Dict:
        """Extract metadata from yt-dlp info dictionary."""
        return {
            'title': info.get('title', 'Video'),
            'description': info.get('description', ''),
            'channel': info.get('channel', info.get('uploader', 'User')),
            'uploader': info.get('uploader', 'User'),
            'uploader_id': info.get('uploader_id', ''),
            'uploader_url': info.get('uploader_url', ''),
            'upload_date': info.get('upload_date', ''),
            'duration': info.get('duration', 0),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'comment_count': info.get('comment_count', 0),
            'webpage_url': info.get('webpage_url', url),
            'id': info.get('id', ''),
        }
    
    @staticmethod
    def _find_best_video_url(info: Dict) -> Optional[str]:
        """
        Find best video URL with both video and audio.
        Uses multiple fallback strategies to handle YouTube's format restrictions.
        """
        formats = info.get('formats', [])
        
        if not formats:
            # If no formats list, try direct URL
            if 'url' in info:
                print("⚠️  Using direct URL from info (no formats available)")
                return info['url']
            return None
        
        print(f"Searching through {len(formats)} available formats...")
        
        # Strategy 1: Try to find mp4 format with both video and audio
        for f in formats:
            if (f.get('ext') == 'mp4' and 
                f.get('vcodec') != 'none' and 
                f.get('acodec') != 'none' and
                f.get('url')):  # Ensure URL exists
                print(f"✅ Found mp4 with video+audio: {f.get('format_note', 'unknown quality')}")
                return f['url']
        
        # Strategy 2: Try any mp4 format (may be video-only or audio-only)
        for f in formats:
            if f.get('ext') == 'mp4' and f.get('url'):
                print(f"⚠️  Using mp4 format (may be video or audio only): {f.get('format_note', 'unknown')}")
                return f['url']
        
        # Strategy 3: Try best format regardless of container
        for f in formats:
            if f.get('vcodec') != 'none' and f.get('url'):
                print(f"⚠️  Using best available video format: {f.get('ext', 'unknown')} - {f.get('format_note', 'unknown')}")
                return f['url']
        
        # Strategy 4: Last resort - any format with a URL
        for f in formats:
            if f.get('url'):
                print(f"⚠️  Using fallback format: {f.get('ext', 'unknown')}")
                return f['url']
        
        # Strategy 5: Check if there's a direct URL in info
        if 'url' in info:
            print("⚠️  Using basic URL fallback from info")
            return info['url']
        
        print("❌ No suitable format found with accessible URL")
        return None
    
    @staticmethod
    def _log_metadata(metadata: Dict) -> None:
        """Log metadata information."""
        print(f"📝 Video Metadata:")
        print(f"   Title: {metadata['title']}")
        print(f"   Uploader: {metadata['uploader']}")
        desc = metadata['description']
        if len(desc) > 200:
            print(f"   Description: {desc[:200]}...")
        else:
            print(f"   Description: {desc}")


class YouTubeVideoDownloader(IVideoDownloader):
    """Service for downloading YouTube videos with Playwright pre-navigation."""
    
    def __init__(self, cookies_file: Optional[str] = None):
        """
        Initialize YouTube downloader.
        
        Args:
            cookies_file: Optional path to cookies file
        """
        self.cookies_file = cookies_file
        self.metadata_extractor = YTDLPMetadataExtractor(cookies_file)
        self.http_downloader = HTTPVideoDownloader()
    
    async def download_video(self, url: str) -> Tuple[Optional[bytes], Optional[Dict]]:
        """
        Download YouTube video.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (video_bytes, metadata_dict)
        """
        try:
            # Pre-navigate with Playwright
            await self._prenavigate_with_playwright(url)
            
            # Extract metadata and direct URL
            print(f"Extracting metadata for: {url}")
            video_url, metadata = self.metadata_extractor.extract_metadata(url, self.cookies_file)
            
            if not video_url:
                print("❌ Error: Could not extract direct video URL from yt-dlp")
                if metadata:
                    print(f"   However, metadata was extracted: {metadata.get('title', 'Unknown')}")
                return None, metadata
            
            print(f"✅ Video URL extracted: {video_url[:80]}...")
            
            # Download video
            print(f"Downloading video bytes from direct URL...")
            video_bytes, _ = await self.http_downloader.download_video(video_url)
            
            if not video_bytes:
                print("❌ Error: Failed to download video bytes")
                return None, metadata
            
            print(f"✅ Video downloaded successfully: {len(video_bytes)} bytes")
            return video_bytes, metadata
            
        except Exception as e:
            print(f"❌ Error in YouTubeVideoDownloader.download_video: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    async def _prenavigate_with_playwright(self, url: str) -> None:
        """
        Pre-navigate to URL using Playwright to establish session.
        
        Args:
            url: YouTube URL
        """
        print(f"Pre-navigating to URL with Playwright: {url}")
        
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-setuid-sandbox",
                        "--disable-blink-features=AutomationControlled",
                        "--user-agent=Chrome/120.0.0.0"
                    ]
                )
                page = await browser.new_page()
                
                # Set additional headers
                await page.set_extra_http_headers({
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
                })
                
                # Load cookies if available
                if self.cookies_file and os.path.exists(self.cookies_file):
                    await self._load_cookies(page)
                
                # Navigate to the URL
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                    page_title = await page.title()
                    print(f"Successfully navigated to: {page_title}")
                    await page.wait_for_timeout(2000)
                except Exception as nav_error:
                    print(f"Navigation error: {nav_error}")
                
                await browser.close()
        except Exception as e:
            print(f"Warning: Playwright navigation failed: {e}")
            print("Proceeding with direct yt-dlp...")
    
    async def _load_cookies(self, page) -> None:
        """Load cookies into Playwright page."""
        try:
            with open(self.cookies_file, 'r') as f:
                if self.cookies_file.endswith('.json'):
                    cookies_data = json.load(f)
                    await page.context.add_cookies(cookies_data)
                else:
                    print("Note: Netscape cookies format detected, will be used by yt-dlp directly")
        except Exception as e:
            print(f"Warning: Could not load cookies for Playwright: {e}")

