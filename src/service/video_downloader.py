import yt_dlp
import aiohttp

def get_instagram_video_url_and_metadata(instagram_post_url, cookies_file=None):
    """
    Extract video URL and metadata (description, comments, etc.) from Instagram post.
    Returns: (video_url, metadata_dict)
    """
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': False,
    }
    if cookies_file:
        ydl_opts['cookiefile'] = cookies_file

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(instagram_post_url, download=False)
        
        if not info:
            return None, None
        
        # Extract metadata
        metadata = {
            'title': info.get('title', 'Instagram Video'),
            'description': info.get('description', ''),
            'uploader': info.get('uploader', 'Instagram User'),
            'uploader_id': info.get('uploader_id', ''),
            'upload_date': info.get('upload_date', ''),
            'duration': info.get('duration', 0),
            'view_count': info.get('view_count', 0),
            'like_count': info.get('like_count', 0),
            'comment_count': info.get('comment_count', 0),
            'webpage_url': info.get('webpage_url', instagram_post_url),
            'id': info.get('id', ''),
        }
        
        print(f"📝 Instagram Metadata:")
        print(f"   Title: {metadata['title']}")
        print(f"   Uploader: {metadata['uploader']}")
        print(f"   Description: {metadata['description'][:200]}..." if len(metadata['description']) > 200 else f"   Description: {metadata['description']}")
        
        # Find best format with BOTH video and audio
        video_url = None
        
        # First, try to get format with both video and audio
        for f in info.get('formats', []):
            # Check if format has both video and audio codecs
            if (f.get('ext') == 'mp4' and 
                f.get('vcodec') != 'none' and 
                f.get('acodec') != 'none'):
                video_url = f['url']
                print(f"✅ Found format with both video and audio: {f.get('format_note', 'unknown quality')}")
                break
        
        # Fallback: try any mp4 format
        if not video_url:
            for f in info.get('formats', []):
                if f.get('ext') == 'mp4':
                    video_url = f['url']
                    print(f"⚠️  Using fallback format (may not have audio): {f.get('format_note', 'unknown')}")
                    break
        
        # Last fallback
        if not video_url and 'url' in info:
            video_url = info['url']
            print("⚠️  Using basic URL fallback")
        
        return video_url, metadata

def get_instagram_video_url(instagram_post_url, cookies_file=None):
    """Legacy function - returns just the URL"""
    url, _ = get_instagram_video_url_and_metadata(instagram_post_url, cookies_file)
    return url


async def download_video(url: str) -> dict:
    """Download video from  URL in byte format."""

    
    print(f"📥 Downloading video from: {url}...")
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(f"📊 Response status: {response.status}")
            
            if response.status != 200:
                raise Exception(f"Failed to download video: {response.status}")
            
            content = await response.read()
            print(f"📦 Downloaded {len(content)} bytes")
                        
            return content
