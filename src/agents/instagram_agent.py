import re
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from typing import Optional, Dict, Tuple, Any
import httpx
import io

from src.service.video_downloader import get_instagram_video_url_and_metadata, download_video

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()

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

async def download_instagram_video_bytes(url: str, cookies_file: Optional[str] = None) -> Tuple[Optional[io.BytesIO], Optional[Dict]]:
    """
    Download Instagram video as bytes using video_downloader utilities.
    Returns: (BytesIO(video_data), metadata_dict)
    """
    try:
        print(f"Downloading Instagram video from: {url}")
        
        # 1. Extract direct video URL and metadata (including description) using yt-dlp
        video_url, yt_metadata = get_instagram_video_url_and_metadata(url, cookies_file)
        if not video_url:
            print("Error: Could not extract video URL")
            return None, None
        
        # 2. Download video bytes
        video_bytes = await download_video(video_url)
        if not video_bytes:
            print("Error: Could not download video bytes")
            return None, None
        
        # 3. Create BytesIO object
        video_file_like = io.BytesIO(video_bytes)
        
        # 4. Create metadata dict with description from yt-dlp
        post_id, post_type = extract_instagram_url_info(url)
        metadata = {
            "title": yt_metadata.get('title', f"Instagram {post_type or 'video'}") if yt_metadata else f"Instagram {post_type or 'video'}",
            "description": yt_metadata.get('description', '') if yt_metadata else '',
            "uploader": yt_metadata.get('uploader', 'Instagram User') if yt_metadata else 'Instagram User',
            "uploader_id": yt_metadata.get('uploader_id', '') if yt_metadata else '',
            "duration": yt_metadata.get('duration', 0) if yt_metadata else 0,
            "view_count": yt_metadata.get('view_count', 0) if yt_metadata else 0,
            "like_count": yt_metadata.get('like_count', 0) if yt_metadata else 0,
            "source_url": url,
            "direct_video_url": video_url,
            "post_id": post_id,
            "post_type": post_type,
        }
        
        print(f"✅ Successfully downloaded {len(video_bytes)} bytes")
        if metadata.get('description'):
            print(f"📝 Description found: {metadata['description'][:150]}...")
        
        return video_file_like, metadata
        
    except Exception as e:
        print(f"Error downloading Instagram video: {e}")
        return None, None
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


class InstagramAgent:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.client_timeout = 120

    async def get_recipe_from_url(self, url: str) -> Dict:
        """
        Process Instagram video URL and return a Guide-compatible dict
        Returns a dict that can be directly converted to Guide model
        """
        # Download video bytes and metadata
        video_file_like, metadata = await download_instagram_video_bytes(url)
        if not video_file_like or not metadata:
            raise Exception("Could not download Instagram video.")

        video_bytes = video_file_like.getvalue()
        print(f"📦 Video downloaded: {len(video_bytes)} bytes")
        
        # Transcribe using Whisper
        print("=" * 60)
        print("STARTING TRANSCRIPTION PROCESS")
        print("=" * 60)
        transcription, whisper_err = await transcribe_video_whisper_api(video_bytes, api_key=self.openai_api_key)
        print("=" * 60)
        print("TRANSCRIPTION PROCESS COMPLETED")
        print("=" * 60)
        
        if whisper_err:
            print(f"❌ Transcription error: {whisper_err}")
            raise Exception(f"Transcription failed: {whisper_err}")
        
        print(f"📝 Transcription result: {len(transcription)} characters")
        if transcription:
            print(f"Transcription preview: {transcription[:200]}...")
        
        # Early warning if no transcription data
        if not transcription or len(transcription) < 50:
            print("⚠️  WARNING: Transcription is empty or very short. AI may not be able to extract meaningful content.")
            print(f"Transcription length: {len(transcription)} characters")

        # Get placeholder for ocr_text (can be improved with actual OCR extraction logic)
        ocr_text = ""
        
        # Get placeholder for comment (function exists, but likely returns dummy)
        comment_info = get_instagram_comments_info(url)

        guide_json, guide_err = await extract_guide_with_openai_async(
            metadata=metadata,
            comment_info=comment_info.get("note", "") if isinstance(comment_info, dict) else str(comment_info),
            transcription=transcription,
            ocr_text=ocr_text,
            api_key=self.openai_api_key
        )
        if guide_err:
            raise Exception(f"Guide extraction failed: {guide_err}")

        # Ensure all required Guide fields are present
        guide_response = {
            "url": metadata.get("source_url", url),
            "title": guide_json.get("title", "Instagram Guide"),
            "description": guide_json.get("description", ""),
            "content_type": guide_json.get("content_type", "general"),
            "materials": guide_json.get("materials", []),
            "steps": guide_json.get("steps", []),
            "metadata": guide_json.get("metadata", {
                "difficulty": "medium",
                "tags": [],
            }),
            "tools": guide_json.get("tools", []),
            "tips": guide_json.get("tips", []),
            "prerequisites": guide_json.get("prerequisites", []),
            "ocrExtractedInfo": str(bool(ocr_text)),
            "channelName": metadata.get("uploader", "Instagram User"),
            "savedDate": datetime.now(),
            "isInstructional": guide_json.get("isInstructional", False),
            "transcription": transcription,
            "video_analysis": transcription,  # Store transcription as video analysis
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }
        
        return guide_response


async def transcribe_video_whisper_api(video_bytes: bytes, api_key: str) -> tuple[str, str]:
    """
    Transcribe video using OpenAI Whisper v1 API. Returns (transcription, error)
    Extracts audio from video first to ensure compatibility with Whisper API.
    Returns empty string (no error) if video has no audio track.
    """
    import tempfile
    import subprocess
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    temp_video = None
    temp_audio = None
    
    try:
        print(f"📹 Starting transcription process for {len(video_bytes)} bytes of video")
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            temp_video = tmp_video.name
        
        print(f"✅ Created temp video file: {temp_video}")
        
        # First, check if video has audio stream
        print("🔍 Checking for audio stream in video...")
        probe_result = subprocess.run(
            ['ffmpeg', '-i', temp_video, '-hide_banner'],
            capture_output=True,
            text=True
        )
        
        # Print ffmpeg output for debugging
        print("📊 FFmpeg probe output:")
        print(probe_result.stderr[:500])  # Print first 500 chars
        
        # Check if there's an audio stream in the ffmpeg output
        has_audio = 'Stream #' in probe_result.stderr and 'Audio:' in probe_result.stderr
        
        if not has_audio:
            print("⚠️ Video has no audio track detected. Skipping transcription.")
            print(f"Debug: Looking for 'Stream #' and 'Audio:' in ffmpeg output")
            return "", ""  # Return empty transcription, no error
        
        print("✅ Audio stream detected!")
        
        # Create temporary audio file path
        temp_audio = tempfile.mktemp(suffix='.m4a')
        
        # Extract audio using ffmpeg
        print("🎵 Extracting audio from video for Whisper API...")
        result = subprocess.run(
            ['ffmpeg', '-i', temp_video, '-vn', '-acodec', 'aac', '-ar', '16000', '-ac', '1', '-y', temp_audio],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"❌ FFmpeg audio extraction failed with return code: {result.returncode}")
            print(f"Error output: {result.stderr[:500]}")
            # Check if error is due to no audio stream
            if 'does not contain any stream' in result.stderr or 'Output file #0 does not contain any stream' in result.stderr:
                print("⚠️ Video has no audio stream. Skipping transcription.")
                return "", ""  # Return empty transcription, no error
            return "", f"Audio extraction failed: {result.stderr}"
        
        print(f"✅ FFmpeg extraction completed successfully")
        
        # Check if audio file was created and has content
        if not os.path.exists(temp_audio):
            print(f"❌ Audio file was not created at: {temp_audio}")
            return "", ""
        
        audio_size = os.path.getsize(temp_audio)
        if audio_size == 0:
            print("⚠️ Audio file is empty (0 bytes). Skipping transcription.")
            return "", ""
        
        print(f"✅ Audio file created: {audio_size} bytes")
        
        # Read extracted audio
        with open(temp_audio, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        print(f"✅ Audio read successfully: {len(audio_bytes)} bytes")
        
        # Create in-memory file-like object from audio bytes
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)  # Reset to beginning
        
        print("🌐 Sending audio to OpenAI Whisper API...")
        async with httpx.AsyncClient(timeout=120) as client:
            # Send as M4A audio format
            files = {"file": ("audio.m4a", audio_buffer, "audio/m4a")}
            data = {"model": "whisper-1"}
            headers = {"Authorization": f"Bearer {api_key}"}
            
            response = await client.post(url, files=files, data=data, headers=headers)
            
            print(f"📡 Whisper API response status: {response.status_code}")
            
            if response.status_code != 200:
                error_msg = f"OpenAI Whisper API error (status {response.status_code}): {response.text}"
                print(f"❌ {error_msg}")
                return "", error_msg
            
            resp = response.json()
            transcription_text = resp.get("text", "")
            print(f"✅ Transcription successful: {len(transcription_text)} characters")
            if transcription_text:
                print(f"Preview: {transcription_text[:100]}...")
            return transcription_text, ""
    except Exception as e:
        print(f"❌ Exception during transcription: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return "", str(e)
    finally:
        # Clean up temporary files
        if temp_video and os.path.exists(temp_video):
            try:
                os.remove(temp_video)
            except Exception as e:
                print(f"Warning: Could not remove temp video file: {e}")
        if temp_audio and os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except Exception as e:
                print(f"Warning: Could not remove temp audio file: {e}")


def build_guide_prompt(metadata: dict, comment_info: str, transcription: str, ocr_text: str) -> str:
    ocr_section = f"\n\nOCR TEXT: {ocr_text}" if ocr_text else ""
    
    # Highlight importance of description when transcription is empty
    description = metadata.get('description', '')
    transcription_note = ""
    if not transcription or len(transcription) < 50:
        if description:
            transcription_note = "\n⚠️ NOTE: Video has no audio transcription. Extract information primarily from DESCRIPTION.\n"
        else:
            transcription_note = "\n⚠️ NOTE: Video has no audio transcription and no description. Limited data available.\n"
    
    return (
        f"Analyze this Instagram video and extract a step-by-step guide in JSON format.\n"
        f"{transcription_note}\n"
        f"TITLE: {metadata.get('title', '')}\n"
        f"DESCRIPTION: {description}\n"
        f"UPLOADER: {metadata.get('uploader', '')}\n"
        f"COMMENT: {comment_info}\n"
        f"TRANSCRIPTION: {transcription}{ocr_section}\n\n"
        f"Extract the guide with this structure:\n"
        f"{{\n"
        f"  \"title\": \"guide name\",\n"
        f"  \"description\": \"brief overview\",\n"
        f"  \"content_type\": \"recipe|tutorial|how-to|programming|educational|general\",\n"
        f"  \"materials\": [{{\"name\": \"item\", \"quantity\": \"amount\", \"notes\": \"notes\", \"optional\": false}}],\n"
        f"  \"steps\": [{{\"step\": 1, \"instruction\": \"description\", \"duration\": \"time\", \"details\": \"extras\", \"code_snippet\": \"code\", \"tips\": [\"tips\"]}}],\n"
        f"  \"metadata\": {{\"duration\": \"\", \"difficulty\": \"easy|medium|hard\", \"category\": \"\", \"tags\": [], \"estimated_time\": \"\", \"skill_level\": \"beginner|intermediate|advanced\", \"language\": \"\", \"framework\": \"\"}},\n"
        f"  \"tools\": [\"tools/equipment\"],\n"
        f"  \"tips\": [\"general tips\"],\n"
        f"  \"prerequisites\": [\"requirements\"],\n"
        f"  \"isInstructional\": true\n"
        f"}}\n\n"
        f"CRITICAL INSTRUCTIONS:\n"
        f"- ONLY extract information that is EXPLICITLY present in the DESCRIPTION, TRANSCRIPTION, COMMENT, or OCR TEXT\n"
        f"- PRIORITIZE: If TRANSCRIPTION is empty, use DESCRIPTION as primary source (Instagram recipes often have full details in description)\n"
        f"- For Instagram posts: Description often contains complete recipe ingredients and steps - extract them carefully\n"
        f"- DO NOT make up, infer, or assume any steps, ingredients, or instructions that are not clearly stated\n"
        f"- If TRANSCRIPTION is empty AND DESCRIPTION is empty, set \"isInstructional\": false and return minimal structure\n"
        f"- Set \"isInstructional\": true when you can extract clear, specific step-by-step instructions from description OR transcription\n"
        f"- For recipes: extract ingredients from description, transcription, or comments\n"
        f"- For tutorials: extract steps from description, transcription, or comments\n"
        f"- Use OCR text for precise measurements, code, or commands when available\n"
        f"- Return only valid JSON\n\n"
        f"Extract now:\n"
    )

async def extract_guide_with_openai_async(metadata: dict, comment_info: str, transcription: str, ocr_text: str, api_key: str) -> tuple[Any, str]:
    """
    Calls OpenAI chat completions (GPT-4o) to extract a guide, requesting JSON-only response as in Go. Returns (json_obj, error)
    """
    url = "https://api.openai.com/v1/chat/completions"
    prompt = build_guide_prompt(metadata, comment_info, transcription, ocr_text)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert content analyzer that extracts step-by-step guides from instructional videos. You MUST ONLY extract information explicitly present in the provided transcription. NEVER make up, infer, or hallucinate content. If there is insufficient information, return a minimal structure with isInstructional set to false. Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 1200,
        "response_format": {"type": "json_object"},
    }
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, headers=headers, content=json.dumps(payload))
            if resp.status_code != 200:
                return None, f"OpenAI GPT-4o API error (status {resp.status_code}): {resp.text}"
            resp_json = resp.json()
            choices = resp_json.get('choices', [])
            if not choices:
                return None, "No choices in OpenAI chat response"
            content = choices[0]["message"].get("content", "")
            json_result = _clean_and_parse_json_response(content)
            return json_result, ""
    except Exception as e:
        return None, str(e)

def _clean_and_parse_json_response(content: str) -> dict:
    """Remove code block markers and parse JSON."""
    text = content.strip()
    if text.startswith('```json'):
        text = text[7:]
    if text.endswith('```'):
        text = text[:-3]
    elif text.startswith('```'):
        text = text[3:]
        if text.endswith('```'):
            text = text[:-3]
    try:
        return json.loads(text)
    except Exception:
        return {"raw": text, "error": "Could not parse JSON"}

