import yt_dlp
from youtube_comment_downloader import YoutubeCommentDownloader
import re
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from datetime import datetime
import time
import cv2
import pytesseract
from PIL import Image
import numpy as np
import tempfile
from typing import Optional, Tuple, Dict
from playwright.async_api import async_playwright
import io
import aiohttp
import httpx

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()


async def download_youtube_video(url: str, cookies: str = "cookies.txt") -> Tuple[Optional[bytes], Optional[Dict]]:
    # Check if cookies file exists
    if not os.path.exists(cookies):
        print(f"Warning: Cookies file '{cookies}' not found. Proceeding without cookies.")
        use_cookies = False
    else:
        use_cookies = True
    # Pre-navigate to URL using Playwright (async)
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
            
            # Navigate to the YouTube URL
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

    # Use yt-dlp to get video metadata and direct URL
    ydl_opts = {
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    if use_cookies:
        ydl_opts['cookiefile'] = cookies
    
    print(f"Extracting video info for: {url}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            
            if not info:
                print("Error: No info returned from yt-dlp")
                return None, None
            
            # Extract metadata
            metadata = {
                "title": info.get("title", "Unknown Title"),
                "description": info.get("description", ""),
                "channel": info.get("channel", ""),
                "uploader": info.get("uploader", ""),
                "uploader_url": info.get("uploader_url", ""),
                "upload_date": info.get("upload_date", ""),
                "duration": info.get("duration", 0),
                "view_count": info.get("view_count", 0),
                "like_count": info.get("like_count", 0),
                "webpage_url": info.get("webpage_url", url),
                "id": info.get("id", ""),
            }
            
            print("--- Video Metadata ---")
            for key, value in metadata.items():
                print(f"{key}: {value}")
            
            # Get direct video URL - prefer mp4 format
            video_url = None
            for f in info.get('formats', []):
                if f.get('ext') == 'mp4' and f.get('vcodec') != 'none' and f.get('acodec') != 'none':
                    video_url = f['url']
                    break
            
            # Fallback to best format
            if not video_url and 'url' in info:
                video_url = info['url']
            elif not video_url and info.get('formats'):
                # Get best format URL
                video_url = info['formats'][-1].get('url')
            
            if not video_url:
                print("Error: Could not extract direct video URL")
                return None, metadata
            
            print(f"📥 Downloading video from direct URL...")
            
            # Download video as bytes using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(video_url) as response:
                    print(f"📊 Response status: {response.status}")
                    
                    if response.status != 200:
                        raise Exception(f"Failed to download video: {response.status}")
                    
                    video_bytes = await response.read()
                    print(f"📦 Downloaded {len(video_bytes)} bytes")
                    
                    return video_bytes, metadata
                    
    except Exception as e:
        print(f"Error during YouTube video download: {e}")
        return None, None

def get_video_id_from_url(url):
    """Extracts the YouTube video ID from a URL."""
    regex = r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    return None

def get_first_comment_from_video(api_key, video_url):
    """Fetches the first comment from a YouTube video using the API."""
    video_id = get_video_id_from_url(video_url)
    if not video_id:
        return "Error: Could not extract Video ID from URL."
    
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=1,
            order="time"
        )
        response = request.execute()
        
        if not response.get("items"):
            return "No comments found on this video."
        
        first_comment = response["items"][0]
        comment_info = {
            "author": first_comment["snippet"]["topLevelComment"]["snippet"]["authorDisplayName"],
            "text": first_comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"],
            "likes": first_comment["snippet"]["topLevelComment"]["snippet"]["likeCount"],
            "published": first_comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
        }
        
        return comment_info
        
    except Exception as e:
        return f"An error occurred: {e}"

def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply denoising
    denoised = cv2.fastNlMeansDenoising(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def extract_text_from_video_frames(video_bytes, trigger_keywords, frame_interval=30):
    """
    Extract text from video frames using OCR from video bytes
    """
    print(f"Starting OCR text extraction from video bytes: {len(video_bytes) if isinstance(video_bytes, bytes) else 'unknown'}")
    
    ocr_keywords = [
        'on screen', 'text', 'written', 'list', 'shown',
        'display', 'appears', 'visible', 'read', 'see', 'shows', 'measurement',
        'quantity', 'amount', 'cup', 'tablespoon', 'teaspoon', 'gram', 'pound',
        'ounce', 'liter', 'milliliter', 'instructions', 'steps', 'directions'
    ]
    
    transcription_lower = trigger_keywords.lower()
    keywords_found = [keyword for keyword in ocr_keywords if keyword in transcription_lower]
    
    if not keywords_found:
        print("No OCR trigger keywords found in transcription. Skipping OCR extraction.")
        return ""
    
    print(f"OCR trigger keywords found: {keywords_found}")
    
    if not isinstance(video_bytes, bytes):
        return f"Error: Expected bytes, got {type(video_bytes)}"
    
    try:
        # Convert bytes to numpy array for OpenCV
        nparr = np.frombuffer(video_bytes, np.uint8)
        
        # Create a temporary file for OpenCV (OpenCV doesn't support direct bytes for video)
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            temp_path = tmp.name
        
        # Open video with OpenCV
        cap = cv2.VideoCapture(temp_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video from bytes for OCR")
            os.remove(temp_path)
            return ""
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video FPS: {fps}, Total frames: {total_frames}")
        
        extracted_text = []
        frame_count = 0
        processed_frames = 0
        
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/: '
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                try:
                    processed_frame = preprocess_image_for_ocr(frame)
                    pil_image = Image.fromarray(processed_frame)
                    text = pytesseract.image_to_string(pil_image, config=custom_config)
                    
                    if text.strip():
                        cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
                        if len(cleaned_text) > 3:
                            timestamp = frame_count / fps
                            extracted_text.append({
                                "timestamp": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                                "frame": frame_count,
                                "text": cleaned_text
                            })
                            print(f"Frame {frame_count} ({timestamp:.1f}s): {cleaned_text[:100]}...")
                    
                    processed_frames += 1
                    
                except Exception as e:
                    print(f"Error processing frame {frame_count}: {e}")
            
            frame_count += 1
            
            if processed_frames >= 100:
                print("Reached maximum frame processing limit")
                break
        
        cap.release()
        
        # Cleanup temp file
        try:
            os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")
        
        if extracted_text:
            combined_text = "\n".join([f"[{item['timestamp']}] {item['text']}" for item in extracted_text])
            print(f"OCR extraction completed. Extracted text from {len(extracted_text)} frames.")
            return combined_text
        else:
            print("No text extracted from video frames.")
            return ""
            
    except Exception as e:
        print(f"Error in extract_text_from_video_frames: {e}")
        return ""

async def transcribe_video_with_openai(video_bytes: bytes) -> str:
    """
    Transcribe video bytes using OpenAI Whisper API
    Extracts audio from video first to ensure compatibility with Whisper API.
    Returns empty string if video has no audio track.
    """
    import tempfile
    import subprocess
    
    url = "https://api.openai.com/v1/audio/transcriptions"
    temp_video = None
    temp_audio = None
    
    try:
        print(f"Transcribing video with OpenAI Whisper (bytes: {len(video_bytes) if isinstance(video_bytes, bytes) else 'unknown'})")
        
        if not isinstance(video_bytes, bytes):
            return f"Error: Expected bytes, got {type(video_bytes)}"
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            tmp_video.write(video_bytes)
            temp_video = tmp_video.name
        
        # First, check if video has audio stream
        print("Checking for audio stream in video...")
        probe_result = subprocess.run(
            ['ffmpeg', '-i', temp_video, '-hide_banner'],
            capture_output=True,
            text=True
        )
        
        # Check if there's an audio stream in the ffmpeg output
        has_audio = 'Stream #' in probe_result.stderr and 'Audio:' in probe_result.stderr
        
        if not has_audio:
            print("⚠️ Video has no audio track. Skipping transcription.")
            return ""  # Return empty transcription
        
        # Create temporary audio file path
        temp_audio = tempfile.mktemp(suffix='.m4a')
        
        # Extract audio using ffmpeg
        print("Extracting audio from video for Whisper API...")
        result = subprocess.run(
            ['ffmpeg', '-i', temp_video, '-vn', '-acodec', 'aac', '-ar', '16000', '-ac', '1', '-y', temp_audio],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            # Check if error is due to no audio stream
            if 'does not contain any stream' in result.stderr or 'Output file #0 does not contain any stream' in result.stderr:
                print("⚠️ Video has no audio stream. Skipping transcription.")
                return ""  # Return empty transcription
            return f"Audio extraction failed: {result.stderr}"
        
        # Check if audio file was created and has content
        if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
            print("⚠️ No audio could be extracted. Skipping transcription.")
            return ""
        
        # Read extracted audio
        with open(temp_audio, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        print(f"Audio extracted: {len(audio_bytes)} bytes")
        
        # Create in-memory file-like object from audio bytes
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)  # Reset to beginning
        
        async with httpx.AsyncClient(timeout=120) as client:
            # Send as M4A audio format
            files = {"file": ("audio.m4a", audio_buffer, "audio/m4a")}
            data = {"model": "whisper-1"}
            headers = {"Authorization": f"Bearer {openai_api_key}"}
            response = await client.post(url, files=files, data=data, headers=headers)
            
            if response.status_code != 200:
                return f"OpenAI Whisper API error (status {response.status_code}): {response.text}"
            
            resp_json = response.json()
            transcription = resp_json.get("text", "")
            print(f"✅ Transcription completed: {len(transcription)} characters")
            return transcription
        
    except Exception as e:
        print(f"Error during video transcription: {str(e)}")
        return f"Error during video transcription: {str(e)}"
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

async def extract_recipe_with_ai(metadata, comment_info=None, video_analysis="", ocr_text=""):
    """Use AI to extract recipe/guide information from video content and analysis using GPT-4o"""
    try:
        print("Analyzing content to extract recipe/guide with OpenAI GPT-4o...")
        
        # Early warning if no transcription data
        transcription_text = str(video_analysis) if video_analysis else ''
        if not transcription_text or len(transcription_text) < 50:
            print("⚠️  WARNING: Transcription is empty or very short. AI may not be able to extract meaningful content.")
            print(f"Transcription length: {len(transcription_text)} characters")
        
        # Prepare the content for AI analysis
        video_title = metadata.get('title', '')
        video_description = metadata.get('description', '')
        channel_name = metadata.get('channel', '')
        
        # Handle comment info
        if isinstance(comment_info, dict):
            first_comment = comment_info.get('text', '')
        else:
            first_comment = str(comment_info) if comment_info else ''
        
        # Prepare video analysis data
        transcription = str(video_analysis) if video_analysis else ''
        
        # Prepare OCR text data
        ocr_section = f"\n\nOCR TEXT: {ocr_text}" if ocr_text else ""
        
        # Add note about using description when transcription is empty
        transcription_note = ""
        if not transcription or len(transcription) < 50:
            if video_description:
                transcription_note = "\n⚠️ NOTE: Video has no/minimal audio transcription. Extract information primarily from DESCRIPTION.\n"
            else:
                transcription_note = "\n⚠️ NOTE: Video has no/minimal audio transcription and no description. Limited data available.\n"
        
        # Create the enhanced prompt for recipe/guide extraction
        prompt = f"""Analyze this video and extract a step-by-step guide in JSON format.
{transcription_note}
TITLE: {video_title}
DESCRIPTION: {video_description}
CHANNEL: {channel_name}
COMMENT: {first_comment}
TRANSCRIPTION: {transcription}{ocr_section}

Extract the guide with this structure:
{{
  "title": "guide name",
  "description": "brief overview",
  "content_type": "recipe|tutorial|how-to|programming|educational|general",
  "materials": [{{"name": "item", "quantity": "amount", "notes": "notes", "optional": false}}],
  "steps": [{{"step": 1, "instruction": "description", "duration": "time", "details": "extras", "code_snippet": "code", "tips": ["tips"]}}],
  "metadata": {{"duration": "", "difficulty": "easy|medium|hard", "category": "", "tags": [], "estimated_time": "", "skill_level": "beginner|intermediate|advanced", "language": "", "framework": ""}},
  "tools": ["tools/equipment"],
  "tips": ["general tips"],
  "prerequisites": ["requirements"],
  "isInstructional": true
}}

CRITICAL INSTRUCTIONS:
- ONLY extract information that is EXPLICITLY present in the DESCRIPTION, TRANSCRIPTION, COMMENT, or OCR TEXT
- PRIORITIZE: If TRANSCRIPTION is empty/minimal, use DESCRIPTION as primary source (many videos have full details in description)
- DO NOT make up, infer, or assume any steps, ingredients, or instructions that are not clearly stated
- If TRANSCRIPTION is empty AND DESCRIPTION is empty, set "isInstructional": false and return minimal structure
- Set "isInstructional": true when you can extract clear, specific step-by-step instructions from description OR transcription
- For recipes: extract ingredients from description, transcription, comments, or OCR text
- For tutorials: extract steps from description, transcription, comments, or OCR text
- Use OCR text for precise measurements, code, or commands when available
- Return only valid JSON

Extract now:
"""
        
        # Call OpenAI GPT-4o with JSON response format
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
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
        
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(url, headers=headers, content=json.dumps(payload))
            
            if response.status_code != 200:
                raise Exception(f"OpenAI GPT-4o API error (status {response.status_code}): {response.text}")
            
            resp_json = response.json()
            choices = resp_json.get('choices', [])
            if not choices:
                raise Exception("No choices in OpenAI chat response")
            
            content = choices[0]["message"].get("content", "")
            
            # Clean and parse JSON response
            text = content.strip()
            if text.startswith('```json'):
                text = text[7:]
            if text.endswith('```'):
                text = text[:-3]
            elif text.startswith('```'):
                text = text[3:]
                if text.endswith('```'):
                    text = text[:-3]
            
            recipe_data = json.loads(text)
            
            # Ensure all required fields are present
            recipe_data.setdefault('title', video_title)
            recipe_data.setdefault('channelName', channel_name)
            recipe_data.setdefault('savedDate', datetime.now().isoformat())
            recipe_data.setdefault('materials', [])
            recipe_data.setdefault('steps', [])
            recipe_data.setdefault('isInstructional', False)
            recipe_data.setdefault('ocrExtractedInfo', bool(ocr_text))
            
            print(f"✅ Guide extraction completed: {recipe_data.get('title', 'Unknown')}")
            return recipe_data
            
    except json.JSONDecodeError as e:
        print(f"Error parsing AI response as JSON: {e}")
        return {
            "title": metadata.get('title', 'Unknown'),
            "description": "",
            "materials": [],
            "steps": [],
            "metadata": {},
            "tools": [],
            "tips": [],
            "channelName": metadata.get('channel', 'Unknown'),
            "savedDate": datetime.now().isoformat(),
            "isInstructional": False,
            "ocrExtractedInfo": bool(ocr_text),
            "error": "Could not parse guide from AI response"
        }
    except Exception as e:
        print(f"Error during guide extraction: {str(e)}")
        return {
            "title": metadata.get('title', 'Unknown'),
            "description": "",
            "materials": [],
            "steps": [],
            "metadata": {},
            "tools": [],
            "tips": [],
            "channelName": metadata.get('channel', 'Unknown'),
            "savedDate": datetime.now().isoformat(),
            "isInstructional": False,
            "ocrExtractedInfo": bool(ocr_text),
            "error": f"Guide extraction failed: {str(e)}"
        }

