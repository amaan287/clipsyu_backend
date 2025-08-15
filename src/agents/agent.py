import yt_dlp
from youtube_comment_downloader import YoutubeCommentDownloader
import re
from googleapiclient.discovery import build
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

# Load environment variables
load_dotenv()
google_cloud_api_key = os.getenv("GOOGLE_CLOUD_API_KEY")
gemini_api_key = os.getenv("GEMINI_APIKEY")

# Configure Gemini API
genai.configure(api_key=gemini_api_key)

def download_youtube_video(url: str, output_path: str = "./downloads"):
    """Enhanced YouTube video download using cookies.txt file"""
 
    os.makedirs(output_path, exist_ok=True)
    
    def clean_filename(title):
        cleaned = re.sub(r'[<>:"/\\|?*#]', '', title)
        cleaned = re.sub(r'[^\w\s-]', '', cleaned)
        return cleaned.strip()
    
    # Check if cookies.txt exists in root folder
    cookies_file = "cookies.txt"
    if not os.path.exists(cookies_file):
        print(f"Warning: {cookies_file} not found in root folder")
        cookies_file = None
    else:
        print(f"Using cookies file: {cookies_file}")
    
    try:
        print("Attempting download with cookies.txt...")
        
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
            'quiet': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        }
        
        # Add cookies file if it exists
        if cookies_file:
            ydl_opts['cookiefile'] = cookies_file
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            safe_title = clean_filename(info.get('title', 'video'))
            ydl_opts['outtmpl'] = f'{output_path}/{safe_title}.%(ext)s'
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
                ydl_download.download([url])
                
                # Return success
                downloaded_file = f"{output_path}/{safe_title}.mp4"
                metadata = {
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "channel": info.get("channel"),
                    "uploader": info.get("uploader"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                }
                
                print(f"Video downloaded successfully: {downloaded_file}")
                return downloaded_file, metadata
                
    except Exception as e:
        print(f"Download with cookies failed: {e}")
        
        # Fallback: Try without cookies
        try:
            print("Attempting download without cookies...")
            ydl_opts_fallback = {
                'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best',
                'quiet': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
            }
            
            with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
                info = ydl.extract_info(url, download=False)
                safe_title = clean_filename(info.get('title', 'video'))
                ydl_opts_fallback['outtmpl'] = f'{output_path}/{safe_title}.%(ext)s'
                
                with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl_download:
                    ydl_download.download([url])
                    
                    downloaded_file = f"{output_path}/{safe_title}.mp4"
                    metadata = {
                        "title": info.get("title"),
                        "description": info.get("description"),
                        "channel": info.get("channel"),
                        "uploader": info.get("uploader"),
                        "duration": info.get("duration"),
                        "view_count": info.get("view_count"),
                    }
                    
                    print(f"Video downloaded successfully: {downloaded_file}")
                    return downloaded_file, metadata
                    
        except Exception as fallback_error:
            print(f"Fallback download also failed: {fallback_error}")
            # Fallback to API-only analysis
            print("All download methods failed. Falling back to API-only analysis...")
            raise Exception("Could not download video. Consider using API-only analysis method.")

# Alternative: API-only workflow (recommended)
def api_only_workflow(video_url, youtube_api_key, gemini_api_key):
    """Complete workflow without downloading video"""
    
    # Get video metadata using YouTube API
    video_id = get_video_id_from_url(video_url)
    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    
    # Get video details
    video_response = youtube.videos().list(
        part='snippet,statistics,contentDetails',
        id=video_id
    ).execute()
    
    video_data = video_response['items'][0]
    
    # Get comments
    comments_response = youtube.commentThreads().list(
        part='snippet',
        videoId=video_id,
        maxResults=10,
        order='relevance'
    ).execute()
    
    comments = []
    for item in comments_response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        comments.append({
            'text': comment['textDisplay'],
            'author': comment['authorDisplayName']
        })
    
    # Prepare content for AI
    content = f"""
    Title: {video_data['snippet']['title']}
    Channel: {video_data['snippet']['channelTitle']}
    Description: {video_data['snippet']['description'][:2000]}
    
    Top Comments:
    {json.dumps(comments[:5], indent=2)}
    """
    
    # Use Gemini to analyze
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    response = model.generate_content(f"""
    Analyze this YouTube video content and extract recipe information if present:
    
    {content}
    
    Return detailed recipe in JSON format if found, or indicate no recipe present.
    """)
    
    return {
        "video_data": video_data,
        "comments": comments,
        "recipe_analysis": response.text
    }

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

def extract_text_from_video_frames(video_path, trigger_keywords, frame_interval=30):
    """Extract text from video frames using OCR when trigger keywords are detected"""
    print(f"Starting OCR text extraction from video: {video_path}")
    
    # OCR trigger keywords (case-insensitive)
    ocr_keywords = [
        'on screen', 'text', 'written', 'list', 'shown',
        'display', 'appears', 'visible', 'read', 'see', 'shows', 'measurement',
        'quantity', 'amount', 'cup', 'tablespoon', 'teaspoon', 'gram', 'pound',
        'ounce', 'liter', 'milliliter', 'instructions', 'steps', 'directions'
    ]
    
    # Check if any trigger keywords are found
    transcription_lower = trigger_keywords.lower()
    keywords_found = [keyword for keyword in ocr_keywords if keyword in transcription_lower]
    
    if not keywords_found:
        print("No OCR trigger keywords found in transcription. Skipping OCR extraction.")
        return ""
    
    print(f"OCR trigger keywords found: {keywords_found}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return ""
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video FPS: {fps}, Total frames: {total_frames}")
    
    extracted_text = []
    frame_count = 0
    processed_frames = 0
    
    # Configure Tesseract for better accuracy
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/: '
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every nth frame based on frame_interval
        if frame_count % frame_interval == 0:
            try:
                # Preprocess image for better OCR
                processed_frame = preprocess_image_for_ocr(frame)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(processed_frame)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(pil_image, config=custom_config)
                
                if text.strip():  # Only add non-empty text
                    # Clean and filter the text
                    cleaned_text = text.strip().replace('\n', ' ').replace('\r', ' ')
                    
                    # Filter out very short or nonsensical text
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
        
        # Limit processing to avoid excessive computation
        if processed_frames >= 100:  # Process max 100 frames
            print("Reached maximum frame processing limit")
            break
    
    cap.release()
    
    # Combine all extracted text
    if extracted_text:
        combined_text = "\n".join([f"[{item['timestamp']}] {item['text']}" for item in extracted_text])
        print(f"OCR extraction completed. Extracted text from {len(extracted_text)} frames.")
        return combined_text
    else:
        print("No text extracted from video frames.")
        return ""

def transcribe_video_with_gemini(video_file_path):
    """
    Transcribe video file using Google Gemini API
    """
    try:
        print(f"Transcribing video: {video_file_path}")
        
        # Check if file exists
        if not os.path.exists(video_file_path):
            return f"Error: Video file not found at {video_file_path}"
        
        # Upload the video file
        print("Uploading video file to Gemini...")
        video_file = genai.upload_file(video_file_path)
        
        # Wait for processing
        print("Processing video...")
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # Generate transcription and analysis
        response = model.generate_content([
            video_file,
            "Please analyze this video and provide: 1) Complete transcription of all speech, 2) Description of cooking activities, objects, and ingredients visible, 3) Any text that appears on screen, 4) Cooking techniques and equipment used. Format as a detailed analysis."
        ])
        
        return response.text
        
    except Exception as e:
        return f"Error during video transcription: {str(e)}"

def extract_recipe_with_ai(metadata, comment_info=None, video_analysis="", ocr_text=""):
    """Use AI to extract recipe information from video content and analysis"""
    try:
        print("Analyzing content to extract recipe...")
        
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
        ocr_section = f"\n\nOCR EXTRACTED TEXT FROM VIDEO FRAMES:\n{ocr_text}" if ocr_text else ""
        
        # Create the enhanced prompt for recipe extraction
        prompt = f"""
        You are a recipe extraction AI. Analyze the following comprehensive content from a YouTube video and extract a recipe if one exists.
        
        VIDEO TITLE: {video_title}
        
        VIDEO DESCRIPTION: {video_description}
        
        CHANNEL NAME: {channel_name}
        
        FIRST COMMENT: {first_comment}
        
        VIDEO TRANSCRIPTION AND ANALYSIS: {transcription}
        {ocr_section}
        
        IMPORTANT: Pay special attention to the OCR extracted text as it may contain precise ingredient measurements, quantities, and recipe details that appear on screen. Use this information to create accurate ingredient lists and measurements.
        
        Based on this comprehensive analysis, extract the recipe information and return it in the following JSON format:
        {{
            "title": "recipe name",
            "description": "brief description of the recipe",
            "ingredients": [
                {{"name": "ingredient name", "quantity": "amount with unit", "notes": "any special notes"}},
                {{"name": "ingredient name", "quantity": "amount with unit", "notes": "any special notes"}}
            ],
            "steps": [
                {{"step": 1, "instruction": "detailed step description", "time": "estimated time", "temperature": "if applicable"}},
                {{"step": 2, "instruction": "detailed step description", "time": "estimated time", "temperature": "if applicable"}}
            ],
            "metadata": {{
                "servings": "number of servings",
                "prep_time": "preparation time",
                "cook_time": "cooking time",
                "total_time": "total time",
                "difficulty": "easy/medium/hard",
                "cuisine": "type of cuisine",
                "dietary_tags": ["vegetarian", "gluten-free", etc.]
            }},
            "equipment": ["list of required equipment"],
            "tips": ["cooking tips and notes"],
            "ocrExtractedInfo": "{bool(ocr_text)}",
            "channelName": "channel name",
            "savedDate": "{datetime.now().isoformat()}",
            "isRecipe": true
        }}
        
        Rules:
        1. If this is clearly a recipe/cooking video, extract the complete recipe with all details
        2. If this is not a recipe video, return the same JSON structure but with "isRecipe": false and empty arrays
        3. Be precise with ingredient quantities and measurements, especially from OCR text
        4. Break down steps clearly and sequentially based on the video content
        5. Include cooking times and temperatures when mentioned
        6. Add relevant dietary tags based on the content
        7. Prioritize OCR extracted text for ingredient measurements and quantities
        8. Only return the JSON, no additional text
        
        Extract the recipe now:
        """
        
        # Create the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Generate recipe extraction
        response = model.generate_content(prompt)
        
        # Try to parse the JSON response
        try:
            # Clean the response text
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            recipe_data = json.loads(response_text)
            
            # Ensure all required fields are present
            recipe_data.setdefault('title', video_title)
            recipe_data.setdefault('channelName', channel_name)
            recipe_data.setdefault('savedDate', datetime.now().isoformat())
            recipe_data.setdefault('ingredients', [])
            recipe_data.setdefault('steps', [])
            recipe_data.setdefault('isRecipe', False)
            recipe_data.setdefault('ocrExtractedInfo', bool(ocr_text))
            
            return recipe_data
            
        except json.JSONDecodeError as e:
            print(f"Error parsing AI response as JSON: {e}")
            print(f"Raw response: {response.text}")
            
            # Return a fallback structure
            return {
                "title": video_title,
                "description": "",
                "ingredients": [],
                "steps": [],
                "metadata": {},
                "equipment": [],
                "tips": [],
                "channelName": channel_name,
                "savedDate": datetime.now().isoformat(),
                "isRecipe": False,
                "ocrExtractedInfo": bool(ocr_text),
                "error": "Could not parse recipe from AI response"
            }
        
    except Exception as e:
        print(f"Error during recipe extraction: {str(e)}")
        return {
            "title": metadata.get('title', 'Unknown'),
            "description": "",
            "ingredients": [],
            "steps": [],
            "metadata": {},
            "equipment": [],
            "tips": [],
            "channelName": metadata.get('channel', 'Unknown'),
            "savedDate": datetime.now().isoformat(),
            "isRecipe": False,
            "ocrExtractedInfo": bool(ocr_text),
            "error": f"Recipe extraction failed: {str(e)}"
        }

def print_recipe(recipe_data):
    """Print the recipe in a formatted way"""
    print("\n" + "="*50)
    print("EXTRACTED RECIPE")
    print("="*50)
    
    if recipe_data.get('isRecipe', False):
        print(f"Title: {recipe_data.get('title', 'N/A')}")
        print(f"Channel: {recipe_data.get('channelName', 'N/A')}")
        print(f"Description: {recipe_data.get('description', 'N/A')}")
        print(f"OCR Text Used: {recipe_data.get('ocrExtractedInfo', False)}")
        
        metadata = recipe_data.get('metadata', {})
        if metadata:
            print("\nMetadata:")
            for key, value in metadata.items():
                print(f"  {key}: {value}")
        
        ingredients = recipe_data.get('ingredients', [])
        if ingredients:
            print("\nIngredients:")
            for i, ingredient in enumerate(ingredients, 1):
                print(f"  {i}. {ingredient.get('quantity', '')} {ingredient.get('name', '')}")
                if ingredient.get('notes'):
                    print(f"     Notes: {ingredient['notes']}")
        
        steps = recipe_data.get('steps', [])
        if steps:
            print("\nSteps:")
            for step in steps:
                print(f"  {step.get('step', '')}. {step.get('instruction', '')}")
                if step.get('time'):
                    print(f"     Time: {step['time']}")
                if step.get('temperature'):
                    print(f"     Temperature: {step['temperature']}")
        
        equipment = recipe_data.get('equipment', [])
        if equipment:
            print(f"\nEquipment: {', '.join(equipment)}")
        
        tips = recipe_data.get('tips', [])
        if tips:
            print("\nTips:")
            for tip in tips:
                print(f"  • {tip}")
    else:
        print("This video does not appear to contain a recipe.")
        print(f"Title: {recipe_data.get('title', 'N/A')}")
        print(f"Channel: {recipe_data.get('channelName', 'N/A')}")
        print(f"OCR Text Used: {recipe_data.get('ocrExtractedInfo', False)}")
    
    print("\nFull JSON:")
    print(json.dumps(recipe_data, indent=2, ensure_ascii=False))
    print("="*50)

def main():
    """Main function to orchestrate the workflow"""
    required_env_vars = [
        ("GOOGLE_CLOUD_API_KEY", google_cloud_api_key),
        ("GEMINI_APIKEY", gemini_api_key)
    ]
    
    for var_name, var_value in required_env_vars:
        if not var_value:
            print(f"Error: {var_name} not found in environment variables")
            return
    
    # Get URL from user
    url = input("Enter YouTube video URL: ").strip()
    
    if not url:
        print("Error: No URL provided")
        return
    
    try:
        # Download video
        print("=== DOWNLOADING VIDEO ===")
        video_file, metadata = download_youtube_video(url)
        
        # Get first comment
        print("\n=== FETCHING FIRST COMMENT ===")
        comment_info = None  # Always define it first
        try:
            comment_info = get_first_comment_from_video(google_cloud_api_key, url)
        except Exception as e:
            print(f"Failed to fetch comment: {e}")
            # comment_info remains None
        
        if isinstance(comment_info, dict):
            print(f"Author: {comment_info['author']}")
            print(f"Comment: {comment_info['text']}")
            print(f"Likes: {comment_info['likes']}")
            print(f"Published: {comment_info['published']}")
        else:
            print(comment_info)
        
        # Analyze video with Gemini
        print("\n=== ANALYZING VIDEO WITH GEMINI ===")
        video_analysis = transcribe_video_with_gemini(video_file)
        print("Video analysis completed!")
        
        # Extract text from video frames using OCR (if triggered)
        print("\n=== EXTRACTING TEXT FROM VIDEO FRAMES ===")
        ocr_text = extract_text_from_video_frames(video_file, video_analysis)
        
        if ocr_text:
            print(f"OCR extracted text ({len(ocr_text)} characters):")
            print(ocr_text[:500] + "..." if len(ocr_text) > 500 else ocr_text)
        
        # Extract recipe using AI (including OCR text)
        print("\n=== EXTRACTING RECIPE WITH AI ===")
        recipe_data = extract_recipe_with_ai(metadata, comment_info, video_analysis, ocr_text)
        print_recipe(recipe_data)
        
        print("\n=== PROCESS COMPLETED ===")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()