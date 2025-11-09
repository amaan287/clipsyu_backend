"""Application configuration and constants."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables (ignore if .env doesn't exist)
try:
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(dotenv_path=env_path)
except Exception:
    pass  # .env file not required


class Config:
    """Application configuration."""
    
    # API Keys
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "").strip()
    
    # Timeouts
    HTTP_CLIENT_TIMEOUT: int = 120
    
    # OpenAI Configuration
    OPENAI_API_BASE_URL: str = "https://api.openai.com/v1"
    WHISPER_MODEL: str = "whisper-1"
    GPT_MODEL: str = "gpt-4o"
    GPT_TEMPERATURE: float = 0.3
    GPT_MAX_TOKENS: int = 1200
    
    # OCR Configuration
    OCR_FRAME_INTERVAL: int = 30
    OCR_MAX_PROCESSED_FRAMES: int = 100
    
    # Server Configuration
    SERVER_HOST: str = "0.0.0.0"
    SERVER_PORT: int = 8080
    
    # Cookies
    COOKIES_FILE: str = "cookies.txt"


class Constants:
    """Application constants."""
    
    # Platform domains
    INSTAGRAM_DOMAINS = ["instagram.com", "www.instagram.com"]
    YOUTUBE_DOMAINS = ["youtube.com", "www.youtube.com", "youtu.be"]
    
    # OCR Keywords
    OCR_TRIGGER_KEYWORDS = [
        'on screen', 'text', 'written', 'list', 'shown',
        'display', 'appears', 'visible', 'read', 'see', 'shows', 'measurement',
        'quantity', 'amount', 'cup', 'tablespoon', 'teaspoon', 'gram', 'pound',
        'ounce', 'liter', 'milliliter', 'instructions', 'steps', 'directions'
    ]
    
    # Tesseract Config
    TESSERACT_CONFIG = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,()-/: '
    
    # Content types
    CONTENT_TYPES = ["recipe", "tutorial", "how-to", "programming", "educational", "general"]
    DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
    SKILL_LEVELS = ["beginner", "intermediate", "advanced"]

