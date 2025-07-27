import os
from dotenv import load_dotenv
load_dotenv(override=True)

JWT_SECRET = os.getenv("JWT_SECRET")  
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")
AUTH_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

GOOGLE_CLOUD_API_KEY = os.getenv("GOOGLE_CLOUD_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_APIKEY")

GOOGLE_CLOUD_PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
MONGODB_URL = os.getenv("MONGODB_URL")


