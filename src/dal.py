# models.py - Pydantic Models and Schema Definitions

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from bson import ObjectId
import pymongo
from config.config import MONGODB_URL
import re

# Load environment variables


# ===== ENUMS =====

class DifficultyLevel(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

# ===== PYDANTIC MODELS FOR MONGODB SCHEMA =====

class Ingredient(BaseModel):
    name: str
    quantity: str
    notes: Optional[str] = None

class RecipeStep(BaseModel):
    step: int
    instruction: str
    time: Optional[str] = None
    temperature: Optional[str] = None

class RecipeMetadata(BaseModel):
    servings: Optional[str] = None
    prep_time: Optional[str] = None
    cook_time: Optional[str] = None
    total_time: Optional[str] = None
    difficulty: Optional[DifficultyLevel] = None
    cuisine: Optional[str] = None
    dietary_tags: List[str] = []

class Recipe(BaseModel):
    title: str
    description: Optional[str] = None
    ingredients: List[Ingredient] = []
    steps: List[RecipeStep] = []
    metadata: RecipeMetadata = RecipeMetadata()
    equipment: List[str] = []
    tips: List[str] = []
    ocrExtractedInfo: bool = False
    channelName: str
    savedDate: datetime
    isRecipe: bool = False
    
    # Additional fields for tracking
    youtube_url: Optional[str] = None
    video_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        # Allow ObjectId to be used
        arbitrary_types_allowed = True
        # JSON encoders for datetime and ObjectId
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }

class User(BaseModel):
    id: Optional[str] = Field(None, alias="_id")  # Store ObjectId as str
    google_id:str
    name: str
    email: str
    picture:str
    refresh_token:Optional[str] = None

# MongoDB Document Model (what gets stored in DB)
class RecipeDocument(Recipe):
    id: Optional[ObjectId] = Field(default_factory=ObjectId, alias="_id")
    
    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            ObjectId: lambda v: str(v)
        }

# ===== REQUEST/RESPONSE MODELS =====

class RecipeExtractionRequest(BaseModel):
    url: str
    """
    @validator('url')
    def validate_youtube_url(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('URL must be a non-empty string')
        # Regex for YouTube and Instagram URLs
        youtube_regex = re.compile(
            r'(https?://)?(www\.)?(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)[\w\-]+', re.IGNORECASE
        )
        instagram_regex = re.compile(
            r'(https?://)?(www\.)?(instagram\.com|instagr\.am)/(p|reel|tv)/[\w\-]+', re.IGNORECASE
        )
        if not (youtube_regex.search(v) or instagram_regex.search(v)):
            raise ValueError('Must be a valid YouTube or Instagram URL')
        return v
"""

class RecipeExtractionResponse(BaseModel):
    success: bool
    message: str
    recipe_id: Optional[str] = None
    recipe_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class GoogleAuthRequest(BaseModel):
    id_token: str

class UserResponse(BaseModel):
    id: str
    google_id:str
    email: str
    name: str
    picture: str
    jwt_token: Optional[str]=None
    refresh_token:Optional[str]=None

# ===== DATABASE CONNECTION =====

# Simple MongoDB connection
mongodb_url = MONGODB_URL
if not mongodb_url:
    raise ValueError("MONGODB_URL environment variable is required")

client = pymongo.MongoClient(
    mongodb_url,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=10000,
    socketTimeoutMS=10000,
    maxPoolSize=10,
    retryWrites=True,
    w="majority"
)

# Test connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

# Get database
db = client.recipe_extractor

recipe_collection = db.recipes
users_collection = db.users


