from pydantic import BaseModel, HttpUrl
from typing import List, Optional
from datetime import datetime

class RecipeRequest(BaseModel):
    url: str


class Material(BaseModel):
    name: str
    quantity: Optional[str] = None
    notes: Optional[str] = None
    optional: Optional[bool] = None

class GuideStep(BaseModel):
    step: int
    instruction: str
    duration: Optional[str] = None
    details: Optional[str] = None
    code_snippet: Optional[str] = None
    tips: Optional[List[str]] = None

class GuideMetadata(BaseModel):
    duration: Optional[str] = None
    difficulty: str
    category: Optional[str] = None
    tags: List[str]
    estimated_time: Optional[str] = None
    skill_level: Optional[str] = None
    language: Optional[str] = None
    framework: Optional[str] = None

class Guide(BaseModel):

    url: str
    title: str
    description: str
    content_type: str
    materials: List[Material]
    steps: List[GuideStep]
    metadata: GuideMetadata
    tools: List[str]
    tips: List[str]
    prerequisites: List[str]
    ocrExtractedInfo: Optional[str] = None
    channelName: Optional[str] = None
    savedDate: Optional[datetime] = None
    isInstructional: Optional[bool] = None
    transcription: Optional[str] = None
    video_analysis: Optional[str] = None


class GuideData(BaseModel):
    title: str
    description: str
    content_type: str
    materials: List[Material]
    steps: List[GuideStep]
    metadata: GuideMetadata
    tools: List[str]
    tips: List[str]
    prerequisites: List[str]
    isInstructional: bool
