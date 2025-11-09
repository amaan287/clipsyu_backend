"""YouTube platform agent."""
from typing import Dict
from datetime import datetime

from src.interfaces import (
    IVideoDownloader,
    ITranscriptionService,
    IOCRService,
    IAIExtractionService,
    ICommentExtractor
)
from src.services import (
    YouTubeVideoDownloader,
    WhisperTranscriptionService,
    OCRService,
    OpenAIExtractionService,
    YouTubeCommentExtractor
)
from src.config import Config


class YouTubeAgent:
    """
    Agent for processing YouTube videos and extracting guides.
    
    This class follows the Dependency Inversion Principle by depending on
    interfaces rather than concrete implementations.
    """
    
    def __init__(
        self,
        video_downloader: IVideoDownloader = None,
        transcription_service: ITranscriptionService = None,
        ocr_service: IOCRService = None,
        ai_extraction_service: IAIExtractionService = None,
        comment_extractor: ICommentExtractor = None
    ):
        """
        Initialize YouTube agent with dependency injection.
        
        Args:
            video_downloader: Video download service
            transcription_service: Transcription service
            ocr_service: OCR service
            ai_extraction_service: AI extraction service
            comment_extractor: Comment extraction service
        """
        # Use provided dependencies or create defaults
        self.video_downloader = video_downloader or YouTubeVideoDownloader(Config.COOKIES_FILE)
        self.transcription_service = transcription_service or WhisperTranscriptionService()
        self.ocr_service = ocr_service or OCRService()
        self.ai_extraction_service = ai_extraction_service or OpenAIExtractionService()
        self.comment_extractor = comment_extractor or YouTubeCommentExtractor()
    
    async def get_guide_from_url(self, url: str) -> Dict:
        """
        Process YouTube video URL and return a guide dictionary.
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary containing guide information
            
        Raises:
            Exception: If video processing fails
        """
        print("=" * 60)
        print(f"PROCESSING YOUTUBE VIDEO: {url}")
        print("=" * 60)
        
        # Step 1: Download video and extract metadata
        video_bytes, metadata = await self._download_video(url)
        
        # Step 2: Transcribe video
        transcription = await self._transcribe_video(video_bytes)
        
        # Step 3: Extract text from frames (OCR)
        ocr_text = self._extract_ocr_text(video_bytes, transcription)
        
        # Step 4: Get comments
        comment_info = self._get_comments(url)
        
        # Step 5: Extract guide using AI
        guide_data = await self._extract_guide(metadata, comment_info, transcription, ocr_text)
        
        # Step 6: Build response
        guide_response = self._build_guide_response(
            url, metadata, guide_data, transcription, ocr_text
        )
        
        print("=" * 60)
        print("YOUTUBE VIDEO PROCESSING COMPLETED")
        print("=" * 60)
        
        return guide_response
    
    async def _download_video(self, url: str) -> tuple:
        """Download video and extract metadata."""
        print("\n[1/5] Downloading video...")
        video_bytes, metadata = await self.video_downloader.download_video(url)
        
        if not video_bytes or not metadata:
            raise Exception("Could not download YouTube video.")
        
        print(f"✅ Video downloaded: {len(video_bytes)} bytes")
        return video_bytes, metadata
    
    async def _transcribe_video(self, video_bytes: bytes) -> str:
        """Transcribe video audio."""
        print("\n[2/5] Transcribing video...")
        transcription, error = await self.transcription_service.transcribe(video_bytes)
        
        if error:
            print(f"⚠️  Transcription error: {error}")
            # Don't raise, continue with empty transcription
        
        print(f"✅ Transcription result: {len(transcription)} characters")
        return transcription
    
    def _extract_ocr_text(self, video_bytes: bytes, transcription: str) -> str:
        """Extract text from video frames using OCR."""
        print("\n[3/5] Extracting text from frames (OCR)...")
        ocr_text = self.ocr_service.extract_text_from_frames(video_bytes, transcription)
        
        if ocr_text:
            print(f"✅ OCR extraction completed: {len(ocr_text)} characters")
        else:
            print("✅ No OCR text extracted (skipped or no trigger keywords)")
        
        return ocr_text
    
    def _get_comments(self, url: str) -> str:
        """Get first comment from video."""
        print("\n[4/5] Fetching comments...")
        comment_info = self.comment_extractor.get_comments(url)
        
        if isinstance(comment_info, dict):
            comment_text = comment_info.get('text', '')
            if comment_text:
                print(f"✅ Comment found: {comment_text[:100]}...")
            return comment_text
        
        return str(comment_info) if comment_info else ''
    
    async def _extract_guide(
        self, 
        metadata: Dict, 
        comment_info: str, 
        transcription: str, 
        ocr_text: str
    ) -> Dict:
        """Extract guide using AI."""
        print("\n[5/5] Extracting guide with AI...")
        guide_data, error = await self.ai_extraction_service.extract_guide(
            metadata, comment_info, transcription, ocr_text
        )
        
        if error:
            raise Exception(f"Guide extraction failed: {error}")
        
        return guide_data
    
    @staticmethod
    def _build_guide_response(
        url: str,
        metadata: Dict,
        guide_data: Dict,
        transcription: str,
        ocr_text: str
    ) -> Dict:
        """Build the final guide response."""
        return {
            "url": metadata.get("webpage_url", url),
            "title": guide_data.get("title", "YouTube Guide"),
            "description": guide_data.get("description", ""),
            "content_type": guide_data.get("content_type", "general"),
            "materials": guide_data.get("materials", []),
            "steps": guide_data.get("steps", []),
            "metadata": guide_data.get("metadata", {
                "difficulty": "medium",
                "tags": [],
            }),
            "tools": guide_data.get("tools", []),
            "tips": guide_data.get("tips", []),
            "prerequisites": guide_data.get("prerequisites", []),
            "ocrExtractedInfo": str(bool(ocr_text)),
            "channelName": metadata.get("channel", metadata.get("uploader", "YouTube User")),
            "savedDate": datetime.now(),
            "isInstructional": guide_data.get("isInstructional", False),
            "transcription": transcription,
            "video_analysis": transcription,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

