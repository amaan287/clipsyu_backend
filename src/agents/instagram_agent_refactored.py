"""Instagram platform agent."""
from typing import Dict, Tuple, Optional
from datetime import datetime
import io

from src.interfaces import (
    IVideoDownloader,
    IVideoMetadataExtractor,
    ITranscriptionService,
    IAIExtractionService,
    ICommentExtractor
)
from src.services import (
    HTTPVideoDownloader,
    YTDLPMetadataExtractor,
    WhisperTranscriptionService,
    OpenAIExtractionService,
    InstagramCommentExtractor
)
from src.config import Config


class InstagramAgent:
    """
    Agent for processing Instagram videos and extracting guides.
    
    This class follows the Dependency Inversion Principle by depending on
    interfaces rather than concrete implementations.
    """
    
    def __init__(
        self,
        metadata_extractor: IVideoMetadataExtractor = None,
        video_downloader: IVideoDownloader = None,
        transcription_service: ITranscriptionService = None,
        ai_extraction_service: IAIExtractionService = None,
        comment_extractor: ICommentExtractor = None
    ):
        """
        Initialize Instagram agent with dependency injection.
        
        Args:
            metadata_extractor: Metadata extraction service
            video_downloader: Video download service
            transcription_service: Transcription service
            ai_extraction_service: AI extraction service
            comment_extractor: Comment extraction service
        """
        # Use provided dependencies or create defaults
        self.metadata_extractor = metadata_extractor or YTDLPMetadataExtractor(Config.COOKIES_FILE)
        self.video_downloader = video_downloader or HTTPVideoDownloader()
        self.transcription_service = transcription_service or WhisperTranscriptionService()
        self.ai_extraction_service = ai_extraction_service or OpenAIExtractionService()
        self.comment_extractor = comment_extractor or InstagramCommentExtractor()
    
    async def get_guide_from_url(self, url: str) -> Dict:
        """
        Process Instagram video URL and return a guide dictionary.
        
        Args:
            url: Instagram video URL
            
        Returns:
            Dictionary containing guide information
            
        Raises:
            Exception: If video processing fails
        """
        print("=" * 60)
        print(f"PROCESSING INSTAGRAM VIDEO: {url}")
        print("=" * 60)
        
        # Step 1: Download video and extract metadata
        video_bytes, metadata = await self._download_video(url)
        
        # Step 2: Transcribe video
        transcription = await self._transcribe_video(video_bytes)
        
        # Step 3: Get comment information (placeholder for Instagram)
        comment_info = self._get_comment_info(url)
        
        # Step 4: Extract guide using AI
        guide_data = await self._extract_guide(metadata, comment_info, transcription)
        
        # Step 5: Build response
        guide_response = self._build_guide_response(url, metadata, guide_data, transcription)
        
        print("=" * 60)
        print("INSTAGRAM VIDEO PROCESSING COMPLETED")
        print("=" * 60)
        
        return guide_response
    
    async def _download_video(self, url: str) -> Tuple[bytes, Dict]:
        """Download Instagram video and extract metadata."""
        print("\n[1/4] Downloading Instagram video...")
        
        # Extract metadata and direct URL using yt-dlp
        video_url, metadata = self.metadata_extractor.extract_metadata(url, Config.COOKIES_FILE)
        
        if not video_url:
            raise Exception("Could not extract video URL from Instagram post.")
        
        # Download video bytes
        video_bytes, _ = await self.video_downloader.download_video(video_url)
        
        if not video_bytes:
            raise Exception("Could not download Instagram video bytes.")
        
        print(f"✅ Video downloaded: {len(video_bytes)} bytes")
        
        # Enhance metadata for Instagram
        metadata['source_url'] = url
        metadata['direct_video_url'] = video_url
        
        return video_bytes, metadata
    
    async def _transcribe_video(self, video_bytes: bytes) -> str:
        """Transcribe video audio."""
        print("\n[2/4] Transcribing video...")
        transcription, error = await self.transcription_service.transcribe(video_bytes)
        
        if error:
            print(f"⚠️  Transcription error: {error}")
            # Don't raise, continue with empty transcription
        
        print(f"✅ Transcription result: {len(transcription)} characters")
        
        if not transcription or len(transcription) < 50:
            print("⚠️  WARNING: Transcription is empty or very short.")
            print("    AI will primarily use description from metadata.")
        
        return transcription
    
    def _get_comment_info(self, url: str) -> str:
        """Get comment information (placeholder for Instagram)."""
        print("\n[3/4] Fetching comment information...")
        comment_data = self.comment_extractor.get_comments(url)
        
        if isinstance(comment_data, dict):
            note = comment_data.get('note', '')
            if note:
                print(f"ℹ️  {note}")
            return note
        
        return str(comment_data) if comment_data else ''
    
    async def _extract_guide(
        self,
        metadata: Dict,
        comment_info: str,
        transcription: str
    ) -> Dict:
        """Extract guide using AI."""
        print("\n[4/4] Extracting guide with AI...")
        
        # Instagram videos often have full recipe/guide in description
        ocr_text = ""  # OCR could be added as an enhancement
        
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
        transcription: str
    ) -> Dict:
        """Build the final guide response."""
        return {
            "url": metadata.get("source_url", url),
            "title": guide_data.get("title", "Instagram Guide"),
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
            "ocrExtractedInfo": "false",
            "channelName": metadata.get("uploader", "Instagram User"),
            "savedDate": datetime.now(),
            "isInstructional": guide_data.get("isInstructional", False),
            "transcription": transcription,
            "video_analysis": transcription,
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

