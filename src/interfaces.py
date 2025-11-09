"""Interfaces and protocols for dependency inversion principle."""
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
from io import BytesIO


class IVideoDownloader(ABC):
    """Interface for video downloading services."""
    
    @abstractmethod
    async def download_video(self, url: str) -> Tuple[Optional[bytes], Optional[Dict]]:
        """
        Download video from URL.
        
        Args:
            url: Video URL
            
        Returns:
            Tuple of (video_bytes, metadata_dict)
        """
        pass


class ITranscriptionService(ABC):
    """Interface for transcription services."""
    
    @abstractmethod
    async def transcribe(self, video_bytes: bytes) -> Tuple[str, Optional[str]]:
        """
        Transcribe video audio.
        
        Args:
            video_bytes: Video content as bytes
            
        Returns:
            Tuple of (transcription_text, error_message)
        """
        pass


class IOCRService(ABC):
    """Interface for OCR services."""
    
    @abstractmethod
    def extract_text_from_frames(
        self, 
        video_bytes: bytes, 
        trigger_keywords: str,
        frame_interval: int = 30
    ) -> str:
        """
        Extract text from video frames using OCR.
        
        Args:
            video_bytes: Video content as bytes
            trigger_keywords: Keywords to trigger OCR
            frame_interval: Interval between frames to process
            
        Returns:
            Extracted text from video frames
        """
        pass


class IAIExtractionService(ABC):
    """Interface for AI-based content extraction."""
    
    @abstractmethod
    async def extract_guide(
        self,
        metadata: Dict,
        comment_info: Optional[str],
        transcription: str,
        ocr_text: str
    ) -> Tuple[Dict, Optional[str]]:
        """
        Extract guide/recipe information using AI.
        
        Args:
            metadata: Video metadata
            comment_info: Comment information
            transcription: Video transcription
            ocr_text: OCR extracted text
            
        Returns:
            Tuple of (guide_dict, error_message)
        """
        pass


class IVideoMetadataExtractor(ABC):
    """Interface for video metadata extraction."""
    
    @abstractmethod
    def extract_metadata(self, url: str, cookies_file: Optional[str] = None) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Extract video metadata and direct URL.
        
        Args:
            url: Video URL
            cookies_file: Optional cookies file path
            
        Returns:
            Tuple of (direct_video_url, metadata_dict)
        """
        pass


class ICommentExtractor(ABC):
    """Interface for comment extraction services."""
    
    @abstractmethod
    def get_comments(self, url: str) -> Dict:
        """
        Extract comments from video URL.
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary containing comment information
        """
        pass

