"""Transcription service implementation."""
import httpx
import io
from typing import Tuple, Optional

from src.interfaces import ITranscriptionService
from src.config import Config
from src.utils.video_processor import VideoProcessor


class WhisperTranscriptionService(ITranscriptionService):
    """Service for transcribing video using OpenAI Whisper API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize transcription service.
        
        Args:
            api_key: OpenAI API key (optional, uses config if not provided)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = Config.OPENAI_API_BASE_URL
        self.model = Config.WHISPER_MODEL
        self.timeout = Config.HTTP_CLIENT_TIMEOUT
    
    async def transcribe(self, video_bytes: bytes) -> Tuple[str, Optional[str]]:
        """
        Transcribe video audio using OpenAI Whisper API.
        
        Args:
            video_bytes: Video content as bytes
            
        Returns:
            Tuple of (transcription_text, error_message)
        """
        print(f"📹 Starting transcription process for {len(video_bytes)} bytes of video")
        
        if not isinstance(video_bytes, bytes):
            return "", f"Expected bytes, got {type(video_bytes)}"
        
        # Extract audio from video
        audio_bytes, error = VideoProcessor.extract_audio_from_video(video_bytes)
        
        if error:
            return "", error
        
        if not audio_bytes:
            print("⚠️ Video has no audio track. Skipping transcription.")
            return "", None  # Empty transcription, no error
        
        print(f"✅ Audio extracted: {len(audio_bytes)} bytes")
        
        # Transcribe audio
        try:
            transcription = await self._transcribe_audio(audio_bytes)
            print(f"✅ Transcription successful: {len(transcription)} characters")
            if transcription:
                print(f"Preview: {transcription[:100]}...")
            return transcription, None
        except Exception as e:
            print(f"❌ Transcription error: {str(e)}")
            return "", str(e)
    
    async def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Transcribe audio using Whisper API.
        
        Args:
            audio_bytes: Audio content as bytes
            
        Returns:
            Transcription text
        """
        url = f"{self.base_url}/audio/transcriptions"
        
        # Create in-memory file-like object
        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            files = {"file": ("audio.m4a", audio_buffer, "audio/m4a")}
            data = {"model": self.model}
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            response = await client.post(url, files=files, data=data, headers=headers)
            
            if response.status_code != 200:
                raise Exception(
                    f"OpenAI Whisper API error (status {response.status_code}): {response.text}"
                )
            
            resp_json = response.json()
            return resp_json.get("text", "")

