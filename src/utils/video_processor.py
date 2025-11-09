"""Video processing utilities."""
import tempfile
import subprocess
import os
from typing import Tuple, Optional


class VideoProcessor:
    """Utilities for video processing operations."""
    
    @staticmethod
    def extract_audio_from_video(video_bytes: bytes) -> Tuple[Optional[bytes], Optional[str]]:
        """
        Extract audio from video bytes using ffmpeg.
        
        Args:
            video_bytes: Video content as bytes
            
        Returns:
            Tuple of (audio_bytes, error_message)
        """
        temp_video = None
        temp_audio = None
        
        try:
            # Create temporary video file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(video_bytes)
                temp_video = tmp_video.name
            
            # Check if video has audio stream
            if not VideoProcessor._has_audio_stream(temp_video):
                print("⚠️ Video has no audio track detected.")
                return None, None  # No audio, but not an error
            
            # Create temporary audio file path
            temp_audio = tempfile.mktemp(suffix='.m4a')
            
            # Extract audio using ffmpeg
            result = subprocess.run(
                ['ffmpeg', '-i', temp_video, '-vn', '-acodec', 'aac', 
                 '-ar', '16000', '-ac', '1', '-y', temp_audio],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if 'does not contain any stream' in result.stderr:
                    return None, None  # No audio, but not an error
                return None, f"Audio extraction failed: {result.stderr}"
            
            # Check if audio file was created and has content
            if not os.path.exists(temp_audio) or os.path.getsize(temp_audio) == 0:
                return None, None  # No audio, but not an error
            
            # Read extracted audio
            with open(temp_audio, 'rb') as audio_file:
                audio_bytes = audio_file.read()
            
            return audio_bytes, None
            
        except Exception as e:
            return None, f"Error extracting audio: {str(e)}"
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
    
    @staticmethod
    def _has_audio_stream(video_path: str) -> bool:
        """
        Check if video has audio stream.
        
        Args:
            video_path: Path to video file
            
        Returns:
            True if video has audio stream, False otherwise
        """
        probe_result = subprocess.run(
            ['ffmpeg', '-i', video_path, '-hide_banner'],
            capture_output=True,
            text=True
        )
        
        return 'Stream #' in probe_result.stderr and 'Audio:' in probe_result.stderr

