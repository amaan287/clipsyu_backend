"""OCR service implementation."""
import cv2
import pytesseract
import numpy as np
import tempfile
import os
from typing import List, Dict

from src.interfaces import IOCRService
from src.config import Config, Constants
from src.utils.image_processor import ImageProcessor


class OCRService(IOCRService):
    """Service for extracting text from video frames using OCR."""
    
    def __init__(self):
        """Initialize OCR service."""
        self.frame_interval = Config.OCR_FRAME_INTERVAL
        self.max_frames = Config.OCR_MAX_PROCESSED_FRAMES
        self.tesseract_config = Constants.TESSERACT_CONFIG
        self.trigger_keywords = Constants.OCR_TRIGGER_KEYWORDS
    
    def extract_text_from_frames(
        self, 
        video_bytes: bytes, 
        trigger_keywords: str,
        frame_interval: int = None
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
        print(f"Starting OCR text extraction from video bytes: {len(video_bytes)}")
        
        if not self._should_perform_ocr(trigger_keywords):
            return ""
        
        if not isinstance(video_bytes, bytes):
            return f"Error: Expected bytes, got {type(video_bytes)}"
        
        interval = frame_interval or self.frame_interval
        
        try:
            extracted_texts = self._process_video_frames(video_bytes, interval)
            return self._combine_extracted_texts(extracted_texts)
        except Exception as e:
            print(f"Error in extract_text_from_frames: {e}")
            return ""
    
    def _should_perform_ocr(self, transcription: str) -> bool:
        """
        Check if OCR should be performed based on trigger keywords.
        
        Args:
            transcription: Video transcription text
            
        Returns:
            True if OCR should be performed
        """
        transcription_lower = transcription.lower()
        keywords_found = [
            keyword for keyword in self.trigger_keywords 
            if keyword in transcription_lower
        ]
        
        if not keywords_found:
            print("No OCR trigger keywords found in transcription. Skipping OCR extraction.")
            return False
        
        print(f"OCR trigger keywords found: {keywords_found}")
        return True
    
    def _process_video_frames(self, video_bytes: bytes, frame_interval: int) -> List[Dict]:
        """
        Process video frames and extract text.
        
        Args:
            video_bytes: Video content as bytes
            frame_interval: Interval between frames to process
            
        Returns:
            List of dictionaries containing extracted text and metadata
        """
        temp_path = self._create_temp_video_file(video_bytes)
        
        try:
            cap = cv2.VideoCapture(temp_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video from bytes for OCR")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Video FPS: {fps}, Total frames: {total_frames}")
            
            extracted_texts = []
            frame_count = 0
            processed_frames = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    text_data = self._extract_text_from_frame(frame, frame_count, fps)
                    if text_data:
                        extracted_texts.append(text_data)
                        print(f"Frame {frame_count} ({text_data['timestamp']}): {text_data['text'][:100]}...")
                    
                    processed_frames += 1
                    
                    if processed_frames >= self.max_frames:
                        print("Reached maximum frame processing limit")
                        break
                
                frame_count += 1
            
            cap.release()
            return extracted_texts
            
        finally:
            self._cleanup_temp_file(temp_path)
    
    def _extract_text_from_frame(self, frame: np.ndarray, frame_count: int, fps: float) -> Dict:
        """
        Extract text from a single frame.
        
        Args:
            frame: Video frame
            frame_count: Frame number
            fps: Video FPS
            
        Returns:
            Dictionary with timestamp and extracted text
        """
        try:
            processed_frame = ImageProcessor.preprocess_for_ocr(frame)
            pil_image = ImageProcessor.numpy_to_pil(processed_frame)
            text = pytesseract.image_to_string(pil_image, config=self.tesseract_config)
            
            if text.strip() and len(text.strip()) > 3:
                timestamp = frame_count / fps
                return {
                    "timestamp": f"{int(timestamp // 60)}:{int(timestamp % 60):02d}",
                    "frame": frame_count,
                    "text": text.strip().replace('\n', ' ').replace('\r', ' ')
                }
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
        
        return None
    
    @staticmethod
    def _create_temp_video_file(video_bytes: bytes) -> str:
        """
        Create temporary video file from bytes.
        
        Args:
            video_bytes: Video content as bytes
            
        Returns:
            Path to temporary video file
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_bytes)
            return tmp.name
    
    @staticmethod
    def _cleanup_temp_file(file_path: str) -> None:
        """
        Clean up temporary file.
        
        Args:
            file_path: Path to temporary file
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file: {e}")
    
    @staticmethod
    def _combine_extracted_texts(extracted_texts: List[Dict]) -> str:
        """
        Combine extracted texts into a single string.
        
        Args:
            extracted_texts: List of dictionaries with extracted text
            
        Returns:
            Combined text string
        """
        if not extracted_texts:
            print("No text extracted from video frames.")
            return ""
        
        combined_text = "\n".join([
            f"[{item['timestamp']}] {item['text']}" 
            for item in extracted_texts
        ])
        print(f"OCR extraction completed. Extracted text from {len(extracted_texts)} frames.")
        return combined_text

