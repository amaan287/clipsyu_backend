"""AI-based guide extraction service."""
import json
import httpx
from typing import Dict, Optional, Tuple
from datetime import datetime

from src.interfaces import IAIExtractionService
from src.config import Config
from src.utils.url_parser import TextCleaner


class OpenAIExtractionService(IAIExtractionService):
    """Service for extracting guide information using OpenAI GPT."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize AI extraction service.
        
        Args:
            api_key: OpenAI API key (optional, uses config if not provided)
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.base_url = Config.OPENAI_API_BASE_URL
        self.model = Config.GPT_MODEL
        self.temperature = Config.GPT_TEMPERATURE
        self.max_tokens = Config.GPT_MAX_TOKENS
        self.timeout = Config.HTTP_CLIENT_TIMEOUT
    
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
        try:
            print("Analyzing content to extract recipe/guide with OpenAI GPT-4o...")
            
            self._log_transcription_warning(transcription, metadata.get('description', ''))
            
            prompt = self._build_extraction_prompt(
                metadata, comment_info, transcription, ocr_text
            )
            
            guide_data = await self._call_openai_api(prompt)
            
            # Enrich guide data with metadata
            guide_data = self._enrich_guide_data(guide_data, metadata, ocr_text)
            
            print(f"✅ Guide extraction completed: {guide_data.get('title', 'Unknown')}")
            return guide_data, None
            
        except json.JSONDecodeError as e:
            print(f"Error parsing AI response as JSON: {e}")
            return self._create_fallback_guide(metadata, ocr_text), None
        except Exception as e:
            print(f"Error during guide extraction: {str(e)}")
            return self._create_fallback_guide(metadata, ocr_text), str(e)
    
    def _build_extraction_prompt(
        self,
        metadata: Dict,
        comment_info: Optional[str],
        transcription: str,
        ocr_text: str
    ) -> str:
        """
        Build the prompt for guide extraction.
        
        Args:
            metadata: Video metadata
            comment_info: Comment information
            transcription: Video transcription
            ocr_text: OCR extracted text
            
        Returns:
            Formatted prompt string
        """
        video_title = metadata.get('title', '')
        video_description = metadata.get('description', '')
        channel_name = metadata.get('channel', metadata.get('uploader', ''))
        
        ocr_section = f"\n\nOCR TEXT: {ocr_text}" if ocr_text else ""
        transcription_note = self._get_transcription_note(transcription, video_description)
        
        return f"""Analyze this video and extract a step-by-step guide in JSON format.
{transcription_note}
TITLE: {video_title}
DESCRIPTION: {video_description}
CHANNEL: {channel_name}
COMMENT: {comment_info or ''}
TRANSCRIPTION: {transcription}{ocr_section}

Extract the guide with this structure:
{{
  "title": "guide name",
  "description": "brief overview",
  "content_type": "recipe|tutorial|how-to|programming|educational|general",
  "materials": [{{"name": "item", "quantity": "amount", "notes": "notes", "optional": false}}],
  "steps": [{{"step": 1, "instruction": "description", "duration": "time", "details": "extras", "code_snippet": "code", "tips": ["tips"]}}],
  "metadata": {{"duration": "", "difficulty": "easy|medium|hard", "category": "", "tags": [], "estimated_time": "", "skill_level": "beginner|intermediate|advanced", "language": "", "framework": ""}},
  "tools": ["tools/equipment"],
  "tips": ["general tips"],
  "prerequisites": ["requirements"],
  "isInstructional": true
}}

CRITICAL INSTRUCTIONS:
- ONLY extract information that is EXPLICITLY present in the DESCRIPTION, TRANSCRIPTION, COMMENT, or OCR TEXT
- PRIORITIZE: If TRANSCRIPTION is empty/minimal, use DESCRIPTION as primary source (many videos have full details in description)
- DO NOT make up, infer, or assume any steps, ingredients, or instructions that are not clearly stated
- If TRANSCRIPTION is empty AND DESCRIPTION is empty, set "isInstructional": false and return minimal structure
- Set "isInstructional": true when you can extract clear, specific step-by-step instructions from description OR transcription
- For recipes: extract ingredients from description, transcription, comments, or OCR text
- For tutorials: extract steps from description, transcription, comments, or OCR text
- Use OCR text for precise measurements, code, or commands when available
- Return only valid JSON

Extract now:
"""
    
    async def _call_openai_api(self, prompt: str) -> Dict:
        """
        Call OpenAI API to extract guide information.
        
        Args:
            prompt: Extraction prompt
            
        Returns:
            Guide data dictionary
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert content analyzer that extracts step-by-step guides from instructional videos. You MUST ONLY extract information explicitly present in the provided transcription. NEVER make up, infer, or hallucinate content. If there is insufficient information, return a minimal structure with isInstructional set to false. Return valid JSON only."
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers=headers, content=json.dumps(payload))
            
            if response.status_code != 200:
                raise Exception(
                    f"OpenAI GPT-4o API error (status {response.status_code}): {response.text}"
                )
            
            resp_json = response.json()
            choices = resp_json.get('choices', [])
            if not choices:
                raise Exception("No choices in OpenAI chat response")
            
            content = choices[0]["message"].get("content", "")
            cleaned_content = TextCleaner.clean_json_response(content)
            
            return json.loads(cleaned_content)
    
    @staticmethod
    def _log_transcription_warning(transcription: str, description: str) -> None:
        """Log warning if transcription is empty or very short."""
        if not transcription or len(transcription) < 50:
            print("⚠️  WARNING: Transcription is empty or very short. AI may not be able to extract meaningful content.")
            print(f"Transcription length: {len(transcription)} characters")
    
    @staticmethod
    def _get_transcription_note(transcription: str, description: str) -> str:
        """Get note about transcription quality."""
        if not transcription or len(transcription) < 50:
            if description:
                return "\n⚠️ NOTE: Video has no/minimal audio transcription. Extract information primarily from DESCRIPTION.\n"
            else:
                return "\n⚠️ NOTE: Video has no/minimal audio transcription and no description. Limited data available.\n"
        return ""
    
    @staticmethod
    def _enrich_guide_data(guide_data: Dict, metadata: Dict, ocr_text: str) -> Dict:
        """Enrich guide data with additional metadata."""
        guide_data.setdefault('title', metadata.get('title', 'Unknown'))
        guide_data.setdefault('channelName', metadata.get('channel', metadata.get('uploader', 'Unknown')))
        guide_data.setdefault('savedDate', datetime.now().isoformat())
        guide_data.setdefault('materials', [])
        guide_data.setdefault('steps', [])
        guide_data.setdefault('isInstructional', False)
        guide_data.setdefault('ocrExtractedInfo', bool(ocr_text))
        
        return guide_data
    
    @staticmethod
    def _create_fallback_guide(metadata: Dict, ocr_text: str) -> Dict:
        """Create a fallback guide structure when extraction fails."""
        return {
            "title": metadata.get('title', 'Unknown'),
            "description": "",
            "materials": [],
            "steps": [],
            "metadata": {},
            "tools": [],
            "tips": [],
            "channelName": metadata.get('channel', metadata.get('uploader', 'Unknown')),
            "savedDate": datetime.now().isoformat(),
            "isInstructional": False,
            "ocrExtractedInfo": bool(ocr_text),
            "error": "Could not extract guide from video content"
        }

