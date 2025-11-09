"""Guide generation service with factory pattern."""
import asyncio
from typing import Dict

from src.agents import YouTubeAgent, InstagramAgent
from src.api.schemas import Guide
from src.utils.url_parser import URLParser


class PlatformAgentFactory:
    """
    Factory for creating platform-specific agents.
    
    This follows the Factory Pattern and Open/Closed Principle,
    making it easy to add new platforms without modifying existing code.
    """
    
    @staticmethod
    def create_agent(url: str):
        """
        Create appropriate agent based on URL.
        
        Args:
            url: Video URL
            
        Returns:
            Platform-specific agent instance
            
        Raises:
            ValueError: If platform is not supported
        """
        if URLParser.is_instagram_url(url):
            return InstagramAgent()
        elif URLParser.is_youtube_url(url):
            return YouTubeAgent()
        else:
            raise ValueError(
                f"Unsupported platform URL. Supported platforms: Instagram, YouTube"
            )


class GuideGeneratorService:
    """
    Service for generating guides from video URLs.
    
    This class follows the Single Responsibility Principle - it only
    handles coordinating guide generation and doesn't know about
    platform-specific implementations.
    """
    
    def __init__(self, agent_factory: PlatformAgentFactory = None):
        """
        Initialize guide generator service.
        
        Args:
            agent_factory: Factory for creating platform agents
        """
        self.agent_factory = agent_factory or PlatformAgentFactory()
    
    async def generate_guide_async(self, url: str) -> Dict:
        """
        Generate guide from video URL asynchronously.
        
        Args:
            url: Video URL
            
        Returns:
            Dictionary containing guide information
            
        Raises:
            ValueError: If platform is not supported
            Exception: If guide generation fails
        """
        # Create appropriate agent for the platform
        agent = self.agent_factory.create_agent(url)
        
        # Generate guide using the agent
        guide_dict = await agent.get_guide_from_url(url)
        
        return guide_dict
    
    def generate_guide(self, url: str) -> Guide:
        """
        Generate guide from video URL (synchronous wrapper).
        
        Args:
            url: Video URL
            
        Returns:
            Guide model instance
            
        Raises:
            ValueError: If platform is not supported
            Exception: If guide generation fails
        """
        # Run async method in event loop
        guide_dict = asyncio.run(self.generate_guide_async(url))
        
        # Convert dict to Guide model
        return Guide(**guide_dict)

