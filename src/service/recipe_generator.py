from src.agents.instagram_agent import InstagramAgent
from src.api.schemas import Guide
import asyncio

INSTAGRAM_DOMAINS = ["instagram.com", "www.instagram.com"]
YOUTUBE_DOMAINS = ["youtube.com", "www.youtube.com", "youtu.be"]

def generate_recipe_from_url(url: str) -> Guide:
    if any(domain in url for domain in INSTAGRAM_DOMAINS):
        agent = InstagramAgent()
        guide_dict = asyncio.run(agent.get_recipe_from_url(url))
        # Convert dict to Guide model
        return Guide(**guide_dict)
    elif any(domain in url for domain in YOUTUBE_DOMAINS):
        # agent = YouTubeAgent()
        # return agent.get_recipe_from_url(url)
        raise NotImplementedError("YouTube recipe generation not yet implemented.")
    else:
        raise ValueError("Unsupported platform URL.")
