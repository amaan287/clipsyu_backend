# Recipe Generation Microservice

This service exposes a FastAPI endpoint to accept an Instagram or YouTube video URL and returns a generated recipe. Database and authentication have been removed.

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg (required for audio extraction from videos)

**Install FFmpeg:**
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)

### Installation

1. **Install Python dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set up environment variables:**

Create a `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
```

3. **Run the service:**

```bash
python src/main.py
```

### Docker Deployment

Alternatively, you can run the service using Docker:

1. **Run the pre-built Docker image:**

```bash
docker run -p 8000:8000 --name clipsy-ai -it --env-file .env amaan287/clipsy-ai:latest
```

Or run in detached mode:

```bash
docker run -d -p 8000:8000 --name clipsy-ai --env-file .env amaan287/clipsy-ai:latest
```

2. **Stop the container:**

```bash
docker stop clipsy-ai
docker rm clipsy-ai
```

**Building locally (optional):**

If you want to build the Docker image locally instead:

```bash
docker build -t clipsy-ai .
docker run -p 8000:8000 --name clipsy-ai -it --env-file .env clipsy-ai
```

## API Usage

**POST /generate-recipe**

- Request JSON:
```json
{
  "url": "https://www.instagram.com/reel/..."
}
```
- Response JSON:
```json
{
  "recipe": "...recipe string..."
}
```

- Only Instagram is currently implemented. YouTube is a TODO.
