"""Main application entry point."""
from fastapi import FastAPI
from src.api.routes import router as guide_router

# Create FastAPI application
app = FastAPI(
    title="Guide Generation Microservice",
    description="Generate step-by-step guides from video URLs (YouTube, Instagram)",
    version="2.0.0"
)

# Include routers
app.include_router(guide_router, tags=["guides"])


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "message": "Guide Generation Microservice",
        "version": "2.0.0",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    from src.config import Config
    
    uvicorn.run(
        app,
        host=Config.SERVER_HOST,
        port=Config.SERVER_PORT
    )
