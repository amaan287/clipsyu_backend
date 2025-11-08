from fastapi import FastAPI
from src.api.routes import router as recipe_router

app = FastAPI(title="Recipe Generation Microservice")

app.include_router(recipe_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
