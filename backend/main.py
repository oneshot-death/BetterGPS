
from fastapi import FastAPI

from app.routes.prediction import router as prediction_router

app = FastAPI(
    title="BetterGPS API",
    description="GPS trajectory map matching backend",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to BetterGPS!",
        "status": "Backend is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


app.include_router(prediction_router)
