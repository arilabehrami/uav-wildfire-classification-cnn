from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ml.classifier import FireClassifier
from .routes.classification_routes import router
from .services.classification_service import ClassificationService


@asynccontextmanager
async def lifespan(app: FastAPI):
    classifier = FireClassifier()
    app.state.classification_service = ClassificationService(classifier)
    yield


app = FastAPI(title="Fire Classification API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}