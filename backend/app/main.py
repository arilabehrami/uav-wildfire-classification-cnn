"""Pika e nisjes e API-se.

TensorFlow importohet vetem brenda 'lifespan', jo ne nivel moduli. Kjo e len
aplikacionin te ndertohet dhe te testohet pa ngarkuar modelin 9 MB, dhe
pa pritur importin e TF-se.

Ekzekutimi:
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings, load_settings
from .routes.classification_routes import router

logger = logging.getLogger("fire.api")
_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s %(message)s"


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=level, format=_LOG_FORMAT)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Importet jane brenda qellimisht: shih docstring-un e modulit.
    from .ml.classifier import FireClassifier
    from .services.classification_service import ClassificationService

    settings: Settings = app.state.settings
    classifier = FireClassifier(
        run=settings.model_run,
        verify_hash=settings.verify_model_hash,
    )
    app.state.classification_service = ClassificationService(classifier)

    logger.info(
        "modeli u ngarkua: run=%s sha256=%s porta=[%.4f, %.4f)",
        classifier.run,
        classifier.model_sha256[:12],
        classifier.low,
        classifier.high,
    )
    try:
        yield
    finally:
        app.state.classification_service = None
        logger.info("sherbimi u ndal")


def create_app(
    settings: Settings | None = None,
    *,
    use_lifespan: bool = True,
) -> FastAPI:
    """Ndertuesi i aplikacionit.

    Testet e thirrin me 'use_lifespan=False' dhe vendosin vete nje sherbim
    ne 'app.state.classification_service', pra nuk kerkohet as TensorFlow
    as modeli i ruajtur.
    """
    settings = settings or load_settings()
    _configure_logging(settings.log_level)

    app = FastAPI(
        title="Fire Classification API",
        version="1.0",
        lifespan=lifespan if use_lifespan else None,
    )
    app.state.settings = settings
    app.state.classification_service = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Gjurme auditimi: metoda, rruga, statusi dhe kohezgjatja."""
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        level = logging.DEBUG if request.url.path == "/health" else logging.INFO
        logger.log(
            level,
            "%s %s -> %s ne %.0f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    app.include_router(router, prefix="/api")

    @app.get("/health")
    def health(request: Request) -> dict:
        ready = request.app.state.classification_service is not None
        return {"status": "ok" if ready else "loading", "model_loaded": ready}

    return app


app = create_app()
