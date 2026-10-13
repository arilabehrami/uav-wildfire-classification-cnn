"""Rruga e klasifikimit.

Kufiri i madhesise lexohet nga konfigurimi, jo nga kodi, dhe fajlli lexohet
vetem deri ne kufi + 1 bajt: nje ngarkese e madhe nuk hyn kurre e plote ne
memorie.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ..schemas.classification_schema import ClassificationResponse

router = APIRouter()
logger = logging.getLogger("fire.api")


@router.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
):
    settings = request.app.state.settings
    limit = settings.max_file_bytes

    try:
        image_bytes = await file.read(limit + 1)
    finally:
        await file.close()

    if len(image_bytes) > limit:
        raise HTTPException(
            413, f"Imazhi duhet të jetë deri në {settings.max_file_mb} MB."
        )

    service = request.app.state.classification_service
    if service is None:
        raise HTTPException(503, "Modeli nuk është i ngarkuar.")

    try:
        result = await run_in_threadpool(service.classify, image_bytes)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error

    logger.info(
        "classify fajll=%r bajte=%d zone=%s score=%.4f harte=%s",
        file.filename,
        len(image_bytes),
        result.get("zone"),
        result.get("fire_score", float("nan")),
        "po" if result.get("explanation") else "jo",
    )
    return result
