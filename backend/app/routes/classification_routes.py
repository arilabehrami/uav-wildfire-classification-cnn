from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from ..schemas.classification_schema import ClassificationResponse

router = APIRouter()
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/classify", response_model=ClassificationResponse)
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
):
    try:
        image_bytes = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, "Imazhi duhet të jetë deri në 10 MB.")

    service = request.app.state.classification_service

    try:
        return await run_in_threadpool(service.classify, image_bytes)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error