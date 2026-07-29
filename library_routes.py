from typing import Literal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models import Library
from storage import upload_library_asset


router = APIRouter(prefix="/library", tags=["Library"])
MEDIA_TYPES = {"photo", "video", "article"}
ARTICLE_CONTENT_TYPES = {"application/pdf", "text/plain"}
ARTICLE_EXTENSIONS = {".pdf", ".txt"}
GENERIC_BINARY_CONTENT_TYPE = "application/octet-stream"


def serialize_library(item: Library) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "type": item.type,
        "media_type": item.media_type,
        "image_url": item.image_url,
        "media_url": item.image_url,
        "size": item.size,
        "size_kb": round((item.size or 0) / 1024, 2),
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


def validate_media_type(media_type: str) -> str:
    normalized_media_type = media_type.strip().lower()
    if normalized_media_type not in MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail="media_type must be one of: photo, video, article",
        )
    return normalized_media_type


def validate_media_file(media: UploadFile, media_type: str) -> None:
    content_type = (media.content_type or "").lower().split(";", 1)[0]
    filename = (media.filename or "").lower()

    if media_type == "photo" and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Photos must be image files")
    if media_type == "video" and not content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Videos must be video files")
    if media_type == "article":
        if not filename.endswith(tuple(ARTICLE_EXTENSIONS)):
            raise HTTPException(status_code=400, detail="Articles must be PDF or TXT files")
        if content_type not in ARTICLE_CONTENT_TYPES | {GENERIC_BINARY_CONTENT_TYPE}:
            raise HTTPException(status_code=400, detail="Articles must be PDF or TXT files")


async def upload_media(media: UploadFile, media_type: str) -> tuple[str, int]:
    validate_media_file(media, media_type)

    media_content = await media.read()
    if not media_content:
        raise HTTPException(status_code=400, detail="Media file is empty")

    try:
        media_url = upload_library_asset(
            file_content=media_content,
            filename=media.filename or "media",
            content_type=media.content_type or "application/octet-stream",
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return media_url, len(media_content)


def get_library_or_404(library_id: str, db: Session) -> Library:
    library_item = db.get(Library, library_id)
    if not library_item:
        raise HTTPException(status_code=404, detail="Library item not found")
    return library_item


@router.post(
    "/",
    status_code=status.HTTP_201_CREATED,
    summary="Upload a library asset",
    description=(
        "Upload one asset using multipart/form-data. Set `media_type` to `photo` "
        "for image files, `video` for video files, or `article` for PDF and TXT files."
    ),
    response_description="The newly created library asset",
    responses={
        400: {"description": "Invalid media type or file type"},
        502: {"description": "Asset upload to storage failed"},
    },
)
async def create_library_item(
    name: str = Form(..., description="Display name for the asset", examples=["Product launch video"]),
    type: str = Form(..., description="Library category selected by the user", examples=["Robotics"]),
    media_type: Literal["photo", "video", "article"] = Form(
        ..., description="Asset kind: photo, video, or article"
    ),
    media: UploadFile = File(
        ...,
        description="Image for photo, video file for video, or PDF/TXT file for article",
    ),
    db: Session = Depends(get_db),
):
    normalized_media_type = validate_media_type(media_type)
    media_url, media_size = await upload_media(media, normalized_media_type)
    library_item = Library(
        name=name,
        type=type,
        media_type=normalized_media_type,
        image_url=media_url,
        size=media_size,
    )
    db.add(library_item)
    db.commit()
    db.refresh(library_item)
    return serialize_library(library_item)


@router.get(
    "/",
    summary="List library assets",
    description=(
        "Return library assets ordered by newest first. Optionally filter the returned "
        "items by `type` and/or `media_type`; the total counts always include all "
        "library assets."
    ),
    response_description="Library assets with media-type counts and storage totals",
    responses={400: {"description": "Invalid media type filter"}},
)
def list_library_items(
    media_type: Literal["photo", "video", "article"] | None = Query(
        None,
        description="Optional filter for photo, video, or article assets",
        examples=["photo"],
    ),
    asset_type: str | None = Query(
        None,
        alias="type",
        description="Optional exact-match library category filter",
        examples=["Robotics"],
    ),
    db: Session = Depends(get_db),
):
    query = db.query(Library)
    if media_type is not None:
        query = query.filter(Library.media_type == validate_media_type(media_type))
    if asset_type is not None:
        query = query.filter(Library.type == asset_type)

    library_items = query.order_by(Library.created_at.desc()).all()
    total_storage_bytes = db.query(func.coalesce(func.sum(Library.size), 0)).scalar()
    return {
        "items": [serialize_library(item) for item in library_items],
        "total_assets": db.query(func.count(Library.id)).scalar(),
        "total_photo": db.query(func.count(Library.id)).filter(Library.media_type == "photo").scalar(),
        "total_article": db.query(func.count(Library.id)).filter(Library.media_type == "article").scalar(),
        "total_video": db.query(func.count(Library.id)).filter(Library.media_type == "video").scalar(),
        "total_storage_bytes": total_storage_bytes,
        "total_storage_kb": round(total_storage_bytes / 1024, 2),
        "total_storage_mb": round(total_storage_bytes / (1024 * 1024), 2),
    }


@router.get("/{library_id}")
def get_library_item(library_id: str, db: Session = Depends(get_db)):
    return serialize_library(get_library_or_404(library_id, db))


@router.put("/{library_id}")
async def update_library_item(
    library_id: str,
    name: str | None = Form(None),
    type: str | None = Form(None),
    media_type: str | None = Form(None),
    media: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    library_item = get_library_or_404(library_id, db)

    if name is not None:
        library_item.name = name
    if type is not None:
        library_item.type = type
    if media_type is not None:
        normalized_media_type = validate_media_type(media_type)
        if normalized_media_type != library_item.media_type and media is None:
            raise HTTPException(
                status_code=400,
                detail="Upload a replacement media file when changing media_type",
            )
        library_item.media_type = normalized_media_type
    if media is not None:
        library_item.image_url, library_item.size = await upload_media(
            media, library_item.media_type
        )

    db.commit()
    db.refresh(library_item)
    return serialize_library(library_item)


@router.delete("/{library_id}")
def delete_library_item(library_id: str, db: Session = Depends(get_db)):
    library_item = get_library_or_404(library_id, db)
    db.delete(library_item)
    db.commit()
    return {"message": "Library item deleted successfully", "id": library_id}