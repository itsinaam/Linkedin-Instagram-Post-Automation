from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from database import get_db
from models import Library
from storage import upload_product_image


router = APIRouter(prefix="/library", tags=["Library"])


def serialize_library(item: Library) -> dict:
    return {
        "id": item.id,
        "name": item.name,
        "type": item.type,
        "image_url": item.image_url,
        "size": item.size,
        "size_kb": round((item.size or 0) / 1024, 2),
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
    }


async def upload_image(image: UploadFile) -> tuple[str, int]:
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    image_content = await image.read()
    if not image_content:
        raise HTTPException(status_code=400, detail="Image file is empty")

    try:
        image_url = upload_product_image(
            file_content=image_content,
            filename=image.filename or "image",
            content_type=image.content_type or "application/octet-stream",
        )
    except Exception as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return image_url, len(image_content)


def get_library_or_404(library_id: str, db: Session) -> Library:
    library_item = db.get(Library, library_id)
    if not library_item:
        raise HTTPException(status_code=404, detail="Library item not found")
    return library_item


@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_library_item(
    name: str = Form(...),
    type: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    image_url, image_size = await upload_image(image)
    library_item = Library(
        name=name,
        type=type,
        image_url=image_url,
        size=image_size,
    )
    db.add(library_item)
    db.commit()
    db.refresh(library_item)
    return serialize_library(library_item)


@router.get("/")
def list_library_items(db: Session = Depends(get_db)):
    library_items = db.query(Library).order_by(Library.created_at.desc()).all()
    total_storage_bytes = db.query(func.coalesce(func.sum(Library.size), 0)).scalar()
    return {
        "items": [serialize_library(item) for item in library_items],
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
    image: UploadFile | None = File(None),
    db: Session = Depends(get_db),
):
    library_item = get_library_or_404(library_id, db)

    if name is not None:
        library_item.name = name
    if type is not None:
        library_item.type = type
    if image is not None:
        library_item.image_url, library_item.size = await upload_image(image)

    db.commit()
    db.refresh(library_item)
    return serialize_library(library_item)


@router.delete("/{library_id}")
def delete_library_item(library_id: str, db: Session = Depends(get_db)):
    library_item = get_library_or_404(library_id, db)
    db.delete(library_item)
    db.commit()
    return {"message": "Library item deleted successfully", "id": library_id}