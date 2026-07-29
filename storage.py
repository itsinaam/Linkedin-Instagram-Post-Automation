from supabase import create_client
import os
from dotenv import load_dotenv
import uuid
from pathlib import Path

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

BUCKET_NAME = "products-images"


def upload_library_asset(
    file_content: bytes, filename: str, content_type: str = "application/octet-stream"
) -> str:
    """
    Upload a library asset to Supabase storage and return the public URL.
    
    Args:
        file_content: The binary content of the uploaded file
        filename: Original filename (will be made unique)
        content_type: MIME type of the uploaded file
    
    Returns:
        Public URL of the uploaded image
    
    Raises:
        Exception: If upload fails
    """
    try:
        # Generate a unique filename to avoid collisions
        file_extension = Path(filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Upload to Supabase storage
        supabase.storage.from_(BUCKET_NAME).upload(
            path=unique_filename,
            file=file_content,
            file_options={
                "content-type": content_type
            }
        )
        
        # Get public URL
        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(unique_filename)
        return public_url
        
    except Exception as e:
        raise Exception(f"Failed to upload library asset: {str(e)}")

