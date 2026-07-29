from fastapi import FastAPI, Depends, HTTPException, Query, UploadFile, File, Form
from fastapi.openapi.utils import get_openapi
from fastapi.responses import RedirectResponse
from urllib.parse import urlencode
from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import requests
import os
import secrets
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from database import get_db, Base, engine
from models import LinkedInToken, XOAuthState, XToken
from library_routes import router as library_router

# Load environment variables from .env file
load_dotenv()

# Create tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Social Media Automation API", version="1.0.0")

app.include_router(library_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    operation = schema["paths"].get("/instagram/post", {}).get("post", {})
    form_schema = operation.get("requestBody", {}).get("content", {}).get(
        "multipart/form-data", {}
    ).get("schema", {})
    schema_reference = form_schema.get("$ref")

    if schema_reference:
        component_name = schema_reference.rsplit("/", 1)[-1]
        files_schema = schema["components"]["schemas"][component_name]["properties"].get("files")
        if files_schema:
            files_schema["items"] = {"type": "string", "format": "binary"}

    app.openapi_schema = schema
    return app.openapi_schema


app.openapi = custom_openapi

# ---------------------------------------------------------------------------
# LinkedIn OAuth configuration from .env
# ---------------------------------------------------------------------------
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI")
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_API_BASE = "https://api.linkedin.com"

SCOPE = "openid profile email w_member_social"

X_CLIENT_ID = os.getenv("X_CLIENT_ID")
X_CLIENT_SECRET = os.getenv("X_CLIENT_SECRET")
X_REDIRECT_URI = os.getenv("X_REDIRECT_URI")
X_AUTH_URL = "https://x.com/i/oauth2/authorize"
X_TOKEN_URL = "https://api.x.com/2/oauth2/token"
X_API_BASE = "https://api.x.com/2"
X_SCOPE = "tweet.read tweet.write users.read offline.access"


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class PostCreate(BaseModel):
    commentary: str
    hashtags: str = ""
    visibility: str = "PUBLIC"
    feedDistribution: str = "MAIN_FEED"


# ---------------------------------------------------------------------------
# Endpoint 1 – Redirect user to LinkedIn for authorization
# ---------------------------------------------------------------------------
@app.get("/auth/linkedin/login")
def linkedin_login():
    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
    }
    url = f"{LINKEDIN_AUTH_URL}?{urlencode(params)}"
    return RedirectResponse(url)


def _require_x_configuration():
    if not X_CLIENT_ID or not X_CLIENT_SECRET:
        raise HTTPException(
            status_code=500,
            detail="X_CLIENT_ID and X_CLIENT_SECRET must be configured.",
        )


def _x_code_challenge(code_verifier: str) -> str:
    digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _x_token_request(data: dict[str, str]) -> requests.Response:
    _require_x_configuration()
    return requests.post(
        X_TOKEN_URL,
        data=data,
        auth=(X_CLIENT_ID, X_CLIENT_SECRET),
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=30,
    )


def _x_expiry(token_data: dict) -> datetime | None:
    expires_in = token_data.get("expires_in")
    if expires_in is None:
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=int(expires_in))


def _refresh_x_token(token_record: XToken, db: Session) -> XToken:
    if not token_record.refresh_token:
        raise HTTPException(
            status_code=401,
            detail="X access token has expired. Complete GET /auth/x/login again.",
        )

    response = _x_token_request(
        {"grant_type": "refresh_token", "refresh_token": token_record.refresh_token}
    )
    if response.status_code != 200:
        token_record.is_active = False
        db.commit()
        raise HTTPException(
            status_code=401,
            detail=f"X token refresh failed. Complete GET /auth/x/login again: {response.text}",
        )

    token_data = response.json()
    token_record.access_token = token_data["access_token"]
    token_record.refresh_token = token_data.get("refresh_token", token_record.refresh_token)
    token_record.token_type = token_data.get("token_type", "Bearer")
    token_record.expires_at = _x_expiry(token_data)
    token_record.scope = token_data.get("scope", token_record.scope)
    db.commit()
    db.refresh(token_record)
    return token_record


def _get_active_x_token(db: Session) -> XToken:
    token_record = (
        db.query(XToken)
        .filter(XToken.is_active == True)
        .order_by(XToken.created_at.desc())
        .first()
    )
    if not token_record:
        raise HTTPException(
            status_code=401,
            detail="No active X token found. Complete GET /auth/x/login first.",
        )

    expires_at = token_record.expires_at
    if expires_at and expires_at <= datetime.now(timezone.utc) + timedelta(seconds=60):
        return _refresh_x_token(token_record, db)
    return token_record


@app.get("/auth/x/login")
def x_login(db: Session = Depends(get_db)):
    """Redirect the user to X's OAuth 2.0 authorization page."""
    _require_x_configuration()
    state = secrets.token_urlsafe(32)
    code_verifier = secrets.token_urlsafe(64)

    db.query(XOAuthState).filter(
        XOAuthState.expires_at < datetime.now(timezone.utc)
    ).delete(synchronize_session=False)
    db.add(
        XOAuthState(
            state=state,
            code_verifier=code_verifier,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=10),
        )
    )
    db.commit()

    query = urlencode(
        {
            "response_type": "code",
            "client_id": X_CLIENT_ID,
            "redirect_uri": X_REDIRECT_URI,
            "scope": X_SCOPE,
            "state": state,
            "code_challenge": _x_code_challenge(code_verifier),
            "code_challenge_method": "S256",
        }
    )
    return RedirectResponse(f"{X_AUTH_URL}?{query}")


@app.get("/auth/x/callback")
def x_callback(
    code: str = Query(...), state: str = Query(...), db: Session = Depends(get_db)
):
    """Exchange the X authorization code and persist the user token pair."""
    oauth_state = db.get(XOAuthState, state)
    if not oauth_state or oauth_state.expires_at < datetime.now(timezone.utc):
        if oauth_state:
            db.delete(oauth_state)
            db.commit()
        raise HTTPException(status_code=400, detail="Invalid or expired X OAuth state.")

    response = _x_token_request(
        {
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": X_REDIRECT_URI,
            "code_verifier": oauth_state.code_verifier,
        }
    )
    db.delete(oauth_state)

    if response.status_code != 200:
        db.commit()
        raise HTTPException(
            status_code=response.status_code,
            detail=f"X token exchange failed: {response.text}",
        )

    token_data = response.json()
    access_token = token_data.get("access_token")
    if not access_token:
        db.commit()
        raise HTTPException(status_code=400, detail="X did not return an access token.")

    user_response = requests.get(
        f"{X_API_BASE}/users/me",
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=30,
    )
    x_user_id = None
    if user_response.status_code == 200:
        x_user_id = user_response.json().get("data", {}).get("id")

    db.query(XToken).filter(XToken.is_active == True).update(
        {XToken.is_active: False}
    )
    token_record = XToken(
        access_token=access_token,
        refresh_token=token_data.get("refresh_token"),
        token_type=token_data.get("token_type", "Bearer"),
        expires_at=_x_expiry(token_data),
        scope=token_data.get("scope"),
        x_user_id=x_user_id,
    )
    db.add(token_record)
    db.commit()

    return {
        "message": "X account connected successfully.",
        "x_user_id": x_user_id,
        "expires_at": token_record.expires_at,
    }


# ---------------------------------------------------------------------------
# Endpoint 2 – Callback: exchange code → access token → save to DB
# ---------------------------------------------------------------------------
@app.get("/auth/linkedin/callback")
def linkedin_callback(code: str = Query(...), db: Session = Depends(get_db)):
    # 1. Exchange authorization code for access token
    token_response = requests.post(
        LINKEDIN_TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    if token_response.status_code != 200:
        raise HTTPException(
            status_code=token_response.status_code,
            detail=f"Token exchange failed: {token_response.text}",
        )

    token_data = token_response.json()
    access_token = token_data.get("access_token")

    if not access_token:
        raise HTTPException(status_code=400, detail="No access_token in response")

    # 2. Use the token to get the LinkedIn person ID (sub)
    userinfo_resp = requests.get(
        f"{LINKEDIN_API_BASE}/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    person_id = None
    if userinfo_resp.status_code == 200:
        person_id = userinfo_resp.json().get("sub")

    # 3. Deactivate any old tokens so only one is active at a time
    db.query(LinkedInToken).filter(LinkedInToken.is_active == True).update(
        {LinkedInToken.is_active: False}
    )

    # 4. Save the new token
    new_token = LinkedInToken(
        access_token=access_token,
        token_type=token_data.get("token_type", "Bearer"),
        expires_in=token_data.get("expires_in"),
        scope=token_data.get("scope"),
        person_id=person_id,
    )
    db.add(new_token)
    db.commit()
    db.refresh(new_token)

    return {
        "message": "LinkedIn account linked successfully ✅",
        "person_id": person_id,
        "token_id": new_token.id,
    }


# ---------------------------------------------------------------------------
# Endpoint 4 – Create a post WITH image on LinkedIn
# ---------------------------------------------------------------------------
@app.post("/linkedin/post")
async def create_linkedin_post_with_image(
    commentary: str = Form(...),
    hashtags: str = Form(""),
    image: UploadFile = File(...),
    visibility: str = Form("PUBLIC"),
    feedDistribution: str = Form("MAIN_FEED"),
    db: Session = Depends(get_db),
):
    # 1. Fetch the most recent active token
    token_record = (
        db.query(LinkedInToken)
        .filter(LinkedInToken.is_active == True)
        .order_by(LinkedInToken.created_at.desc())
        .first()
    )

    if not token_record:
        raise HTTPException(
            status_code=401,
            detail="No active LinkedIn token found. Please authenticate via /auth/linkedin/login",
        )

    # 2. Get person_id if missing (fallback)
    person_id = token_record.person_id
    if not person_id:
        userinfo_resp = requests.get(
            f"{LINKEDIN_API_BASE}/v2/userinfo",
            headers={"Authorization": f"Bearer {token_record.access_token}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Could not fetch LinkedIn person ID. Re-authenticate please.",
            )
        person_id = userinfo_resp.json().get("sub")
        token_record.person_id = person_id
        db.commit()

    access_token = token_record.access_token
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "LinkedIn-Version": "202607",
    }

    # 3. Register image upload
    image_bytes = await image.read()
    image_filename = image.filename or "image.png"

    register_resp = requests.post(
        f"{LINKEDIN_API_BASE}/rest/images?action=initializeUpload",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "initializeUploadRequest": {
                "owner": f"urn:li:person:{person_id}",
            }
        },
    )

    if register_resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=register_resp.status_code,
            detail=f"Image register failed: {register_resp.text}",
        )

    upload_data = register_resp.json().get("value", {})
    upload_url = upload_data.get("uploadUrl")
    image_urn = upload_data.get("image")

    if not upload_url or not image_urn:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to get upload URL from LinkedIn: {register_resp.text}",
        )

    # 4. Upload the actual image binary
    upload_headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream",
    }

    # LinkedIn expects a PUT to the upload URL
    upload_resp = requests.put(upload_url, headers=upload_headers, data=image_bytes)

    if upload_resp.status_code not in (200, 201):
        raise HTTPException(
            status_code=upload_resp.status_code,
            detail=f"Image upload failed: {upload_resp.text}",
        )

    # 5. Build commentary with hashtags
    commentary_text = commentary
    if hashtags:
        commentary_text += "\n\n" + hashtags

    # 6. Create the post with the image
    post_body = {
        "author": f"urn:li:person:{person_id}",
        "commentary": commentary_text,
        "visibility": visibility,
        "distribution": {
            "feedDistribution": feedDistribution,
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
        "content": {
            "media": {
                "title": "Post Image",
                "id": image_urn,
            }
        },
    }

    response = requests.post(
        f"{LINKEDIN_API_BASE}/rest/posts",
        headers={**headers, "Content-Type": "application/json"},
        json=post_body,
    )

    if response.status_code not in (200, 201):
        raise HTTPException(
            status_code=response.status_code,
            detail=f"LinkedIn post with image failed: {response.text}",
        )

    return {
        "message": "Post with image created successfully ✅",
        "status_code": response.status_code,
        "image_urn": image_urn,
        "response": response.text,
    }


# ---------------------------------------------------------------------------
# Endpoint 5 – Create a post on Instagram
# ---------------------------------------------------------------------------
INSTAGRAM_USERNAME = os.getenv("INSTAGRAM_USERNAME")
INSTAGRAM_PASSWORD = os.getenv("INSTAGRAM_PASSWORD")
INSTAGRAM_SESSION_ID = os.getenv("INSTAGRAM_SESSION_ID")
INSTAGRAM_SESSION_FILE = Path(__file__).with_name("instagram_session.json")

_instagram_client = None


def _get_instagram_client():
    global _instagram_client
    if _instagram_client is not None:
        return _instagram_client

    if not INSTAGRAM_SESSION_FILE.exists():
        raise RuntimeError("Instagram session is missing. Call POST /instagram/login once.")

    from instagrapi import Client

    cl = Client()
    cl.load_settings(INSTAGRAM_SESSION_FILE)
    _instagram_client = cl
    return _instagram_client


@app.post("/instagram/login")
def login_instagram():
    """Authenticate once and persist the Instagram session for future posts."""
    global _instagram_client

    if INSTAGRAM_SESSION_FILE.exists():
        return {
            "message": "Instagram session is already active.",
            "next_step": "Use POST /instagram/post to create a post.",
        }

    if not INSTAGRAM_SESSION_ID and (not INSTAGRAM_USERNAME or not INSTAGRAM_PASSWORD):
        raise HTTPException(status_code=500, detail="Instagram credentials are not configured.")

    try:
        from instagrapi import Client

        cl = Client()
        if INSTAGRAM_SESSION_ID:
            authenticated = cl.login_by_sessionid(INSTAGRAM_SESSION_ID)
        else:
            authenticated = cl.login(
                INSTAGRAM_USERNAME,
                INSTAGRAM_PASSWORD,
            )

        if not authenticated:
            raise HTTPException(status_code=401, detail="Instagram login was not accepted.")

        cl.dump_settings(INSTAGRAM_SESSION_FILE)
        _instagram_client = cl
        return {"message": "Instagram session saved successfully."}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=401, detail=f"Instagram login failed: {str(error)}")


def _upload_instagram_media(cl, media_type: str, media_paths: list[Path], caption: str):
    if media_type == "photo":
        return cl.photo_upload(media_paths[0], caption)
    if media_type == "reel":
        return cl.video_upload(media_paths[0], caption)
    return cl.album_upload(media_paths, caption)


@app.post("/instagram/post")
async def create_instagram_post(
    caption: str = Form(...),
    media_type: str = Form("photo"),
    files: list[UploadFile] = File(...),
):
    global _instagram_client

    media_type = media_type.lower().strip()
    if media_type not in {"photo", "carousel", "reel"}:
        raise HTTPException(
            status_code=422,
            detail="media_type must be one of: photo, carousel, reel.",
        )

    if media_type in {"photo", "reel"} and len(files) != 1:
        raise HTTPException(
            status_code=422,
            detail=f"{media_type} posts require exactly one file.",
        )

    if media_type == "carousel" and not 2 <= len(files) <= 10:
        raise HTTPException(
            status_code=422,
            detail="Carousel posts require between 2 and 10 files.",
        )

    if media_type == "photo" and files[0].content_type and not files[0].content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="A photo post requires an image file.")

    if media_type == "reel" and files[0].content_type and not files[0].content_type.startswith("video/"):
        raise HTTPException(status_code=422, detail="A reel post requires a video file.")

    try:
        cl = _get_instagram_client()

        with TemporaryDirectory() as temp_dir:
            media_paths = []
            for index, upload in enumerate(files):
                extension = Path(upload.filename or "").suffix
                if not extension:
                    extension = ".mp4" if media_type == "reel" else ".jpg"

                temp_path = Path(temp_dir) / f"media_{index}{extension}"
                temp_path.write_bytes(await upload.read())
                media_paths.append(temp_path)

            result = _upload_instagram_media(cl, media_type, media_paths, caption)

        return {
            "message": f"Instagram {media_type} post created successfully",
            "media_type": media_type,
            "media_id": result.id,
            "code": result.code,
        }
    except HTTPException:
        raise
    except Exception as error:
        if "login_required" in str(error):
            _instagram_client = None
            INSTAGRAM_SESSION_FILE.unlink(missing_ok=True)
            raise HTTPException(
                status_code=401,
                detail=(
                    "Instagram session has expired or requires verification. "
                    "Complete POST /instagram/login, then try the post again."
                ),
            )
        raise HTTPException(status_code=400, detail=f"Instagram post failed: {str(error)}")


# ---------------------------------------------------------------------------
# Endpoint 6 – Create a post with an image on X
# ---------------------------------------------------------------------------
@app.post("/x/post")
async def create_x_post(
    caption: str = Form(...),
    image: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload one image to X and publish a post with the supplied caption."""
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="X posts require an image file.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")

    token_record = _get_active_x_token(db)
    headers = {"Authorization": f"Bearer {token_record.access_token}"}
    media_type = image.content_type or "image/jpeg"

    init_response = requests.post(
        f"{X_API_BASE}/media/upload",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "media_category": "tweet_image",
            "media_type": media_type,
            "total_bytes": len(image_bytes),
        },
        timeout=60,
    )
    if init_response.status_code not in (200, 201, 202):
        raise HTTPException(
            status_code=init_response.status_code,
            detail=f"X media initialization failed: {init_response.text}",
        )

    media_id = init_response.json().get("data", {}).get("id")
    if not media_id:
        raise HTTPException(status_code=400, detail="X did not return a media ID.")

    append_response = requests.post(
        f"{X_API_BASE}/media/upload/{media_id}/append",
        headers=headers,
        data={"segment_index": "0"},
        files={"media": (image.filename or "image.jpg", image_bytes, media_type)},
        timeout=60,
    )
    if append_response.status_code not in (200, 201, 202, 204):
        raise HTTPException(
            status_code=append_response.status_code,
            detail=f"X media upload failed: {append_response.text}",
        )

    finalize_response = requests.post(
        f"{X_API_BASE}/media/upload/{media_id}/finalize",
        headers=headers,
        timeout=60,
    )
    if finalize_response.status_code not in (200, 201, 202):
        raise HTTPException(
            status_code=finalize_response.status_code,
            detail=f"X media finalization failed: {finalize_response.text}",
        )

    tweet_response = requests.post(
        f"{X_API_BASE}/tweets",
        headers={**headers, "Content-Type": "application/json"},
        json={"text": caption, "media": {"media_ids": [media_id]}},
        timeout=60,
    )
    if tweet_response.status_code not in (200, 201):
        raise HTTPException(
            status_code=tweet_response.status_code,
            detail=f"X post creation failed: {tweet_response.text}",
        )

    tweet_data = tweet_response.json().get("data", {})
    return {
        "message": "X post created successfully.",
        "post_id": tweet_data.get("id"),
        "text": tweet_data.get("text"),
        "media_id": media_id,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)