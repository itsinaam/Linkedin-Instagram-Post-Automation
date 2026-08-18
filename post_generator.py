import os
import io
import json
import random
import base64
import requests
from pathlib import Path
from sqlalchemy.orm import Session
from fastapi import HTTPException
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from models import Library, GeneratedPost
from storage import upload_library_asset

BASE_DIR = Path(__file__).resolve().parent

def load_system_prompt() -> str:
    system_prompt_path = BASE_DIR / "system_prompt.txt"
    if system_prompt_path.exists():
        return system_prompt_path.read_text(encoding="utf-8")
    return (
        "You are an expert social media manager for PIX Moving. "
        "Match PIX Moving's official corporate, engineering-focused, and thought-leadership visual & caption style."
    )

def load_dataset_content(db: Session | None = None) -> str:
    dataset_path = BASE_DIR / "pix_moving.txt"
    if dataset_path.exists():
        return dataset_path.read_text(encoding="utf-8")
    
    if db:
        item = db.query(Library).filter(
            (Library.media_type == "article") | (Library.type == "pix_moving website content")
        ).first()
        if item and item.image_url:
            try:
                resp = requests.get(item.image_url, timeout=10)
                if resp.status_code == 200:
                    return resp.text
            except Exception:
                pass
    return ""

def call_gemini_text_model(
    system_prompt: str,
    user_prompt: str,
    dataset_text: str,
    platform: str,
    language: str = "English (US)",
    tone: str = "Professional",
    image_bytes: bytes | None = None,
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")

    image_instruction_note = ""
    if image_bytes:
        image_instruction_note = "- An attached reference image is provided. Analyze this image carefully and incorporate its specific visual features, object details, and subject into the title, headline, subtitle, and caption!"

    full_instruction = f"""{system_prompt}

### RELEVANT PIX MOVING COMPANY DATASET CONTEXT:
{dataset_text[:4000]}

### USER TOPIC & PROMPT FOR {platform.upper()}:
{user_prompt}

### SPECIFIC POST CONSTRAINTS:
- Language: {language} (Write title, headline, subtitle, caption, and hashtags strictly in this language/dialect).
- Tone of Voice: {tone} (Adopt this tone throughout the title, headline, subtitle, and caption, e.g., professional, casual, informative, persuasive, corporate).
{image_instruction_note}

Please generate the response as a strict JSON object with six keys:
1. "title": A short, catchy 4 to 7 word professional title for this social media post (e.g. "PIX RoboBus Autonomous Mobility Launch").
2. "ai_safety_score": An integer score out of 100 (e.g. 98 or 95) representing AI brand safety, tone compliance, and content safety.
3. "headline": A short, punchy upper-case headline for the post concept written in {language} matching the {tone} tone.
4. "subtitle": A subtle 1-line subtitle explaining the concept in {language}.
5. "caption": A caption of approximately 120–180 words for {platform.title()} written in {language} with a {tone} tone.
6. "hashtags": An array of 5 to 8 relevant hashtags written in or relevant to {language}.

Return ONLY valid JSON without markdown code blocks.
"""

    parts: list[dict] = [{"text": full_instruction}]
    if image_bytes:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        parts.append({"inlineData": {"mimeType": "image/jpeg", "data": b64_img}})

    models_to_try = [
        "gemini-2.5-flash",
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-3.5-flash",
        "gemini-2.5-pro"
    ]

    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": parts
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    last_error = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        try:
            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code == 200:
                data = res.json()
                text_out = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if text_out.startswith("```"):
                    lines = text_out.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    text_out = "\n".join(lines).strip()
                
                parsed = json.loads(text_out)
                return parsed
            else:
                last_error = f"Model {model_name} returned {res.status_code}: {res.text}"
        except Exception as err:
            last_error = str(err)

    raise HTTPException(status_code=500, detail=f"Failed to generate post text via Gemini. Error: {last_error}")

def render_post_image_composition(
    headline: str = "",
    subtitle: str = "",
    reference_image_url: str = "",
    reference_image_bytes: bytes | None = None
) -> bytes:
    """Returns the clean reference image bytes without text or logo overlays."""
    if reference_image_bytes:
        return reference_image_bytes
    if reference_image_url:
        try:
            resp = requests.get(reference_image_url, timeout=15)
            if resp.status_code == 200:
                return resp.content
        except Exception:
            pass
    return b""

def generate_post_image_with_ai(
    headline: str = "",
    subtitle: str = "",
    reference_image_url: str = "",
    reference_image_bytes: bytes | None = None
) -> bytes:
    """Returns the clean reference image bytes without text or logo overlays."""
    return render_post_image_composition(
        headline=headline,
        subtitle=subtitle,
        reference_image_url=reference_image_url,
        reference_image_bytes=reference_image_bytes
    )

def generate_post_content(
    prompt: str,
    platform: str,
    db: Session,
    language: str = "English (US)",
    tone: str = "Professional",
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    uploaded_images: list[tuple[bytes, str]] | None = None,
) -> dict:
    platform_norm = platform.strip().lower()
    if platform_norm not in {"linkedin", "instagram", "x"}:
        platform_norm = "linkedin"

    system_prompt = load_system_prompt()
    dataset_text = load_dataset_content(db)

    # 1. Enhance User Prompt & Generate Text (Headline, Subtitle, Caption, Hashtags)
    enhanced_prompt = f"""Create a new PIX Moving {platform_norm.upper()} post around this topic:

**{prompt.upper()}**

Language: {language}
Tone: {tone}
Scheduled Date: {date or 'N/A'}
Window: {start_time or 'N/A'} - {end_time or 'N/A'}

Use the reference images.
Match the existing PIX Moving {platform_norm.title()} visual style.
The tone must strictly be: {tone}.
The language must strictly be: {language}.
Write a caption of ~120-180 words and 5-8 hashtags.
"""

    primary_image_bytes = None
    if uploaded_images and len(uploaded_images) > 0:
        primary_image_bytes = uploaded_images[0][0]

    ai_text_result = call_gemini_text_model(
        system_prompt=system_prompt,
        user_prompt=enhanced_prompt,
        dataset_text=dataset_text,
        platform=platform_norm,
        language=language,
        tone=tone,
        image_bytes=primary_image_bytes,
    )

    title = ai_text_result.get("title", prompt.title()).strip()
    raw_safety = ai_text_result.get("ai_safety_score", 98)
    try:
        ai_safety_score = int(str(raw_safety).replace("%", "").strip())
    except Exception:
        ai_safety_score = 98

    headline = ai_text_result.get("headline", prompt.upper()).strip()
    subtitle = ai_text_result.get("subtitle", "Where intelligence meets robotics, mobility, and the physical world.").strip()
    caption = ai_text_result.get("caption", "").strip()
    raw_hashtags = ai_text_result.get("hashtags", [])

    # Process hashtags
    hashtags_list = []
    if isinstance(raw_hashtags, list):
        for tag in raw_hashtags:
            t = str(tag).strip()
            if not t.startswith("#"):
                t = f"#{t}"
            hashtags_list.append(t)
    elif isinstance(raw_hashtags, str):
        for tag in raw_hashtags.split():
            t = tag.strip()
            if t:
                if not t.startswith("#"):
                    t = f"#{t}"
                hashtags_list.append(t)

    formatted_hashtags = " ".join(hashtags_list)
    full_post_text = f"{caption}\n\n{formatted_hashtags}".strip()

    # 2. Select Reference Image(s) (User Uploaded vs Library DB)
    selected_image = None
    reference_image_url = ""
    reference_image_id = None
    uploaded_urls: list[str] = []

    if uploaded_images:
        for idx, (img_bytes, filename) in enumerate(uploaded_images):
            fname = filename or f"uploaded_user_image_{idx}.jpg"
            uploaded_url = upload_library_asset(
                file_content=img_bytes,
                filename=fname,
                content_type="image/jpeg"
            )
            lib_item = Library(
                name=fname,
                type="User Uploaded Image",
                media_type="photo",
                image_url=uploaded_url,
                size=len(img_bytes)
            )
            db.add(lib_item)
            db.commit()
            db.refresh(lib_item)

            uploaded_urls.append(uploaded_url)
            if idx == 0:
                reference_image_url = uploaded_url
                reference_image_id = lib_item.id
    else:
        photo_items = db.query(Library).filter(Library.media_type == "photo").all()
        if photo_items:
            selected_image = random.choice(photo_items)
            reference_image_url = selected_image.image_url or ""
            reference_image_id = selected_image.id

    # 3. Clean Post Image (No text or logo overlay on top of the image)
    created_image_url = reference_image_url
    if not created_image_url and primary_image_bytes:
        created_image_url = upload_library_asset(
            file_content=primary_image_bytes,
            filename=f"post_{platform_norm}.jpg",
            content_type="image/jpeg"
        )

    # 5. Save Record in Database (GeneratedPost Table)
    db_post = GeneratedPost(
        prompt=prompt,
        platform=platform_norm,
        title=title,
        ai_safety_score=ai_safety_score,
        headline=headline,
        subtitle=subtitle,
        caption=caption,
        hashtags=formatted_hashtags,
        image_url=created_image_url,
        reference_image_id=reference_image_id,
        reference_image_url=reference_image_url,
        language=language,
        tone=tone,
        date=date,
        start_time=start_time,
        end_time=end_time,
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)

    return {
        "id": db_post.id,
        "platform": platform_norm,
        "original_prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "title": title,
        "ai_safety_score": ai_safety_score,
        "language": language,
        "tone": tone,
        "date": date,
        "start_time": start_time,
        "end_time": end_time,
        "headline": headline,
        "subtitle": subtitle,
        "caption": caption,
        "hashtags": hashtags_list,
        "formatted_hashtags": formatted_hashtags,
        "full_post_text": full_post_text,
        "image_url": created_image_url,
        "uploaded_image_urls": uploaded_urls,
        "reference_image": {
            "id": reference_image_id,
            "url": reference_image_url
        },
        "created_at": db_post.created_at.isoformat() if db_post.created_at else None
    }
