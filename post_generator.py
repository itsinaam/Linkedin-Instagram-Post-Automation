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

def call_gemini_text_model(system_prompt: str, user_prompt: str, dataset_text: str, platform: str) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not configured.")

    full_instruction = f"""{system_prompt}

### RELEVANT PIX MOVING COMPANY DATASET CONTEXT:
{dataset_text[:4000]}

### USER TOPIC & PROMPT FOR {platform.upper()}:
{user_prompt}

Please generate the response as a strict JSON object with four keys:
1. "headline": A short, punchy upper-case headline for the post image (e.g. "THE FUTURE OF MOBILITY IS MODULAR" or "BRINGING AI INTO THE REAL WORLD").
2. "subtitle": A subtle 1-line subtitle explaining the concept (e.g. "Where intelligence meets robotics, mobility, and the physical world.").
3. "caption": A professional {platform.title()} caption of approximately 120–180 words following PIX Moving's tone (Professional, Intelligent, Engineering-focused, Innovative, B2B, Thought-leadership).
4. "hashtags": An array of 5 to 8 relevant, high-impact professional hashtags (e.g. ["#PIXMoving", "#AutonomousVehicles", "#ModularMobility", "#PhysicalAI", "#Robotics", "#SmartCities"]).

Return ONLY valid JSON without markdown code blocks.
"""

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
                "parts": [
                    {"text": full_instruction}
                ]
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
    headline: str,
    subtitle: str,
    reference_image_url: str
) -> bytes:
    """
    Renders a 4:5 format (1080x1350 px) social media post image matching the 
    exact visual layout of PIX Moving's LinkedIn brand style.
    """
    target_width, target_height = 1080, 1350
    
    # 1. Fetch reference image
    try:
        resp = requests.get(reference_image_url, timeout=15)
        ref_img = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception:
        ref_img = Image.new("RGBA", (target_width, target_height), (40, 44, 52, 255))

    # Scale reference image to fill canvas nicely
    ref_ratio = ref_img.width / ref_img.height
    canvas_ratio = target_width / target_height

    if ref_ratio > canvas_ratio:
        new_height = target_height
        new_width = int(new_height * ref_ratio)
    else:
        new_width = target_width
        new_height = int(new_width / ref_ratio)

    ref_resized = ref_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Crop to center
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    ref_cropped = ref_resized.crop((left, top, left + target_width, top + target_height))

    # Canvas
    canvas = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 255))
    canvas.paste(ref_cropped, (0, 0))

    # 2. Add Top Slate/Dark Overlay for Header Readability
    overlay = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
    draw_overlay = ImageDraw.Draw(overlay)
    
    # Gradient overlay for top 45% and bottom 20%
    for y in range(int(target_height * 0.5)):
        alpha = int(210 * (1 - (y / (target_height * 0.5)) ** 1.5))
        draw_overlay.line([(0, y), (target_width, y)], fill=(30, 38, 46, alpha))

    for y in range(int(target_height * 0.8), target_height):
        progress = (y - target_height * 0.8) / (target_height * 0.2)
        alpha = int(180 * (progress ** 1.2))
        draw_overlay.line([(0, y), (target_width, y)], fill=(20, 24, 28, alpha))

    canvas = Image.alpha_composite(canvas, overlay)
    draw = ImageDraw.Draw(canvas)

    # 3. Draw Typography (Headline & Subtitle)
    margin_x = 90
    margin_y = 110

    # Load default truetype fonts or fallback
    try:
        font_headline = ImageFont.truetype("arialbd.ttf", 64)
        font_subtitle = ImageFont.truetype("arial.ttf", 36)
        font_logo_bold = ImageFont.truetype("arialbd.ttf", 44)
        font_logo_sub = ImageFont.truetype("arial.ttf", 24)
    except Exception:
        font_headline = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_logo_bold = ImageFont.load_default()
        font_logo_sub = ImageFont.load_default()

    # Wrap Headline text
    words = headline.upper().split()
    lines = []
    current_line = []
    for word in words:
        current_line.append(word)
        test_str = " ".join(current_line)
        bbox = draw.textbbox((0, 0), test_str, font=font_headline)
        if bbox[2] > (target_width - 2 * margin_x):
            current_line.pop()
            lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    # Draw headline lines
    curr_y = margin_y
    for line in lines:
        # Subtle text drop shadow
        draw.text((margin_x + 2, curr_y + 2), line, fill=(0, 0, 0, 160), font=font_headline)
        draw.text((margin_x, curr_y), line, fill=(255, 255, 255, 255), font=font_headline)
        curr_y += 75

    curr_y += 25

    # Wrap Subtitle text
    sub_words = subtitle.split()
    sub_lines = []
    sub_curr = []
    for w in sub_words:
        sub_curr.append(w)
        test_s = " ".join(sub_curr)
        bbox = draw.textbbox((0, 0), test_s, font=font_subtitle)
        if bbox[2] > (target_width - 2 * margin_x):
            sub_curr.pop()
            sub_lines.append(" ".join(sub_curr))
            sub_curr = [w]
    if sub_curr:
        sub_lines.append(" ".join(sub_curr))

    for sub_l in sub_lines:
        draw.text((margin_x + 1, curr_y + 1), sub_l, fill=(0, 0, 0, 140), font=font_subtitle)
        draw.text((margin_x, curr_y), sub_l, fill=(230, 235, 240, 240), font=font_subtitle)
        curr_y += 48

    # 4. Draw Official PIX MOVING Logo in Bottom Right
    logo_x = target_width - 240
    logo_y = target_height - 120
    draw.text((logo_x + 1, logo_y + 1), "PIX", fill=(0, 0, 0, 180), font=font_logo_bold)
    draw.text((logo_x, logo_y), "PIX", fill=(255, 255, 255, 255), font=font_logo_bold)
    draw.text((logo_x + 1, logo_y + 45), "MOVING", fill=(0, 0, 0, 180), font=font_logo_sub)
    draw.text((logo_x, logo_y + 44), "MOVING", fill=(220, 225, 230, 230), font=font_logo_sub)

    # Output JPEG bytes
    buffer = io.BytesIO()
    canvas.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()

def generate_post_image_with_ai(
    headline: str,
    subtitle: str,
    reference_image_url: str
) -> bytes:
    """
    Attempts image generation using Nano Banana Pro / Gemini vision model.
    Falls back to precision rendering composition.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return render_post_image_composition(headline, subtitle, reference_image_url)

    try:
        ref_bytes = requests.get(reference_image_url, timeout=10).content
        b64_ref = base64.b64encode(ref_bytes).decode("utf-8")

        prompt_text = (
            f"Create a 4:5 social media post image (1080x1350) for PIX Moving based on this reference photo. "
            f"Add bold headline text '{headline}' in white at top, subtitle '{subtitle}', "
            f"and official PIX MOVING logo in bottom right corner."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt_text},
                        {"inlineData": {"mimeType": "image/jpeg", "data": b64_ref}}
                    ]
                }
            ]
        }

        models_to_try = [
            "gemini-3.1-flash-image",
            "nano-banana-pro-preview",
            "gemini-3-pro-image-preview",
            "gemini-2.5-flash-image"
        ]

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            res = requests.post(url, json=payload, timeout=25)
            if res.status_code == 200:
                data = res.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for part in parts:
                    if "inlineData" in part and part["inlineData"].get("data"):
                        return base64.b64decode(part["inlineData"]["data"])
    except Exception:
        pass

    # Fallback to precision composition
    return render_post_image_composition(headline, subtitle, reference_image_url)

def generate_post_content(
    prompt: str,
    platform: str,
    db: Session,
    image_id: str | None = None
) -> dict:
    platform_norm = platform.strip().lower()
    if platform_norm not in {"linkedin", "instagram", "x"}:
        platform_norm = "linkedin"

    system_prompt = load_system_prompt()
    dataset_text = load_dataset_content(db)

    # 1. Enhance User Prompt & Generate Text (Headline, Subtitle, Caption, Hashtags)
    enhanced_prompt = f"""Create a new PIX Moving {platform_norm.upper()} post around this topic:

**{prompt.upper()}**

Use the original PIX Moving images uploaded as reference.
Match the existing PIX Moving {platform_norm.title()} visual style.
The tone should be: Professional, Intelligent, Engineering-focused, Innovative, B2B, Thought-leadership.
Write a professional caption of ~120-180 words and 5-8 hashtags.
"""

    ai_text_result = call_gemini_text_model(
        system_prompt=system_prompt,
        user_prompt=enhanced_prompt,
        dataset_text=dataset_text,
        platform=platform_norm
    )

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

    # 2. Select Reference Image from Database
    selected_image = None
    if image_id:
        selected_image = db.get(Library, image_id)
        if not selected_image:
            raise HTTPException(status_code=404, detail=f"Image with id '{image_id}' not found in Library.")
    else:
        photo_items = db.query(Library).filter(Library.media_type == "photo").all()
        if photo_items:
            selected_image = random.choice(photo_items)

    reference_image_url = selected_image.image_url if selected_image else ""
    reference_image_id = selected_image.id if selected_image else None

    # 3. Generate 4:5 Social Media Post Image with Nano Banana Pro / Vision Model
    image_bytes = generate_post_image_with_ai(
        headline=headline,
        subtitle=subtitle,
        reference_image_url=reference_image_url
    )

    # 4. Upload Generated Image to Supabase Storage
    created_image_url = upload_library_asset(
        file_content=image_bytes,
        filename=f"post_{platform_norm}.jpg",
        content_type="image/jpeg"
    )

    # 5. Save Record in Database (GeneratedPost Table)
    db_post = GeneratedPost(
        prompt=prompt,
        platform=platform_norm,
        headline=headline,
        subtitle=subtitle,
        caption=caption,
        hashtags=formatted_hashtags,
        image_url=created_image_url,
        reference_image_id=reference_image_id,
        reference_image_url=reference_image_url
    )
    db.add(db_post)
    db.commit()
    db.refresh(db_post)

    return {
        "id": db_post.id,
        "platform": platform_norm,
        "original_prompt": prompt,
        "enhanced_prompt": enhanced_prompt,
        "headline": headline,
        "subtitle": subtitle,
        "caption": caption,
        "hashtags": hashtags_list,
        "formatted_hashtags": formatted_hashtags,
        "full_post_text": full_post_text,
        "image_url": created_image_url,
        "reference_image": {
            "id": reference_image_id,
            "url": reference_image_url
        },
        "created_at": db_post.created_at.isoformat() if db_post.created_at else None
    }
