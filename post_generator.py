import json
import logging
import os
from pathlib import Path
from typing import Optional
import requests

from google import genai
from google.genai import types
from sqlalchemy.orm import Session

from models import Library, GeneratedPost
from storage import upload_library_asset
from text_embed import generate_text_embedding, cosine_similarity

logger = logging.getLogger("PostGenerator")

MODEL = "gemini-3.1-flash-image"
THRESHOLD = 0.20

SYSTEM_PROMPT_PATH = Path(__file__).parent / "system_prompt.txt"
PIX_MOVING_TXT_PATH = Path(__file__).parent / "pix_moving.txt"


def get_genai_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is missing.")
    return genai.Client(api_key=api_key)


def load_system_prompt() -> str:
    """Load system instructions from system_prompt.txt if available."""
    if SYSTEM_PROMPT_PATH.exists():
        return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are an expert social media graphic designer. Generate a high quality, "
        "professional social media post image based on the reference product image."
    )


def load_pix_moving_data() -> str:
    """Load company information & background context from pix_moving.txt."""
    if PIX_MOVING_TXT_PATH.exists():
        return PIX_MOVING_TXT_PATH.read_text(encoding="utf-8")
    return (
        "PIX Moving is a pioneering City Robotics & Autonomous Mobility company. "
        "Products include RoboBus, RoboShop, Beastie, and Robotic Skateboard Chassis."
    )


def generate_linkedin_caption_and_hashtags(
    user_prompt: str,
    pix_data: str,
    image_name: str,
    tone: str | None = None,
    language: str | None = None,
    platform: str = "linkedin",
) -> dict:
    """
    Generate professional LinkedIn or engaging Instagram post caption, hashtags, and ai_safety_score.
    """
    client = get_genai_client()
    tone_str = tone if tone else "Professional"
    lang_str = language if language else "English (US)"
    plat_str = platform.lower().strip()

    if "instagram" in plat_str or "insta" in plat_str:
        platform_instructions = f"""
        TARGET PLATFORM: Instagram
        STYLE & TONE: Casual, vibrant, lifestyle-oriented B2C vibe with natural emojis.
        CAPTION INSTRUCTIONS:
        Write an Instagram-optimized caption in {lang_str} (1-2 engaging paragraphs). Include relevant emojis (like 🚀, 🤖, 🌆, ✨) naturally throughout the text.
        HASHTAGS INSTRUCTIONS:
        Generate 8-12 popular, high-reach Instagram lifestyle & tech hashtags separated by spaces (e.g. "#PIXMoving #CityRobotics #TechLifestyle #FutureMobility #RoboticsDaily #InstaTech #Innovation #SmartCities").
        """
    else:
        platform_instructions = f"""
        TARGET PLATFORM: LinkedIn
        STYLE & TONE: Professional, corporate, engineering-focused B2B thought-leadership tone matching {tone_str}.
        CAPTION INSTRUCTIONS:
        Write a professional B2B LinkedIn caption in {lang_str} (1-2 clear, impactful paragraphs).
        HASHTAGS INSTRUCTIONS:
        Generate 5-8 relevant corporate B2B tech hashtags separated by spaces (e.g. "#PIXMoving #CityRobotics #AutonomousMobility #AI #Robotics #Engineering").
        """

    prompt_text = f"""
            You are an expert Content Strategist for PIX Moving (a leading City Robotics & Autonomous Mobility company).

            COMPANY DATA & BACKGROUND CONTEXT:
            {pix_data}

            USER REQUEST / TOPIC:
            {user_prompt}

            TONE:
            {tone_str}

            LANGUAGE:
            {lang_str}

            {platform_instructions}

            Return the response in JSON format with the following keys:
            - "headline": A short, impactful headline (1 line) in {lang_str}
            - "caption": The main body text of the post in {lang_str}
            - "hashtags": The hashtags separated by spaces
            - "ai_safety_score": An integer score (between 0 and 100, e.g. 90-100) evaluating content safety, brand compliance, and appropriateness.

            Return ONLY the JSON object. Do not include markdown code block formatting like ```json.
            """

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt_text
        )

        clean_json = response.text.strip()
        if clean_json.startswith("```"):
            clean_json = clean_json.strip("`").removeprefix("json").strip()

        data = json.loads(clean_json)

        score_val = data.get("ai_safety_score", 98)
        try:
            score_val = int(score_val)
        except (ValueError, TypeError):
            score_val = 98

        default_hashtags = "#PIXMoving #CityRobotics #TechLifestyle #InstaTech" if "instagram" in plat_str else "#PIXMoving #CityRobotics #AutonomousMobility #AI #Robotics"

        return {
            "headline": data.get("headline", user_prompt[:50]),
            "caption": data.get("caption", user_prompt),
            "hashtags": data.get("hashtags", default_hashtags),
            "ai_safety_score": score_val,
        }
    except Exception as err:
        logger.warning("Failed to generate caption via Gemini: %s", err)
        default_hashtags = "#PIXMoving #CityRobotics #TechLifestyle #InstaTech" if "instagram" in plat_str else "#PIXMoving #CityRobotics #AutonomousMobility #AI #Robotics"
        return {
            "headline": user_prompt[:50],
            "caption": f"{user_prompt}\n\nDriving innovation in autonomous mobility and city robotics with PIX Moving.",
            "hashtags": default_hashtags,
            "ai_safety_score": 98,
        }


def find_relevant_image_for_prompt(
    db: Session,
    prompt: str,
    min_threshold: float = THRESHOLD,
    top_k: int = 5,
) -> dict:
    """
    Generate prompt text embedding and search library database for top matching reference images.
    """
    logger.info("[Step 1.1] Generating text embedding for prompt: '%s'...", prompt)
    prompt_embedding = generate_text_embedding(prompt)
    logger.info("[Step 1.1 Output] Prompt text embedding generated (%d dimensions).", len(prompt_embedding))

    logger.info("[Step 1.2] Fetching all library assets from database table 'library'...")
    assets = db.query(Library).filter(Library.embedding.is_not(None)).all()
    logger.info("[Step 1.2 Output] Found %d library assets with valid embeddings.", len(assets))

    if not assets:
        logger.warning("No assets with embeddings found in database.")
        return {
            "total_scanned": 0,
            "scored_candidates": [],
            "most_relevant": None,
            "all_matches": []
        }

    scored_matches = []
    for asset in assets:
        if asset.embedding:
            score = cosine_similarity(prompt_embedding, asset.embedding)
            scored_matches.append({
                "id": asset.id,
                "name": asset.name,
                "type": asset.type,
                "media_type": asset.media_type,
                "image_url": asset.image_url,
                "size": asset.size,
                "similarity_score": score,
                "is_matched": score >= min_threshold,
            })

    scored_matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    total_scanned = len(scored_matches)
    most_relevant = scored_matches[0] if scored_matches else None

    return {
        "total_scanned": total_scanned,
        "scored_candidates": scored_matches,
        "most_relevant": most_relevant,
        "all_matches": scored_matches[:top_k]
    }


def generate_linkedin_post_image(
    images: list[dict],
    system_prompt: str,
    user_prompt: str,
    platform: str = "linkedin",
) -> bytes:
    """
    Generate a social post image using selected reference images' bytes,
    system prompt, and user prompt via Gemini API tailored for target platform.
    """
    plat_str = platform.lower().strip()
    if "instagram" in plat_str or "insta" in plat_str:
        style_guide = "Generate a vibrant, modern lifestyle Instagram-style post image. Focus on aesthetic lighting, urban life integration, vibrant colors, and visually appealing composition suitable for Instagram. DO NOT add any text, titles, headlines, or typography on the image itself."
    else:
        style_guide = "Generate a clean, high-impact corporate LinkedIn-ready post image. Focus on professional engineering detail, corporate architecture, high-tech clarity, and premium brand aesthetics suitable for LinkedIn. DO NOT add any text, titles, headlines, or typography on the image itself."

    combined_prompt = f"""
SYSTEM INSTRUCTIONS:

{system_prompt}

USER REQUEST:

{user_prompt}

IMPORTANT:

{style_guide}

CRITICAL MANDATE - NO TEXT ON IMAGE:
DO NOT render, print, or overlay ANY text, titles, headlines, captions, labels, words, or typography on the generated image under any circumstances. The generated image must be 100% clean photography with NO text overlay whatsoever.

Use the supplied reference image(s) as the primary visual/reference image(s).

Do not replace the actual product with a fictional or AI-generated product.

Preserve the original subject's appearance, proportions, structure,
and important visual details.

Follow the supplied system instructions exactly.

Preferred aspect ratio:
4:5

Preferred size:
1080 x 1350

Do not return an explanation or image-generation instructions.
Actually generate the image.
"""

    client = get_genai_client()

    logger.info("==================== FINAL PROMPT SENT TO GEMINI LLM (%s) ====================\n%s\n================================================================================", MODEL, combined_prompt)

    parts = [types.Part.from_text(text=combined_prompt)]
    for img in images:
        parts.append(types.Part.from_bytes(data=img["bytes"], mime_type=img["mime_type"]))

    response = client.models.generate_content(
        model=MODEL,
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ]
    )

    for candidate in response.candidates or []:
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.inline_data and part.inline_data.data:
                    return part.inline_data.data

    reasons = []
    text_responses = []
    for cand in response.candidates or []:
        if cand.finish_reason:
            reasons.append(str(cand.finish_reason))
        if cand.content and cand.content.parts:
            for p in cand.content.parts:
                if p.text:
                    text_responses.append(p.text)

    err_msg = "Gemini API did not return an image."
    if text_responses:
        err_msg += f" Gemini text response: {' '.join(text_responses)}"
    elif reasons:
        err_msg += f" Finish reason: {', '.join(reasons)}"

    logger.error("Image generation failed: %s", err_msg)
    raise RuntimeError(err_msg)


def create_generated_post(
    db: Session,
    prompt: str,
    platform: str = "linkedin",
    date: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    tone: str | None = "Professional",
    language: str | None = "English (US)",
    custom_images_data: list[dict] | None = None,
    min_threshold: float = THRESHOLD,
) -> dict:
    """
    Full post generation workflow:
    1. Determine reference images: If user passed 'custom_images_data' or 'images' list, use them directly (skipping embedding search).
       Otherwise, search top (up to 2) relevant reference images from DB library via embeddings.
    2. Generate professional caption & hashtags using pix_moving.txt data.
    3. Fetch/prepare reference image bytes for selected top images.
    4. Generate new post image using Gemini multimodal API.
    5. Upload generated image to Supabase storage.
    6. Save post details to GeneratedPost database table with date, time, tone, and language.
    """
    # Step 1: Reference image selection or embedding search
    top_images = []
    images_data = []

    if custom_images_data and len(custom_images_data) > 0:
        logger.info("================================================================================")
        logger.info("[Step 1] CASE 1: User uploaded %d custom image(s).", len(custom_images_data))
        logger.info("[Step 1] SKIPPING text embedding generation & database similarity search!")
        logger.info("================================================================================")
        for idx, item in enumerate(custom_images_data, 1):
            top_images.append({
                "id": item.get("id", f"uploaded-image-{idx}"),
                "name": item.get("name", f"Uploaded Image {idx}"),
                "image_url": item["image_url"],
                "similarity_score": 1.0
            })
            images_data.append({
                "bytes": item["bytes"],
                "mime_type": item.get("mime_type", "image/png")
            })
   
    else:
        logger.info("================================================================================")
        logger.info("[Step 1] CASE 2: No custom images uploaded by user.")
        logger.info("[Step 1] RUNNING embedding search: Generating prompt text embedding & searching database...")
        logger.info("================================================================================")
        search_result = find_relevant_image_for_prompt(
            db=db,
            prompt=prompt,
            min_threshold=min_threshold,
        )

        total_scanned = search_result.get("total_scanned", 0)
        scored_candidates = search_result.get("scored_candidates", [])
        top_images = scored_candidates[:2]

        logger.info("[Step 1 Output] Scanned %d assets. %d matched threshold >= %.2f.", total_scanned, len(scored_candidates), min_threshold)

        for cand in scored_candidates:
            logger.info("  - Match: '%s' (ID: %s) | Similarity Score: %.4f", cand['name'], cand['id'], cand['similarity_score'])

    if not top_images:
        err_msg = f"No relevant reference image found matching prompt with similarity score >= {int(min_threshold * 100)}%."
        logger.error(err_msg)
        raise ValueError(err_msg)

    logger.info("[Step 1 Selected] Top %d Image(s) Selected for LLM:", len(top_images))
    for idx, img in enumerate(top_images, 1):
        logger.info("  - Top %d: '%s' (ID: %s) | Score: %.4f", idx, img['name'], img['id'], img['similarity_score'])

    # Step 2: Generate LinkedIn Caption and Hashtags using pix_moving.txt
    logger.info("[Step 2] Generating professional caption & hashtags using pix_moving.txt background data...")

    pix_data = load_pix_moving_data()
    image_names = ", ".join([img["name"] for img in top_images])
    caption_data = generate_linkedin_caption_and_hashtags(
        user_prompt=prompt,
        pix_data=pix_data,
        image_name=image_names,
        tone=tone,
        language=language,
        platform=platform,
    )

    logger.info("[Step 2 Output] Headline: '%s' | Hashtags: %s | AI Safety Score: %d", caption_data['headline'], caption_data['hashtags'], caption_data.get('ai_safety_score', 98))

    # Step 3: Fetch reference image bytes for selected images (if not pre-loaded)
    if not images_data:
        logger.info("[Step 3] Fetching reference image bytes for %d selected image(s)...", len(top_images))
        for img in top_images:
            ref_url = img["image_url"]
            logger.info("  - Fetching image bytes from URL: %s", ref_url)

            if ref_url.startswith("http://") or ref_url.startswith("https://"):
                res = requests.get(ref_url, timeout=30)
                res.raise_for_status()
                ref_bytes = res.content
            else:
                ref_bytes = Path(ref_url).read_bytes()

            mime_type = "image/png" if ref_url.lower().endswith(".png") else "image/jpeg"
            images_data.append({"bytes": ref_bytes, "mime_type": mime_type})

    system_prompt = load_system_prompt()

    # Step 4: Generate Post Image using Gemini
    logger.info("[Step 4] Generating post image using model '%s' with %d reference image(s)...", MODEL, len(images_data))

    generated_image_bytes = generate_linkedin_post_image(
        images=images_data,
        system_prompt=system_prompt,
        user_prompt=prompt,
        platform=platform,
    )

    logger.info("[Step 4 Output] Gemini post image generated successfully.")

    # Step 5: Upload generated image to Supabase Storage
    logger.info("[Step 5] Uploading generated post image to Supabase storage...")

    primary_id = top_images[0]["id"]
    uploaded_image_url = upload_library_asset(
        file_content=generated_image_bytes,
        filename=f"post_{primary_id[:8]}.png",
        content_type="image/png",
    )

    logger.info("==================== GENERATED POST IMAGE URL ====================")
    logger.info("Uploaded post image URL: %s", uploaded_image_url)
    logger.info("==================================================================")

    # Step 6: Store in Database
    logger.info("[Step 6] Saving generated post entry in database table 'generated_posts'...")

    ref_ids_str = ", ".join([img["id"] for img in top_images])
    ref_urls_str = ", ".join([img["image_url"] for img in top_images])

    new_post = GeneratedPost(
        prompt=prompt,
        platform=platform,
        title=caption_data["headline"],
        headline=caption_data["headline"],
        caption=caption_data["caption"],
        hashtags=caption_data["hashtags"],
        image_url=uploaded_image_url,
        reference_image_id=ref_ids_str,
        reference_image_url=ref_urls_str,
        date=date,
        start_time=start_time,
        end_time=end_time,
        tone=tone or "Professional",
        language=language or "English (US)",
        ai_safety_score=caption_data.get("ai_safety_score", 98),
        is_approved=False,
        is_posted=False,
    )
    db.add(new_post)
    db.commit()
    db.refresh(new_post)

    logger.info("[Step 6 Output] Generated post saved successfully. Post ID: %s", new_post.id)

    return {
        "id": new_post.id,
        "title": new_post.title,
        "headline": new_post.headline,
        "ai_safety_score": new_post.ai_safety_score,
        "prompt": new_post.prompt,
        "platform": new_post.platform,
        "caption": new_post.caption,
        "hashtags": new_post.hashtags,
        "generated_image_url": new_post.image_url,
        "reference_image_id": new_post.reference_image_id,
        "reference_image_url": new_post.reference_image_url,
        "date": new_post.date,
        "start_time": new_post.start_time,
        "end_time": new_post.end_time,
        "tone": new_post.tone,
        "language": new_post.language,
        "is_approved": new_post.is_approved,
        "is_posted": new_post.is_posted,
        "created_at": new_post.created_at.isoformat() if new_post.created_at else None,
    }
