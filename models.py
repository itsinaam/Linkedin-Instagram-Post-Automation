"""Database models."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column

from database import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LinkedInToken(Base):
    """Stores LinkedIn OAuth access tokens."""
    __tablename__ = "linkedin_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_type: Mapped[str] = mapped_column(String(50), default="Bearer")
    expires_in: Mapped[int | None] = mapped_column(default=None)
    scope: Mapped[str | None] = mapped_column(String(500), default=None)
    person_id: Mapped[str | None] = mapped_column(String(200), default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )



class XToken(Base):
    """Stores OAuth 2.0 user tokens issued by X."""
    __tablename__ = "x_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, default=None)
    token_type: Mapped[str] = mapped_column(String(50), default="Bearer")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), default=None)
    scope: Mapped[str | None] = mapped_column(String(500), default=None)
    x_user_id: Mapped[str | None] = mapped_column(String(100), default=None)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class XOAuthState(Base):
    """Short-lived PKCE verifier storage for an X OAuth authorization attempt."""
    __tablename__ = "x_oauth_states"

    state: Mapped[str] = mapped_column(String(255), primary_key=True)
    code_verifier: Mapped[str] = mapped_column(String(255), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now
    )


class Library(Base):
    __tablename__ = "library"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    type: Mapped[str] = mapped_column(String(50), nullable=False)
    media_type: Mapped[str] = mapped_column(String(20), nullable=False, default="photo", server_default="photo")
    image_url: Mapped[str | None] = mapped_column(String(500), default=None)
    size: Mapped[int | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column( DateTime(timezone=True), default=_now )
    updated_at: Mapped[datetime] = mapped_column( DateTime(timezone=True), default=_now, onupdate=_now)


class GeneratedPost(Base):
    """Stores generated social media posts in the database."""
    __tablename__ = "generated_posts"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    platform: Mapped[str] = mapped_column(String(50), nullable=False, default="linkedin")
    headline: Mapped[str | None] = mapped_column(String(300), default=None)
    subtitle: Mapped[str | None] = mapped_column(String(500), default=None)
    caption: Mapped[str] = mapped_column(Text, nullable=False)
    hashtags: Mapped[str] = mapped_column(Text, nullable=False)
    image_url: Mapped[str | None] = mapped_column(String(500), default=None)
    reference_image_id: Mapped[str | None] = mapped_column(String(100), default=None)
    reference_image_url: Mapped[str | None] = mapped_column(String(500), default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)

