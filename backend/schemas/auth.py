"""Authentication request and response schemas."""
from __future__ import annotations

import datetime
from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class TokenData(BaseModel):
    username: str | None = None
    role: str | None = None


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=6)
    role: str = Field("engineer", pattern="^(admin|engineer|viewer)$")


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime.datetime

    class Config:
        from_attributes = True


class LoginRequest(BaseModel):
    username: str
    password: str
