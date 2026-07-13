"""Incoming request shapes, validated by Pydantic before reaching the model."""

from typing import Literal

from pydantic import BaseModel, Field

from ..configs.config import settings


class GenerationParams(BaseModel):
    max_new_tokens: int = Field(default=settings.max_new_tokens, ge=1, le=4096)
    temperature: float = Field(default=settings.temperature, ge=0.0, le=2.0)
    top_p: float = Field(default=settings.top_p, ge=0.0, le=1.0)
    top_k: int = Field(default=settings.top_k, ge=0)
    repetition_penalty: float = Field(default=settings.repetition_penalty, ge=1.0, le=2.0)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    params: GenerationParams = GenerationParams()
    stream: bool = False


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)
    params: GenerationParams = GenerationParams()
    stream: bool = False
