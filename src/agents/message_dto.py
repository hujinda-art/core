from enum import Enum

from pydantic import BaseModel, Field


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class MessageDTO(BaseModel):
    role: Role
    content: str = Field(..., description="消息具体内容")
