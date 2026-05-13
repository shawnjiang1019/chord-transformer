from typing import Literal
from pydantic import BaseModel, Field


class ChordIn(BaseModel):
    root: str = Field(..., examples=["C", "C#", "Bb"])
    q: str = Field(..., examples=["maj", "min", "7", "maj7"])


class GenerateRequest(BaseModel):
    prompt: list[ChordIn]
    section: Literal["verse", "chorus", "bridge"] = "verse"
    key_root: str = "C"
    key_mode: Literal["major", "minor"] = "major"
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    top_k: int = Field(4, ge=1, le=50)
    num_chords: int = Field(8, ge=1, le=32)


class ChordOut(BaseModel):
    root: str
    q: str
    roman: str | None = None
