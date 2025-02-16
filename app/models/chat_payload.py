from pydantic import BaseModel
from typing import Optional, List, Literal


class ChatPayload(BaseModel):
    query: str
    chat_type: Literal["naive", "rag"] = "rag"  # Default to RAG
    ground_truth: Optional[str] = None
    key_facts: Optional[List[str]] = None
