from pydantic import BaseModel
from typing import Optional, List

class ChatPayload(BaseModel):
    query: str
    ground_truth: Optional[str] = None
    key_facts: Optional[List[str]] = None 