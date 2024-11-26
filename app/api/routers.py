from fastapi import APIRouter
from pydantic import BaseModel
from app.models.embeddings import get_embedding

router = APIRouter()


class QueryPayload(BaseModel):
    query: str


@router.post("/embed/")
async def embed_query(payload: QueryPayload):
    """
    API endpoint to process a query and return its vector embedding.

    Args:
        payload (QueryPayload): JSON body containing the user query.

    Returns:
        dict: The vector embedding of the query.
    """
    try:
        embedding = await get_embedding(payload.query)
        return {"query": payload.query, "embedding": embedding}
    except Exception as e:
        return {"error": str(e)}
