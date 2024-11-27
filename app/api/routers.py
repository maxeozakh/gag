from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.embeddings import get_embedding
from app.models.vector_search import find_similar_vectors
from app.models.database import database

router = APIRouter()


async def vectorize_query(query: str):
    """
    Reusable function to vectorize a query.
    """
    try:
        return await get_embedding(query)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during embedding: {str(e)}")


class EmbedPayload(BaseModel):
    query: str


@router.post("/embed/")
async def embed_query(payload: EmbedPayload):
    """
    API endpoint to process a query and return its vector embedding.
    """
    embedding = await vectorize_query(payload.query)
    return {"query": payload.query, "embedding": embedding}


class SearchPayload(BaseModel):
    query: str


@router.post("/search/")
async def search_query(payload: SearchPayload):
    """
    API endpoint to search for similar vectors and return relevant data.
    """
    embedding = await vectorize_query(payload.query)

    result = await find_similar_vectors(embedding)

    if result is None:
        return {"message": "No relevant data found.", "placeholder": True}

    return {
        "message": "Relevant data found.",
        "username": result["username"],
        "content": result["content"],
        "metadata": result["metadata"],
        "similarity": result["similarity"],
    }


class EmbedAndSavePayload(BaseModel):
    ig_data_id: int  # ID of the related IG data
    content: str     # Content to embed and save


@router.post("/embed_and_save/")
async def embed_and_save(payload: EmbedAndSavePayload):
    """
    API endpoint to create an embedding from content and save it to the vectors table.

    Args:
        payload (EmbedAndSavePayload): JSON body containing ig_data_id and content.

    Returns:
        dict: Confirmation of saved vector with metadata.
    """
    try:
        embedding = await get_embedding(payload.content)

        # Convert embedding to a PostgreSQL-compatible format
        vector_as_string = "[" + ",".join(map(str, embedding)) + "]"

        query = f"""
        INSERT INTO vectors (ig_data_id, vector)
        VALUES ({payload.ig_data_id}, '{vector_as_string}'::VECTOR)
        RETURNING id, ig_data_id, created_at;
        """
        result = await database.fetch_one(query)

        if result is None:
            raise HTTPException(
                status_code=500, detail="Failed to save the vector to the database."
            )

        return {"message": "Vector saved successfully.", "vector_data": result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}"
        )
