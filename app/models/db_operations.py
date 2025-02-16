from app.models.database import database
from fastapi import HTTPException
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def save_vector(query: str, vector_embedding: List[float]) -> int:
    """
    Save a vector and its original query to the database.
    
    Args:
        query: Original query text
        vector_embedding: List of vector embeddings
        
    Returns:
        int: ID of the inserted vector
    """
    try:
        # Format vector string directly in the query
        vector_as_string = "[" + ",".join(map(str, vector_embedding)) + "]"
        vector_query = f"""
        INSERT INTO vectors (vector, original)
        VALUES ('{vector_as_string}'::vector, :original)
        RETURNING id;
        """
        
        values = {"original": query}
        
        result = await database.fetch_one(query=vector_query, values=values)
        if not result:
            logger.error("Failed to save vector: No result returned")
            raise HTTPException(status_code=500, detail="Failed to save the query vector.")
        
        logger.debug(f"Successfully saved vector with id: {result['id']}")
        return result["id"]
    except Exception as e:
        logger.error(f"Error saving vector: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save vector: {str(e)}")

async def save_answer(content: str, vector_id: int) -> int:
    """
    Save an answer associated with a vector to the database.
    
    Args:
        content: Answer content
        vector_id: ID of the associated vector
        
    Returns:
        int: ID of the inserted answer
    """
    try:
        query = """
        INSERT INTO answers (content, vector_id)
        VALUES (:content, :vector_id)
        RETURNING id;
        """
        values = {"content": content, "vector_id": vector_id}
        
        result = await database.fetch_one(query=query, values=values)
        if not result:
            logger.error("Failed to save answer: No result returned")
            raise HTTPException(status_code=500, detail="Failed to save the answer.")
        
        # logger.info(f"Successfully saved answer with id: {result['id']}")
        return result["id"]
    except Exception as e:
        logger.error(f"Error saving answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save answer: {str(e)}") 