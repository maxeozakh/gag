import logging
from app.models.database import database

logging.basicConfig(level=logging.INFO)


async def find_similar_vectors(embedding, limit: int = 5, threshold: float = 0.6):
    """
    Find similar vectors in ecommerce data.
    
    Args:
        embedding: Vector to compare against
        limit: Maximum number of results to return
        threshold: Similarity threshold (0-1). Lower values are more similar.
    """
    try:
        embedding_string = "[" + ",".join(map(str, embedding)) + "]"
        
        query = f"""
        SELECT v.id, v.original as text, v.vector, p.description as content,
               (v.vector <=> '{embedding_string}'::vector) as similarity
        FROM ecom_vectors v
        LEFT JOIN ecom_products p ON v.id = p.vector_id
        WHERE (v.vector <=> '{embedding_string}'::vector) <= :threshold
        ORDER BY similarity ASC
        LIMIT :limit
        """
        
        results = await database.fetch_all(
            query,
            {
                "limit": limit,
                "threshold": threshold
            }
        )
        
        if not results:
            return None
            
        best_match = results[0]
        return {
            "content": best_match["content"],
            "similarity": float(best_match["similarity"])
        }
        
    except Exception as e:
        print(f"Error in vector search: {str(e)}")
        return None
