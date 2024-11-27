import logging
from app.models.database import database

logging.basicConfig(level=logging.INFO)


async def find_similar_vectors(query_embedding, threshold=0.4, limit=1):
    """
    Search the database for vectors similar to the query embedding.
    """
    # Convert the Python list to a PostgreSQL-compatible array string
    vector_as_string = "[" + ",".join(map(str, query_embedding)) + "]"

    # Inline the vector directly into the query string
    query = f"""
    SELECT v.id AS vector_id, a.id AS answer_id, a.content, v.vector <=> '{vector_as_string}'::VECTOR AS similarity
    FROM vectors v
    JOIN answers a ON a.vector_id = v.id
    WHERE v.vector <=> '{vector_as_string}'::VECTOR < {threshold}
    ORDER BY similarity
    LIMIT {limit};
    """

    try:
        result = await database.fetch_one(query)
        return result
    except Exception as e:
        logging.error(f"Error in find_similar_vectors: {e}")
        return None
