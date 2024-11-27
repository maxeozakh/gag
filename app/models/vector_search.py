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
    SELECT v.id, v.answers_id, i.content, i.metadata, v.vector <=> '{vector_as_string}'::VECTOR AS similarity
    FROM vectors v
    JOIN answers i ON v.answers_id = i.id
    WHERE v.vector <=> '{vector_as_string}'::VECTOR < {threshold}
    ORDER BY similarity
    LIMIT {limit};
    """

    result = await database.fetch_one(query)
    return result
