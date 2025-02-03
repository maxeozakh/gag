from langfuse.openai import openai  # OpenAI integration
from langfuse.decorators import observe

# Configure your OpenAI API key
from app.utils.helpers import get_env_variable
openai.api_key = get_env_variable("OPENAI_API_KEY")


@observe()
async def get_embedding(query: str, model: str = "text-embedding-3-small"):
    """
    Fetch vector embeddings for a given query using OpenAI API.

    Args:
        query (str): The user query to be vectorized.
        model (str): The OpenAI embedding model to use (default: text-embedding-ada-002).

    Returns:
        list: A 1D list representing the vector embedding of the query.
    """
    try:
        response = openai.embeddings.create(input=query, model=model)
        embedding = response.data[0].embedding
        return embedding
    except Exception as e:
        print(f"Error fetching embedding: {e}")
        raise
