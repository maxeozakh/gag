import openai
from openai import OpenAI
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.embeddings import get_embedding
from app.models.vector_search import find_similar_vectors
from app.models.database import database

from app.utils.helpers import get_env_variable

router = APIRouter()
client = OpenAI()


@router.get("/vectors_original/")
async def get_vectors_original():
    """

    Returns:
        list: A list of vectors original.
    """
    query = "SELECT original FROM vectors;"
    try:
        results = await database.fetch_all(query)
        if not results:
            return {"message": "No vectors found."}
        return {"vectors_original": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


@router.get("/answers/")
async def get_answers():
    """
    API endpoint to fetch all answers from the database.

    Returns:
        list: A list of answers.
    """
    query = "SELECT * FROM answers ORDER BY created_at DESC;"
    try:
        results = await database.fetch_all(query)
        if not results:
            return {"message": "No comments found."}
        return {"comments": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")


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
        "content": result["content"],
        "similarity": result["similarity"],
    }


class ChatPayload(BaseModel):
    query: str


client = OpenAI()


@router.post("/chat/")
async def chat(payload: ChatPayload):
    """
    Handles a user query by searching for similar vectors, generating a response,
    and storing the results in the database.
    """
    try:
        # Step 1: Embed the user query
        query_embedding = await get_embedding(payload.query)

        # Step 2: Search for similar vectors
        search_result = await find_similar_vectors(query_embedding)

        # Step 3: Build the OpenAI prompt
        if search_result:
            context = f"Relevant content: {search_result['content']}\n\n"
        else:
            context = "No relevant content found.\n\n"

        prompt = f"""You are a smart and stylistically consistent assistant. Your task is to respond to user queries or comments in a way that closely matches the tone, style, and language of the provided context. Whenever a relevant answer from a previous query is available, use its style, phrasing, and structure as a blueprint for your response.
                Instructions:
                1. Always prioritize consistency with the provided context's style and structure. Match the tone (e.g., formal, casual, humorous, technical).
                2. When relevant content from a prior query is provided, heavily incorporate its phrasing, sentence patterns, and stylistic choices into your response.
                3. If no prior context is available, generate a response that is clear, concise, and helpful.
                4. Avoid repeating content verbatim unless explicitly requested.

                Context for the user query:
                {context}

                User Query:
                {payload.query}

                Your Response:"""

        # Step 4: Call OpenAI API
        openai.api_key = get_env_variable("OPENAI_API_KEY")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": prompt,
                }
            ]
        )

        # Extract OpenAI's response content
        answer_content = response.choices[0].message.content

        # Step 5: Save the vector for the query
        vector_as_string = "[" + ",".join(map(str, query_embedding)) + "]"
        vector_query = f"""
        INSERT INTO vectors (vector, original)
        VALUES ('{vector_as_string}'::VECTOR, '{payload.query}')
        RETURNING id;
        """
        vector_result = await database.fetch_one(vector_query)
        if not vector_result:
            raise HTTPException(
                status_code=500, detail="Failed to save the query vector.")

        vector_id = vector_result["id"]

        # Step 6: Save the answer
        query = """
        INSERT INTO answers (content, vector_id)
        VALUES (:content, :vector_id)
        RETURNING id;
        """
        values = {"content": answer_content, "vector_id": vector_id}
        answer_result = await database.fetch_one(query, values)
        if not answer_result:
            raise HTTPException(
                status_code=500, detail="Failed to save the answer.")

        print('response...', payload.query, answer_content, context)
        return {
            "message": "Chat processed successfully.",
            "query": payload.query,
            "answer": answer_content,
            "similarity": search_result["similarity"] if search_result else None,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
