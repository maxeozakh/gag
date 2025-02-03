from langfuse.openai import openai  # OpenAI integration
from langfuse.decorators import observe, langfuse_context
from fastapi import HTTPException
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.models.embeddings import get_embedding
from app.models.vector_search import find_similar_vectors
from app.models.database import database
from fastapi.responses import JSONResponse
from app.models.db_operations import save_vector, save_answer

from app.utils.helpers import get_env_variable

router = APIRouter()


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



async def categorize_query(query: str) -> str:
    """
    Categorizes the user query into predefined categories using LLM.
    Returns one of: "lore", "film", "director personality", "not relevant question"
    """
    prompt = """Categorize the following query into one of these exact categories:
    - lore
    - film
    - director personality
    - not relevant question

    Rules:
    - If the query is about story, characters, or world-building, categorize as "lore"
    - If the query is about cinematography, scenes, or movie production, categorize as "film"
    - If the query is about director's style, opinions, or personal life, categorize as "director personality"
    - If the query doesn't fit any of above categories, categorize as "not relevant question"

    Return ONLY the category name, nothing else.

    Query: {query}
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a precise categorization assistant."},
                {"role": "user", "content": prompt.format(query=query)}
            ]
        )
        category = response.choices[0].message.content.strip().lower()
        
        # Validate the category
        valid_categories = {"lore", "film", "director personality", "not relevant question"}
        return category if category in valid_categories else "not relevant question"
    except Exception as e:
        print(f"Categorization error: {str(e)}")
        return "not relevant question"

@router.post("/chat/")
@observe()
async def chat(payload: ChatPayload):
    """
    Handles a user query by searching for similar vectors, generating a response,
    and storing the results in the database.
    """
    try:
        # Step 1: Embed the user query
        query_embedding = await get_embedding(payload.query)
        
        # New Step: Categorize the query
        category = await categorize_query(payload.query)

        # Update the current trace with the category tag
        langfuse_context.update_current_trace(tags=[category])

        # Step 2: Search for similar vectors
        search_result = await find_similar_vectors(query_embedding)

        # Step 3: Build the OpenAI prompt
        if search_result:
            context = f"Relevant content: {search_result['content']}\n\n"
        else:
            context = "No relevant content found.\n\n"
            return {"answer": "Sorry we didn't find any similar queries so cannot mimic here"}

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
        response = openai.chat.completions.create(
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

        try:
            # Step 5: Save the vector for the query
            vector_id = await save_vector(payload.query, query_embedding)

            # Step 6: Save the answer
            answer_id = await save_answer(answer_content, vector_id)
        except Exception as db_error:
            # Fixed the formatting
            print(f"Database operation failed: {str(db_error)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save to database: {str(db_error)}"
            )

        print('response...', payload.query, answer_content, context)
        return {
            "message": "Chat processed successfully.",
            "query": payload.query,
            "answer": answer_content,
            "similarity": search_result["similarity"] if search_result else None,
        }

    except Exception as e:
        print(f"Chat endpoint error: {str(e)}")  # For debugging
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


@router.post("/reinit_db/")
async def reinit_db():
    """
    Endpoint to reinitialize the database using the init.sql file.
    """
    try:
        # Path to your init.sql file
        init_sql_path = os.path.join(
            os.path.dirname(__file__), "../../init.sql")

        # Check if the file exists
        if not os.path.exists(init_sql_path):
            raise HTTPException(
                status_code=404, detail="init.sql file not found.")

        # Read the SQL file
        with open(init_sql_path, "r") as sql_file:
            sql_commands = sql_file.read()

        # Split the commands by semicolon
        commands = [cmd.strip()
                    for cmd in sql_commands.split(";") if cmd.strip()]

        # Execute each command separately
        async with database.transaction():
            for command in commands:
                await database.execute(command)

        return JSONResponse(
            content={"message": "Database reinitialized successfully."}, status_code=200
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to reinitialize the database: {str(e)}"
        )
