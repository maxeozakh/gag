import os

from typing import Optional, List
from langfuse.openai import openai  # type: ignore
from langfuse.decorators import langfuse_context  # type: ignore
from langfuse import Langfuse  # type: ignore
from fastapi.responses import JSONResponse
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.models.embeddings import get_embedding
from app.models.db_operations import save_vector, save_answer
from app.models.vector_search import find_similar_vectors
from app.models.database import database
from app.utils.helpers import get_env_variable
from app.evaluation.token_metrics import TokenMetrics
from app.models import ChatPayload  # Add this import
from app.api.metrics import EnhancedKeyFactsValidator

router = APIRouter()
langfuse = Langfuse()

RAG_PROMPT = """You are a precise and knowledgeable assistant specializing in multi-domain information retrieval. Your goal is to provide accurate, well-structured responses based on the retrieved context while maintaining consistency in style and format.

Instructions:
1. ACCURACY FIRST: Always prioritize factual accuracy. Base your response primarily on the provided context.
2. DOMAIN ADAPTATION: Adjust your response style to match the domain (e.g., technical for product specs, conversational for customer service).
3. STRUCTURE:
   - Start with the most relevant information
   - Use clear sections when appropriate
   - Include specific details and measurements when available
4. HONESTY: If the context doesn't fully address the query, acknowledge limitations while providing available information.
5. KEY FACTS: Ensure all numerical values, specifications, and critical details from the context are preserved accurately.
6. STYLE MATCHING: Mirror the terminology and tone of the provided context while maintaining clarity.

Retrieved Context:
{context}

User Query:
{query}

Format your response to:
1. Address the query directly
2. Include specific details from the context
3. Maintain consistent terminology
4. Acknowledge any information gaps

Your Response:"""

NAIVE_PROMPT = """You are a helpful assistant. Please provide a clear and concise response to the following query.

User Query:
{query}

Your Response:"""


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


# its garbage but fine for now
class TraceContext:
    """Simple class to store the current trace context"""

    def __init__(self):
        self.current_trace_id = None

    def set_trace_id(self, trace_id: str):
        self.current_trace_id = trace_id

    def get_trace_id(self) -> str:
        return self.current_trace_id


# Create a global instance
trace_context = TraceContext()


async def categorize_query(query: str) -> str:
    """
    Categorizes the user query into predefined categories using LLM.
    Returns one of: "lore", "film", "director personality", "not relevant question"
    """
    prompt = """Categorize the following query into one of these exact categories:
    - lore
    - film
    - director_personality
    - not_relevant_question

    Rules:
    - If the query is about story, characters, or world-building, categorize as "lore"
    - If the query is about cinematography, scenes, or movie production, categorize as "film"
    - If the query is about director's style, opinions, or personal life, categorize as "director_personality"
    - If the query doesn't fit any of above categories, categorize as "not_relevant_question"

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
        valid_categories = {"lore", "film",
                            "director personality", "not relevant question"}
        return category if category in valid_categories else "not relevant question"
    except Exception as e:
        print(f"Categorization error: {str(e)}")
        return "not relevant question"

# Initialize TokenMetrics at module level
token_metrics = TokenMetrics()


async def handle_evaluation(answer_content: str, payload: ChatPayload, trace_id: str, is_evaluation: bool = False) -> dict:
    """Handle evaluation logic for both naive and RAG chat endpoints."""
    response_data: dict = {
        "answer": answer_content,
        "trace_id": trace_id
    }

    if payload.ground_truth:
        metrics = token_metrics.calculate_f1(
            prediction=answer_content,
            reference=payload.ground_truth
        )
        

        response_data["evaluation"] = dict({
            "f1_score": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"]
        })

        if payload.key_facts:
            validator = EnhancedKeyFactsValidator()
            key_facts_results = validator.validate_key_facts(
                prediction=answer_content,
                key_facts=payload.key_facts
            )
            response_data["evaluation"]["key_facts"] = key_facts_results
    elif not is_evaluation:
        print("⚠️ No ground truth available for evaluation")

    return response_data


async def save_chat_data(query: str, answer: str, embedding: Optional[List[float]] = None):
    """Handle database operations for both chat endpoints."""
    try:
        # Use zero vector if no embedding provided
        vector = embedding if embedding else [0.0] * 1536
        vector_id = await save_vector(query, vector)
        answer_id = await save_answer(answer, vector_id)
        return vector_id, answer_id
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save to database: {str(e)}"
        )


async def handle_trace_context(current_trace: Optional[str], category: str, tags: Optional[List[str]] = []) -> str:
    """Handle trace context and tagging for both endpoints."""
    if current_trace:
        trace_context.set_trace_id(current_trace)
        all_tags = [category]
        if tags:
            all_tags.extend(tags)
        langfuse_context.update_current_trace(tags=all_tags)
    return trace_context.get_trace_id()


@router.post("/rag_chat/")
async def rag_chat(payload: ChatPayload):
    """RAG-enhanced chat endpoint that uses vector search and retrieval."""
    try:
        print("-----------------------------------------------------")
        print(f"RAG chat endpoint called with query: {payload.query}")
        
        # Handle trace context and categorization
        current_trace = langfuse_context.get_current_trace_id()
        category = "other"
        trace_id = await handle_trace_context(current_trace, category)

        # Get embeddings and search for similar vectors
        query_embedding = await get_embedding(payload.query)
        search_result = await find_similar_vectors(query_embedding)

        print("\n=== RAG RETRIEVAL INFO ===")
        print(f"Query: {payload.query}")
        print(f"Retrieved content: {search_result['content'][:50] if search_result else 'No content retrieved'}")
        print(f"Similarity score: {search_result['similarity'] if search_result else 'N/A'}")
        print("========================\n")

        # Handle context and early return for no results
        if search_result:
            context = f"Relevant content: {search_result['content']}\n\n"
        else:
            await handle_trace_context(current_trace, category, ['no_similar_embeds'])
            return {"answer": "I'm not sure about that", "trace_id": trace_id}

        # Call OpenAI API
        openai.api_key = get_env_variable("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": RAG_PROMPT.format(
                    context=context,
                    query=payload.query
                )}
            ]
        )

        answer_content = response.choices[0].message.content

        # Handle evaluation and prepare response
        response_data = await handle_evaluation(answer_content, payload, trace_id)

        # Save to database
        await save_chat_data(payload.query, answer_content, query_embedding)

        return response_data

    except Exception as e:
        print(f"RAG chat endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred: {str(e)}"
        )


@router.post("/naive_chat/")
async def naive_chat(payload: ChatPayload):
    """Naive chat endpoint that uses only the LLM without retrieval."""
    try:
        print("-----------------------------------------------------")
        print(f"Naive chat endpoint called with query: {payload.query}")
        # Handle trace context and categorization
        current_trace = langfuse_context.get_current_trace_id()
        # category = await categorize_query(payload.query)
        category = "other"
        trace_id = await handle_trace_context(current_trace, category)

        # Call OpenAI API
        openai.api_key = get_env_variable("OPENAI_API_KEY")
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": NAIVE_PROMPT.format(
                    query=payload.query)}
            ]
        )

        answer_content = response.choices[0].message.content

        # Handle evaluation and prepare response
        response_data = await handle_evaluation(answer_content, payload, trace_id)

        # Save to database with zero vector
        await save_chat_data(payload.query, answer_content)

        return response_data

    except Exception as e:
        print(f"Naive chat endpoint error: {str(e)}")
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


class FeedbackPayload(BaseModel):
    score: float  # Only need score from the frontend now


@router.post("/feedback/")
async def submit_feedback(payload: FeedbackPayload):
    """
    API endpoint to collect user feedback scores for responses.
    """
    try:
        if not 0 <= payload.score <= 1:
            raise HTTPException(
                status_code=400,
                detail="Score must be between 0 and 1"
            )

        # Get trace_id from our local context
        trace_id = trace_context.get_trace_id()
        if not trace_id:
            raise HTTPException(
                status_code=400,
                detail="No active trace found"
            )

        langfuse.score(
            trace_id=trace_id,
            data_type="BOOLEAN",  # required for boolean scores
            name="user_feedback_bool",
            value=payload.score
        )

        return {
            "message": "Feedback submitted successfully",
            "trace_id": trace_id,
            "score": payload.score
        }

    except Exception as e:
        print(f"Feedback submission error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )


def safe_calculate_metrics(token_metrics, prediction, reference):
    try:
        return token_metrics.calculate_f1(prediction=prediction, reference=reference)
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }


@router.get("/ecom_products/")
async def get_ecom_products():
    """
    API endpoint to fetch all ecommerce products from the database.
    """
    query = """
    SELECT p.id, p.description, v.original as text
    FROM ecom_products p
    JOIN ecom_vectors v ON p.vector_id = v.id
    ORDER BY p.id;
    """
    try:
        results = await database.fetch_all(query)
        if not results:
            return {"message": "No products found."}
        return {"products": results}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"An error occurred: {str(e)}")
