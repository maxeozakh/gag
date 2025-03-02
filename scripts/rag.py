import os
import argparse
import json
import sys
import time
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import subprocess

# Load environment variables from .env file
load_dotenv()

# Define the RAG prompt template based on the one from routers.py
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

def search_vectors(query: str, embeddings_file: str, top_n: int = 3, threshold: float = 0.0, model: str = "text-embedding-3-small") -> List[Dict[str, Any]]:
    """
    Search for similar vectors using the vector_search.py script.
    
    Args:
        query: The search query
        embeddings_file: Path to the embeddings file
        top_n: Number of top results to return
        threshold: Minimum similarity threshold
        model: OpenAI embedding model to use
        
    Returns:
        List of search results
    """
    try:
        print(f"Searching for content similar to query: '{query}'")
        
        # Build command to run vector_search.py
        cmd = [
            sys.executable,
            "scripts/vector_search.py",
            "--embeddings_file", embeddings_file,
            "--query", query,
            "--top_n", str(top_n),
            "--threshold", str(threshold),
            "--model", model
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error running vector search: {result.stderr}")
            return []
            
        # Parse the output to extract results
        # Note: This is a simplified approach - we're just going to parse the JSON output from stdout
        # The actual implementation might need to be adjusted based on the output format
        
        # Look for JSON content in the output
        output_lines = result.stdout.split('\n')
        results = []
        
        # Process each line and try to extract structured data
        for line in output_lines:
            if "Match (Score:" in line:
                # Start of a new result
                score_text = line.split("Score: ")[1].strip().strip(')')
                current_result: Dict[str, Any] = {"similarity_score": float(score_text)}
                results.append(current_result)
            elif ": " in line and not line.startswith("-") and not line.startswith("="):
                # Extract key-value pairs
                key, value = line.split(": ", 1)
                if results and key.strip():
                    results[-1][key.strip()] = value.strip()
                    
        return results
                
    except Exception as e:
        print(f"Error during vector search: {e}")
        return []
        
def call_openai(client: OpenAI, prompt: str, model: str = "gpt-4o") -> str:
    """
    Call the OpenAI API to generate a response.
    
    Args:
        client: OpenAI client
        prompt: The prompt to send to OpenAI
        model: The OpenAI model to use
        
    Returns:
        Generated response
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        if content is None:
            return "No response generated"
        return content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return f"Error generating response: {str(e)}"

def format_context(results: List[Dict[str, Any]]) -> str:
    """
    Format search results into a context string for the RAG prompt.
    
    Args:
        results: List of search results
        
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant information found."
        
    context_parts = []
    
    for i, result in enumerate(results, 1):
        # Extract the text content - could be in "text" field or other fields
        content = result.get("text", "")
        
        # If no text field, try to construct from other fields
        if not content:
            # Skip similarity score and id fields
            content_parts = []
            for key, value in result.items():
                if key not in ["similarity_score", "id", "embedding"]:
                    content_parts.append(f"{key}: {value}")
            content = " ".join(content_parts)
            
        # Add to context with separator
        if content:
            context_parts.append(f"[Result {i}] {content}")
    
    return "\n\n".join(context_parts)

def main():
    parser = argparse.ArgumentParser(description="Retrieval-Augmented Generation (RAG) using vector search and OpenAI")
    parser.add_argument("--query", type=str, required=True, help="The user query")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings file")
    parser.add_argument("--top_n", type=int, default=3, help="Number of top results to retrieve")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold (0.0 to 1.0)")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--llm_model", type=str, default="gpt-3.5-turbo", help="OpenAI LLM model")
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return 1
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Step 1: Search for similar vectors
    start_time = time.time()
    results = search_vectors(
        query=args.query,
        embeddings_file=args.embeddings_file,
        top_n=args.top_n,
        threshold=args.threshold,
        model=args.embed_model
    )
    search_time = time.time() - start_time
    
    # Step 2: Format the context
    context = format_context(results)
    
    # Print retrieval information
    print("\n=== RAG RETRIEVAL INFO ===")
    print(f"Query: {args.query}")
    print(f"Retrieved {len(results)} results in {search_time:.2f} seconds")
    print(f"Similarity scores: {[round(r.get('similarity_score', 0), 4) for r in results]}")
    print("========================\n")
    
    # Step 3: Generate response using OpenAI
    print("Generating response...")
    prompt = RAG_PROMPT.format(context=context, query=args.query)
    
    start_time = time.time()
    response = call_openai(client, prompt, args.llm_model)
    generation_time = time.time() - start_time
    
    # Step 4: Return the result
    print("\n=== RESPONSE ===")
    print(response)
    print("\n========================")
    print(f"Generated in {generation_time:.2f} seconds")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 