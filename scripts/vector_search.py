import os
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from typing import List, Dict, Any, Tuple
import time

# Load environment variables from .env file
load_dotenv()

def get_embedding(text: str, client: OpenAI, model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for a text using OpenAI API."""
    try:
        text = str(text).replace("\n", " ")
        
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        raise

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(v1) # type: ignore
    v2 = np.array(v2) # type: ignore
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    return dot_product / (norm_v1 * norm_v2)

def search_similar_vectors(
    query_embedding: List[float],
    embeddings_data: Dict[str, Dict[str, Any]],
    top_n: int = 5,
    threshold: float = 0.0
) -> List[Tuple[str, Dict[str, Any], float]]:
    """Search for similar vectors in embeddings data using cosine similarity.
    
    Args:
        query_embedding: The query vector to compare against
        embeddings_data: Dictionary containing embeddings data
        top_n: Number of top results to return
        threshold: Minimum similarity threshold
    
    Returns:
        List of tuples (id, data, similarity_score)
    """
    results = []
    
    for id_key, data in embeddings_data.items():
        # Skip entries without embeddings
        if "embedding" not in data:
            continue
        
        vector = data["embedding"]
        similarity = cosine_similarity(query_embedding, vector)
        
        if similarity >= threshold:
            results.append((id_key, data, similarity))
    
    # Sort by similarity score in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    
    # Return top N results
    return results[:top_n]

def format_result(result_id: str, data: Dict[str, Any], score: float) -> Dict[str, Any]:
    """Format a search result for display."""
    # Create a copy of the data without the embedding to make output cleaner
    formatted = {k: v for k, v in data.items() if k != "embedding"}
    formatted["id"] = result_id
    formatted["similarity_score"] = f"{score:.4f}"
    return formatted

def main():
    parser = argparse.ArgumentParser(description="Search for similar vectors by query using cosine similarity")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings JSON file")
    parser.add_argument("--query", type=str, required=True, help="Search query")
    parser.add_argument("--top_n", type=int, default=5, help="Number of top results to return")
    parser.add_argument("--threshold", type=float, default=0.0, help="Minimum similarity threshold (0.0 to 1.0)")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="OpenAI embedding model to use")
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load embeddings data
    print(f"Loading embeddings from {args.embeddings_file}...")
    try:
        with open(args.embeddings_file, "r") as f:
            embeddings_data = json.load(f)
    except Exception as e:
        print(f"Error loading embeddings file: {e}")
        return
    
    # Get embedding for query
    print(f"Generating embedding for query: '{args.query}'")
    start_time = time.time()
    query_embedding = get_embedding(args.query, client, args.model)
    
    # Search for similar vectors
    print("Searching for similar items...")
    results = search_similar_vectors(
        query_embedding,
        embeddings_data,
        top_n=args.top_n,
        threshold=args.threshold
    )
    end_time = time.time()
    
    # Display results
    print(f"\nFound {len(results)} results in {end_time - start_time:.2f} seconds:")
    print("="*80)
    
    for i, (id_key, data, score) in enumerate(results, 1):
        formatted = format_result(id_key, data, score)
        print(f"\n{i}. Match (Score: {formatted['similarity_score']})")
        print("-"*50)
        
        # Print text representation first if available
        if "text" in formatted:
            print(f"Text: {formatted['text']}")
            del formatted["text"]  # Remove from dictionary to avoid printing twice
        
        # Print other fields
        for key, value in formatted.items():
            if key != "similarity_score" and key != "id":  # Already printed these
                print(f"{key}: {value}")
        
    print("\n" + "="*80)

if __name__ == "__main__":
    main() 