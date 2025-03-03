import os
import sys
import json
import time
import argparse
import subprocess
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import importlib.util
from openai import OpenAI
import traceback

# Try to import datasets for dataset creation
try:
    import datasets
    from datasets import Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")
    # Define a dummy Dataset class as a fallback
    class HFDataset:
        @classmethod
        def from_dict(cls, data):
            return data

# Try to import RAGAS with proper error handling
try:
    # Import the latest RAGAS components
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_recall
    )
    
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError as e:
    print(f"Error importing RAGAS: {e}")
    print("Please install RAGAS with: pip install ragas")
    RAGAS_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

def check_rag_script(script_path: str) -> bool:
    """
    Check if the specified RAG script exists.
    
    Args:
        script_path: Path to the RAG script
        
    Returns:
        True if the script exists, False otherwise
    """
    # Check if the file exists
    return os.path.exists(script_path) and os.path.isfile(script_path)

def run_rag_script(script_path: str, query: str, embeddings_file: str, llm_model: str) -> Tuple[str, Dict[str, Any], float]:
    """
    Run the specified RAG script with the given query and return the response.
    
    Args:
        script_path: Path to the RAG script
        query: The user query
        embeddings_file: Path to the embeddings file
        llm_model: LLM model to use
        
    Returns:
        Tuple of (response, retrieved_context, response_time)
    """
    start_time = time.time()
    
    # Setup base command
    cmd = [
        sys.executable,
        script_path,
        "--query", query,
        "--embeddings_file", embeddings_file,
        "--llm_model", llm_model
    ]
    
    # Check if script is openai-based-rag.py and adjust parameters if needed
    if "openai-based-rag.py" in script_path:
        # Replace --embeddings_file with --input_file for openai-based-rag.py
        cmd = [c if c != "--embeddings_file" else "--input_file" for c in cmd]
        
    # Run the command and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check for errors
    if result.returncode != 0:
        print(f"Error running RAG script: {result.stderr}")
        return "Error: Failed to run RAG script", {}, 0.0
    
    # Parse the JSON output
    try:
        output_data = json.loads(result.stdout.strip())
        
        # Check if there was an error in the RAG script
        if "error" in output_data:
            print(f"Error in RAG script: {output_data['error']}")
            return f"Error: {output_data['error']}", {}, 0.0
        
        # Extract response and context
        response = output_data.get("response", "")
        
        # Prepare context info for RAGAS
        context_info = {
            "context_text": output_data.get("context_text", []),
            "similarity_scores": output_data.get("similarity_scores", []),
            "metadata": output_data.get("metadata", []),
            "search_time": output_data.get("search_time", 0),
            "generation_time": output_data.get("generation_time", 0),
            "total_time": output_data.get("total_time", 0)
        }
        
        total_time = output_data.get("total_time", time.time() - start_time)
        
        # Log successful response
        resp_len = len(response)
        print(f"Successfully processed response (length: {resp_len} chars) with {len(context_info['context_text'])} context items")
        
        return response, context_info, total_time
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON output from RAG script: {e}")
        print("RAG script output:", result.stdout)
        return "Error: Failed to parse JSON output", {}, 0.0
    except Exception as e:
        print(f"Unexpected error processing RAG script output: {e}")
        print("RAG script output:", result.stdout)
        return f"Error: {str(e)}", {}, 0.0

def load_qa_pairs(qa_file_path: str) -> List[Dict[str, Any]]:
    """
    Load question-answer pairs from a file.
    
    Args:
        qa_file_path: Path to the file containing question-answer pairs
        
    Returns:
        List of dictionaries containing question-answer pairs
    """
    try:
        valid_pairs = []
        
        # Load based on file extension
        if qa_file_path.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(qa_file_path)
            
            # Ensure required columns exist
            required_columns = ['question', 'answer']
            if not all(col in df.columns for col in required_columns):
                print(f"Error: CSV file must contain columns: {required_columns}")
                return []
            
            # Convert DataFrame to list of dictionaries
            for _, row in df.iterrows():
                qa_pair = {str(k): v for k, v in row.items()}  # Ensure keys are strings
                if 'question' in qa_pair and 'answer' in qa_pair:
                    valid_pairs.append(qa_pair)
        
        elif qa_file_path.endswith('.json'):
            # Read JSON file
            with open(qa_file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            if isinstance(data, list):
                # List of QA pairs
                for item in data:
                    if isinstance(item, dict) and 'question' in item and 'answer' in item:
                        # Ensure all keys are strings
                        qa_pair = {str(k): v for k, v in item.items()}
                        valid_pairs.append(qa_pair)
            elif isinstance(data, dict):
                # Dictionary with questions as keys and answers as values
                for q, a in data.items():
                    valid_pairs.append({'question': str(q), 'answer': str(a)})
        else:
            print(f"Unsupported file format: {qa_file_path}")
            return []
        
        print(f"Loaded {len(valid_pairs)} valid QA pairs from {qa_file_path}")
        return valid_pairs
        
    except Exception as e:
        print(f"Error loading QA pairs: {str(e)}")
        return []

def prepare_ragas_dataset(
    qa_pairs: List[Dict[str, Any]], 
    rag_responses: List[str],
    contexts: List[Dict[str, Any]]
) -> Optional[Any]:
    """
    Prepare dataset for RAGAS evaluation.
    
    Args:
        qa_pairs: List of QA pairs
        rag_responses: List of RAG responses
        contexts: List of context information from RAG system
        
    Returns:
        RAGAS dataset if successful, None otherwise
    """
    try:
        if not DATASETS_AVAILABLE:
            print("Error: datasets library not available for RAGAS evaluation")
            return None
    
        # Make sure we have the same number of elements
        if not (len(qa_pairs) == len(rag_responses) == len(contexts)):
            print(f"Error: Mismatched lengths - QA pairs: {len(qa_pairs)}, "
                  f"RAG responses: {len(rag_responses)}, Contexts: {len(contexts)}")
            return None
            
        # Prepare data in the format required by RAGAS
        data = {
            "question": [],
            "ground_truths": [],
            "answer": [],
            "contexts": []
        }
        
        for qa_pair, response, context_info in zip(qa_pairs, rag_responses, contexts):
            # Extract question and ground truth
            question = qa_pair.get("question", "")
            ground_truth = qa_pair.get("answer", "")
            
            # Skip if missing data
            if not question or not ground_truth:
                continue
                
            # Get context text from the context info dictionary
            context_text = context_info.get("context_text", [])
            if not context_text:
                # If no context was found, add an empty list to avoid errors in RAGAS
                context_text = ["No relevant context found"]
            
            # Add to dataset
            data["question"].append(question)
            data["ground_truths"].append([ground_truth])  # RAGAS expects a list of ground truths
            data["answer"].append(response)
            data["contexts"].append(context_text)
        
        # Make sure we have at least one valid data point
        if not data["question"]:
            print("Error: No valid data points for RAGAS evaluation")
            return None
            
        # Create dataset
        dataset = HFDataset.from_dict(data)
        return dataset
        
    except Exception as e:
        print(f"Error preparing RAGAS dataset: {e}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return None

def evaluate_with_ragas(
    eval_dataset: Any,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate the RAG system using RAGAS metrics.
    
    Args:
        eval_dataset: Dataset prepared for RAGAS
        model: Model to use for evaluation
        api_key: OpenAI API key
        
    Returns:
        Dictionary of evaluation metrics
    """
    if not RAGAS_AVAILABLE:
        return {"error": 0.0}
        
    print("\nEvaluating with RAGAS...")
    
    # Set up OpenAI client for RAGAS
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    # Initialize metrics according to latest RAGAS API
    metrics = [
        faithfulness,
        answer_relevancy,
        context_recall
    ]
    
    # Run evaluation
    try:
        # Direct evaluation with RAGAS using the simplest possible call
        result = evaluate(
            eval_dataset,
            metrics=metrics
        )
        
        # Convert result to dictionary based on its type
        scores = {}
        
        # Known non-metric columns to ignore
        non_metric_columns = [
            'question', 'answer', 'contexts', 'ground_truth', 
            'user_input', 'retrieved_contexts', 'response', 'reference'
        ]
        
        # Handle different result types from different RAGAS versions
        if hasattr(result, "to_pandas"):
            # For newer RAGAS versions that return a DataFrame-like object
            result_df = result.to_pandas()
            
            # Process only numeric metric columns
            for col in result_df.columns:
                if col not in non_metric_columns:
                    try:
                        # First check if the column is numeric
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            value = result_df[col].mean()
                            if pd.notna(value):  # Only add if it's not NaN
                                scores[col] = float(value)
                    except Exception:
                        pass
            
        elif hasattr(result, "items"):
            # For some RAGAS versions that return a dictionary
            for k, v in result.items():
                if k not in non_metric_columns:
                    try:
                        if isinstance(v, (int, float, np.number)):
                            scores[k] = float(v)
                        elif isinstance(v, (list, np.ndarray)) and all(isinstance(x, (int, float, np.number)) for x in v):
                            scores[k] = float(np.mean(v))
                    except (ValueError, TypeError):
                        pass
                    
        else:
            # For the latest RAGAS that might return a custom object
            for attr_name in dir(result):
                if attr_name.startswith('_') or callable(getattr(result, attr_name)) or attr_name in non_metric_columns:
                    continue
                
                try:
                    attr_value = getattr(result, attr_name)
                    if isinstance(attr_value, (int, float, np.number)):
                        scores[attr_name] = float(attr_value)
                    elif hasattr(attr_value, 'mean') and callable(attr_value.mean):
                        # Check if the mean result is numeric
                        mean_value = attr_value.mean()
                        if isinstance(mean_value, (int, float, np.number)):
                            scores[attr_name] = float(mean_value)
                except Exception:
                    pass
        
        # If we still have no scores but the object is a DataFrame, try extracting just the known metric columns
        if not scores and hasattr(result, "to_pandas"):
            df = result.to_pandas()
            known_metrics = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']
            for metric in known_metrics:
                if metric in df.columns:
                    try:
                        # Get only numeric values if any
                        numeric_values = pd.to_numeric(df[metric], errors='coerce').dropna()
                        if not numeric_values.empty:
                            scores[metric] = float(numeric_values.mean())
                    except Exception:
                        pass
        
        return scores
            
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        traceback_str = traceback.format_exc()
        print(f"Traceback: {traceback_str}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG systems using RAGAS framework")
    parser.add_argument("--qa_file", type=str, required=True, help="Path to the QA pairs file (CSV or JSON)")
    parser.add_argument("--embeddings_file", type=str, required=True, help="Path to the embeddings file")
    parser.add_argument("--rag_script", type=str, default="scripts/rag.py", help="Path to the RAG script to evaluate")
    parser.add_argument("--llm_model", type=str, default="gpt-4o-mini", help="OpenAI model for generation")
    parser.add_argument("--eval_model", type=str, default="gpt-4o-mini", help="OpenAI model for evaluation")
    parser.add_argument("--sample_size", type=int, default=0, help="Number of QA pairs to sample (0 for all)")
    parser.add_argument("--output_file", type=str, help="Path to save evaluation results")
    args = parser.parse_args()
    
    # Check if the RAG script exists
    if not check_rag_script(args.rag_script):
        print(f"Error: RAG script '{args.rag_script}' not found.")
        return 1
        
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.qa_file)
    
    # Sample QA pairs if specified
    if args.sample_size > 0 and args.sample_size < len(qa_pairs):
        import random
        random.shuffle(qa_pairs)
        qa_pairs = qa_pairs[:args.sample_size]
        print(f"Sampled {args.sample_size} QA pairs for evaluation")
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        return 1
        
    # Run RAG script for each QA pair
    print(f"Evaluating RAG script: {args.rag_script}")
    print(f"Using LLM model: {args.llm_model}")
    print(f"Using {len(qa_pairs)} QA pairs")
    
    rag_responses = []
    context_infos = []
    response_times = []
    
    for i, qa_pair in enumerate(tqdm(qa_pairs, desc="Generating RAG responses")):
        query = qa_pair["question"]
        
        # Run the RAG script
        response, context_info, response_time = run_rag_script(
            args.rag_script, 
            query, 
            args.embeddings_file,
            args.llm_model
        )
        
        rag_responses.append(response)
        context_infos.append(context_info)
        response_times.append(response_time)
    
    # Output results before RAGAS evaluation
    results_data = {
        "config": {
            "rag_script": args.rag_script,
            "llm_model": args.llm_model,
            "eval_model": args.eval_model,
            "embeddings_file": args.embeddings_file,
            "qa_file": args.qa_file,
            "sample_size": args.sample_size if args.sample_size > 0 else len(qa_pairs)
        },
        "qa_pairs": [
            {
                "question": qa["question"],
                "expected_answer": qa["answer"],
                "rag_response": resp,
                "response_time": time,
                # Include simplified context for reference
                "context_summary": {
                    "num_chunks": len(ctx.get("context_text", [])),
                    "similarity_scores": ctx.get("similarity_scores", [])
                }
            } 
            for qa, resp, ctx, time in zip(qa_pairs, rag_responses, context_infos, response_times)
        ]
    }
    
    # Run RAGAS evaluation if available
    if RAGAS_AVAILABLE and DATASETS_AVAILABLE:
        print("Preparing RAGAS evaluation dataset...")
        ragas_dataset = prepare_ragas_dataset(qa_pairs, rag_responses, context_infos)
        
        if ragas_dataset is not None:
            print(f"Evaluating with RAGAS using model: {args.eval_model}")
            ragas_scores = evaluate_with_ragas(ragas_dataset, args.eval_model, api_key)
            
            # Add RAGAS scores to results
            results_data["ragas_scores"] = ragas_scores
            
            # Print RAGAS scores
            print("\n=== RAGAS Evaluation Results ===")
            for metric, score in ragas_scores.items():
                print(f"{metric}: {score:.4f}")
    else:
        print("RAGAS evaluation skipped: Required libraries not available")
    
    # Calculate basic metrics (even without RAGAS)
    avg_response_time = sum(response_times) / len(response_times)
    results_data["basic_metrics"] = {
        "avg_response_time": avg_response_time,
        "total_queries": len(qa_pairs)
    }
    
    print(f"\nAverage response time: {avg_response_time:.2f} seconds")
    
    # Save results to file if specified
    if args.output_file:
        try:
            with open(args.output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results to file: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 