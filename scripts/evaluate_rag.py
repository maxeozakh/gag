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
    """Check if the specified RAG script exists."""
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
    
    # If using our default rag.py script
    if script_path.endswith("rag.py"):
        cmd = [
            sys.executable,
            script_path,
            "--query", query,
            "--embeddings_file", embeddings_file,
            "--llm_model", llm_model
        ]
        
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check for errors
        if result.returncode != 0:
            print(f"Error running RAG script: {result.stderr}")
            return "Error: Failed to run RAG script", {}, 0.0
        
        # Parse the output to extract the response
        output = result.stdout
        
        # Extract context from output
        context_info = {}
        if "=== RAG RETRIEVAL INFO ===" in output and "=== RESPONSE ===" in output:
            retrieval_section = output.split("=== RAG RETRIEVAL INFO ===")[1].split("========================")[0]
            context_lines = [line for line in retrieval_section.strip().split("\n") if line.strip()]
            
            # Extract context content for RAGAS
            context_text = []
            for line in context_lines:
                # Skip lines that are just metadata
                if line.startswith("Time taken for search:") or line.startswith("Similarity scores:"):
                    continue
                    
                # If line appears to be content, add it
                if ":" in line and not line.startswith("CONTEXT"):
                    parts = line.split(":", 1)
                    if len(parts) > 1:
                        content = parts[1].strip()
                        if content:
                            context_text.append(content)
                elif not any(header in line for header in ["CONTEXT", "Similarity", "Time"]):
                    # If not a header line, it's probably content
                    clean_line = line.strip()
                    if clean_line:
                        context_text.append(clean_line)
            
            # Store formatted context text
            context_info["context_text"] = context_text
            
            # Extract similarity scores 
            for line in context_lines:
                if "Similarity scores:" in line:
                    try:
                        scores_str = line.split("Similarity scores:")[1].strip()
                        scores = eval(scores_str)  # Convert string representation of list to actual list
                        context_info["similarity_scores"] = scores
                    except Exception as e:
                        print(f"Error parsing similarity scores: {e}")
        
        # Extract the response
        if "=== RESPONSE ===" in output:
            response = output.split("=== RESPONSE ===")[1].split("========================")[0].strip()
        else:
            response = "Failed to extract response from output"
        
    else:
        # For custom RAG implementations, you would need to adapt this part
        # This is just a placeholder for custom RAG scripts
        try:
            # Try to import the custom script as a module
            spec = importlib.util.spec_from_file_location("custom_rag", script_path)
            if spec is None or spec.loader is None:
                return f"Error: Could not load {script_path}", {}, 0.0
                
            custom_rag = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_rag)
            
            # Assume the custom script has a function called "generate_response"
            if not hasattr(custom_rag, "generate_response"):
                return f"Error: {script_path} does not have a generate_response function", {}, 0.0
                
            response, context_info = custom_rag.generate_response(query, embeddings_file, llm_model)
        except Exception as e:
            return f"Error running custom RAG script: {str(e)}", {}, 0.0
    
    end_time = time.time()
    return response, context_info, end_time - start_time

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
    Prepare a dataset for RAGAS evaluation.
    
    Args:
        qa_pairs: List of QA pairs
        rag_responses: List of responses from the RAG system
        contexts: List of contexts used for generation
        
    Returns:
        RAGAS-compatible dataset or None if datasets library is not available
    """
    if not DATASETS_AVAILABLE:
        print("Warning: datasets library not available. Install with: pip install datasets")
        return None
        
    # Format contexts properly - RAGAS expects a list of lists of strings
    formatted_contexts = []
    for context in contexts:
        # Try to extract text content from the context in various formats
        context_texts = []
        
        if isinstance(context, dict):
            # Handle dictionary format
            if "context_text" in context and isinstance(context["context_text"], list):
                # Use pre-extracted context texts
                context_texts = [str(text) for text in context["context_text"] if text]
            elif "text" in context:
                # Text field is present
                context_texts.append(str(context["text"]))
            elif "similarity_scores" in context:
                # Our format has similarity scores but might not have proper context
                # Try to find other fields that might contain actual content
                for key, value in context.items():
                    if isinstance(value, str) and key not in ["id", "embedding", "similarity_scores"]:
                        if value and len(value) > 10:  # Avoid short metadata fields
                            context_texts.append(value)
            else:
                # Try to combine all text fields
                context_text = ""
                for key, value in context.items():
                    if isinstance(value, str) and key not in ["id", "embedding"]:
                        context_text += f"{value} "
                if context_text:
                    context_texts.append(context_text.strip())
        elif isinstance(context, list):
            # If already a list, make sure all items are strings
            for item in context:
                if isinstance(item, str):
                    context_texts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    context_texts.append(str(item["text"]))
                elif isinstance(item, dict):
                    # Extract all string values
                    txt = " ".join([str(v) for k, v in item.items() 
                                   if isinstance(v, str) and k not in ["id", "embedding"]])
                    if txt:
                        context_texts.append(txt)
        elif isinstance(context, str):
            # Single string context
            context_texts.append(context)
            
        # Ensure we have at least one context text
        if not context_texts:
            # As a last resort, add an actual informative placeholder
            # but it's better to have real context
            context_texts = ["Note: The RAG system did not provide any retrievable context for this query."]
            
        # Filter out very short contexts that might just be metadata
        context_texts = [text for text in context_texts if len(text) > 15]
        
        # Ensure there's always at least one context even after filtering
        if not context_texts:
            context_texts = ["Context unavailable for this query."]
            
        formatted_contexts.append(context_texts)
    
    # Show sample of contexts being evaluated
    if len(formatted_contexts) > 0:
        print("\nSample context for first query:")
        for i, context in enumerate(formatted_contexts[0][:2]):  # Show first 2 contexts for first query
            print(f"  Context {i+1}: {context[:100]}...")  # Truncate long contexts
        if len(formatted_contexts[0]) > 2:
            print(f"  ...and {len(formatted_contexts[0])-2} more context segments")
    
    # Ensure all questions, answers and responses are strings
    questions = [str(pair["question"]) for pair in qa_pairs]
    ground_truths = [str(pair["answer"]) for pair in qa_pairs]
    answers = [str(response) for response in rag_responses]
    
    # Prepare data in the format expected by RAGAS
    data = {
        "question": questions,
        "ground_truth": ground_truths,
        "answer": answers,
        "contexts": formatted_contexts
    }
    
    try:
        # Create a Hugging Face dataset
        return HFDataset.from_dict(data)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        print("Trying alternate format...")
        
        # Try with minimal required fields
        minimal_data = {
            "question": questions,
            "ground_truth": ground_truths,
            "answer": answers,
            "contexts": formatted_contexts  # Keep contexts in the minimal dataset
        }
        try:
            return HFDataset.from_dict(minimal_data)
        except Exception as e2:
            print(f"Error creating minimal dataset: {e2}")
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
    
    # Run RAG system for each question
    print(f"\nRunning {len(qa_pairs)} queries through the RAG system...")
    rag_responses = []
    contexts = []
    response_times = []
    
    for i, pair in enumerate(tqdm(qa_pairs, desc="Evaluating")):
        question = pair["question"]
        ground_truth = pair["answer"]
        
        # Print query and ground truth for each evaluation
        print(f"\n[Query {i+1}/{len(qa_pairs)}]")
        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth}")
        
        response, context, response_time = run_rag_script(
            args.rag_script, 
            question, 
            args.embeddings_file,
            args.llm_model
        )
        rag_responses.append(response)
        contexts.append(context)
        response_times.append(response_time)
        
        # Print the response
        print(f"Response: {response}")
        print("---")
    
    evaluation_results = {}
    
    # Create dataset for RAGAS evaluation
    if RAGAS_AVAILABLE:
        eval_dataset = prepare_ragas_dataset(qa_pairs, rag_responses, contexts)
        
        if eval_dataset is not None:
            # Evaluate with RAGAS
            ragas_results = evaluate_with_ragas(eval_dataset, args.eval_model, api_key)
            
            # Print results
            print("\n=== EVALUATION RESULTS ===")
            
            # Make sure we have at least one metric
            if not ragas_results:
                print("No metrics were calculated. This could be because:")
                print("- Your RAGAS version returns metrics in a different format")
                print("- There was an issue with the OpenAI API during evaluation")
                print("- The dataset doesn't have the required structure")
                print("\nCheck the output above for more details on what happened.")
            else:
                for metric, score in ragas_results.items():
                    if metric != "error" and isinstance(score, (int, float)):
                        print(f"{metric}: {score:.4f}")
                
                evaluation_results = ragas_results
        else:
            print("\n=== EVALUATION RESULTS ===")
            print("Failed to prepare dataset for RAGAS evaluation.")
    else:
        print("\n=== EVALUATION RESULTS ===")
        print("RAGAS is not available. Please install with: pip install ragas")
    
    # Print response time statistics
    avg_time = sum(response_times) / len(response_times)
    min_time = min(response_times)
    max_time = max(response_times)
    print(f"\nResponse Time: avg={avg_time:.2f}s, min={min_time:.2f}s, max={max_time:.2f}s")
    
    # Add response time statistics to evaluation results
    evaluation_results.update({
        "avg_response_time": avg_time,
        "min_response_time": min_time,
        "max_response_time": max_time
    })
    
    # Save results if output file specified
    if args.output_file:
        try:
            results = {
                "metrics": evaluation_results,
                "config": vars(args),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save based on file extension
            if args.output_file.endswith('.json'):
                with open(args.output_file, 'w') as f:
                    json.dump(results, f, indent=2)
            elif args.output_file.endswith('.csv'):
                # Flatten metrics for CSV
                flat_results = {
                    **{f"metric_{k}": v for k, v in evaluation_results.items()},
                    **{f"config_{k}": v for k, v in vars(args).items() if k != "output_file"}
                }
                pd.DataFrame([flat_results]).to_csv(args.output_file, index=False)
            else:
                print(f"Warning: Unknown output format. Saving as JSON.")
                with open(f"{args.output_file}.json", 'w') as f:
                    json.dump(results, f, indent=2)
                    
            print(f"Evaluation results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 