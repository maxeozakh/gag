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
import random
import logging
import datetime
import dotenv

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

def validate_file_path(file_path: str) -> bool:
    """
    Check if a file exists and is readable.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file exists and is readable, False otherwise
    """
    return os.path.exists(file_path) and os.path.isfile(file_path)

def validate_rag_script(script_path: str) -> bool:
    """
    Check if a RAG script exists and is executable.
    
    Args:
        script_path: Path to the RAG script
        
    Returns:
        True if the script exists and is executable, False otherwise
    """
    return os.path.exists(script_path) and os.path.isfile(script_path)

def run_rag_script(
    script_path: str, 
    query: str, 
    embeddings_file: str = "",
    products_file: str = "",
    llm_model: str = "gpt-3.5-turbo",
    log_level: str = "ERROR"
) -> Tuple[str, Dict[str, Any], float]:
    """
    Run the RAG script with a query and return the response, context, and time taken.
    
    Args:
        script_path: Path to the RAG script
        query: User query to run
        embeddings_file: Path to embeddings file
        products_file: Path to products file
        llm_model: Language model to use
        log_level: Logging level
        
    Returns:
        Tuple of (response, context_info, response_time)
    """
    start_time = time.time()
    
    try:
        # Prepare command to run the script
        cmd = [sys.executable, script_path, "--query", query]
        
        # Special handling for different RAG scripts
        script_name = os.path.basename(script_path)
        
        if script_name == "openai-based-rag.py":
            # OpenAI-based RAG uses different parameters
            if products_file and os.path.exists(products_file):
                cmd.extend(["--input_file", products_file])
            elif embeddings_file:
                raise ValueError("Products file is required for OpenAI-based RAG")
                
            # Add LLM model
            if llm_model:
                cmd.extend(["--llm_model", llm_model])
        else:
            # Standard RAG script
            if embeddings_file:
                cmd.extend(["--embeddings_file", embeddings_file])
                
            
            # Add LLM model if specified
            if llm_model:
                cmd.extend(["--llm_model", llm_model])
                
            # Add log level if accepted by the script
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
                if "--log_level" in script_content:
                    cmd.extend(["--log_level", log_level])
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run the command and capture the output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            print(f"Error running RAG script: {stderr} {stdout} {cmd} {script_path} {query} {embeddings_file} {products_file} {llm_model} {log_level}")
            
            return f"Error: {stderr}", {"context_text": [], "similarity_scores": []}, time.time() - start_time
        
        if stderr:
            print(f"Warning from RAG script: {stderr}")
        
        # Parse the JSON output
        try:
            # Extract only the JSON part if there's other output
            # The script might output logs before the JSON
            json_start = stdout.find('{')
            json_end = stdout.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = stdout[json_start:json_end]
                try:
                    output_data = json.loads(json_str)
                    print(f"Successfully parsed JSON output of length {len(json_str)}")
                except json.JSONDecodeError:
                    print(f"Failed to parse JSON output: {json_str[:100]}...")
                    # Try again with full output as a fallback
                    output_data = json.loads(stdout)
            else:
                # Try with the full output
                output_data = json.loads(stdout)
                
            response = output_data.get("response", "")
            context_info = {
                "context_text": output_data.get("context_text", []),
                "similarity_scores": output_data.get("similarity_scores", []),
                "metadata": output_data.get("metadata", []),
            }
            
            # Debug output to help diagnose issues
            print(f"Debug - Response type: {type(response)}, length: {len(response)}")
            print(f"Debug - Response preview: '{response[:50]}...'")
            print(f"Debug - Context items: {len(context_info['context_text'])}")
            if context_info['context_text']:
                print(f"Debug - First context item: '{context_info['context_text'][0][:50]}...'")
            
            # Check if structure looks valid
            if not isinstance(response, str):
                print(f"Warning: Response is not a string, it's {type(response)}")
                response = str(response)
                
            print(f"Successfully processed RAG response of length {len(response)} with {len(context_info['context_text'])} context items")
            return response, context_info, time.time() - start_time
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON output: {e}")
            print(f"Output: {stdout[:500]}...")
            return stdout, {"context_text": [], "similarity_scores": []}, time.time() - start_time
        except Exception as e:
            print(f"Unexpected error processing RAG output: {e}")
            traceback.print_exc()
            return str(e), {"context_text": [], "similarity_scores": []}, time.time() - start_time
    
    except Exception as e:
        print(f"Error running RAG script: {e}")
        traceback.print_exc()
        return str(e), {"context_text": [], "similarity_scores": []}, time.time() - start_time

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
    contexts: List[Dict[str, Any]],
    products_file: str = "data/ecommerce_products_test.csv"
) -> Optional[Any]:
    """
    Prepare dataset for RAGAS evaluation.
    
    Args:
        qa_pairs: List of QA pairs
        rag_responses: List of RAG responses
        contexts: List of context information from RAG system
        products_file: Path to the products CSV file to use as reference
        
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
            
        # Load products data from CSV file to use as reference documents
        products_data = []
        if products_file and os.path.exists(products_file):
            try:
                import csv
                with open(products_file, 'r', encoding='utf-8') as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        # Convert each product row to a string description
                        product_text = " ".join([f"{k}: {v}" for k, v in row.items() if v and v.lower() != "unknown"])
                        products_data.append(product_text)
                print(f"Loaded {len(products_data)} products from {products_file}")
            except Exception as e:
                print(f"Warning: Could not load products file: {e}")
                
        # Prepare data in the format required by RAGAS
        data = {
            "question": [],
            "ground_truths": [],  # RAGAS expects this field for context_recall
            "answer": [],
            "contexts": [],
            "reference": []  # Required by context_recall metric
        }
        
        for idx, (qa_pair, response, context_info) in enumerate(zip(qa_pairs, rag_responses, contexts)):
            # Extract question and ground truth
            question = qa_pair.get("question", "")
            ground_truth = qa_pair.get("answer", "")
            
            # Debug prints to identify issues
            print(f"Debug - QA pair {idx}: Question type: {type(question)}, Ground truth type: {type(ground_truth)}")
            
            # Skip if missing data
            if not question or not ground_truth:
                print(f"Warning: Skipping QA pair {idx} due to missing question or ground truth")
                continue
                
            # Get context text from the context info dictionary
            context_text = context_info.get("context_text", [])
            if not context_text:
                # If no context was found, add an empty list to avoid errors in RAGAS
                processed_context = ["No relevant context found"]
                print(f"Warning: No context found for QA pair {idx}")
            else:
                print(f"Debug - Context for QA {idx} - Type: {type(context_text)}, Length: {len(context_text)}")
                if context_text and len(context_text) > 0:
                    print(f"Debug - First context item type: {type(context_text[0])}")
                    print(f"Debug - First context preview: {str(context_text[0])[:100]}...")
                
                # Process context to ensure we have a list of strings
                processed_context = []
                for ctx in context_text:
                    # Handle different context formats
                    if isinstance(ctx, str):
                        # If it's already a string, use it directly
                        processed_context.append(ctx)
                    elif isinstance(ctx, list) and ctx and isinstance(ctx[0], dict) and 'text' in ctx[0]:
                        # Extract text from list containing dictionaries with 'text' key
                        # This handles format like [{'text': 'content', 'type': 'text'}]
                        processed_context.append(ctx[0]['text'])
                    elif isinstance(ctx, dict) and 'text' in ctx:
                        # Extract text from dictionary with 'text' key
                        processed_context.append(ctx['text'])
                    else:
                        # Fall back to string representation
                        processed_context.append(str(ctx))
                
                # Ensure we have at least one context item
                if not processed_context:
                    processed_context = ["No processable context found"]
                
                print(f"Debug - Processed first context item: {processed_context[0][:100]}...")
            
            # Create a comprehensive reference document from all products
            # This gives context_recall a real reference to evaluate against
            if products_data:
                # Join all product data into one reference document
                reference_doc = " ".join(products_data)
            else:
                # If no products data, fall back to using the context
                reference_doc = " ".join(processed_context)
            
            # Add to dataset
            data["question"].append(question)
            data["ground_truths"].append([ground_truth])  # RAGAS expects a list of ground truths
            data["answer"].append(response)
            data["contexts"].append(processed_context)  # Use the processed context
            data["reference"].append(reference_doc)  # Use comprehensive reference document
            
            # Debug what's being added to ensure it's properly formatted
            print(f"Debug - Added to dataset: Q: '{question[:30]}...', A: '{response[:30]}...'")
        
        # Make sure we have at least one valid data point
        if not data["question"]:
            print("Error: No valid data points for RAGAS evaluation")
            return None
        
        # Verify the question and answers are valid for OpenAI
        print("Debug - Sample data check:")
        print(f"  - First question: '{data['question'][0][:50]}...'")  
        print(f"  - First answer: '{data['answer'][0][:50]}...'")
        print(f"  - First ground truth: '{data['ground_truths'][0][0][:50]}...'")
        # raw data
        print(f"  - Raw data: '{data['answer']}...'")
        
        # Create dataset
        dataset = HFDataset.from_dict(data)
        return dataset
        
    except Exception as e:
        print(f"Error preparing RAGAS dataset: {e}")
        traceback.print_exc()
        return None

def evaluate_with_ragas(
    eval_dataset: Any,
    model: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None
) -> Dict[str, Union[float, str]]:
    """
    Evaluate the RAG system using RAGAS metrics.
    
    Args:
        eval_dataset: Dataset prepared for RAGAS
        model: Model to use for evaluation
        api_key: OpenAI API key
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\nEvaluating with RAGAS using model: {model}")
    
    if eval_dataset is None:
        raise ValueError("Cannot evaluate with RAGAS: dataset is None")
    
    # Set up OpenAI API key for RAGAS
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    
    try:
        # Import the necessary modules
        try:
            from langchain_openai import ChatOpenAI
            # Initialize metrics according to latest RAGAS API
            metrics = [
                faithfulness, 
                answer_relevancy,
                context_recall
            ]
            print("Successfully imported LangChain OpenAI integration")
        except ImportError:
            print("Warning: Could not import LangChain OpenAI. Using default metrics.")
            # Fall back to default metrics
            metrics = [
                faithfulness, 
                answer_relevancy,
                context_recall
            ]
        
        # For debugging
        print(f"eval_dataset {eval_dataset}")
        
        # Call evaluate() with the dataset and metrics
        # The model name should be properly handled by RAGAS
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics
        )
        
        print(f"RAGAS evaluation completed: {type(result)}")
        
        # Convert result to dictionary
        if hasattr(result, "to_pandas"):
            # Newer RAGAS versions return a DataFrame-like object
            df = result.to_pandas()
            metrics_dict: Dict[str, Union[float, str]] = {}
            for col in df.columns:
                if col not in ["question", "contexts", "answer", "ground_truths", "reference"]:
                    try:
                        metrics_dict[str(col)] = float(df[col].mean())
                    except (ValueError, TypeError):
                        metrics_dict[str(col)] = 0.0
            return metrics_dict
        
        # Try to convert directly to dict
        try:
            result_dict: Dict[str, Union[float, str]] = {}
            for k, v in dict(result).items():
                if isinstance(v, (int, float)):
                    result_dict[str(k)] = float(v)
                else:
                    result_dict[str(k)] = str(v)
            return result_dict
        except (ValueError, TypeError, AttributeError):
            pass
            
        # Last resort
        print(f"RAGAS result type: {type(result)}, value: {result}")
        return {"evaluation_completed": 1.0, "note": "Results format unclear, check logs"}
    
    except Exception as e:
        print(f"Error during RAGAS evaluation: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        
        # Instead of raising an error, return an error object
        # This allows the script to continue running
        return {
            "error": 1.0,
            "error_message": str(e)
        }

def main():
    """Main entry point for script execution."""
    parser = argparse.ArgumentParser(description="Evaluate RAG system against a QA dataset")
    
    # Required arguments
    parser.add_argument("--qa_file", required=True, help="Path to the file containing QA pairs")
    parser.add_argument("--embeddings_file", required=True, help="Path to the embeddings file")
    parser.add_argument("--rag_script", required=True, help="Path to the RAG script to evaluate")
    
    # Optional arguments
    parser.add_argument("--products_file", default="data/ecommerce_products_test.csv", 
                        help="Path to the products CSV file for reference")
    parser.add_argument("--sample_size", type=int, default=0, 
                        help="Number of QA pairs to sample (0 for all)")
    parser.add_argument("--output_file", help="Path to save evaluation results")
    parser.add_argument("--api_key", help="OpenAI API key (if not in .env file)")
    parser.add_argument("--llm_model", default="gpt-3.5-turbo", 
                        help="Language model to use for the RAG system")
    parser.add_argument("--eval_model", default="gpt-4o-mini", 
                        help="Model to use for evaluation (RAGAS metrics)")
    parser.add_argument("--log_level", default="ERROR",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level")
    
    args = parser.parse_args()
    
    # Generate timestamp for the run
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Validate input files
    if not validate_file_path(args.qa_file):
        raise FileNotFoundError(f"QA file not found: {args.qa_file}")
        
    if not validate_file_path(args.embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {args.embeddings_file}")
        
    if not validate_rag_script(args.rag_script):
        raise ValueError(f"RAG script not found or not executable: {args.rag_script}")
        
    # Check if products_file exists if provided
    if args.products_file and not os.path.exists(args.products_file):
        print(f"Warning: Products file not found: {args.products_file}")
        print("Will proceed without products file for reference")
        args.products_file = None
    
    # Load API key from .env file if not provided
    if not args.api_key:
        dotenv.load_dotenv()
        args.api_key = os.getenv("OPENAI_API_KEY")
    
    # Set Logging
    log_level = getattr(logging, args.log_level.upper(), logging.ERROR)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load QA pairs
    qa_pairs = load_qa_pairs(args.qa_file)
    if not qa_pairs:
        raise ValueError(f"No valid QA pairs found in {args.qa_file}")
    
    print(f"Loaded {len(qa_pairs)} valid QA pairs from {args.qa_file}")
    
    # Sample QA pairs if needed
    if args.sample_size > 0 and args.sample_size < len(qa_pairs):
        print(f"Sampling {args.sample_size} QA pairs for evaluation")
        qa_pairs = random.sample(qa_pairs, args.sample_size)
    
    # Run evaluation
    print(f"Evaluating RAG script at {args.rag_script} using model {args.llm_model}")
    responses = []
    contexts = []
    
    response_times = []
    for i, qa_pair in enumerate(qa_pairs):
        query = qa_pair.get("question", "")
        print(f"\nEvaluating QA pair {i+1}/{len(qa_pairs)}")
        print(f"Query: {query}")
        
        response, context, response_time = run_rag_script(
            args.rag_script, 
            query, 
            args.embeddings_file,
            args.products_file,
            args.llm_model,
            args.log_level
        )
        
        responses.append(response)
        contexts.append(context)
        response_times.append(response_time)
        
        print(f"Response generated (length: {len(response)} chars)")
        print(f"Retrieved {len(context.get('context_text', []))} context items")
    
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    print(f"\nAverage response time: {avg_response_time:.2f} seconds")
    
    # Create config dictionary with all parameters
    config = {
        "rag_script": args.rag_script,
        "qa_file": args.qa_file,
        "embeddings_file": args.embeddings_file,
        "products_file": args.products_file,
        "llm_model": args.llm_model,
        "eval_model": args.eval_model,
        "sample_size": args.sample_size,
        "actual_samples": len(qa_pairs)
    }
    
    # Prepare results structure
    evaluation_results = {
        "timestamp": timestamp,
        "config": config,
        "avg_response_time": avg_response_time,
        "sample_size": len(qa_pairs)
    }
    
    # Prepare RAGAS dataset
    try:
        print(f"Preparing RAGAS dataset with {len(qa_pairs)} QA pairs")
        ragas_dataset = prepare_ragas_dataset(qa_pairs, responses, contexts, args.products_file)
        
        # Evaluate with RAGAS
        print(f"Evaluating with RAGAS using model: {args.eval_model}")
        ragas_results = evaluate_with_ragas(ragas_dataset, args.eval_model, args.api_key)
        
        # Add RAGAS results
        evaluation_results["ragas_results"] = ragas_results
    except Exception as e:
        print(f"RAGAS evaluation failed: {e}")
        evaluation_results["ragas_results"] = {
            "error": 1.0,
            "error_message": str(e)
        }
    
    # Print results
    print("\n=== Evaluation Results ===")
    print(json.dumps(evaluation_results, indent=2))
    
    # If no output file is specified, create one with timestamp
    output_file = args.output_file
    if not output_file:
        output_dir = "evaluation_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/rag_eval_{timestamp}.json"
        print(f"No output file specified, using: {output_file}")
    
    # Save results
    with open(output_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1) 