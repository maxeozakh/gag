import os
import argparse
import json
import sys
import time
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, cast, Literal
from dotenv import load_dotenv
from openai import OpenAI
import uuid
import csv
import re

# Configure logging with a null handler by default
logging.basicConfig(level=logging.CRITICAL, handlers=[])  # Prevent root logger from handling logs
logger = logging.getLogger("openai-rag")
logger.setLevel(logging.CRITICAL)  # Default to critical only
logger.addHandler(logging.NullHandler())
logger.propagate = False  # Prevent propagation to root logger

# Load environment variables from .env file
load_dotenv()

# Import csv_to_json module
try:
    # First try to import directly if the script is in the same directory
    try:
        from specific_csv_to_json import convert_csv_to_json
    except ImportError:
        # Try to import from scripts directory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from specific_csv_to_json import convert_csv_to_json
except ImportError:
    # Define a placeholder function if the module is not available
    def convert_csv_to_json(csv_file, json_file=None):
        logger.error("csv_to_json module not found, cannot convert CSV files.")
        return None

# Define the RAG prompt template (same as in rag.py for compatibility)
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

class OpenAIVectorStoreManager:
    def __init__(self, client: OpenAI):
        self.client = client
        self.vector_store_cache: Dict[str, str] = {}  # Cache vector store IDs

    def create_vector_store(self, name: str) -> str:
        """
        Creates a vector store and returns its ID.
        If a vector store with the same name has already been created, returns the cached ID.
        """
        # Check if vector store with this name is already in cache
        if name in self.vector_store_cache:
            logger.info(f"Using cached vector store: {name}")
            return self.vector_store_cache[name]
        
        try:
            # Create a new vector store
            logger.info(f"Creating new vector store: {name}")
            vector_store = self.client.beta.vector_stores.create(name=name)
            vector_store_id = vector_store.id
            self.vector_store_cache[name] = vector_store_id
            logger.info(f"Vector store created with ID: {vector_store_id}")
            return vector_store_id
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def upload_file_to_vector_store(self, vector_store_id: str, file_path: str) -> bool:
        """
        Uploads a file to a vector store and waits for processing to complete.
        """
        try:
            # Open the file for upload
            logger.info(f"Uploading file {file_path} to vector store {vector_store_id}")
            with open(file_path, "rb") as file:
                # Upload the file and poll until processing is complete
                file_batch = self.client.beta.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=[file]
                )
                
                logger.info(f"File batch status: {file_batch.status}")
                if hasattr(file_batch, 'file_counts'):
                    logger.info(f"File counts: {file_batch.file_counts}")
                
                # Check if the upload was successful
                success = file_batch.status == "completed"
                if success:
                    logger.info("File upload completed successfully")
                else:
                    logger.error(f"File upload failed with status: {file_batch.status}")
                return success
        except Exception as e:
            logger.error(f"Error uploading file to vector store: {e}")
            logger.error(traceback.format_exc())
            return False

class OpenAIAssistantManager:
    def __init__(self, client: OpenAI, model: str = "gpt-4o", input_file: Optional[str] = None):
        self.client = client
        self.model = model
        self.assistant_id: Optional[str] = None
        self.assistant_name = f"RAG-Assistant-{str(uuid.uuid4())[:8]}"
        
    def create_assistant(self, vector_store_id: str) -> str:
        """
        Create an OpenAI assistant with file search capability
        
        Args:
            vector_store_id: ID of the vector store to use
            
        Returns:
            Assistant ID
        """
        try:
            logger.info(f"Creating assistant with name: {self.assistant_name}")
            
            # Create assistant using proper tool type format
            assistant = self.client.beta.assistants.create(
                name=self.assistant_name,
                instructions="""You are a helpful product information assistant. 
                Use the provided documents to answer questions about products.
                If you don't know the answer, say so. Do not make up information.""",
                model=self.model,
                tools=[
                    {
                        "type": "file_search",
                        # Use extended parameters as keyword arguments
                        "file_search": {
                            "max_num_results": 20 if self.model.startswith(("gpt-4", "gpt-4o")) else 5,
                            "ranking_options": {
                                "ranker": "auto",
                                "score_threshold": 0.7
                            }
                        }
                    }
                ],
                tool_resources={
                    "file_search": {
                        "vector_store_ids": [vector_store_id]
                    }
                }
            )
            
            self.assistant_id = assistant.id
            logger.info(f"Assistant created with ID: {self.assistant_id}")
            return self.assistant_id
            
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def delete_assistant(self) -> None:
        """
        Delete the assistant to clean up resources.
        """
        if self.assistant_id:
            try:
                logger.info(f"Deleting assistant: {self.assistant_id}")
                self.client.beta.assistants.delete(self.assistant_id)
                logger.info("Assistant deleted successfully")
                self.assistant_id = None
            except Exception as e:
                logger.error(f"Error deleting assistant: {e}")
                logger.error(traceback.format_exc())

    def _process_file_search_results(self, results) -> List[Dict[str, Any]]:
        """
        Process file search results into a consistent format.
        
        Args:
            results: File search results from the API
            
        Returns:
            List of normalized result dictionaries with content, file_id, and score
        """
        normalized_results: List[Dict[str, Any]] = []
        
        if not results:
            logger.warning("No results to process")
            return normalized_results
            
        for result in results:
            try:
                # Skip results without required attributes
                if not hasattr(result, "file_id") or not hasattr(result, "content"):
                    logger.warning("Result missing required attributes, skipping")
                    continue
                    
                file_id = result.file_id
                content = result.content
                score = getattr(result, "score", 0.0)
                
                # Process different content formats
                if isinstance(content, str):
                    # String content is already in the right format
                    pass
                elif isinstance(content, list) and content and isinstance(content[0], dict) and "text" in content[0]:
                    # Handle content as a list of dictionaries with text field
                    content = content[0].get("text", "")
                    logger.info(f"Extracted text content from structured content for file {file_id}")
                elif isinstance(content, dict) and "text" in content:
                    # Handle content as a dictionary with text field
                    content = content.get("text", "")
                    logger.info(f"Extracted text from dictionary content for file {file_id}")
                
                # Add normalized result
                normalized_results.append({
                    "file_id": file_id,
                    "content": content,
                    "score": score
                })
                
                logger.info(f"Processed result for file {file_id} with score {score}")
            except Exception as e:
                logger.warning(f"Error processing search result: {e}")
                continue
                
        return normalized_results

    def search_and_generate(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Use the assistant to search files and generate a response.
        
        Args:
            query: User query
            
        Returns:
            Tuple of (generated response, context_info dictionary)
        """
        # Initialize context info
        context_info: Dict[str, List] = {
            "context_text": [],
            "similarity_scores": [],
            "metadata": []
        }
        
        if not self.assistant_id:
            error_msg = "Assistant not created"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Create a thread for search and generate
            logger.info("Creating thread for search and generate")
            thread = self.client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"Thread created with ID: {thread_id}")
            
            # Add user message to thread
            logger.info(f"Adding user message to thread: '{query}'")
            self.client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content=query
            )
            
            # Run the assistant on the thread
            logger.info(f"Running assistant (ID: {self.assistant_id}) on thread")
            run = self.client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=self.assistant_id
            )
            run_id = run.id
            logger.info(f"Run created with ID: {run_id}")
            
            # Poll for run completion with proper error handling
            while True:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run_id
                )
                logger.info(f"Run status: {run.status}")
                
                if run.status == "completed":
                    logger.info("Run completed successfully")
                    break
                elif run.status == "requires_action":
                    logger.error("Run requires action but this is not supported in this implementation")
                    raise ValueError("Run requires action but this is not supported")
                elif run.status in ["failed", "cancelled", "expired"]:
                    error_details = ""
                    if hasattr(run, "last_error") and run.last_error:
                        error_details = f": {run.last_error.code} - {run.last_error.message}"
                    error_msg = f"Run ended with status: {run.status}{error_details}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                # Wait before polling again
                logger.info("Waiting for run to complete...")
                time.sleep(1)
            
            # Get run steps to extract file search results
            logger.info("Retrieving run steps to extract context")
            try:
                run_steps = self.client.beta.threads.runs.steps.list(
                    thread_id=thread_id,
                    run_id=run_id,
                    include=["step_details.tool_calls[*].file_search.results[*].content"]
                )
            except Exception as e:
                logger.error(f"Error retrieving run steps: {e}")
                raise RuntimeError(f"Failed to retrieve run steps: {str(e)}")
            
            logger.info(f"Found {len(run_steps.data)} run steps")
            
            # Process run steps to extract context
            search_results_found = False
            for step in run_steps.data:
                if step.type != "tool_calls":
                    logger.info(f"Skipping non-tool-calls step of type: {step.type}")
                    continue
                
                if not hasattr(step, "step_details") or not hasattr(step.step_details, "tool_calls"):
                    logger.info("Step has no tool_calls attribute, skipping")
                    continue
                
                # Process file search tool calls
                for tool_call in step.step_details.tool_calls:
                    # Skip non-file-search tool calls
                    if not hasattr(tool_call, "type") or tool_call.type != "file_search":
                        logger.info(f"Skipping non-file-search tool call of type: {getattr(tool_call, 'type', 'unknown')}")
                        continue
                    
                    # Ensure file_search attribute exists - we've verified this is a file_search type
                    if not hasattr(tool_call, "file_search"):
                        logger.warning("File search tool call missing file_search attribute")
                        continue
                    
                    # Process file search results
                    file_search = getattr(tool_call, "file_search", None)
                    if not file_search or not hasattr(file_search, "results"):
                        logger.warning("File search missing results attribute")
                        continue
                    
                    results = file_search.results
                    if not results:
                        logger.info("File search returned no results")
                        continue
                    
                    search_results_found = True
                    logger.info(f"Found {len(results)} results in file search")
                    
                    # Process results using the helper method
                    normalized_results = self._process_file_search_results(results)
                    logger.info(f"Processed {len(normalized_results)} valid results")
                    
                    # Add results to context info
                    for result in normalized_results:
                        file_id = result["file_id"]
                        content = result["content"]
                        score = result["score"]
                        
                        # Add content to context info
                        context_info["context_text"].append(content)
                        context_info["similarity_scores"].append(score)
                        
                        # Get file metadata if available
                        try:
                            file_info = self.client.files.retrieve(file_id)
                            file_name = file_info.filename
                        except Exception as e:
                            logger.warning(f"Could not retrieve file info for {file_id}: {e}")
                            file_name = "unknown_file"
                        
                        context_info["metadata"].append({
                            "source": file_name,
                            "file_id": file_id
                        })
                        
                        logger.info(f"Added content from {file_id} ({file_name}) with score {score}")
            
            if not search_results_found:
                logger.warning("No file search results found in any run steps")
            
            # Get assistant's response
            try:
                messages = self.client.beta.threads.messages.list(
                    thread_id=thread_id
                )
                
                # Get assistant's reply (most recent message from assistant)
                assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
                if not assistant_messages:
                    error_msg = "No assistant response found"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                # Extract the most recent assistant message
                assistant_message = assistant_messages[0]
                
                # Extract text content from the message
                response_text = self._extract_message_text(assistant_message)
                
                if not response_text:
                    logger.warning("Assistant response contained no text content")
                    response_text = "No text content found in assistant response."
            except Exception as e:
                logger.error(f"Error retrieving assistant response: {e}")
                raise RuntimeError(f"Failed to retrieve assistant response: {str(e)}")
            
            # Clean up: delete the thread
            try:
                logger.info(f"Deleting thread: {thread_id}")
                self.client.beta.threads.delete(thread_id)
            except Exception as e:
                logger.warning(f"Failed to delete thread {thread_id}: {e}")
                # Don't raise an exception here, as we already have our response
            
            return response_text, context_info
        
        except Exception as e:
            logger.error(f"Error during search and generate: {e}")
            logger.error(traceback.format_exc())
            return f"Error during search: {str(e)}", context_info
            
    def _extract_message_text(self, message) -> str:
        """
        Extract text content from an assistant message.
        
        Args:
            message: The message object from the API
            
        Returns:
            The extracted text content
        """
        response_text = ""
        
        # Check if message has content
        if not hasattr(message, "content") or not message.content:
            logger.warning("Message has no content")
            return response_text
            
        # Process each content item
        for content_item in message.content:
            # Handle text content
            if hasattr(content_item, "type") and content_item.type == "text":
                if hasattr(content_item, "text") and hasattr(content_item.text, "value"):
                    response_text += content_item.text.value
            else:
                logger.info(f"Skipping non-text content item of type: {getattr(content_item, 'type', 'unknown')}")
                
        return response_text

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
        logger.info(f"Calling OpenAI API with model: {model}")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        if content is None:
            logger.warning("No response generated")
            return "No response generated"
        logger.info(f"Generated response of length {len(content)}")
        return content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}")
        logger.error(traceback.format_exc())
        return f"Error generating response: {str(e)}"

def format_context(context_info: Dict[str, Any]) -> str:
    """
    Format context information into a string for the RAG prompt.
    
    Args:
        context_info: Dictionary with context information
        
    Returns:
        Formatted context string
    """
    if not context_info or not context_info.get("context_text"):
        logger.warning("No relevant information found for context formatting")
        return "No relevant information found."
        
    context_parts = []
    
    for i, text in enumerate(context_info.get("context_text", []), 1):
        if text:
            # Ensure text is a string
            if not isinstance(text, str):
                try:
                    text = str(text)
                except Exception as e:
                    logger.warning(f"Could not convert text to string: {e}")
                    continue
            
            # Format context with numbers
            context_parts.append(f"[{i}] {text}")
    
    formatted_context = "\n\n".join(context_parts)
    logger.info(f"Formatted context of length {len(formatted_context)}")
    return formatted_context

def ensure_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable types.
    
    Args:
        obj: The object to convert
        
    Returns:
        A JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {ensure_json_serializable(k): ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return [ensure_json_serializable(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        # Convert objects to dictionaries
        return ensure_json_serializable(obj.__dict__)
    elif hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        # Handle objects with to_dict method
        return ensure_json_serializable(obj.to_dict())
    elif hasattr(obj, "__str__"):
        # Try converting to string as a last resort
        try:
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)
    else:
        # Return the object if it's a primitive type
        return obj

def main():
    parser = argparse.ArgumentParser(description="OpenAI-based Retrieval-Augmented Generation (RAG)")
    parser.add_argument("--query", type=str, required=True, help="The user query")
    parser.add_argument("--input_file", type=str, required=False, 
                        help="Path to the file to search (json, txt, pdf, etc.)")
    parser.add_argument("--embeddings_file", type=str, required=False,
                        help="[DEPRECATED] Use --input_file instead. Path to the file to search")
    parser.add_argument("--file_type", type=str, required=False, 
                        help="Explicitly specify file type (json, csv, pdf, txt). If not provided, will be inferred from extension.")
    parser.add_argument("--top_n", type=int, default=3, 
                        help="Number of top results to retrieve (unused, kept for compatibility)")
    parser.add_argument("--threshold", type=float, default=0.0, 
                        help="Minimum similarity threshold (unused, kept for compatibility)")
    parser.add_argument("--embed_model", type=str, default="text-embedding-3-small", 
                        help="Embedding model (unused, kept for compatibility)")
    parser.add_argument("--llm_model", type=str, default="gpt-4o", 
                        help="OpenAI LLM model to use")
    parser.add_argument("--terminal_output", action="store_true", 
                        help="Display human-readable output in terminal")
    args = parser.parse_args()
    
    # Configure logging based on terminal_output flag
    if args.terminal_output:
        # Configure detailed logging for terminal mode
        logger.setLevel(logging.INFO)
        # Clear any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add new handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)
        logger.info(f"Starting OpenAI-based RAG with query: {args.query}")
        logger.info(f"Arguments: {args}")
    else:
        # Ensure ALL logging is disabled when not in terminal mode
        logger.setLevel(logging.CRITICAL)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.addHandler(logging.NullHandler())
        # Also disable other libraries' logging
        logging.getLogger().setLevel(logging.CRITICAL)
    
    try:
        # Determine which file to use (prioritize input_file if both are provided)
        if args.input_file:
            file_to_use = args.input_file
        else:
            error_msg = "Error: Either --input_file or --embeddings_file must be provided"
            if args.terminal_output:
                logger.error(error_msg)
                return 1
            else:
                print(json.dumps({"error": error_msg}))
                return 1

        # Get file type (either explicitly specified or inferred from extension)
        if args.file_type:
            file_type = args.file_type.lower()
        else:
            _, file_extension = os.path.splitext(file_to_use)
            file_type = file_extension[1:].lower()  # Remove the leading dot
        
        if args.terminal_output:
            logger.info(f"Using file: {file_to_use} with type: {file_type}")
        
        # Handle CSV files by converting them to JSON
        if file_type == 'csv':
            if args.terminal_output:
                logger.info(f"CSV file detected: {file_to_use}")
                logger.info("Converting to JSON format for compatibility with OpenAI vector store...")
            json_file = convert_csv_to_json(file_to_use)
            if json_file:
                file_to_use = json_file
                file_type = 'json'
                if args.terminal_output:
                    logger.info(f"Using converted file: {file_to_use}")
            else:
                error_msg = "Error: Failed to convert CSV to JSON"
                if args.terminal_output:
                    logger.error(error_msg)
                    return 1
                else:
                    print(json.dumps({"error": error_msg}))
                    return 1

        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            error_msg = "Error: OPENAI_API_KEY not found in .env file"
            if args.terminal_output:
                logger.error(error_msg)
                return 1
            else:
                print(json.dumps({"error": error_msg}))
                return 1
        
        # Initialize components and run the RAG pipeline
        if args.terminal_output:
            logger.info("Initializing OpenAI client and components")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)
        
        # Initialize vector store and assistant managers
        vector_store_manager = OpenAIVectorStoreManager(client)
        assistant_manager = OpenAIAssistantManager(client, args.llm_model, args.input_file)
        
        # Create a vector store
        if args.terminal_output:
            logger.info("Creating vector store...")
        vector_store_id = vector_store_manager.create_vector_store(f"RAG-VectorStore-{uuid.uuid4().hex[:8]}")
        
        # Upload file to vector store
        if args.terminal_output:
            logger.info(f"Uploading file to vector store: {file_to_use}")
        
        try:
            vector_store_manager.upload_file_to_vector_store(vector_store_id, file_to_use)
        except Exception as e:
            error_msg = f"Error: Failed to upload file to vector store: {e}"
            if args.terminal_output:
                logger.error(error_msg)
                return 1
            else:
                print(json.dumps({"error": error_msg}))
                return 1
        
        # Create assistant with the vector store
        if args.terminal_output:
            logger.info("Creating assistant...")
        assistant_manager.create_assistant(vector_store_id)
        
        # Step 1: Search and generate response
        start_time = time.time()
        if args.terminal_output:
            logger.info(f"Searching for content related to query: '{args.query}'")
        response, context_info = assistant_manager.search_and_generate(args.query)
        search_time = time.time() - start_time
        
        # Format for compatibility with evaluate_rag.py
        formatted_context = format_context(context_info)
        
        # Prepare output data structure
        output_data = {
            "query": args.query,
            "retrieved_results": len(context_info.get("context_text", [])), 
            "search_time": round(search_time, 2),
            "similarity_scores": context_info.get("similarity_scores", []),
            "context_text": context_info.get("context_text", []),
            "metadata": context_info.get("metadata", []),
        }
        
        if args.terminal_output:
            logger.info("\n=== RAG RETRIEVAL INFO ===")
            logger.info(f"Query: {args.query}")
            logger.info(f"Retrieved {len(context_info.get('context_text', []))} results in {search_time:.2f} seconds")
            logger.info(f"Similarity scores: {context_info.get('similarity_scores', [])}")
            logger.info("========================\n")
        
        
        # Add response to output data
        output_data["response"] = response
        output_data["generation_time"] = 'N/A'
        output_data["total_time"] = round(output_data["search_time"], 2)
        
        # Ensure output_data is JSON serializable
        serializable_output = ensure_json_serializable(output_data)
        
        # Output based on mode
        if args.terminal_output:
            # Human-readable format for terminal
            logger.info("\n=== RESPONSE ===")
            # Always use print for the actual response to ensure it's visible
            print(response)  
            logger.info("\n========================")
            
            # Also print JSON for completeness
            logger.info("\n=== JSON OUTPUT ===")
            print(json.dumps(serializable_output))
            
            # Clean up resources
            logger.info("Cleaning up resources...")
        else:
            # ONLY output JSON for programmatic use, nothing else - no logging, no extra prints
            # This MUST be the ONLY output when not in terminal mode
            # Clean up resources first to avoid any logs after JSON
            assistant_manager.delete_assistant()
            # Now print the clean JSON output
            sys.stdout.write(json.dumps(serializable_output))
            sys.stdout.flush()
            return 0
        
        # Only reach here in terminal mode
        assistant_manager.delete_assistant()
        
        return 0
    
    except Exception as e:
        # Handle exceptions
        error_data = {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        
        if args.terminal_output:
            logger.error(f"Error in main function: {e}")
            logger.error(traceback.format_exc())
            print(f"Error: {str(e)}")
            
            # Also print JSON error for completeness in terminal mode
            logger.info("\n=== JSON ERROR OUTPUT ===")
            print(json.dumps(error_data))
        else:
            # ONLY output clean JSON error with no other output
            error_data = ensure_json_serializable(error_data)
            sys.stdout.write(json.dumps(error_data))
            sys.stdout.flush()
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 