import os
import argparse
import json
import sys
import time
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dotenv import load_dotenv
from openai import OpenAI
import uuid
import csv
import re

# Configure logging with a null handler by default
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger("openai-rag")
logger.addHandler(logging.NullHandler())

# Load environment variables from .env file
load_dotenv()

# Import csv_to_json module
try:
    # First try to import directly if the script is in the same directory
    try:
        from csv_to_json import convert_csv_to_json
    except ImportError:
        # Try to import from scripts directory
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from csv_to_json import convert_csv_to_json
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
        # No need to store product_data since we're using pure file search
        
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
            assistant = self.client.beta.assistants.create(
                name=self.assistant_name,
                instructions="""You are a helpful product information assistant. 
                Use the provided documents to answer questions about products.
                If you don't know the answer, say so. Do not make up information.""",
                model=self.model,
                tools=[{"type": "file_search"}],
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
        
        logger.info(f"Searching for content related to query: '{query}'")
        
        # Use the OpenAI Assistant API for file search
        if not self.assistant_id:
            logger.error("Assistant not created")
            raise Exception("Assistant not created")
        
        try:
            # Create a thread for search and generate
            logger.info("Creating thread for search and generate")
            thread = self.client.beta.threads.create()
            logger.info(f"Thread created with ID: {thread.id}")
            
            # Add user message to thread
            logger.info(f"Adding user message to thread: '{query}'")
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            
            # Run the assistant on the thread
            logger.info(f"Running assistant (ID: {self.assistant_id}) on thread")
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            logger.info(f"Run created with ID: {run.id}")
            
            # Initialize counters
            total_tool_calls = 0
            total_file_searches = 0
            total_results = 0
            
            # Poll for run completion
            while True:
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                logger.info(f"Run status: {run.status}")
                
                if run.status == "completed":
                    logger.info("Run completed successfully")
                    break
                elif run.status in ["failed", "cancelled", "expired"]:
                    logger.error(f"Run ended with status: {run.status}")
                    if hasattr(run, "last_error") and run.last_error:
                        logger.error(f"Error: {run.last_error}")
                    raise Exception(f"Run ended with status: {run.status}")
                
                # Wait before polling again
                logger.info("Waiting for run to complete...")
                time.sleep(1)
            
            # Get run steps to extract file search results
            logger.info(f"Retrieving run steps to extract context")
            run_steps = self.client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id,
                include=["step_details.tool_calls[*].file_search.results[*].content"]
            )
            
            logger.info(f"Found {len(run_steps.data)} run steps")
            
            # Process run steps to extract context
            for step in run_steps.data:
                logger.info(f"Processing step: {step.type}")
                
                if step.type == "tool_calls" and hasattr(step, "step_details"):
                    # Check if step_details has tool_calls attribute
                    if not hasattr(step.step_details, "tool_calls"):
                        logger.info("Step details has no tool_calls attribute")
                        continue
                    
                    tool_calls = step.step_details.tool_calls
                    logger.info(f"Found {len(tool_calls)} tool calls in step")
                    total_tool_calls += len(tool_calls)
                    
                    for tool_call in tool_calls:
                        # Detailed logging of tool call structure
                        logger.info(f"Tool call object type: {type(tool_call)}")
                        
                        # For dictionary type tool calls
                        if isinstance(tool_call, dict):
                            logger.info(f"Tool call keys: {tool_call.keys()}")
                            
                            # Handle new API format with 'id', 'type', 'file_search' keys
                            if 'type' in tool_call and tool_call['type'] == 'file_search' and 'file_search' in tool_call:
                                logger.info(f"Processing file_search tool call with id: {tool_call.get('id')}")
                                total_file_searches += 1
                                
                                file_search_data = tool_call['file_search']
                                if isinstance(file_search_data, dict) and 'results' in file_search_data:
                                    results = file_search_data['results']
                                    if isinstance(results, list):
                                        logger.info(f"Found {len(results)} results in file_search")
                                        total_results += len(results)
                                        
                                        for result in results:
                                            if isinstance(result, dict):
                                                file_id = result.get('file_id', 'unknown')
                                                file_name = result.get('file_name', 'unknown_file')
                                                score = result.get('score', 0.0)
                                                
                                                # Extract content from the result
                                                content = result.get('content', '')
                                                
                                                # Log search result details
                                                content_preview = content[:100] + "..." if len(content) > 100 else content
                                                logger.info(f"  File ID: {file_id}, File: {file_name}, Score: {score}")
                                                logger.info(f"  Content preview: {content_preview}")
                                                
                                                # Add content to context
                                                context_info["context_text"].append(content)
                                                context_info["similarity_scores"].append(score)
                                                context_info["metadata"].append({
                                                    "source": file_name,
                                                    "file_id": file_id
                                                })
                                else:
                                    logger.warning("No results found in file_search data")
                                    logger.info(f"file_search data: {file_search_data}")
                                
                                continue
                            
                        # Handle object-style tool call with no type attribute
                        if not hasattr(tool_call, "type"):
                            logger.info("Tool call has no type attribute")
                            
                            # Try to access common attributes/structure
                            if hasattr(tool_call, "id"):
                                logger.info(f"Tool call id: {tool_call.id}")
                            
                            # Check if it has a file_search attribute
                            if hasattr(tool_call, "file_search"):
                                logger.info("Tool call has file_search attribute")
                                file_search = tool_call.file_search
                                logger.info(f"File search type: {type(file_search)}")
                                logger.info(f"File search attributes: {dir(file_search)}")
                                
                                # Check for results
                                if hasattr(file_search, "results"):
                                    results = file_search.results
                                    logger.info(f"File search results found: {len(results)}")
                                    
                                    # Process the results
                                    for result in results:
                                        logger.info(f"Result type: {type(result)}")
                                        logger.info(f"Result attributes: {dir(result)}")
                                        
                                        # Extract file_id and content
                                        file_id = getattr(result, "file_id", "unknown")
                                        content = getattr(result, "content", "")
                                        
                                        logger.info(f"Found result - file_id: {file_id}")
                                        logger.info(f"Content preview: {content[:100]}...")
                                        
                                        # Add to context
                                        context_info["context_text"].append(content)
                                        context_info["similarity_scores"].append(0.95)
                                        context_info["metadata"].append({
                                            "source": "file_search",
                                            "file_id": file_id
                                        })
                                        
                                        total_results += 1
                            else:
                                logger.info("Tool call doesn't have file_search attribute")
                            
                            continue
            
            # Log search statistics
            logger.info(f"Search summary: {total_tool_calls} tool calls, {total_file_searches} file searches, {total_results} total results")
            
            # If no context items were found, add a note
            if not context_info["context_text"]:
                logger.warning("No context items found")
            
            # Get messages
            logger.info("Retrieving messages from thread")
            messages = self.client.beta.threads.messages.list(
                thread_id=thread.id
            )
            
            # Get assistant's reply (most recent message from assistant)
            assistant_messages = [msg for msg in messages.data if msg.role == "assistant"]
            if not assistant_messages:
                logger.error("No assistant response found")
                raise Exception("No assistant response found")
            
            # Extract the most recent assistant message
            assistant_message = assistant_messages[0]
            
            # Extract text content from the message
            response_text = ""
            if hasattr(assistant_message, "content") and assistant_message.content:
                for content_item in assistant_message.content:
                    if hasattr(content_item, "text") and content_item.text:
                        response_text += content_item.text.value
            
            # Clean up: delete the thread
            logger.info(f"Deleting thread: {thread.id}")
            self.client.beta.threads.delete(thread.id)
            
            return response_text, context_info
        
        except Exception as e:
            logger.error(f"Error during search and generate: {e}")
            logger.error(traceback.format_exc())
            return f"Error during search: {str(e)}", context_info

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
            context_parts.append(f"[Result {i}] {text}")
    
    formatted_context = "\n\n".join(context_parts)
    logger.info(f"Formatted context of length {len(formatted_context)}")
    return formatted_context

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
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.handlers = [handler]
        logger.info(f"Starting OpenAI-based RAG with query: {args.query}")
        logger.info(f"Arguments: {args}")
    else:
        # Ensure ALL logging is disabled when not in terminal mode
        logger.setLevel(logging.CRITICAL)
        logger.handlers = [logging.NullHandler()]
        # Also disable other libraries' logging
        logging.getLogger().setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)
        logging.getLogger("openai").setLevel(logging.CRITICAL)
    
    try:
        # Determine which file to use (prioritize input_file if both are provided)
        if args.input_file:
            file_to_use = args.input_file
        elif args.embeddings_file:
            file_to_use = args.embeddings_file
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
        if not vector_store_manager.upload_file_to_vector_store(vector_store_id, file_to_use):
            error_msg = "Error: Failed to upload file to vector store"
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
        
        # For compatibility with the evaluation script, use the standard RAG prompt
        if context_info and context_info.get("context_text"):
            if args.terminal_output:
                logger.info("Formatting response with RAG prompt template...")
            prompt = RAG_PROMPT.format(context=formatted_context, query=args.query)
            
            start_time = time.time()
            response = call_openai(client, prompt, args.llm_model)
            generation_time = time.time() - start_time
        else:
            # If no context was found, use the direct response from the assistant
            generation_time = 0
        
        # Add response to output data
        output_data["response"] = response
        output_data["generation_time"] = round(generation_time, 2)
        output_data["total_time"] = round(output_data["search_time"] + output_data["generation_time"], 2)
        
        # Output based on mode
        if args.terminal_output:
            # Human-readable format for terminal
            logger.info("\n=== RESPONSE ===")
            # Always use print for the actual response to ensure it's visible
            print(response)  
            logger.info("\n========================")
            logger.info(f"Generated in {generation_time:.2f} seconds")
            
            # Also print JSON for completeness
            logger.info("\n=== JSON OUTPUT ===")
            print(json.dumps(output_data))
            
            # Clean up resources
            logger.info("Cleaning up resources...")
        else:
            # ONLY output JSON for programmatic use, nothing else - no logging, no extra prints
            # This MUST be the ONLY output when not in terminal mode
            # Clean up resources first to avoid any logs after JSON
            assistant_manager.delete_assistant()
            # Now print the clean JSON output
            sys.stdout.write(json.dumps(output_data))
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
            sys.stdout.write(json.dumps(error_data))
            sys.stdout.flush()
        
        return 1

if __name__ == "__main__":
    sys.exit(main()) 