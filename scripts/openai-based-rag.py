import os
import argparse
import json
import sys
import time
import traceback
import logging
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("openai-rag")

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
    def __init__(self, client: OpenAI, model: str = "gpt-4o"):
        self.client = client
        self.model = model
        self.assistant_id: Optional[str] = None
        
    def create_assistant(self, vector_store_id: str) -> str:
        """
        Create an assistant with file search capability.
        """
        try:
            # Create a new assistant with file search
            assistant_name = f"RAG-Assistant-{uuid.uuid4().hex[:8]}"
            logger.info(f"Creating assistant: {assistant_name} with model {self.model}")
            assistant = self.client.beta.assistants.create(
                name=assistant_name,
                description="An assistant for retrieval-augmented generation",
                model=self.model,
                tools=[{"type": "file_search"}]
            )
            self.assistant_id = assistant.id
            logger.info(f"Assistant created with ID: {self.assistant_id}")
            
            # Update the assistant to use the vector store
            logger.info(f"Updating assistant to use vector store: {vector_store_id}")
            self.client.beta.assistants.update(
                assistant_id=self.assistant_id,
                tool_resources={"file_search": {"vector_store_ids": [vector_store_id]}}
            )
            logger.info("Assistant updated successfully")
                
            return assistant.id
        except Exception as e:
            logger.error(f"Error creating assistant: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def delete_assistant(self):
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
        Returns the response and context information.
        """
        if not self.assistant_id:
            logger.error("Assistant not created")
            raise ValueError("Assistant not created")
        
        try:
            # Create a thread
            logger.info("Creating thread for search and generate")
            thread = self.client.beta.threads.create()
            logger.info(f"Thread created with ID: {thread.id}")
            
            # Add a message to the thread
            logger.info(f"Adding user message to thread: {query}")
            self.client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=query
            )
            
            # Run the assistant
            logger.info(f"Running assistant on thread")
            run = self.client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=self.assistant_id
            )
            logger.info(f"Run created with ID: {run.id}")
            
            # Poll for completion
            while True:
                run_status = self.client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                logger.info(f"Run status: {run_status.status}")
                
                if run_status.status == "completed":
                    logger.info("Run completed successfully")
                    break
                elif run_status.status in ["failed", "cancelled", "expired"]:
                    error_message = f"Run ended with status: {run_status.status}"
                    if hasattr(run_status, 'last_error'):
                        error_message += f", Error: {run_status.last_error}"
                    logger.error(error_message)
                    raise Exception(error_message)
                    
                time.sleep(1)
            
            # Get context information from run steps with file search results
            logger.info("Retrieving context information from run steps")
            context_info: Dict[str, Any] = {"context_text": [], "similarity_scores": []}
            
            # Get the file search results from run steps
            logger.info("Retrieving run steps")
            run_steps = self.client.beta.threads.runs.steps.list(
                thread_id=thread.id,
                run_id=run.id,
                include=["step_details.tool_calls[*].file_search.results[*].content"]
            )
            logger.info(f"Retrieved {len(run_steps.data)} run steps")
            
            # Process the search results and build context
            for step in run_steps.data:
                logger.debug(f"Processing step: {step.id}")
                if hasattr(step, 'step_details') and hasattr(step.step_details, 'type') and step.step_details.type == "tool_calls":
                    for tool_call in step.step_details.tool_calls:
                        if hasattr(tool_call, 'type') and tool_call.type == "file_search" and hasattr(tool_call, 'file_search'):
                            if hasattr(tool_call.file_search, 'results') and tool_call.file_search.results is not None:
                                logger.info(f"Found {len(tool_call.file_search.results)} search results")
                                for result in tool_call.file_search.results:
                                    if hasattr(result, 'content') and result.content:
                                        context_info["context_text"].append(result.content)
                                        # Use placeholder similarity score since actual scores aren't provided
                                        context_info["similarity_scores"].append(0.95)
                                        logger.debug(f"Added context: {result.content[:50]}...")
            
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
            
            logger.info(f"Found {len(assistant_messages)} assistant messages")
            response_content = ""
            
            # Extract text and file citations from the message
            for message in assistant_messages:
                for content_part in message.content:
                    if hasattr(content_part, 'type') and content_part.type == "text":
                        response_content += content_part.text.value
                        logger.debug(f"Added response content: {content_part.text.value[:50]}...")
                        
                        # Try to extract file citations if they exist
                        if hasattr(content_part.text, 'annotations'):
                            for annotation in content_part.text.annotations:
                                if hasattr(annotation, 'type') and annotation.type == "file_citation":
                                    # Extract file content and add to context if not already present
                                    quoted_text = annotation.text
                                    if quoted_text not in context_info["context_text"]:
                                        context_info["context_text"].append(quoted_text)
                                        # Since we don't have real similarity scores, use placeholder
                                        context_info["similarity_scores"].append(0.95)
                                        logger.debug(f"Added citation context: {quoted_text[:50]}...")
            
            # Clean up
            logger.info(f"Deleting thread: {thread.id}")
            self.client.beta.threads.delete(thread.id)
            
            # Log the final results
            logger.info(f"Generated response of length {len(response_content)}")
            logger.info(f"Collected {len(context_info['context_text'])} context items")
            
            return response_content, context_info
        
        except Exception as e:
            logger.error(f"Error using assistant: {e}")
            logger.error(traceback.format_exc())
            raise

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
    args = parser.parse_args()
    
    logger.info(f"Starting OpenAI-based RAG with query: {args.query}")
    logger.info(f"Arguments: {args}")
    
    # Determine which file to use (prioritize input_file if both are provided)
    if args.input_file:
        file_to_use = args.input_file
    elif args.embeddings_file:
        file_to_use = args.embeddings_file
    else:
        logger.error("Error: Either --input_file or --embeddings_file must be provided")
        return 1

    # Get file type (either explicitly specified or inferred from extension)
    if args.file_type:
        file_type = args.file_type.lower()
    else:
        _, file_extension = os.path.splitext(file_to_use)
        file_type = file_extension[1:].lower()  # Remove the leading dot
    
    logger.info(f"Using file: {file_to_use} with type: {file_type}")
    
    # Handle CSV files by converting them to JSON
    if file_type == 'csv':
        logger.info(f"CSV file detected: {file_to_use}")
        logger.info("Converting to JSON format for compatibility with OpenAI vector store...")
        json_file = convert_csv_to_json(file_to_use)
        if json_file:
            file_to_use = json_file
            file_type = 'json'
            logger.info(f"Using converted file: {file_to_use}")
        else:
            logger.error("Error: Failed to convert CSV to JSON")
            return 1

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("Error: OPENAI_API_KEY not found in .env file")
        return 1
    
    # Initialize OpenAI client
    logger.info("Initializing OpenAI client")
    client = OpenAI(api_key=api_key)
    
    try:
        # Initialize vector store and assistant managers
        logger.info("Initializing vector store and assistant managers")
        vector_store_manager = OpenAIVectorStoreManager(client)
        assistant_manager = OpenAIAssistantManager(client, args.llm_model)
        
        # Create a vector store
        logger.info("Creating vector store...")
        vector_store_id = vector_store_manager.create_vector_store(f"RAG-VectorStore-{uuid.uuid4().hex[:8]}")
        
        # Upload file to vector store
        logger.info(f"Uploading file to vector store: {file_to_use}")
        if not vector_store_manager.upload_file_to_vector_store(vector_store_id, file_to_use):
            logger.error("Error: Failed to upload file to vector store")
            return 1
        
        # Create assistant with the vector store
        logger.info("Creating assistant...")
        assistant_manager.create_assistant(vector_store_id)
        
        # Step 1: Search and generate response
        start_time = time.time()
        logger.info(f"Searching for content related to query: '{args.query}'")
        response, context_info = assistant_manager.search_and_generate(args.query)
        search_time = time.time() - start_time
        
        # Format for compatibility with evaluate_rag.py
        formatted_context = format_context(context_info)
        
        # Print retrieval information
        logger.info("\n=== RAG RETRIEVAL INFO ===")
        logger.info(f"Query: {args.query}")
        logger.info(f"Retrieved {len(context_info.get('context_text', []))} results in {search_time:.2f} seconds")
        logger.info(f"Similarity scores: {context_info.get('similarity_scores', [])}")
        logger.info("========================\n")
        
        # For compatibility with the evaluation script, use the standard RAG prompt
        if context_info and context_info.get("context_text"):
            logger.info("Formatting response with RAG prompt template...")
            prompt = RAG_PROMPT.format(context=formatted_context, query=args.query)
            
            start_time = time.time()
            response = call_openai(client, prompt, args.llm_model)
            generation_time = time.time() - start_time
        else:
            # If no context was found, use the direct response from the assistant
            generation_time = 0
        
        # Print the response
        logger.info("\n=== RESPONSE ===")
        print(response)  # Use print for the actual response to ensure it's captured by the evaluation script
        logger.info("\n========================")
        logger.info(f"Generated in {generation_time:.2f} seconds")
        
        # Clean up resources
        logger.info("Cleaning up resources...")
        assistant_manager.delete_assistant()
        
        return 0
    
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")  # Ensure the error is printed for the evaluation script
        return 1

if __name__ == "__main__":
    sys.exit(main()) 