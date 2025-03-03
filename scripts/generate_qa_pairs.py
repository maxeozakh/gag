import os
import pandas as pd
import argparse
from tqdm import tqdm # type: ignore
from typing import List, Dict, Any
import json
from dotenv import load_dotenv
import random
import re

# For text processing
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_loaders.csv_loader import CSVLoader
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

def extract_rating(rating_text: str) -> int:
    """Extract numerical rating from evaluation text."""
    numbers = re.findall(r'\d+', rating_text.split('\n')[0])
    if numbers:
        rating = int(numbers[0])
        if 1 <= rating <= 5:
            return rating
    raise ValueError(f"Failed to extract valid rating (1-5) from text: {rating_text}")

def call_openai(client: OpenAI, prompt: str, model: str = "gpt-3.5-turbo") -> str:
    """Call OpenAI API with a prompt and return the response."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that follows instructions precisely."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ""

def load_and_process_data(file_path: str, chunk_size: int = 2000, chunk_overlap: int = 200) -> List[LangchainDocument]:
    """Load data from CSV and split into chunks."""
    # Load the dataset using CSVLoader
    loader = CSVLoader(file_path=file_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")
    
    # Convert loaded documents to LangchainDocument format
    langchain_docs = [
        LangchainDocument(page_content=doc.page_content, metadata=doc.metadata)
        for doc in tqdm(documents, desc="Processing documents")
    ]
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    
    # Split documents into chunks
    processed_docs = []
    for doc in langchain_docs:
        processed_docs.extend(text_splitter.split_documents([doc]))
    
    print(f"Processed into {len(processed_docs)} document chunks.")
    return processed_docs

def generate_qa_pairs(
    client: OpenAI,
    documents: List[LangchainDocument], 
    num_pairs: int, 
    model: str = "gpt-3.5-turbo"
) -> List[Dict[str, Any]]:
    """Generate QA pairs from document chunks using OpenAI."""
    
    # Prompt template for generating QA pairs
    qa_generation_base_prompt = """
    Your task is to write a factoid question, an answer, and a key fact given a context.
    Your factoid question should be answerable with a specific, concise piece of factual information from the context.
    Your factoid question should be formulated in the same style as questions users could ask in a search engine.
    This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

    IMPORTANT: You must ALWAYS extract a key fact that represents the main product being discussed.
    The key fact should be a single, concise phrase (2-5 words) that captures the main idea of the context.
    Examples of good key facts:
    - "Logo baseball hat"
    - "Material: nylon"
    - "Cotton t-shirt"

    {avoid_similar}

    Provide your answer in EXACTLY this format:

    Output:::
    Factoid question: (your factoid question)
    Answer: (your answer to the factoid question)
    Key fact: (your key fact)

    Now here is the context.

    Context: {context}
    Output:::
    """
    
    # Prompts for evaluating quality of generated QA pairs
    evaluation_prompts = {
        "groundedness": """
        You will be given a context and a question.
        Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here are the question and context.

        Question: {question}
        Context: {context}
        Answer::: 
        """,
        
        "relevance": """
        You will be given a question.
        Your task is to provide a 'total rating' representing how useful this question can be for information retrieval.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}
        Answer::: 
        """,
        
        "standalone": """
        You will be given a question.
        Your task is to provide a 'total rating' representing how context-independent this question is.
        Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
        For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
        The questions can contain technical terms and still be a 5: it must simply be clear what the question is about.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}
        Answer::: 
        """
    }
    
    print(f"Generating {num_pairs} QA pairs...")
    outputs = []
    
    # Keep track of previously generated questions and key facts
    previous_questions = []
    previous_key_facts = []
    
    # Sample random documents
    sampled_docs = random.sample(documents, min(num_pairs, len(documents)))
    
    for idx, doc in enumerate(tqdm(sampled_docs)):
        try:
            # Create the avoid similar instruction
            avoid_similar_text = ""
            if previous_questions:
                avoid_similar_text = "IMPORTANT: Avoid generating questions similar to these previously generated questions:\n"
                # Add at most 10 previous questions to avoid making the prompt too long
                for i, (prev_q, prev_kf) in enumerate(zip(previous_questions, previous_key_facts)):
                    if i >= 10:
                        break
                    avoid_similar_text += f"- Question: {prev_q} (Key fact: {prev_kf})\n"
                avoid_similar_text += "\nCreate a question on a different aspect or category than those listed above."
            
            # Format the prompt with avoid similar instruction
            qa_generation_prompt = qa_generation_base_prompt.format(
                context=doc.page_content,
                avoid_similar=avoid_similar_text
            )
            
            # Generate QA pair
            qa_output = call_openai(
                client,
                qa_generation_prompt,
                model
            )
            
            # Parse the output
            output_parts = qa_output.split("\n")
            question = ""
            answer = ""
            key_fact = ""
            
            for part in output_parts:
                part = part.strip()
                if part.startswith("Factoid question:"):
                    question = part.replace("Factoid question:", "").strip()
                elif part.startswith("Answer:"):
                    answer = part.replace("Answer:", "").strip()
                elif part.startswith("Key fact:"):
                    key_fact = part.replace("Key fact:", "").strip()
            
            if not question or not answer or not key_fact:
                print(f"Skip: Missing question, answer, or key fact at index {idx + 1}")
                continue
                
            if len(answer) > 300:
                print(f"Skip: Answer too long at index {idx + 1}")
                continue
            
            # Add the new question and key fact to our tracking lists
            previous_questions.append(question)
            previous_key_facts.append(key_fact)
            
            # Create QA pair entry
            qa_pair = {
                "context": doc.page_content,
                "question": question,
                "answer": answer,
                "key_fact": key_fact,
                "source_doc": doc.metadata.get("source", ""),
                "groundedness_score": 0,
                "relevance_score": 0,
                "standalone_score": 0,
                "groundedness_eval": "",
                "relevance_eval": "",
                "standalone_eval": "",
            }
            
            # Evaluate QA pair quality
            print(f"Evaluating QA pair {idx + 1}")
            for criterion, prompt in evaluation_prompts.items():
                # Format the prompt based on the criterion
                formatted_prompt = prompt.format(
                    question=qa_pair["question"],
                    context=qa_pair["context"] if criterion == "groundedness" else ""
                )
                
                # Get evaluation
                eval_output = call_openai(client, formatted_prompt, model)
                
                # Extract rating and evaluation
                if "Total rating:" in eval_output and "Evaluation:" in eval_output:
                    rating_part = eval_output.split("Total rating:")[-1].split('\n')[0].strip()
                    eval_part = eval_output.split("Evaluation:")[-1].split("Total rating:")[0].strip()
                    
                    try:
                        score = extract_rating(rating_part)
                        qa_pair[f"{criterion}_score"] = score
                        qa_pair[f"{criterion}_eval"] = eval_part
                    except Exception as e:
                        print(f"Error parsing rating for {criterion}: {e}")
            
            outputs.append(qa_pair)
            print(f"Successfully generated QA pair {idx + 1}")
            
        except Exception as e:
            print(f"Error generating QA pair at index {idx + 1}: {e}")
            continue
    
    return outputs

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs from a dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the generated QA pairs")
    parser.add_argument("--num_pairs", type=int, default=20, help="Number of QA pairs to generate")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use")
    parser.add_argument("--quality_threshold", type=int, default=10, help="Minimum sum of quality scores (out of 15)")
    parser.add_argument("--chunk_size", type=int, default=2000, help="Size of text chunks")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between text chunks")
    args = parser.parse_args()
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load and process data
    processed_docs = load_and_process_data(args.input_file, args.chunk_size, args.chunk_overlap)
    
    # Generate QA pairs
    qa_pairs = generate_qa_pairs(client, processed_docs, args.num_pairs, args.model)
    
    if not qa_pairs:
        print("No QA pairs were generated successfully. Exiting...")
        return
    
    # Convert to DataFrame
    qa_df = pd.DataFrame(qa_pairs)
    
    # Filter by quality threshold
    print("Filtering QA pairs by quality...")
    filtered_qa_df = qa_df[
        (qa_df["groundedness_score"] + qa_df["relevance_score"] + qa_df["standalone_score"]) >= args.quality_threshold
    ]
    
    print(f"Generated {len(qa_df)} QA pairs, {len(filtered_qa_df)} passed quality threshold.")
    
    # Print evaluation statistics
    print("\nQuality metrics for filtered QA pairs:")
    print(filtered_qa_df[["groundedness_score", "relevance_score", "standalone_score"]].describe())
    
    # Add ID column
    filtered_qa_df['id'] = range(1, len(filtered_qa_df) + 1)
    
    # Save QA pairs to CSV
    columns_to_save = ["id", "question", "answer", "key_fact"]
    print(f"Saving {len(filtered_qa_df)} QA pairs to {args.output_file}")
    filtered_qa_df[columns_to_save].to_csv(args.output_file, index=False)
    
    print(f"QA pairs successfully saved to {args.output_file}")

if __name__ == "__main__":
    main() 