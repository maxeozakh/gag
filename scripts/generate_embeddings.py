import os
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import argparse
from tqdm import tqdm # type: ignore
import pickle
import json

# Load environment variables from .env file
load_dotenv()

def get_embedding(text, client, model="text-embedding-3-small"):
    """Get embedding for a text using OpenAI API."""
    try:
        # Handle NaN values
        if pd.isna(text):
            text = ""
        
        text = str(text).replace("\n", " ")
        
        response = client.embeddings.create(
            model=model,
            input=text
        )
        
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

def format_text_for_embedding(row, columns, template=None):
    """Format multiple columns into a single text representation.
    
    Args:
        row: A pandas Series representing a row from DataFrame
        columns: List of column names to include
        template: Optional template string with {column_name} placeholders
                  If None, will use a default template
    
    Returns:
        Formatted text combining values from specified columns
    """
    # Create a dictionary with column values, handling missing values
    values = {}
    for col in columns:
        if col in row and not pd.isna(row[col]):
            values[col] = row[col]
        else:
            values[col] = ""
    
    # If no template provided, create a default one
    if not template:
        parts = []
        for col in columns:
            if col in values and values[col]:
                # Format numeric values appropriately
                if isinstance(values[col], (int, float)):
                    if col.lower().find('price') >= 0:
                        parts.append(f"{col}: ${values[col]}")
                    else:
                        parts.append(f"{col}: {values[col]}")
                else:
                    parts.append(f"{col}: {values[col]}")
        return " ".join(parts)
    else:
        # Use provided template
        try:
            return template.format(**values)
        except KeyError as e:
            print(f"Warning: Template references undefined column {e}")
            # Fall back to default formatting
            return " ".join([f"{col}: {values[col]}" for col in columns if col in values])

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for CSV data using OpenAI API")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file for embeddings")
    parser.add_argument("--columns", type=str, nargs='+', required=True, 
                        help="Names of columns to include in the text representation (space-separated)")
    parser.add_argument("--id_column", type=str, help="Optional column to use as the identifier instead of row index")
    parser.add_argument("--template", type=str, help="Optional template string for formatting text. Use {column_name} as placeholder.")
    parser.add_argument("--model", type=str, default="text-embedding-3-small", help="OpenAI embedding model to use")
    parser.add_argument("--format", type=str, choices=["pickle", "json"], default="json", help="Output format (pickle or json)")
    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = pd.read_csv(args.input_file)
    
    # Validate columns
    missing_columns = [col for col in args.columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Columns not found in the CSV file: {missing_columns}. Available columns: {df.columns.tolist()}")
    
    if args.id_column and args.id_column not in df.columns:
        raise ValueError(f"ID column '{args.id_column}' not found in the CSV file.")
    
    # Generate embeddings
    print(f"Generating embeddings for {len(df)} records using model {args.model}...")
    embeddings = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Combine multiple columns into a single text representation
        text = format_text_for_embedding(row, args.columns, args.template)
        
        # Generate embedding
        embedding = get_embedding(text, client, args.model)
        
        if embedding:
            # Use specified ID column or row index as key
            key = row[args.id_column] if args.id_column else idx
            
            # Create a dictionary to store both the original values and the embedding
            record = {
                "text": text,
                "embedding": embedding
            }
            
            # Add original column values for reference
            for col in args.columns:
                record[col] = row[col] if col in row else None
                
            embeddings[key] = record
    
    # Save embeddings
    print(f"Saving embeddings to {args.output_file}...")
    if args.format == "pickle":
        with open(args.output_file, "wb") as f:
            pickle.dump(embeddings, f)
    else:  # json
        # Convert numpy arrays to lists for JSON serialization
        json_compatible = {}
        for k, v in embeddings.items():
            json_compatible[k] = {**v, "embedding": list(v["embedding"])}
            
        with open(args.output_file, "w") as f:
            json.dump(json_compatible, f, indent=2)
    
    print(f"Successfully generated and saved embeddings for {len(embeddings)} records.")

if __name__ == "__main__":
    main() 