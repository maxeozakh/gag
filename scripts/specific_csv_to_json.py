import csv
import json
import argparse
import os

def convert_csv_to_json(csv_file, json_file=None):
    """
    Convert a CSV file to a JSON format that can be used with OpenAI's vector store.
    
    Args:
        csv_file: Path to the CSV file
        json_file: Optional path to save the JSON file, defaults to same filename with .json extension
    
    Returns:
        Path to the generated JSON file
    """
    # If json_file is not specified, use the same name as csv_file but with .json extension
    if json_file is None:
        base_name = os.path.splitext(csv_file)[0]
        json_file = f"{base_name}.json"
    
    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Create JSON documents with metadata
    documents = []
    for i, row in enumerate(rows):
        # Combine all fields into a text document
        text = "Product Information:\n"
        for key, value in row.items():
            if value and value.lower() != "unknown":
                text += f"{key}: {value}\n"
        
        # Create a document with text and metadata
        document = {
            "id": f"doc_{i}",
            "text": text,
        }
        documents.append(document)
    
    # Write to JSON file
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    
    print(f"Converted {len(documents)} rows from {csv_file} to {json_file}")
    return json_file

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to JSON for OpenAI vector store")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the CSV file to convert")
    parser.add_argument("--json_file", type=str, required=False, help="Path to save the JSON file (optional)")
    
    args = parser.parse_args()
    convert_csv_to_json(args.csv_file, args.json_file)

if __name__ == "__main__":
    main() 