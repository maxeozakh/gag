import csv
from typing import Callable, Any

def process_csv_file(
    file_path: str,
    row_processor: Callable[[dict], Any],
    encoding: str = 'utf-8',
    delimiter: str = ','
) -> None:
    """
    Process a CSV file row by row and apply a given function to each row.
    
    Args:
        file_path: Path to the CSV file
        row_processor: Function to process each row
        encoding: File encoding (default: utf-8)
        delimiter: CSV delimiter (default: comma)
    """
    try:
        with open(file_path, 'r', encoding=encoding) as csv_file:
            csv_reader = csv.DictReader(csv_file, delimiter=delimiter)
            
            for row in csv_reader:
                row_processor(row)
                
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"Error processing CSV file: {str(e)}") 