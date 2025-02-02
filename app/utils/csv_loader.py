import csv
import logging
from typing import Optional
from app.models.embeddings import get_embedding
from app.models.db_operations import save_vector, save_answer
from app.models.database import database

logger = logging.getLogger(__name__)

async def is_data_already_loaded(database) -> bool:
    """Check if mythology data is already in the database."""
    query = "SELECT COUNT(*) as count FROM vectors;"
    result = await database.fetch_one(query)
    return result['count'] > 0

async def load_mythology_data(csv_path: str) -> None:
    """
    Load mythology data from CSV file and save to database using vector embeddings.
    
    Args:
        csv_path: Path to the CSV file
    """
    if await is_data_already_loaded(database):
        logger.info("Mythology data already loaded, skipping...")
        return
        
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                try:
                    # Get embeddings for the question
                    question_embedding = await get_embedding(row['question'])
                    
                    # Save the question vector
                    vector_id = await save_vector(row['question'], question_embedding)
                    
                    # Save the answer
                    await save_answer(row['answer'], vector_id)
                    
                    logger.info(f"Successfully processed row with question: {row['question'][:50]}...")
                    
                except Exception as row_error:
                    logger.error(f"Error processing row: {str(row_error)}")
                    continue  # Continue with next row even if current one fails
                    
    except Exception as e:
        logger.error(f"Error loading mythology data: {str(e)}")
        raise 