from typing import Dict, List, Optional
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")
    pass

def calculate_f1(prediction: str, reference: str) -> Dict[str, float]:
    """Calculate F1 score between prediction and reference."""
    try:
        # Simple fallback tokenization if NLTK fails
        def simple_tokenize(text: str) -> set:
            return set(text.lower().split())
            
        try:
            pred_tokens = set(word_tokenize(prediction.lower()))
            ref_tokens = set(word_tokenize(reference.lower()))
        except:
            # Fallback to simple tokenization
            pred_tokens = simple_tokenize(prediction)
            ref_tokens = simple_tokenize(reference)
        
        # Calculate intersection
        intersection = pred_tokens.intersection(ref_tokens)
        
        # Calculate precision and recall
        precision = len(intersection) / len(pred_tokens) if pred_tokens else 0
        recall = len(intersection) / len(ref_tokens) if ref_tokens else 0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }
    except Exception as e:
        print(f"Error calculating F1: {str(e)}")
        return {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0
        }

class TokenMetrics:
    @staticmethod
    def calculate_f1(prediction: str, reference: str) -> Dict[str, float]:
        return calculate_f1(prediction, reference)
    
    @staticmethod
    def validate_key_facts(prediction: str, key_facts: List[str]) -> Dict[str, bool]:
        """Check if key facts are present in the prediction."""
        prediction_lower = prediction.lower()
        return {
            fact: fact.lower() in prediction_lower
            for fact in key_facts if fact
        } 