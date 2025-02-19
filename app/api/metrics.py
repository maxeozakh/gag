from thefuzz import fuzz # type: ignore
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Union
import os

class EnhancedKeyFactsValidator:
    def __init__(self):
        # Disable progress bar for sentence transformers
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        # Threshold for semantic similarity (0-1)
        self.semantic_threshold = 0.8
        # Threshold for fuzzy matching (0-100)
        self.fuzzy_threshold = 85

    def validate_key_facts(self, 
                          prediction: str, 
                          key_facts: List[str]) -> Dict[str, Dict[str, Union[bool, float]]]:
        """
        Enhanced validation of key facts using multiple approaches:
        1. Semantic similarity using sentence embeddings
        2. Fuzzy string matching for handling typos and minor variations
        3. Exact substring matching as a fallback
        """
        results = {}
        
        # Convert prediction to lowercase for better matching
        prediction_lower = prediction.lower()
        
        for fact in key_facts:
            fact_lower = fact.lower()
            
            # 1. Try semantic similarity first
            fact_embedding = self.model.encode([fact_lower])[0]
            pred_embedding = self.model.encode([prediction_lower])[0]
            semantic_similarity = np.dot(fact_embedding, pred_embedding) / (
                np.linalg.norm(fact_embedding) * np.linalg.norm(pred_embedding)
            )
            
            # 2. Try fuzzy matching
            fuzzy_ratio = fuzz.partial_ratio(fact_lower, prediction_lower)
            fuzzy_score = fuzzy_ratio / 100.0
            
            # 3. Exact substring match
            exact_match = fact_lower in prediction_lower
            
            # Combine all approaches
            is_match = (
                semantic_similarity >= self.semantic_threshold or
                fuzzy_score >= (self.fuzzy_threshold / 100) or
                exact_match
            )
            
            # Store detailed results
            results[fact] = {
                "match": is_match,
                "semantic_similarity": float(semantic_similarity),
                "fuzzy_score": fuzzy_score,
                "exact_match": exact_match
            }
            
        return results

    def get_best_match_explanation(self, 
                                 fact: str, 
                                 result: Union[Dict[str, Union[bool, float]], bool]) -> str:
        """Generate human-readable explanation for the matching result"""
        if isinstance(result, bool):
            return "Exact match found" if result else "No match found"
        
        if result["exact_match"]:
            return "Exact match found"
        elif result["semantic_similarity"] >= self.semantic_threshold:
            return f"Semantic similarity: {result['semantic_similarity']:.2f}"
        elif result["fuzzy_score"] >= (self.fuzzy_threshold / 100):
            return f"Fuzzy match score: {result['fuzzy_score']:.2f}"
        return "No significant match found" 