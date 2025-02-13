from typing import Dict, List, Optional, Tuple
import pandas as pd
from app.evaluation.token_metrics import calculate_f1
from app.api.routers import chat
from app.models import ChatPayload
from app.evaluation.token_metrics import TokenMetrics

class Evaluator:
    def __init__(self, ground_truth_path: str):
        """Initialize evaluator with ground truth data."""
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        
    async def evaluate_single_query(self, 
                                  query: str, 
                                  ground_truth: str, 
                                  key_facts: Optional[List[str]] = None) -> Dict:
        """Evaluate a single query against its ground truth."""
        try:
            # Create payload
            payload = ChatPayload(
                query=query,
                ground_truth=ground_truth,
                key_facts=key_facts
            )
            
            # Get response from chat endpoint
            response = await chat(payload)
            
            # Calculate metrics using our token_metrics
            metrics = calculate_f1(prediction=response["answer"], reference=ground_truth)
            
            # Debug print
            print(f"\nQuery: {query[:50]}...")
            print(f"Predicted: {response['answer'][:50]}...")
            print(f"Ground Truth: {ground_truth[:50]}...")
            print(f"Calculated Metrics: {metrics}")
            
            return {
                "query": query,
                "predicted": response["answer"],
                "ground_truth": ground_truth,
                "f1_score": metrics["f1"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "key_facts_match": TokenMetrics.validate_key_facts(response["answer"], key_facts) if key_facts else {}
            }
            
        except Exception as e:
            print(f"Error evaluating query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0
            }

    async def evaluate_all(self, limit: Optional[int] = None) -> Tuple[List[Dict], Dict]:
        """Evaluate queries in the ground truth dataset.
        
        Args:
            limit: Optional number of queries to evaluate. If None, evaluates all queries.
        """
        results = []
        # Get random sample if limit is specified, otherwise use all data
        df = self.ground_truth_df.sample(n=limit) if limit else self.ground_truth_df
        total = len(df)
        
        for idx, row in df.iterrows():
            print(f"\n📝 Processing query {idx + 1}/{total}")
            result = await self.evaluate_single_query(
                query=row["question"],
                ground_truth=row["answer"],
                key_facts=[row["key_fact"]] if pd.notna(row["key_fact"]) else None
            )
            results.append(result)
            
            if "error" in result:
                print(f"❌ Error in query {idx + 1}: {result['error']}")
        
        print("\n✅ Evaluation complete!")
        
        # Calculate aggregate metrics
        aggregate_metrics = self.calculate_aggregate_metrics(results)
        
        return results, aggregate_metrics
    
    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all results."""
        # Consider a result valid if it has non-zero metrics
        valid_results = [r for r in results if "error" not in r and r["f1_score"] > 0]
        
        if not valid_results:
            return {
                "avg_f1": 0.0,
                "avg_precision": 0.0,
                "avg_recall": 0.0,
                "total_queries": len(results),
                "successful_queries": 0,
                "key_facts_success_rate": 0.0
            }
            
        # Calculate key facts success rate
        key_facts_results = [
            r for r in valid_results 
            if "key_facts_match" in r and r["key_facts_match"]
        ]
        total_facts = sum(len(r["key_facts_match"]) for r in key_facts_results)
        matched_facts = sum(
            sum(1 for matched in r["key_facts_match"].values() if matched)
            for r in key_facts_results
        )
        key_facts_rate = matched_facts / total_facts if total_facts > 0 else 0
        
        return {
            "avg_f1": sum(r["f1_score"] for r in valid_results) / len(valid_results),
            "avg_precision": sum(r["precision"] for r in valid_results) / len(valid_results),
            "avg_recall": sum(r["recall"] for r in valid_results) / len(valid_results),
            "total_queries": len(results),
            "successful_queries": len(valid_results),
            "key_facts_success_rate": key_facts_rate
        } 