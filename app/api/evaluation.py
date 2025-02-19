from typing import Dict, List, Optional, Literal
import pandas as pd
from app.evaluation.token_metrics import calculate_f1
from app.api.routers import naive_chat, rag_chat, ChatPayload
from app.evaluation.token_metrics import TokenMetrics
from app.api.metrics import EnhancedKeyFactsValidator

class EnhancedEvaluator:
    def __init__(self, ground_truth_path: str):
        """Initialize evaluator with ground truth data."""
        # Load the CSV but don't group yet - we'll group during evaluation
        self.ground_truth_df = pd.read_csv(ground_truth_path)
        self.key_facts_validator = EnhancedKeyFactsValidator()

    async def evaluate_single_query(self,
                                  query: str,
                                  ground_truths: List[str],
                                  chat_type: Literal["naive", "rag"],
                                  key_facts: Optional[List[str]] = None) -> Dict:
        """Evaluate a single query against multiple ground truths using specified chat type."""
        try:
            # Get the key_fact (product name) from ground truth data for this query
            query_data = self.ground_truth_df[self.ground_truth_df['question'] == query]
            product_name = query_data['key_fact'].iloc[0] if not query_data.empty else None

            # Skip evaluation if product is N/A
            if product_name == "N/A":
                return {
                    "chat_type": chat_type,
                    "query": query,
                    "predicted": None,
                    "ground_truths": ground_truths,
                    "best_matching_ground_truth": ground_truths[0],
                    "f1_score": None,
                    "precision": None,
                    "recall": None,
                    "key_facts_match": {},
                    "key_facts_details": {},
                    "trace_id": None
                }

            # Create payload with first ground truth (for context)
            payload = ChatPayload(
                query=query,
                chat_type=chat_type,
                ground_truth=ground_truths[0],
                key_facts=key_facts
            )

            # Get response from appropriate chat endpoint
            chat_func = naive_chat if chat_type == "naive" else rag_chat
            response = await chat_func(payload)

            # Calculate metrics against all ground truths and take the best match
            all_metrics = [
                calculate_f1(prediction=response["answer"], reference=gt)
                for gt in ground_truths
            ]
            
            # Select best matching metrics based on F1 score
            best_metrics = max(all_metrics, key=lambda x: x["f1"])
            best_ground_truth = ground_truths[all_metrics.index(best_metrics)]

            # Debug print
            print(f"\nChat Type: {chat_type.upper()}")
            print(f"Query: {query[:50]}...")
            print(f"Predicted: {response['answer'][:50]}...")
            print(f"Metrics: F1={best_metrics['f1']:.3f}, P={best_metrics['precision']:.3f}, R={best_metrics['recall']:.3f}")
            if key_facts:
                print(f"\nProcessing key facts for {chat_type}:")
                print(f"Key facts to check: {key_facts}")
                key_facts_results = self.key_facts_validator.validate_key_facts(
                    response["answer"], 
                    key_facts
                )
                print(f"Validation results: {key_facts_results}")

            return {
                "chat_type": chat_type,
                "query": query,
                "predicted": response["answer"],
                "ground_truths": ground_truths,
                "best_matching_ground_truth": best_ground_truth,
                "f1_score": best_metrics["f1"],
                "precision": best_metrics["precision"],
                "recall": best_metrics["recall"],
                "key_facts_match": key_facts_results if key_facts else {},
                "key_facts_details": {
                    fact: self.key_facts_validator.get_best_match_explanation(fact, result)
                    for fact, result in key_facts_results.items()
                } if key_facts else {},
                "trace_id": response.get("trace_id")
            }

        except Exception as e:
            print(f"Error evaluating {chat_type} query: {str(e)}")
            return {
                "chat_type": chat_type,
                "query": query,
                "error": str(e),
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "key_facts_match": {},
                "key_facts_details": {},
                "trace_id": None
            }

    async def evaluate_all_approaches(self, limit: Optional[int] = None) -> Dict[str, Dict]:
        """Evaluate both naive and RAG approaches on the dataset."""
        results: Dict[str, List[Dict]] = {
            "naive": [],
            "rag": []
        }

        # Get unique questions if limit is specified
        unique_questions = self.ground_truth_df[['id', 'question']].drop_duplicates()
        if limit:
            unique_questions = unique_questions.sample(n=limit)
        
        total = len(unique_questions)

        counter = 0
        for idx, row in unique_questions.iterrows():
            counter += 1
            print(f"\n📝 Processing query {counter}/{total}")
            
            # Get all ground truths for this question
            question_answers = self.ground_truth_df[
                (self.ground_truth_df['id'] == row['id']) & 
                (self.ground_truth_df['question'] == row['question'])
            ]
            
            ground_truths = question_answers['answer'].tolist()
            key_facts = question_answers['key_fact'].dropna().tolist()

            # Evaluate both approaches
            for chat_type in ["naive", "rag"]:
                result = await self.evaluate_single_query(
                    query=row["question"],
                    ground_truths=ground_truths,
                    chat_type=chat_type,  # type: ignore # TODO
                    key_facts=key_facts if key_facts else None
                )
                results[chat_type].append(result)

                if "error" in result:
                    print(f"❌ Error in {chat_type} query {counter}: {result['error']}")

        # Calculate aggregate metrics for both approaches
        metrics = {
            "naive": self.calculate_aggregate_metrics(results["naive"]),
            "rag": self.calculate_aggregate_metrics(results["rag"])
        }

        return {"results": results, "metrics": metrics}

    def calculate_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all results."""
        valid_results = [
            r for r in results 
            if "error" not in r and r["f1_score"] is not None
        ]

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
            if r.get("key_facts_match") and isinstance(r["key_facts_match"], dict)
        ]
        
        if key_facts_results:
            total_matches = 0
            total_facts = 0
            
            for result in key_facts_results:
                matches = result["key_facts_match"]
                
                for fact, fact_result in matches.items():
                    total_facts += 1
                    if fact_result.get("match", False):
                        total_matches += 1
            
            key_facts_rate = total_matches / total_facts if total_facts > 0 else 0
        else:
            key_facts_rate = 0.0

        return {
            "avg_f1": sum(r["f1_score"] for r in valid_results) / len(valid_results),
            "avg_precision": sum(r["precision"] for r in valid_results) / len(valid_results),
            "avg_recall": sum(r["recall"] for r in valid_results) / len(valid_results),
            "total_queries": len(results),
            "successful_queries": len(valid_results),
            "key_facts_success_rate": float(key_facts_rate)  # Ensure it's JSON serializable
        }
