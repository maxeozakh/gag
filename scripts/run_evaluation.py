import asyncio
import sys
import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Configure logging to suppress INFO messages from sentence_transformers
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()

# Debug: Print original DATABASE_URL
print(f"Original DATABASE_URL: {os.getenv('DATABASE_URL')}")

# Override DATABASE_URL for local execution
if os.getenv("DATABASE_URL"):
    # Replace db with localhost and ensure port is correct
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        if "@db/" in db_url:
            db_url = db_url.replace("@db/", "@localhost:5442/")
        elif "@db:" in db_url:
            db_url = db_url.replace("@db:", "@localhost:5442/")
        os.environ["DATABASE_URL"] = db_url
        # Debug: Print modified DATABASE_URL
        print(f"Modified DATABASE_URL: {os.environ['DATABASE_URL']}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.evaluation import EnhancedEvaluator
from app.models.database import connect_db, disconnect_db


def print_comparison_results(metrics: dict):
    """Print comparison between naive and RAG approaches."""
    print("\n📊 Comparison Results:")
    print("=" * 50)

    headers = ["Metric", "Naive LLM", "RAG", "Improvement"]
    row_format = "{:<20} {:<15} {:<15} {:<15}"

    print(row_format.format(*headers))
    print("-" * 65)

    metrics_to_compare = [
        ("F1 Score", "avg_f1"),
        ("Precision", "avg_precision"),
        ("Recall", "avg_recall"),
        ("Key Facts", "key_facts_success_rate")
    ]

    for label, metric in metrics_to_compare:
        naive_value = metrics["naive"][metric]
        rag_value = metrics["rag"][metric]
        improvement = ((rag_value - naive_value) / naive_value * 100
                       if naive_value > 0 else float('inf'))

        print(row_format.format(
            label,
            f"{naive_value:.3f}",
            f"{rag_value:.3f}",
            f"{improvement:+.1f}%" if improvement != float('inf') else "N/A"
        ))

def prepare_for_json(obj):
    """Convert all values to JSON serializable format"""
    if isinstance(obj, dict):
        return {k: prepare_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [prepare_for_json(item) for item in obj]
    elif isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    else:
        return str(obj)

async def main():
    # Get limit from command line argument, default to None if not provided
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    
    print(f"Running evaluation with limit: {limit if limit else 'No limit'}")
    
    start_time = datetime.now()
    print(f"\n🚀 Starting evaluation at {start_time.strftime('%H:%M:%S')}")

    # Connect to database
    print("\n📡 Connecting to database...")
    await connect_db()
    print("✅ Database connected")

    try:
        # Initialize evaluator
        print("\n📚 Loading QA pairs")
        evaluator = EnhancedEvaluator("data/qa_dev.csv")
        print(f"✅ Loaded {len(evaluator.ground_truth_df)} test cases")

        # Add debug prints
        print("\nDebug: Checking ground truth data")
        print(f"Columns: {evaluator.ground_truth_df.columns}")
        print("\nSample key_facts:")
        print(evaluator.ground_truth_df['key_fact'].head())

        print(f"\n🔄 Running evaluations (limited to {limit if limit else 'No limit'} queries)...")
        evaluation_results = await evaluator.evaluate_all_approaches(limit=limit)

        # Print comparison results
        print_comparison_results(evaluation_results["metrics"])

        # Save results to file
        with open("evaluation_results.json", "w") as f:
            # Convert results to JSON-serializable format
            serializable_results = prepare_for_json(evaluation_results)
            json.dump(serializable_results, f, indent=2)
        print(f"\n💾 Detailed results saved to evaluation_results.json")

    finally:
        # Disconnect from database
        print("\n📡 Disconnecting from database...")
        await disconnect_db()
        print("✅ Database disconnected")

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print("\n⏱️ Evaluation completed in {:.2f} seconds".format(duration))

if __name__ == "__main__":
    asyncio.run(main())
