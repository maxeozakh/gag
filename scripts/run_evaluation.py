import asyncio
import sys
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Debug: Print original DATABASE_URL
print(f"Original DATABASE_URL: {os.getenv('DATABASE_URL')}")

# Override DATABASE_URL for local execution
if os.getenv("DATABASE_URL"):
    # Replace db with localhost and ensure port is correct
    db_url = os.getenv("DATABASE_URL")
    if "@db/" in db_url:
        db_url = db_url.replace("@db/", "@localhost:5442/")
    elif "@db:" in db_url:
        db_url = db_url.replace("@db:", "@localhost:5442/")
    os.environ["DATABASE_URL"] = db_url
    # Debug: Print modified DATABASE_URL
    print(f"Modified DATABASE_URL: {os.environ['DATABASE_URL']}")

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.api.evaluation import Evaluator
from app.models.database import connect_db, disconnect_db

async def main():
    start_time = datetime.now()
    print(f"\n🚀 Starting evaluation at {start_time.strftime('%H:%M:%S')}")
    
    # Connect to database
    print("\n📡 Connecting to database...")
    await connect_db()
    print("✅ Database connected")
    
    try:
        # Initialize evaluator
        print("\n📚 Loading ground truth data from ecom_ground_truth.csv...")
        evaluator = Evaluator(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/ecom_ground_truth.csv"))
        print(f"✅ Loaded {len(evaluator.ground_truth_df)} test cases")
        
        limit = 10
        print(f"\n🔄 Running evaluations (limited to {limit} queries)...")
        results, metrics = await evaluator.evaluate_all(limit)
        
        # Calculate success rate
        success_rate = (metrics['successful_queries'] / metrics['total_queries']) * 100
        key_facts_rate = metrics['key_facts_success_rate'] * 100
        
        print("\n📊 Evaluation Results:")
        print("=" * 50)
        print(f"Average F1 Score: {metrics['avg_f1']:.3f}")
        print(f"Average Precision: {metrics['avg_precision']:.3f}")
        print(f"Average Recall: {metrics['avg_recall']:.3f}")
        print(f"Key Facts Success Rate: {key_facts_rate:.1f}%")
        print(f"Total Queries: {metrics['total_queries']}")
        print(f"Successful Queries: {metrics['successful_queries']} ({success_rate:.1f}%)")
    
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