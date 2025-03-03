# Evals

openai

```
{
  "faithfulness": 0.5657142857142856,
  "answer_relevancy": 0.7932489612943341,
  "context_recall": 0.7617196209587513
}
{
  "faithfulness": 0.3416666666666667,
  "answer_relevancy": 0.20000000000000004,
  "context_recall": 0.3666666666666667
}
```

homemade

```
{
  "faithfulness": 0.8458333333333332,
  "answer_relevancy": 0.7845947987159569,
  "context_recall": 0.36837218337218336
}
```

# Few commands to eval

python scripts/openai-based-rag.py --query "What is the price of the Men Slim Fit Checkered Casual Shirt?" --input_file data/ecommerce_products_test.csv --llm_model gpt-4o --terminal_output

python scripts/evaluate_rag.py --qa_file data/generated_qa_pairs.csv --embeddings_file product_embeddings.json --products_file data/ecommerce_products_test.csv --rag_script scripts/openai-based-rag.py --llm_model gpt-4o-mini --sample_size 1

# Basic usage with required parameters

python scripts/evaluate_rag.py --qa_file data/generated_qa_pairs.csv --embeddings_file product_embeddings.json

# Advanced usage with all options

python scripts/evaluate_rag.py \
 --qa_file data/generated_qa_pairs.csv \
 --embeddings_file product_embeddings.json \
 --rag_script scripts/rag.py \
 --llm_model gpt-4o \
 --eval_model gpt-4 \
 --sample_size 10 \
 --output_file evaluation_results.json

gag is like RAG but we exploring

## Task

Evaluate four different pipelines for a multi-domain FAQ system—(1) Naive LLM, (2) Enhanced LLM (Prompt-Engineered or Fine-Tuned), (3) Naive LLM + Retrieval-Augmentation (RAG), and (4) Enhanced LLM + RAG - across four distinct domains (game rules, e-commerce, legal text, recipes). Determine how each approach impacts factual correctness, retrieval accuracy, and latency, thereby quantifying the combined (and separate) benefits of both model enhancement and retrieval

## Pipeline

- Naive LLM
- Enhanced LLM (Prompt-Engineered/Fine Tuned)
- Naive LLM + RAG
- Enhanced LLM + RAG

## Metrics

- token-level F1 - main one
  - F1 tells you “Did the answer match the ground truth?”
- Precision@k
  - Tells you how “clean” your top retrieval results are—if you retrieve lots of irrelevant chunks, your LLM sees unhelpful context
- Recall@k
  - Tells you if you’re actually getting all the crucial information that exists in your knowledge base
- Todo: describe key_facts check as a metric

## Ground truth definition

- Multiple Reference Answers (a “Reference Set”), created manually
- Key-Fact or “Slot” Checking
  - Evaluate whether the model output includes each key fact (and doesn’t contradict it). For instance, if the question is “What is the recommended oven temperature for this recipe?” the key fact is “350°F” (or “180°C”).

## Todo

- [ ] Handle multiple retrievals
- [ ] Handle "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}"
- [ ] query_embedding should be sanitized because vector embedded directly in query
- [ ] "Consider explicitly re-raising using 'raise HTTPException(status_code=500, detail=f'Error during embedding: {str(e)}') from e'"
- [ ] Untackle routers
