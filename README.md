gag is like RAG but we exploring. we want:

1. To learn how to build naive RAG and how to evaluate it performance
2. Later, compare homemade RAG vs openai-assistants-file-search-RAG, with help of RAGAS

### Commands to run RAG/evals

```
python scripts/openai-based-rag.py --query "What is the price of the Men Slim Fit Checkered Casual Shirt?" --input_file data/ecommerce_products_test.csv --llm_model gpt-4o --terminal_output

python scripts/evaluate_rag.py --qa_file data/generated_qa_pairs.csv --embeddings_file product_embeddings.json --products_file data/ecommerce_products_test.csv --rag_script scripts/openai-based-rag.py --llm_model gpt-4o-mini --sample_size 1

python scripts/evaluate_rag.py --qa_file data/generated_qa_pairs.csv --embeddings_file product_embeddings.json
```
