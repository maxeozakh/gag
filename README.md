gag is like RAG but quality is kinda yet garbage

## TODO

- [ ] Create ground truth dataset (start from q-a examples)
- [ ] BERTscore
- [ ] Handle multiple retrievals

- [ ] Handle "Error code: 429 - {'error': {'message': 'You exceeded your current quota, please check your plan and billing details. For more information on this error, read the docs: https://platform.openai.com/docs/guides/error-codes/api-errors.', 'type': 'insufficient_quota', 'param': None, 'code': 'insufficient_quota'}}"
- [ ] query_embedding should be sanitized because vector embedded directly in query
- [ ] "Consider explicitly re-raising using 'raise HTTPException(status_code=500, detail=f'Error during embedding: {str(e)}') from e'"
- [ ] Move things from routers

- [ ] Put all context/prev answers, not only first one lol
