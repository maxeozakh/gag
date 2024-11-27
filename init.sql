-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE answers (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


INSERT INTO answers (content, metadata) VALUES
('Sample post 1', '{"likes": 100}'),
('Sample post 2', '{"likes": 200}');

CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    answers_id INT REFERENCES answers(id) ON DELETE CASCADE,
    vector VECTOR(1536), 
    original TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
