-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS vectors CASCADE;
DROP TABLE IF EXISTS answers CASCADE;

-- Create the vectors table first
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    vector VECTOR(1536), 
    original TEXT NOT NULL, -- User query
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- Create the answers table, referencing the vectors table
CREATE TABLE answers (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL, 
    vector_id INT REFERENCES vectors(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
