-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create the ig_data table
CREATE TABLE ig_data (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,  -- Renamed from "user"
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the vectors table
CREATE TABLE vectors (
    id SERIAL PRIMARY KEY,
    ig_data_id INT REFERENCES ig_data(id) ON DELETE CASCADE,
    vector VECTOR(128),  -- Use the vector type
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Seed the ig_data table with initial data
INSERT INTO ig_data (username, content, metadata) VALUES
('test_user', 'Sample post 1', '{"likes": 100}'),
('test_user', 'Sample post 2', '{"likes": 200}');


CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    price FLOAT NOT NULL,
    is_offer BOOLEAN
);

INSERT INTO items (name, price, is_offer) VALUES
('Sample Item 1', 9.99, TRUE),
('Sample Item 2', 19.99, FALSE);