#!/bin/bash
set -e

echo "🚀 Starting database initialization..."

# Enable vector extension
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS vector;
EOSQL
echo "✅ Vector extension enabled"

# Drop existing tables if they exist
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    DROP TABLE IF EXISTS ecom_products CASCADE;
    DROP TABLE IF EXISTS ecom_vectors CASCADE;
    DROP TABLE IF EXISTS vectors CASCADE;
    DROP TABLE IF EXISTS answers CASCADE;
EOSQL
echo "🗑️ Cleaned up existing tables"

# Restore data from dump with clean option
pg_restore -v --clean --if-exists --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" /docker-entrypoint-initdb.d/02-data.sqlc
echo "✅ Data restored"

# Verify data
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT COUNT(*) as product_count FROM ecom_products;
EOSQL