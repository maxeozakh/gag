#!/bin/bash

# Configuration
CONTAINER_NAME="postgres_db_gag"
DB_NAME="ig_data_db"
DB_USER="ig_user"
DUMP_FILE="db_dump.sqlc"

# Check if the database container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "🚨 PostgreSQL container ($CONTAINER_NAME) is not running! Starting it now..."
    docker-compose up -d db
    sleep 5  # Wait for it to start
fi

# Command to create a database dump inside the container
echo "📦 Creating a database dump..."
docker exec -t $CONTAINER_NAME pg_dump -U $DB_USER -d $DB_NAME -F c -f /$DUMP_FILE

# Copy the dump file from the container to the local machine
echo "📂 Copying dump file from container..."
docker cp $CONTAINER_NAME:/$DUMP_FILE .

# Restore function
restore_db() {
    echo "🔍 Checking if the database is empty..."
    TABLE_COUNT=$(docker exec -t $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -tAc "SELECT count(*) FROM pg_tables WHERE schemaname = 'public';")

    if [ "$TABLE_COUNT" -eq "0" ]; then
        echo "🛠️ Database is empty. Restoring from dump..."
        docker exec -t $CONTAINER_NAME pg_restore -U $DB_USER -d $DB_NAME /$DUMP_FILE
    else
        echo "✅ Database is not empty. Skipping restore."
    fi
}

# Ask if user wants to restore the database
read -p "Do you want to restore the database now? (y/N): " confirm
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    restore_db
else
    echo "🛑 Skipping restore."
fi

echo "🎉 Done!"
