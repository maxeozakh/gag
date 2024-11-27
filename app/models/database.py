import os
from databases import Database

# Get the database URL from environment variables
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is not set.")

# Initialize the Database object
database = Database(DATABASE_URL)


async def connect_db():
    """
    Establish a connection to the database.
    """
    await database.connect()


async def disconnect_db():
    """
    Disconnect from the database.
    """
    await database.disconnect()
