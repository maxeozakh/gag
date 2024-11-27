import os
from databases import Database

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise EnvironmentError("DATABASE_URL environment variable is not set.")

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
