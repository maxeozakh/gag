from fastapi import FastAPI, HTTPException, Request
from app.utils.helpers import get_env_variable

from langfuse.openai import openai  # OpenAI integration
from langfuse import Langfuse
import langfuse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

from app.api.routers import router
from app.models.database import connect_db, disconnect_db, database
from app.utils.csv_loader import load_mythology_data

templates = Jinja2Templates(directory="app/templates")

app = FastAPI()

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    print("Starting application startup process... 🚀")
    # Try to initialize Langfuse first
    print('Initializing Langfuse...')
    try:
        # Make sure these environment variables are set
        public_key = get_env_variable("LANGFUSE_PUBLIC_KEY")
        secret_key = get_env_variable("LANGFUSE_SECRET_KEY")
        host = get_env_variable("LANGFUSE_HOST")  # Optional, defaults to cloud

        langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )
        # Initialize Langfuse with configuration
        print("✅ Successfully connected to Langfuse")
    except Exception as e:
        print(f"⚠️ Warning: Failed to connect to Langfuse: {str(e)}")
        # Continue running the application even if Langfuse fails

    # Connect to database
    await connect_db()

    # Path to your CSV file
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "..data/mythology_of_star_wars_ready_to_work_with.csv"
    )

    try:
        await load_mythology_data(csv_path)
        print("Successfully loaded mythology data")
    except Exception as e:
        print(f"Error loading mythology data: {str(e)}")
        # You might want to exit the application here if this is critical
        # import sys
        # sys.exit(1)


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})
