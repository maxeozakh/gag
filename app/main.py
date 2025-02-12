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

templates = Jinja2Templates(directory="app/templates")

app = FastAPI()

app.include_router(router)


@app.on_event("startup")
async def startup_event():
    print("Starting application startup process... 🚀")
    
    # Connect to database first
    await connect_db()
    print("✅ Database connected")
    
    # Verify data exists
    try:
        result = await database.fetch_one("SELECT COUNT(*) as count FROM ecom_products;")
        print(f"✅ Found {result['count']} products in database")
    except Exception as e:
        print(f"❌ Error checking products: {str(e)}")


    # Load ecommerce data
    # ecom_csv_path = os.path.join(
    #     os.path.dirname(os.path.dirname(__file__)),
    #     "data/ecom.csv"
        # "data/ecom_mini.csv"
    # )

    # try:
        # print("🛍️ Loading ecommerce data...")
        # await load_ecom_data(ecom_csv_path)
        # print("✅ Ecommerce data loaded")
    # except Exception as e:
        # print(f"❌ Error loading data: {str(e)}")
        # You might want to exit the application here if this is critical
        # import sys
        # sys.exit(1)

    # Initialize Langfuse last
    print('Initializing Langfuse...')
    try:
        public_key = get_env_variable("LANGFUSE_PUBLIC_KEY")
        secret_key = get_env_variable("LANGFUSE_SECRET_KEY")
        host = get_env_variable("LANGFUSE_HOST")

        langfuse = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=host
        )
        print("✅ Successfully connected to Langfuse")
    except Exception as e:
        print(f"⚠️ Warning: Failed to connect to Langfuse: {str(e)}")


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})
