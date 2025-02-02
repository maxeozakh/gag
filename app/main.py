from fastapi import FastAPI, HTTPException, Request
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
    await connect_db()
    
    # Path to your CSV file
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "../mythology_of_star_wars_ready_to_work_with.csv"
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
