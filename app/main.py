from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routers import router
from app.models.database import connect_db, disconnect_db, database

templates = Jinja2Templates(directory="app/templates")

app = FastAPI()

app.include_router(router)


@app.on_event("startup")
async def startup():
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Render the index.html page.
    """
    return templates.TemplateResponse("index.html", {"request": request, "title": "Home"})


# @app.get("/vectors/{vector_id}")
# async def read_vector(vector_id: int):
#     query = "SELECT * FROM vectors WHERE id = :vector_id"
#     vector = await database.fetch_one(query, values={"vector_id": vector_id})

#     if vector is None:
#         raise HTTPException(status_code=404, detail="Vector not found")

#     return {"vector": vector}
