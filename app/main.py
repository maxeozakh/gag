# will use this for compatibility
# but more convenient syntax is already released: a | b
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.api.routers import router
from app.models.database import connect_db, disconnect_db, database

app = FastAPI()

app.include_router(router)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.on_event("startup")
async def startup():
    await connect_db()


@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()


@app.get("/")
def read_root():
    return {"Hey": "Hey"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: Union[str, None] = None):
    query = "SELECT * FROM items WHERE id = :item_id"
    item = await database.fetch_one(query, values={"item_id": item_id})

    if item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"item": item, "q": q}


@app.get("/vectors/{vector_id}")
async def read_vector(vector_id: int):
    query = "SELECT * FROM vectors WHERE id = :vector_id"
    vector = await database.fetch_one(query, values={"vector_id": vector_id})

    if vector is None:
        raise HTTPException(status_code=404, detail="Vector not found")

    return {"vector": vector}


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    query = """
    UPDATE items
    SET name = :name, price = :price, is_offer = :is_offer
    WHERE id = :item_id
    RETURNING id, name, price, is_offer
    """
    values = {
        "name": item.name,
        "price": item.price,
        "is_offer": item.is_offer,
        "item_id": item_id,
    }
    updated_item = await database.fetch_one(query, values)

    if updated_item is None:
        raise HTTPException(status_code=404, detail="Item not found")

    return {"updated_item": updated_item}
