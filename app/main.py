# will use this for compatibility
# but more convenient syntax is already released: a | b
from typing import Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from databases import Database
from app.utils.helpers import get_env_variable
from app.api.routers import router

DATABASE_URL = get_env_variable('DATABASE_URL')
database = Database(DATABASE_URL)

app = FastAPI()

app.include_router(router)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


@app.on_event("startup")
async def startup():
    await database.connect()


@app.on_event("shutdown")
async def shutdown():
    await database.disconnect()


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
