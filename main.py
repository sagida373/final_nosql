from fastapi import FastAPI, HTTPException
from db import airbnb_col
from bson.decimal128 import Decimal128
from bson import ObjectId
from datetime import datetime, date
from pydantic import BaseModel, Field
from typing import Optional
import re


app = FastAPI()


def to_json_safe(x):
    # dict
    if isinstance(x, dict):
        return {k: to_json_safe(v) for k, v in x.items()}

    # list/tuple
    if isinstance(x, (list, tuple)):
        return [to_json_safe(v) for v in x]

    # Mongo Decimal128 -> float
    if isinstance(x, Decimal128):
        return float(x.to_decimal())

    # ObjectId -> str (на всякий случай)
    if isinstance(x, ObjectId):
        return str(x)

    # datetime/date -> ISO string
    if isinstance(x, (datetime, date)):
        return x.isoformat()

    return x


# ROOT
@app.get("/")
async def root():
    return {"status": "ok"}


# READ: listings
@app.get("/api/listings")
async def get_listings(limit: int = 10):
    cursor = airbnb_col.find(
        {},
        {
            "name": 1,
            "price": 1,
            "room_type": 1,
            "address.market": 1
        }
    ).limit(limit)

    listings = await cursor.to_list(length=limit)
    return to_json_safe(listings)


#эндпоинт поиска /api/listings/search с фильтрами и сортировко
@app.get("/api/listings/search")
async def search_listings(
    city: Optional[str] = None,
    market: Optional[str] = None,
    country: Optional[str] = None,
    room_type: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    min_rating: Optional[int] = None,
    sort: Optional[str] = None,
    limit: int = 20,
    skip: int = 0
):
    q = {}

    def exact_i(v: str):
        return {"$regex": f"^{re.escape(v.strip())}$", "$options": "i"}

    def contains_i(v: str):
        return {"$regex": re.escape(v.strip()), "$options": "i"}

    if country:
        q["address.country"] = contains_i(country)

    if market and market.strip():
        q["address.market"] = contains_i(market)

    if city:
        q["$or"] = [
            {"address.market": contains_i(city)},
            {"address.suburb": contains_i(city)},
            {"address.government_area": contains_i(city)},
            {"address.street": contains_i(city)},
        ]

    if room_type:
        q["room_type"] = contains_i(room_type)

    if property_type:
        q["property_type"] = contains_i(property_type)

    if min_rating is not None:
        q["review_scores.review_scores_rating"] = {"$gte": min_rating}

    if min_price is not None or max_price is not None:
        expr = []
        if min_price is not None:
            expr.append({"$gte": [{"$toDecimal": "$price"}, Decimal128(str(min_price))]})
        if max_price is not None:
            expr.append({"$lte": [{"$toDecimal": "$price"}, Decimal128(str(max_price))]})
        q["$expr"] = {"$and": expr}

    sort_spec = None
    if sort == "price_asc":
        sort_spec = [("price", 1)]
    elif sort == "price_desc":
        sort_spec = [("price", -1)]
    elif sort == "rating_desc":
        sort_spec = [("review_scores.review_scores_rating", -1)]
    elif sort:
        raise HTTPException(400, "Invalid sort")

    projection = {
        "_id": 1,
        "name": 1,
        "price": 1,
        "room_type": 1,
        "property_type": 1,
        "address.market": 1,
        "address.country": 1,
        "review_scores.review_scores_rating": 1,
        "images.picture_url": 1,
    }

    cursor = airbnb_col.find(q, projection).skip(skip).limit(limit)
    if sort_spec:
        cursor = cursor.sort(sort_spec)

    docs = await cursor.to_list(length=limit)
    return to_json_safe(docs)


# READ: single listing
@app.get("/api/listings/{listing_id}")
async def get_listing(listing_id: str):
    doc = await airbnb_col.find_one(
        {"_id": listing_id},
        {
            "name": 1,
            "price": 1,
            "room_type": 1,
            "address": 1,
            "reviews": {"$slice": 5}
        }
    )

    if not doc:
        raise HTTPException(status_code=404, detail="Listing not found")

    return to_json_safe(doc)


# MODELS
class ReviewCreate(BaseModel):
    review_id: str = Field(..., description="Unique review _id, e.g. my_review_001")
    reviewer_id: str
    reviewer_name: str
    comments: str
    verified: bool = True
    date: Optional[datetime] = None


class ReviewUpdate(BaseModel):
    comments: Optional[str] = None
    verified: Optional[bool] = None


# CREATE review: $push
@app.post("/api/listings/{listing_id}/reviews")
async def add_review(listing_id: str, body: ReviewCreate):
    review_doc = {
        "_id": body.review_id,
        "date": body.date or datetime.utcnow(),
        "listing_id": listing_id,
        "reviewer_id": body.reviewer_id,
        "reviewer_name": body.reviewer_name,
        "comments": body.comments,
        "verified": body.verified
    }

    res = await airbnb_col.update_one(
        {"_id": listing_id, "reviews._id": {"$ne": body.review_id}},
        {"$push": {"reviews": review_doc}}
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found OR review_id already exists")

    return {"message": "Review added", "modified": res.modified_count}


# UPDATE review
@app.patch("/api/listings/{listing_id}/reviews/{review_id}")
async def update_review(listing_id: str, review_id: str, body: ReviewUpdate):
    set_fields = {}
    if body.comments is not None:
        set_fields["reviews.$.comments"] = body.comments
    if body.verified is not None:
        set_fields["reviews.$.verified"] = body.verified

    if not set_fields:
        raise HTTPException(status_code=400, detail="Nothing to update")

    res = await airbnb_col.update_one(
        {"_id": listing_id, "reviews._id": review_id},
        {"$set": set_fields}
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing or review not found")

    return {"message": "Review updated", "modified": res.modified_count}


# DELETE
@app.delete("/api/listings/{listing_id}/reviews/{review_id}")
async def delete_review(listing_id: str, review_id: str):
    res = await airbnb_col.update_one(
        {"_id": listing_id, "reviews._id": review_id},
        {"$pull": {"reviews": {"_id": review_id}}}
    )

    if res.matched_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Listing or review not found"
        )

    if res.modified_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Review not found"
        )

    return {
        "message": "Review deleted",
        "listing_id": listing_id,
        "review_id": review_id
    }
