import os
import re
from datetime import datetime, date
from statistics import mean
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Query, Path, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import FastAPI, Depends, Header, HTTPException

from bson.decimal128 import Decimal128
from bson import ObjectId

from db import airbnb_col
from fastapi import Header, HTTPException




app = FastAPI(title="Airbnb API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # для учебного проекта ок
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "dev-admin-token")

def require_admin(x_admin_token: str = Header(default="")):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Admin token invalid")
    return True

#REVIEW ENDPOINTS
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


# READ: single listing Тень Аската
@app.get("/api/listings/by-id/{listing_id}")
async def get_listing(listing_id: str):
    query = {"_id": listing_id}

    # если id выглядит как число — пробуем ещё и int
    if listing_id.isdigit():
        doc = await airbnb_col.find_one({"_id": int(listing_id)}, {
            "name": 1, "price": 1, "room_type": 1, "address": 1, "reviews": {"$slice": 5}
        })
        if doc:
            return to_json_safe(doc)

    doc = await airbnb_col.find_one(query, {
        "name": 1, "price": 1, "room_type": 1, "address": 1, "reviews": {"$slice": 5}
    })

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

#добавил обновлять рейтинг
class ReviewUpdate(BaseModel):
    comments: Optional[str] = None
    verified: Optional[bool] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)



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


# UPDATE review Здесь Аскат
@app.patch("/api/listings/{listing_id}/reviews/{review_id}")
async def update_review(listing_id: str, review_id: str, body: ReviewUpdate):
    update = {}
    if body.comments is not None:
        update["reviews.$.comments"] = body.comments
    if body.verified is not None:
        update["reviews.$.verified"] = body.verified
    if body.rating is not None:
        update["reviews.$.rating"] = body.rating

    if not update:
        raise HTTPException(status_code=400, detail="No fields to update")

    res = await airbnb_col.update_one(
        {"_id": listing_id, "reviews._id": review_id},
        {"$set": update}
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing or review not found")

    return {"message": "Review updated", "listing_id": listing_id, "review_id": review_id}


# DELETE Тута?
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


#HOST ENDPOINTS

#HOST ENDPOINTS (и дальше: дополнительные endpoints по reviews)

# ВАЖНО:
# 1) НЕ создаём app второй раз
# 2) Используем MongoDB (airbnb_col)
# 3) Никаких _get_listing_reviews

class ReviewOut(BaseModel):
    _id: str
    listing_id: Optional[str] = None
    reviewer_id: Optional[str] = None
    reviewer_name: Optional[str] = None
    comments: Optional[str] = None
    verified: Optional[bool] = True
    date: Optional[datetime] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)  # если у тебя в датасете есть rating


class ReviewStatsOut(BaseModel):
    listing_id: str
    count: int
    average_rating: Optional[float] = None
    min_rating: Optional[int] = None
    max_rating: Optional[int] = None
    rating_distribution: Dict[int, int]

#Askat

from typing import List, Optional
from fastapi import Query, HTTPException
from datetime import datetime, date

@app.get("/api/listings/{listing_id}/reviews")
async def get_listing_reviews(
    listing_id: str,
    min_rating: Optional[int] = Query(default=None, ge=1, le=5),
    max_rating: Optional[int] = Query(default=None, ge=1, le=5),
    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
    newest_first: bool = True
):
    """
    Возвращает отзывы конкретного listing из поля reviews внутри документа.
    Поддерживает фильтр по rating (если он есть) + пагинацию.
    """
    doc = await airbnb_col.find_one({"_id": listing_id}, {"reviews": 1})
    if not doc or "reviews" not in doc:
        raise HTTPException(status_code=404, detail="Listing not found or no reviews")

    reviews = doc.get("reviews", [])

    # сортировка по date
    def safe_date(r):
        d = r.get("date")
        return d if isinstance(d, (datetime, date)) else datetime.min

    reviews = sorted(reviews, key=safe_date, reverse=newest_first)

    # фильтр по rating (если rating отсутствует — такие отзывы пропускаем при фильтрации)
    if min_rating is not None or max_rating is not None:
        filtered = []
        for r in reviews:
            rating = r.get("rating")
            if rating is None:
                continue
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue
            filtered.append(r)
        reviews = filtered

    page = reviews[skip: skip + limit]

    # Добавляем review_id, чтобы фронт мог PATCH/DELETE /reviews/{review_id}
    out = []
    for r in page:
        rr = dict(r)
        if "_id" in rr:
            rr["review_id"] = str(rr["_id"])
            rr.pop("_id", None)  # чтобы _id не потерялся/не скрывался Pydantic-ом
        out.append(rr)

    return to_json_safe(out)



@app.get("/api/listings/{listing_id}/reviews/stats", response_model=ReviewStatsOut)
async def get_listing_reviews_stats(listing_id: str):
    """
    Статистика по отзывам:
    - count
    - average/min/max rating
    - distribution 1..5

    ВАЖНО: статистика считается только по тем отзывам, где есть поле rating.
    """
    doc = await airbnb_col.find_one({"_id": listing_id}, {"reviews": 1})
    if not doc or "reviews" not in doc:
        raise HTTPException(status_code=404, detail="Listing not found or no reviews")

    reviews = doc.get("reviews", [])

    ratings = [r.get("rating") for r in reviews if isinstance(r.get("rating"), int)]
    distribution = {i: 0 for i in range(1, 6)}

    for rt in ratings:
        if 1 <= rt <= 5:
            distribution[rt] += 1

    if not ratings:
        return ReviewStatsOut(
            listing_id=listing_id,
            count=0,
            average_rating=None,
            min_rating=None,
            max_rating=None,
            rating_distribution=distribution
        )

    avg_rating = round(sum(ratings) / len(ratings), 2)

    return ReviewStatsOut(
        listing_id=listing_id,
        count=len(ratings),
        average_rating=avg_rating,
        min_rating=min(ratings),
        max_rating=max(ratings),
        rating_distribution=distribution
    )
#POST /api/listings/{listing_id}/reviews/{review_id}
from datetime import datetime
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional

class ReviewCreate2(BaseModel):
    reviewer_id: str
    reviewer_name: str
    comments: str
    verified: bool = True
    date: Optional[datetime] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)  # если хочешь рейтинг


@app.post("/api/listings/{listing_id}/reviews/{review_id}")
async def create_review_with_id(listing_id: str, review_id: str, body: ReviewCreate2):
    review_doc = {
        "_id": review_id,
        "listing_id": listing_id,
        "reviewer_id": body.reviewer_id,
        "reviewer_name": body.reviewer_name,
        "comments": body.comments,
        "verified": body.verified,
        "date": body.date or datetime.utcnow(),
    }
    if body.rating is not None:
        review_doc["rating"] = body.rating

    # 1) проверяем что listing существует
    listing = await airbnb_col.find_one({"_id": listing_id}, {"_id": 1})
    if not listing:
        raise HTTPException(status_code=404, detail="Listing not found")

    # 2) запрещаем дубликаты review_id в одном listing
    dup = await airbnb_col.find_one({"_id": listing_id, "reviews._id": review_id}, {"_id": 1})
    if dup:
        raise HTTPException(status_code=409, detail="Review with this id already exists in this listing")

    # 3) пушим
    res = await airbnb_col.update_one(
        {"_id": listing_id},
        {"$push": {"reviews": review_doc}}
    )

    return {"message": "Review created", "listing_id": listing_id, "review_id": review_id, "modified": res.modified_count}
#PATCH /api/listings/{listing_id}/reviews/{review_id}
class ReviewPatch2(BaseModel):
    comments: Optional[str] = None
    verified: Optional[bool] = None
    rating: Optional[int] = Field(default=None, ge=1, le=5)


@app.patch("/api/listings/{listing_id}/reviews/{review_id}")
async def patch_review(listing_id: str, review_id: str, body: ReviewPatch2):
    set_fields = {}

    if body.comments is not None:
        set_fields["reviews.$.comments"] = body.comments
    if body.verified is not None:
        set_fields["reviews.$.verified"] = body.verified
    if body.rating is not None:
        set_fields["reviews.$.rating"] = body.rating

    if not set_fields:
        raise HTTPException(status_code=400, detail="Nothing to update")

    res = await airbnb_col.update_one(
        {"_id": listing_id, "reviews._id": review_id},
        {"$set": set_fields}
    )

    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing or review not found")

    return {"message": "Review updated", "listing_id": listing_id, "review_id": review_id, "modified": res.modified_count}




#сводка по хосту
from fastapi import HTTPException, Query
from typing import Optional, List, Dict

@app.get("/api/hosts/{host_id}")
async def get_host_summary(host_id: str):
    """
    Возвращает сводку по хосту (из любого listing этого host).
    """
    doc = await airbnb_col.find_one(
        {"host.host_id": host_id},
        {
            "_id": 0,
            "host.host_id": 1,
            "host.host_name": 1,
            "host.host_location": 1,
            "host.host_is_superhost": 1,
            "host.host_response_time": 1,
            "host.host_response_rate": 1,
            "host.host_verifications": 1,
            "host.host_listings_count": 1,
            "host.host_total_listings_count": 1,
            "host.host_thumbnail_url": 1,
            "host.host_picture_url": 1,
            "host.host_identity_verified": 1,
            "host.verified_status": 1,
            "host.profile_complete": 1,
            "host.host_url": 1,
        }
    )

    if not doc or "host" not in doc:
        raise HTTPException(status_code=404, detail="Host not found")

    return to_json_safe(doc["host"])





#объявления хоста
@app.get("/api/hosts/{host_id}/listings")
async def get_host_listings(
    host_id: str,
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """
    Возвращает список listings конкретного host_id.
    """
    projection = {
        "_id": 1,
        "name": 1,
        "listing_url": 1,
        "price": 1,
        "property_type": 1,
        "room_type": 1,
        "address.market": 1,
        "address.country": 1,
        "images.picture_url": 1,
        "review_scores.review_scores_rating": 1,
        "number_of_reviews": 1,
        "accommodates": 1,
        "bedrooms": 1,
        "beds": 1,
    }

    cursor = (
        airbnb_col.find({"host.host_id": host_id}, projection)
        .skip(skip)
        .limit(limit)
        .sort([("review_scores.review_scores_rating", -1)])
    )

    docs = await cursor.to_list(length=limit)

    if skip == 0 and not docs:
        # если ничего не нашли — скорее всего host_id не существует
        raise HTTPException(status_code=404, detail="Host not found or no listings")

    return to_json_safe(docs)

#поиск через availability
@app.get("/api/amenities")
async def get_amenities():
    """
    Возвращает список уникальных amenities (для чекбоксов на фронте).
    """
    pipeline = [
        {"$unwind": "$amenities"},
        {"$match": {"amenities": {"$type": "string"}}},
        {"$group": {"_id": "$amenities"}},
        {"$sort": {"_id": 1}},
    ]

    cursor = airbnb_col.aggregate(pipeline)
    rows = await cursor.to_list(length=None)

    amenities = [r["_id"] for r in rows]
    return {"count": len(amenities), "amenities": amenities}



#GET /api/listings/availability
from typing import Optional
from fastapi import Query


@app.get("/api/listings/availability")
async def get_listings_by_availability(
    min_30: Optional[int] = Query(default=None, ge=0),
    min_60: Optional[int] = Query(default=None, ge=0),
    min_90: Optional[int] = Query(default=None, ge=0),
    min_365: Optional[int] = Query(default=None, ge=0),

    limit: int = Query(default=20, ge=1, le=100),
    skip: int = Query(default=0, ge=0),
):
    # 1) фильтр
    q = {}
    if min_30 is not None:
        q["availability.availability_30"] = {"$gte": min_30}
    if min_60 is not None:
        q["availability.availability_60"] = {"$gte": min_60}
    if min_90 is not None:
        q["availability.availability_90"] = {"$gte": min_90}
    if min_365 is not None:
        q["availability.availability_365"] = {"$gte": min_365}

    # 2) какие поля вернуть
    projection = {
        "_id": 1,
        "name": 1,
        "price": 1,
        "property_type": 1,
        "room_type": 1,
        "address.market": 1,
        "address.country": 1,
        "images.picture_url": 1,
        "review_scores.review_scores_rating": 1,
        "number_of_reviews": 1,
        "availability": 1,
    }

    # 3) запрос к Mongo
    cursor = (
        airbnb_col
        .find(q, projection)
        .skip(skip)
        .limit(limit)
    )

    docs = await cursor.to_list(length=limit)

    return {
        "filters": {
            "min_30": min_30,
            "min_60": min_60,
            "min_90": min_90,
            "min_365": min_365,
            "limit": limit,
            "skip": skip,
        },
        "count_returned": len(docs),
        "items": to_json_safe(docs),
    }

#ADD, UPDATE, DELETE for host
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

class HostUpdate(BaseModel):
    # common fields (optional)
    host_name: Optional[str] = None
    host_location: Optional[str] = None
    host_is_superhost: Optional[bool] = None
    host_response_time: Optional[str] = None
    host_response_rate: Optional[int] = None
    host_verifications: Optional[List[str]] = None

    # extra custom fields admin wants to add/update
    extra: Optional[Dict[str, Any]] = None


@app.patch("/api/admin/hosts/{host_id}")
async def admin_update_host(host_id: str, body: HostUpdate, admin=Depends(require_admin)):
    set_fields = {}

    if body.host_name is not None:
        set_fields["host.host_name"] = body.host_name
    if body.host_location is not None:
        set_fields["host.host_location"] = body.host_location
    if body.host_is_superhost is not None:
        set_fields["host.host_is_superhost"] = body.host_is_superhost
    if body.host_response_time is not None:
        set_fields["host.host_response_time"] = body.host_response_time
    if body.host_response_rate is not None:
        set_fields["host.host_response_rate"] = body.host_response_rate
    if body.host_verifications is not None:
        set_fields["host.host_verifications"] = body.host_verifications

    # add/update arbitrary new host fields (host.<key>)
    if body.extra:
        for k, v in body.extra.items():
            if k and k.strip():
                set_fields[f"host.{k.strip()}"] = v

    if not set_fields:
        raise HTTPException(status_code=400, detail="Nothing to update")

    res = await airbnb_col.update_many(
        {"host.host_id": host_id},
        {"$set": set_fields}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Host not found")

    return {"message": "Host updated", "matched": res.matched_count, "modified": res.modified_count}

#DELETE for host
class HostDeleteFields(BaseModel):
    fields: List[str]  # e.g. ["host_about", "phone", "custom_note"]


@app.delete("/api/admin/hosts/{host_id}/fields")
async def admin_delete_host_fields(host_id: str, body: HostDeleteFields, admin=Depends(require_admin)):
    # Build $unset: {"host.field": "" }
    unset_fields = {}
    for f in body.fields:
        f = (f or "").strip()
        if not f:
            continue
        unset_fields[f"host.{f}"] = ""

    if not unset_fields:
        raise HTTPException(status_code=400, detail="No valid fields provided")

    res = await airbnb_col.update_many(
        {"host.host_id": host_id},
        {"$unset": unset_fields}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Host not found")

    return {"message": "Host fields deleted", "matched": res.matched_count, "modified": res.modified_count}

#UPDATE, DELETE, ADD for amenities per listing
#POST
class AmenitiesAdd(BaseModel):
    amenities: List[str]

@app.post("/api/admin/listings/{listing_id}/amenities")
async def admin_add_amenities(listing_id: str, body: AmenitiesAdd, admin=Depends(require_admin)):
    cleaned = [a.strip() for a in body.amenities if a and a.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="Amenities list is empty")

    res = await airbnb_col.update_one(
        {"_id": listing_id},
        {"$addToSet": {"amenities": {"$each": cleaned}}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")

    return {"message": "Amenities added", "requested": cleaned, "modified": res.modified_count}
#patch
class AmenitiesReplace(BaseModel):
    amenities: List[str]

@app.patch("/api/admin/listings/{listing_id}/amenities")
async def admin_replace_amenities(listing_id: str, body: AmenitiesReplace, admin=Depends(require_admin)):
    cleaned = [a.strip() for a in body.amenities if a and a.strip()]
    # remove duplicates keep order
    seen = set()
    cleaned = [x for x in cleaned if not (x in seen or seen.add(x))]

    res = await airbnb_col.update_one(
        {"_id": listing_id},
        {"$set": {"amenities": cleaned}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")

    return {"message": "Amenities replaced", "new_count": len(cleaned), "modified": res.modified_count}
#delete one amenity
@app.delete("/api/admin/listings/{listing_id}/amenities/{amenity}")
async def admin_remove_amenity(listing_id: str, amenity: str, admin=Depends(require_admin)):
    res = await airbnb_col.update_one(
        {"_id": listing_id},
        {"$pull": {"amenities": amenity}}
    )
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")
    if res.modified_count == 0:
        raise HTTPException(status_code=404, detail="Amenity not found in listing")

    return {"message": "Amenity removed", "listing_id": listing_id, "amenity": amenity}



#ADD, UPDATE, DELETE availability per listing
#patch update availability 
class AvailabilityPatch(BaseModel):
    availability_30: Optional[int] = None
    availability_60: Optional[int] = None
    availability_90: Optional[int] = None
    availability_365: Optional[int] = None

@app.patch("/api/admin/listings/{listing_id}/availability")
async def admin_patch_availability(listing_id: str, body: AvailabilityPatch, admin=Depends(require_admin)):
    set_fields = {}
    if body.availability_30 is not None:
        set_fields["availability.availability_30"] = body.availability_30
    if body.availability_60 is not None:
        set_fields["availability.availability_60"] = body.availability_60
    if body.availability_90 is not None:
        set_fields["availability.availability_90"] = body.availability_90
    if body.availability_365 is not None:
        set_fields["availability.availability_365"] = body.availability_365

    if not set_fields:
        raise HTTPException(status_code=400, detail="Nothing to update")

    res = await airbnb_col.update_one({"_id": listing_id}, {"$set": set_fields})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")

    return {"message": "Availability updated", "modified": res.modified_count}

#patch add new fields availability 
class AvailabilityExtra(BaseModel):
    extra: Dict[str, Any]  # e.g. {"custom_note":"...", "last_updated_by":"admin1"}

@app.patch("/api/admin/listings/{listing_id}/availability/extra")
async def admin_add_availability_fields(listing_id: str, body: AvailabilityExtra, admin=Depends(require_admin)):
    if not body.extra:
        raise HTTPException(status_code=400, detail="extra is empty")

    set_fields = {}
    for k, v in body.extra.items():
        k = (k or "").strip()
        if k:
            set_fields[f"availability.{k}"] = v

    if not set_fields:
        raise HTTPException(status_code=400, detail="No valid extra fields")

    res = await airbnb_col.update_one({"_id": listing_id}, {"$set": set_fields})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")

    return {"message": "Availability extra fields added/updated", "modified": res.modified_count}

#delete remove from availability 
class AvailabilityDeleteFields(BaseModel):
    fields: List[str]  # e.g. ["custom_note", "last_updated_by"]

@app.delete("/api/admin/listings/{listing_id}/availability/fields")
async def admin_delete_availability_fields(listing_id: str, body: AvailabilityDeleteFields, admin=Depends(require_admin)):
    unset_fields = {}
    for f in body.fields:
        f = (f or "").strip()
        if f:
            unset_fields[f"availability.{f}"] = ""

    if not unset_fields:
        raise HTTPException(status_code=400, detail="No valid fields provided")

    res = await airbnb_col.update_one({"_id": listing_id}, {"$unset": unset_fields})
    if res.matched_count == 0:
        raise HTTPException(status_code=404, detail="Listing not found")

    return {"message": "Availability fields deleted", "modified": res.modified_count}
