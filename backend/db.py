from motor.motor_asyncio import AsyncIOMotorClient

MONGO_URL = "mongodb://localhost:27017"

client = AsyncIOMotorClient(MONGO_URL)
db = client["airbnb"]

airbnb_col = db["airbnb"]
market_avg_price_col = db["market_avg_price"]
roomtype_counts_col = db["roomtype_counts"]
host_review_summary_col = db["host_review_summary"]
