# database.py - MongoDB Database Configuration and Connection

import pymongo
from config.config import MONGODB_URL

# ===== DATABASE CONNECTION =====

# Simple MongoDB connection
mongodb_url = MONGODB_URL
if not mongodb_url:
    raise ValueError("MONGODB_URL environment variable is required")

client = pymongo.MongoClient(
    mongodb_url,
    serverSelectionTimeoutMS=5000,
    connectTimeoutMS=10000,
    socketTimeoutMS=10000,
    maxPoolSize=10,
    retryWrites=True,
    w="majority"
)

# Test connection
try:
    client.admin.command('ping')
    print("Successfully connected to MongoDB Atlas")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise

# Get database
db = client.recipe_extractor

# Collections
recipe_collection = db.recipes
users_collection = db.users
