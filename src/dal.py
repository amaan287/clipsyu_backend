# dal.py - Data Access Layer - Imports and re-exports models and database

# Import all models
from models import (
    DifficultyLevel,
    Ingredient,
    RecipeStep,
    RecipeMetadata,
    Recipe,
    User,
    RecipeDocument,
    RecipeExtractionRequest,
    RecipeExtractionResponse,
    GoogleCodeAuthRequest,
    GoogleAuthRequest,
    UserResponse,
    RefreshTokenRequest
)

# Import database configuration
from database import (
    client,
    db,
    recipe_collection,
    users_collection
)

# Re-export everything for backward compatibility
__all__ = [
    'DifficultyLevel',
    'Ingredient',
    'RecipeStep',
    'RecipeMetadata',
    'Recipe',
    'User',
    'RecipeDocument',
    'RecipeExtractionRequest',
    'RecipeExtractionResponse',
    'GoogleCodeAuthRequest',
    'GoogleAuthRequest',
    'UserResponse',
    'client',
    'db',   
    'recipe_collection',
    'users_collection',
    'RefreshTokenRequest'
]


