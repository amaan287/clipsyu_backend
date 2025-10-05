from fastapi import HTTPException, Header
from typing import Optional
import jwt
from config.config import JWT_SECRET, JWT_ALGORITHM


# ==================== Authentication Utilities ====================

def verify_access_token(authorization: Optional[str] = Header(None)) -> str:
    """
    Verify and decode JWT access token from Authorization header
    
    Args:
        authorization: Authorization header value (Bearer token)
        
    Returns:
        User ID from token payload
        
    Raises:
        HTTPException: If token is missing, invalid, or expired
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header"
        )
    
    token = authorization.split(" ", 1)[1]
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=401,
                detail="Invalid token type"
            )
        
        user_id = payload.get("user_id")
        if not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload"
            )
        
        return user_id
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Access token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid access token"
        )
