"""Authentication endpoints."""

import logging
from datetime import timedelta
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from src.api.middleware.auth import (
    authenticate_user, 
    create_access_token, 
    require_auth, 
    require_permission,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str

class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_info: dict

class UserInfo(BaseModel):
    """User info model."""
    username: str
    permissions: list
    is_active: bool

@router.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint."""
    user = authenticate_user(request.username, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, 
        expires_delta=access_token_expires
    )
    
    return LoginResponse(
        access_token=access_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user_info={
            "username": user["username"],
            "permissions": user["permissions"],
            "is_active": user["is_active"]
        }
    )

@router.get("/auth/me", response_model=UserInfo)
async def get_current_user_info(current_user: dict = Depends(require_auth)):
    """Get current user information."""
    if current_user.get("type") == "api_key":
        return UserInfo(
            username=current_user.get("name", "api_key"),
            permissions=current_user.get("permissions", []),
            is_active=True
        )
    else:
        return UserInfo(
            username=current_user.get("username", "unknown"),
            permissions=current_user.get("permissions", []),
            is_active=True
        )

@router.post("/auth/logout")
async def logout(current_user: dict = Depends(require_auth)):
    """Logout endpoint (token invalidation)."""
    # In a real implementation, you would add the token to a blacklist
    # For JWT tokens, you can maintain a blacklist in Redis/database
    logging.info(f"User logged out: {current_user}")
    return {"message": "Successfully logged out"}

@router.get("/auth/permissions")
async def check_permissions(current_user: dict = Depends(require_auth)):
    """Check user permissions."""
    return {
        "user": current_user.get("username") or current_user.get("name"),
        "type": current_user.get("type"),
        "permissions": current_user.get("permissions", [])
    }
