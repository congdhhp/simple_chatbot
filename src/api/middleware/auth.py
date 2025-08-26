"""Authentication middleware for API service."""

import os
import jwt
import time
import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Security scheme
security = HTTPBearer(auto_error=False)

# In-memory user store (replace with database in production)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production
        "is_active": True,
        "permissions": ["read", "write", "admin"]
    },
    "user": {
        "username": "user", 
        "hashed_password": pwd_context.hash("user123"),   # Change in production
        "is_active": True,
        "permissions": ["read"]
    }
}

# API Keys store (replace with database in production)
API_KEYS_DB = {
    "sk-simple-llm-demo-key-123": {
        "name": "Demo Key",
        "is_active": True,
        "permissions": ["read", "write"],
        "created_at": datetime.now(),
        "last_used": None
    }
}

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token")
        return payload
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid token")

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with username/password."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key."""
    key_info = API_KEYS_DB.get(api_key)
    if not key_info or not key_info["is_active"]:
        return None
    
    # Update last used
    key_info["last_used"] = datetime.now()
    return key_info

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    token = credentials.credentials
    
    # Try API key first
    if token.startswith("sk-"):
        key_info = verify_api_key(token)
        if key_info:
            return {
                "type": "api_key",
                "name": key_info["name"],
                "permissions": key_info["permissions"]
            }
    
    # Try JWT token
    try:
        payload = verify_token(token)
        username = payload.get("sub")
        user = USERS_DB.get(username)
        if user and user["is_active"]:
            return {
                "type": "jwt",
                "username": username,
                "permissions": user["permissions"]
            }
    except AuthenticationError:
        pass
    
    return None

async def require_auth(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require authentication."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

def require_permission(permission: str):
    """Require specific permission."""
    async def check_permission(current_user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return check_permission

# Optional auth (for public endpoints with optional authentication)
async def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional authentication."""
    try:
        return await get_current_user(credentials)
    except:
        return NoneTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timedelta

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

# Security scheme
security = HTTPBearer(auto_error=False)

# In-memory user store (replace with database in production)
USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin123"),  # Change in production
        "is_active": True,
        "permissions": ["read", "write", "admin"]
    },
    "user": {
        "username": "user", 
        "hashed_password": pwd_context.hash("user123"),   # Change in production
        "is_active": True,
        "permissions": ["read"]
    }
}

# API Keys store (replace with database in production)
API_KEYS_DB = {
    "sk-simple-llm-demo-key-123": {
        "name": "Demo Key",
        "is_active": True,
        "permissions": ["read", "write"],
        "created_at": datetime.now(),
        "last_used": None
    }
}

class AuthenticationError(Exception):
    """Custom authentication error."""
    pass

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise AuthenticationError("Invalid token")
        return payload
    except jwt.PyJWTError:
        raise AuthenticationError("Invalid token")

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """Authenticate user with username/password."""
    user = USERS_DB.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Verify API key."""
    key_info = API_KEYS_DB.get(api_key)
    if not key_info or not key_info["is_active"]:
        return None
    
    # Update last used
    key_info["last_used"] = datetime.now()
    return key_info

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    token = credentials.credentials
    
    # Try API key first
    if token.startswith("sk-"):
        key_info = verify_api_key(token)
        if key_info:
            return {
                "type": "api_key",
                "name": key_info["name"],
                "permissions": key_info["permissions"]
            }
    
    # Try JWT token
    try:
        payload = verify_token(token)
        username = payload.get("sub")
        user = USERS_DB.get(username)
        if user and user["is_active"]:
            return {
                "type": "jwt",
                "username": username,
                "permissions": user["permissions"]
            }
    except AuthenticationError:
        pass
    
    return None

async def require_auth(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    """Require authentication."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

def require_permission(permission: str):
    """Require specific permission."""
    def check_permission(current_user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        if permission not in current_user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return current_user
    return check_permission

# Optional auth (for public endpoints with optional authentication)
async def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional authentication."""
    try:
        return await get_current_user(credentials)
    except:
        return None

# Optional auth (for public endpoints with optional authentication)
async def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Optional authentication."""
    try:
        return await get_current_user(credentials)
    except:
        return None
