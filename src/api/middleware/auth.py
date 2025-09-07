"""Authentication and authorization utilities (cleaned Wave 2)."""
import os, jwt
from typing import Optional, Dict, Any
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from datetime import datetime, timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
security = HTTPBearer(auto_error=False)

USERS_DB = {
    "admin": {"username": "admin", "hashed_password": pwd_context.hash("admin123"), "is_active": True, "permissions": ["read", "write", "admin"]},
    "user": {"username": "user", "hashed_password": pwd_context.hash("user123"), "is_active": True, "permissions": ["read"]},
}

API_KEYS_DB = {
    "sk-simple-llm-demo-key-123": {
        "name": "Demo Key",
        "is_active": True,
        "permissions": ["read", "write", "admin"],  # grant admin for config validation
        "created_at": datetime.now(),
        "last_used": None,
    }
}

class AuthenticationError(Exception):
    pass

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if payload.get("sub") is None:
            raise AuthenticationError("Invalid token")
        return payload
    except jwt.PyJWTError as e:
        raise AuthenticationError(str(e))

def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = USERS_DB.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return None
    return user

def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    info = API_KEYS_DB.get(api_key)
    if not info or not info["is_active"]:
        return None
    info["last_used"] = datetime.now()
    return info

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    if not credentials:
        return None
    token = credentials.credentials
    if token.startswith("sk-"):
        info = verify_api_key(token)
        if info:
            return {"type": "api_key", "name": info["name"], "permissions": info["permissions"]}
    try:
        payload = verify_token(token)
        username = payload.get("sub")
        user = USERS_DB.get(username)
        if user and user["is_active"]:
            return {"type": "jwt", "username": username, "permissions": user["permissions"]}
    except AuthenticationError:
        return None
    return None

async def require_auth(current_user: Optional[Dict[str, Any]] = Depends(get_current_user)) -> Dict[str, Any]:
    if not current_user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    return current_user

def require_permission(permission: str):
    async def checker(user: Dict[str, Any] = Depends(require_auth)) -> Dict[str, Any]:
        if permission not in user.get("permissions", []):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Permission '{permission}' required")
        return user
    return checker

async def optional_auth(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    try:
        return await get_current_user(credentials)
    except Exception:
        return None

__all__ = [
    'create_access_token','authenticate_user','verify_token','verify_api_key','require_auth','require_permission','optional_auth'
]
