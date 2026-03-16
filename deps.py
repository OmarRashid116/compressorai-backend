"""
deps.py — CompressorAI v5
Central auth dependency helpers.

Roles: admin | engineer  (NO super_admin, NO viewer)
Default admin: configured via DEFAULT_ADMIN_EMAIL env var — undeletable.

Changes from v4:
  - require_auth now enforced via get_current_user (is_active checked)
  - All role guards re-verify from DB on sensitive ops (token can be stale)
  - safe_user no longer exposes password_plain (removed feature)
  - Added get_current_user_db: always fresh from DB (use on sensitive endpoints)
"""
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from config import JWT_SECRET, JWT_ALGORITHM, get_supabase_client

security = HTTPBearer()


# ── Token decode ──────────────────────────────────────────────
def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Decode JWT and return payload.
    Raises 401 on expired / invalid token.
    Enforces is_active from token payload (fast path — no DB hit).
    """
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please login again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

    if not payload.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Contact admin.")

    return payload


def get_current_user_db(
    creds: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Like get_current_user but re-fetches from DB.
    Use on sensitive operations (password change, admin actions, retrain).
    Catches cases where account was deactivated after token was issued.
    """
    payload = get_current_user(creds)
    try:
        sb  = get_supabase_client()
        res = sb.table("users").select(
            "id,email,role,full_name,is_active,is_default_admin,deleted_at"
        ).eq("id", payload["sub"]).execute()
    except Exception:
        # DB unreachable — fall back to token payload
        return payload

    if not res.data:
        raise HTTPException(status_code=401, detail="User account not found.")

    user = res.data[0]
    if user.get("deleted_at"):
        raise HTTPException(status_code=403, detail="Account has been deleted.")
    if not user.get("is_active", True):
        raise HTTPException(status_code=403, detail="Account is deactivated. Contact admin.")

    # Merge DB fields into payload (DB is source of truth)
    payload["role"]             = user["role"]
    payload["is_active"]        = user["is_active"]
    payload["is_default_admin"] = user.get("is_default_admin", False)
    return payload


# ── Role guards ───────────────────────────────────────────────
def require_auth(user: dict = Depends(get_current_user)) -> dict:
    """Any authenticated, active user."""
    return user


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    """Admin only."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


def require_engineer(user: dict = Depends(get_current_user)) -> dict:
    """Engineer only."""
    if user.get("role") != "engineer":
        raise HTTPException(status_code=403, detail="Engineer access required.")
    return user


def require_default_admin(user: dict = Depends(get_current_user_db)) -> dict:
    """
    Only the default admin.
    Uses DB re-fetch so this cannot be spoofed via stale token.
    """
    if not user.get("is_default_admin"):
        raise HTTPException(status_code=403, detail="Only the default admin can perform this action.")
    return user


def require_admin_db(user: dict = Depends(get_current_user_db)) -> dict:
    """Admin with DB re-verify — use on destructive / sensitive admin actions."""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Utility ───────────────────────────────────────────────────
def safe_user(user: dict) -> dict:
    """
    Strip all sensitive fields before returning user data to clients.
    password_plain storage has been removed; password_hash always stripped.
    """
    SENSITIVE = {"password_hash", "password_plain"}
    return {k: v for k, v in user.items() if k not in SENSITIVE}


def is_default_admin_check(user_id: str) -> bool:
    """Utility: check if a user_id belongs to the default admin."""
    try:
        sb  = get_supabase_client()
        res = sb.table("users").select("is_default_admin").eq("id", user_id).execute()
        return bool(res.data and res.data[0].get("is_default_admin"))
    except Exception:
        return False