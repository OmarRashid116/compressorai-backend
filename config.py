"""
config.py — CompressorAI v5
Production-grade configuration with startup validation.

Replit fix:
  - python-jose removed (was in requirements but not needed — pyjwt is used everywhere)
  - Client | None syntax replaced with Optional (Python 3.9 compatibility)
"""
import os
import sys
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# ── Required settings ─────────────────────────────────────────
SUPABASE_URL         = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY         = os.getenv("SUPABASE_ANON_KEY", "").strip()
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
JWT_SECRET           = os.getenv("JWT_SECRET", "").strip()
JWT_ALGORITHM        = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = int(os.getenv("ACCESS_TOKEN_EXPIRE_HOURS", "24"))

DEFAULT_ADMIN_EMAIL = os.getenv("DEFAULT_ADMIN_EMAIL", "ali.rashid.fyp@gmail.com")

# ── Supabase Storage Buckets ──────────────────────────────────
BUCKET_DATASETS  = "datasets"
BUCKET_ML_MODELS = "ml-models"

# ── Environment ───────────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development").lower()
IS_PROD = APP_ENV == "production"

# ── CORS origins ──────────────────────────────────────────────
_raw_origins = os.getenv(
    "CORS_ORIGINS",
    (
        "http://localhost:5173,"
        "http://localhost:3000,"
        "http://127.0.0.1:5173,"
        "https://compressorai-frontend-kappa.vercel.app"
    )
)
CORS_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]


# ── Startup validation ────────────────────────────────────────
def _validate_config():
    missing = []
    if not SUPABASE_URL:
        missing.append("SUPABASE_URL")
    if not SUPABASE_KEY:
        missing.append("SUPABASE_ANON_KEY")
    if not SUPABASE_SERVICE_KEY:
        missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not JWT_SECRET:
        missing.append("JWT_SECRET")
    elif len(JWT_SECRET) < 32:
        print(
            "⚠  WARNING: JWT_SECRET is shorter than 32 characters — "
            "use a strong random secret in production.",
            file=sys.stderr,
        )

    if missing:
        msg = (
            f"❌ Missing required environment variables: {', '.join(missing)}\n"
            f"   In Replit: Go to Tools → Secrets and add these variables."
        )
        if IS_PROD:
            sys.exit(msg)
        else:
            print(f"\n{msg}\n", file=sys.stderr)


_validate_config()


# ── Client factories ──────────────────────────────────────────
def get_supabase_client() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("Supabase credentials not configured. Check Replit Secrets.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def get_supabase_admin_client() -> Client:
    """Service-role client — bypasses RLS."""
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError("Supabase service role key not configured.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ── Shared singletons ─────────────────────────────────────────
supabase:       Optional[Client] = get_supabase_client()       if SUPABASE_URL and SUPABASE_KEY else None
supabase_admin: Optional[Client] = get_supabase_admin_client() if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
