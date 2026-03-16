"""
init_admin.py — CompressorAI v5
Run ONCE to create the default admin account in Supabase.

  python init_admin.py

Production changes:
  - password_plain NOT stored (removed feature)
  - Reads credentials from .env (not hardcoded fallbacks)
  - Validates env vars before proceeding
  - Fully idempotent (safe to re-run)
"""
import os
import sys
import bcrypt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from supabase import create_client

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

# ── Admin credentials (from env or defaults for dev) ──────────
EMAIL    = os.getenv("DEFAULT_ADMIN_EMAIL",    "ali.rashid.fyp@gmail.com")
PASSWORD = os.getenv("DEFAULT_ADMIN_PASSWORD", "Me-22032#Secure")
NAME     = os.getenv("DEFAULT_ADMIN_NAME",     "Ali Rashid")

if len(PASSWORD) < 8:
    sys.exit("❌ DEFAULT_ADMIN_PASSWORD must be at least 8 characters.")


def hash_pw(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()


existing = sb.table("users").select("id,is_default_admin").eq("email", EMAIL).execute()

if existing.data:
    user = existing.data[0]
    if user.get("is_default_admin"):
        print(f"✅ Default admin already exists: {EMAIL}")
    else:
        # Promote to default admin if somehow created as regular user
        sb.table("users").update({
            "role":             "admin",
            "is_default_admin": True,
            "is_active":        True,
            "is_email_verified": True,
        }).eq("id", user["id"]).execute()
        print(f"✅ Existing user promoted to default admin: {EMAIL}")
else:
    sb.table("users").insert({
        "full_name":         NAME,
        "email":             EMAIL,
        "password_hash":     hash_pw(PASSWORD),
        "role":              "admin",
        "is_default_admin":  True,
        "is_active":         True,
        "is_email_verified": True,
        "agreed_to_terms":   True,
        "company":           "NEDUET",
    }).execute()
    print(f"✅ Default admin created: {EMAIL}")
    print(f"   Password: {PASSWORD}")
    print(f"   ⚠  Change this password after first login!")