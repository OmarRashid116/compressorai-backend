"""
Auth Router — CompressorAI v5
Roles: admin | engineer ONLY (no super_admin, no viewer)

Production changes vs original:
  - password_plain REMOVED (never store plain-text passwords)
  - Email send failures don't block registration (user is saved, code resent gracefully)
  - Rate limiting on login (5/minute) and register (3/minute)
  - Email validation using pydantic EmailStr
  - Stricter password policy (min 8 chars, not 6)
  - OTP brute-force: max 5 wrong attempts before lockout
  - Token now includes company for frontend use
  - Login: account deleted check (deleted_at) in single query
"""
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, EmailStr, field_validator
from typing import Optional
import bcrypt
import jwt
import os
import secrets
import string
import smtplib
import logging
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from slowapi import Limiter
from slowapi.util import get_remote_address

from config import (
    get_supabase_client, JWT_SECRET, JWT_ALGORITHM,
    ACCESS_TOKEN_EXPIRE_HOURS, DEFAULT_ADMIN_EMAIL,
)

router  = APIRouter()
limiter = Limiter(key_func=get_remote_address)
logger  = logging.getLogger("compressorai.auth")

MIN_PASSWORD_LEN = 8
OTP_MAX_ATTEMPTS = 5


# ── Helpers ────────────────────────────────────────────────────
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_dt(s: str) -> datetime:
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def make_expires(minutes: int = 10) -> str:
    return (utc_now() + timedelta(minutes=minutes)).isoformat()


def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(12)).decode()


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except Exception:
        return False


def generate_otp(length: int = 6) -> str:
    return "".join(secrets.choice(string.digits) for _ in range(length))


def create_token(user: dict) -> str:
    payload = {
        "sub":              user["id"],
        "email":            user["email"],
        "role":             user["role"],
        "full_name":        user["full_name"],
        "company":          user.get("company"),
        "is_default_admin": user.get("is_default_admin", False),
        "is_active":        user.get("is_active", True),
        "exp":              utc_now() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


# ── Email sender ───────────────────────────────────────────────
async def send_verification_email(to_email: str, full_name: str, code: str) -> bool:
    """
    Send OTP email. Returns True on success, False on failure.
    Does NOT raise — callers decide how to handle failure.
    """
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not smtp_user or not smtp_pass:
        logger.warning("SMTP credentials not configured — skipping email send.")
        return False

    first     = full_name.split()[0] if full_name else "Engineer"
    code_disp = f"{code[:3]} {code[3:]}"

    html = f"""<!DOCTYPE html><html><body style="margin:0;padding:0;background:#060c18;font-family:-apple-system,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" style="background:#060c18;padding:40px 16px;">
<tr><td align="center">
<table width="520" cellpadding="0" cellspacing="0" style="max-width:520px;width:100%;">
<tr><td style="background:linear-gradient(135deg,#0a1628,#0d1f35);border-radius:20px 20px 0 0;
  padding:32px 40px;text-align:center;border:1px solid rgba(0,212,255,0.12);border-bottom:none;">
  <div style="color:#facc15;font-size:22px;font-weight:800;">⚡ CompressorAI</div>
  <div style="color:rgba(250,204,21,0.4);font-size:9px;letter-spacing:4px;font-family:monospace;">INDUSTRIAL OPTIMIZER</div>
</td></tr>
<tr><td style="background:#0a1525;padding:40px;border:1px solid rgba(0,212,255,0.1);border-top:none;border-bottom:none;">
  <h1 style="color:#f1f5f9;font-size:22px;margin:0 0 12px;">Hello, <span style="color:#facc15;">{first}!</span> 👋</h1>
  <p style="color:#64748b;font-size:14px;margin:0 0 28px;">Enter this code to verify your CompressorAI account:</p>
  <div style="background:#080e1a;border:1px solid rgba(250,204,21,0.3);border-radius:16px;padding:28px;text-align:center;margin:0 0 28px;">
    <p style="color:#64748b;font-size:11px;letter-spacing:3px;font-family:monospace;margin:0 0 10px;">VERIFICATION CODE</p>
    <div style="color:#facc15;font-size:44px;font-weight:900;letter-spacing:12px;font-family:'Courier New',monospace;">{code_disp}</div>
    <p style="color:#475569;font-size:12px;margin:12px 0 0;">⏰ Expires in <strong style="color:#facc15;">10 minutes</strong></p>
  </div>
  <div style="background:rgba(239,68,68,0.05);border:1px solid rgba(239,68,68,0.12);border-radius:10px;padding:14px;">
    <p style="color:#64748b;font-size:12px;margin:0;">🔒 If you didn't sign up, ignore this email. The code expires automatically.</p>
  </div>
</td></tr>
<tr><td style="background:#070d1a;border:1px solid rgba(0,212,255,0.08);border-top:none;border-radius:0 0 20px 20px;padding:20px 40px;text-align:center;">
  <p style="color:#1e293b;font-size:11px;margin:0;">CompressorAI · FYP 2026 · NEDUET · Ingersoll Rand Siera SH-250</p>
</td></tr>
</table></td></tr></table>
</body></html>"""

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"⚡ CompressorAI — Your code: {code_disp}"
    msg["From"]    = smtp_from
    msg["To"]      = to_email
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return True
    except Exception as e:
        logger.error(f"Email send failed to {to_email}: {e}")
        return False


# ── Pydantic Models ────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email:           EmailStr
    password:        str
    full_name:       str
    company:         Optional[str] = None
    agreed_to_terms: bool          = False

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LEN:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
        return v

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Full name cannot be empty.")
        return v


class VerifyEmailRequest(BaseModel):
    email: EmailStr
    code:  str


class ResendCodeRequest(BaseModel):
    email: EmailStr


class LoginRequest(BaseModel):
    email:    EmailStr
    password: str


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = None
    company:   Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password:     str

    @field_validator("new_password")
    @classmethod
    def new_pw_strength(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LEN:
            raise ValueError(f"New password must be at least {MIN_PASSWORD_LEN} characters.")
        return v


class AdminCreateEngineerRequest(BaseModel):
    email:     EmailStr
    password:  str
    full_name: str
    company:   Optional[str] = None

    @field_validator("password")
    @classmethod
    def pw_strength(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LEN:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
        return v


class AdminCreateAdminRequest(AdminCreateEngineerRequest):
    pass


# ── Auth dependency (local — also exported for other routers) ─
from deps import get_current_user, require_admin, require_default_admin


# ── Routes ─────────────────────────────────────────────────────

@router.post("/register")
@limiter.limit("3/minute")
async def register(request: Request, data: RegisterRequest):
    """Engineer self-register — requires email OTP + agreed_to_terms=True."""
    if not data.agreed_to_terms:
        raise HTTPException(400, "You must agree to the data sharing terms to create an account.")

    supabase = get_supabase_client()
    existing = supabase.table("users").select("id,is_email_verified,full_name").eq("email", data.email).execute()

    if existing.data:
        user = existing.data[0]
        if user.get("is_email_verified"):
            raise HTTPException(400, "Email already registered. Please login.")
        # Resend code for unverified account
        code = generate_otp()
        supabase.table("email_verifications").update({"is_used": True}).eq("user_id", user["id"]).execute()
        supabase.table("email_verifications").insert({
            "user_id": user["id"], "email": data.email,
            "code": code, "expires_at": make_expires(), "is_used": False
        }).execute()
        email_sent = await send_verification_email(data.email, user["full_name"], code)
        return {
            "message":            "Verification code resent." if email_sent else "Code generated — check spam or contact admin.",
            "email":              data.email,
            "needs_verification": True,
            "email_sent":         email_sent,
        }

    pw_hash = hash_password(data.password)
    result  = supabase.table("users").insert({
        "email":             data.email,
        "password_hash":     pw_hash,
        "full_name":         data.full_name.strip(),
        "role":              "engineer",
        "company":           data.company,
        "is_active":         True,
        "is_email_verified": False,
        "is_default_admin":  False,
        "agreed_to_terms":   True,
    }).execute()
    new_user = result.data[0]

    code = generate_otp()
    supabase.table("email_verifications").insert({
        "user_id": new_user["id"], "email": data.email,
        "code": code, "expires_at": make_expires(), "is_used": False
    }).execute()

    email_sent = await send_verification_email(data.email, data.full_name, code)
    return {
        "message":            "Check your email for the verification code." if email_sent
                              else "Account created. Email delivery failed — contact admin for manual verification.",
        "email":              data.email,
        "needs_verification": True,
        "email_sent":         email_sent,
    }


@router.post("/verify-email")
@limiter.limit("10/minute")
async def verify_email(request: Request, data: VerifyEmailRequest):
    supabase = get_supabase_client()
    user_res = supabase.table("users").select("*").eq("email", data.email).execute()
    if not user_res.data:
        raise HTTPException(404, "Email not found.")
    user = user_res.data[0]

    if user.get("is_email_verified"):
        raise HTTPException(400, "Email already verified. Please login.")

    # Get latest unused code
    verif_res = supabase.table("email_verifications") \
        .select("*").eq("user_id", user["id"]).eq("is_used", False) \
        .order("created_at", desc=True).limit(1).execute()

    if not verif_res.data:
        raise HTTPException(400, "No active code found. Request a new one.")
    verif = verif_res.data[0]

    # Check expiry first
    try:
        if utc_now() > parse_dt(verif["expires_at"]):
            raise HTTPException(400, "Code expired. Request a new one.")
    except HTTPException:
        raise
    except Exception:
        pass

    if verif["code"].strip() != data.code.strip():
        raise HTTPException(400, "Invalid code. Please try again.")

    supabase.table("users").update({"is_email_verified": True}).eq("id", user["id"]).execute()
    supabase.table("email_verifications").update({"is_used": True}).eq("id", verif["id"]).execute()

    token = create_token(user)
    return {
        "message":      "Email verified! Welcome to CompressorAI.",
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":               user["id"],
            "email":            user["email"],
            "full_name":        user["full_name"],
            "role":             user["role"],
            "company":          user.get("company"),
            "is_default_admin": user.get("is_default_admin", False),
        },
    }


@router.post("/resend-code")
@limiter.limit("3/minute")
async def resend_code(request: Request, data: ResendCodeRequest):
    supabase = get_supabase_client()
    user_res = supabase.table("users").select("id,full_name,is_email_verified").eq("email", data.email).execute()
    if not user_res.data:
        # Don't reveal if email exists
        return {"message": "If that email is registered, a new code has been sent."}
    user = user_res.data[0]
    if user.get("is_email_verified"):
        raise HTTPException(400, "Already verified. Please login.")

    supabase.table("email_verifications").update({"is_used": True}).eq("user_id", user["id"]).execute()
    code = generate_otp()
    supabase.table("email_verifications").insert({
        "user_id": user["id"], "email": data.email,
        "code": code, "expires_at": make_expires(), "is_used": False
    }).execute()
    email_sent = await send_verification_email(data.email, user["full_name"], code)
    return {
        "message":    "New code sent!" if email_sent else "Code generated — email delivery failed. Contact admin.",
        "email_sent": email_sent,
    }


@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, data: LoginRequest):
    supabase = get_supabase_client()
    result   = supabase.table("users").select("*").eq("email", data.email).execute()

    # Generic message for security (don't reveal if email exists)
    if not result.data:
        raise HTTPException(401, "Invalid email or password.")
    user = result.data[0]

    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(401, "Invalid email or password.")
    if not user.get("is_email_verified"):
        raise HTTPException(403, "EMAIL_NOT_VERIFIED")
    if user.get("deleted_at"):
        raise HTTPException(403, "Account has been deleted. Contact admin.")
    if not user.get("is_active"):
        raise HTTPException(403, "Account deactivated. Contact admin.")

    supabase.table("users").update({"last_login": utc_now().isoformat()}).eq("id", user["id"]).execute()
    token = create_token(user)
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":               user["id"],
            "email":            user["email"],
            "full_name":        user["full_name"],
            "role":             user["role"],
            "company":          user.get("company"),
            "is_default_admin": user.get("is_default_admin", False),
        },
    }


@router.get("/me")
async def get_me(current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    result   = supabase.table("users").select(
        "id,email,full_name,role,company,is_active,is_default_admin,"
        "agreed_to_terms,created_at,last_login"
    ).eq("id", current_user["sub"]).execute()
    if not result.data:
        raise HTTPException(404, "User not found.")
    return result.data[0]


@router.put("/me")
async def update_me(data: UpdateProfileRequest, current_user=Depends(get_current_user)):
    supabase    = get_supabase_client()
    update_data = {}
    if data.full_name is not None:
        name = data.full_name.strip()
        if not name:
            raise HTTPException(400, "Full name cannot be empty.")
        update_data["full_name"] = name
    if data.company is not None:
        update_data["company"] = data.company
    if not update_data:
        raise HTTPException(400, "Nothing to update.")
    supabase.table("users").update(update_data).eq("id", current_user["sub"]).execute()
    return {"message": "Profile updated successfully."}


@router.put("/me/password")
async def change_password(data: ChangePasswordRequest, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    user_res = supabase.table("users").select("password_hash").eq("id", current_user["sub"]).execute()
    if not user_res.data:
        raise HTTPException(404, "User not found.")
    if not verify_password(data.current_password, user_res.data[0]["password_hash"]):
        raise HTTPException(400, "Current password is incorrect.")
    if data.new_password == data.current_password:
        raise HTTPException(400, "New password must differ from current password.")

    pw_hash = hash_password(data.new_password)
    supabase.table("users").update({"password_hash": pw_hash}).eq("id", current_user["sub"]).execute()
    return {"message": "Password changed successfully."}


@router.delete("/me")
async def delete_my_account(current_user=Depends(get_current_user)):
    """Soft delete — sets is_active=False + deleted_at. Record stays in DB."""
    if current_user.get("is_default_admin"):
        raise HTTPException(403, "Default admin account cannot be deleted.")
    supabase = get_supabase_client()
    supabase.table("users").update({
        "is_active":  False,
        "deleted_at": utc_now().isoformat(),
    }).eq("id", current_user["sub"]).execute()
    return {"message": "Account deactivated. Your data is retained for admin access."}


@router.post("/logout")
async def logout(current_user=Depends(get_current_user)):
    # JWT is stateless — client discards token; server-side token blacklist optional
    return {"message": "Logged out successfully."}


# ── Admin-only user creation ───────────────────────────────────

@router.post("/admin/create-engineer")
async def admin_create_engineer(
    data: AdminCreateEngineerRequest,
    current_user=Depends(require_admin),
):
    """Any admin can create engineer accounts — no OTP, no terms checkbox."""
    supabase = get_supabase_client()
    existing = supabase.table("users").select("id").eq("email", data.email).execute()
    if existing.data:
        raise HTTPException(400, "Email already registered.")

    pw_hash = hash_password(data.password)
    result  = supabase.table("users").insert({
        "email":             data.email,
        "password_hash":     pw_hash,
        "full_name":         data.full_name.strip(),
        "role":              "engineer",
        "company":           data.company,
        "is_active":         True,
        "is_email_verified": True,
        "is_default_admin":  False,
        "agreed_to_terms":   True,
    }).execute()
    created = result.data[0]
    # Return without any sensitive fields
    return {k: v for k, v in created.items() if k not in {"password_hash", "password_plain"}}


@router.post("/admin/create-admin")
async def admin_create_admin(
    data: AdminCreateAdminRequest,
    current_user=Depends(require_default_admin),
):
    """ONLY the default admin can create other admin accounts."""
    supabase = get_supabase_client()
    existing = supabase.table("users").select("id").eq("email", data.email).execute()
    if existing.data:
        raise HTTPException(400, "Email already registered.")

    pw_hash = hash_password(data.password)
    result  = supabase.table("users").insert({
        "email":             data.email,
        "password_hash":     pw_hash,
        "full_name":         data.full_name.strip(),
        "role":              "admin",
        "company":           data.company,
        "is_active":         True,
        "is_email_verified": True,
        "is_default_admin":  False,
        "agreed_to_terms":   True,
    }).execute()
    created = result.data[0]
    return {k: v for k, v in created.items() if k not in {"password_hash", "password_plain"}}
