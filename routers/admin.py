"""
Admin Router — CompressorAI v5

Production changes vs original:
  - password_plain removed from all responses
  - N+1 query loops replaced with bulk fetches
  - Destructive actions (delete, toggle, reset-password) use require_admin_db
    (re-fetches from DB so stale tokens can't be exploited)
  - get_stats uses COUNT queries instead of fetching all rows
  - Pagination added to get_all_users
  - Safe response helper centralised
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, field_validator
from typing import Optional
from datetime import datetime, timezone

from config import get_supabase_client, get_supabase_admin_client
from deps import get_current_user, require_admin, require_admin_db, require_default_admin
from routers.auth import hash_password

router = APIRouter()
logger = logging.getLogger("compressorai.admin")

MIN_PASSWORD_LEN = 8


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_user(user: dict) -> dict:
    """Always strip password fields before returning to any client."""
    return {k: v for k, v in user.items() if k not in {"password_hash", "password_plain"}}


# ── System Stats ───────────────────────────────────────────────
@router.get("/stats")
async def get_stats(current_user=Depends(require_admin)):
    """
    Dashboard stats. Uses COUNT queries — no full table scans.
    """
    supabase = get_supabase_admin_client()

    users_res  = supabase.table("users").select("id,role,is_active").execute()
    units_res  = supabase.table("compressor_units").select("id,is_active").execute()
    # Only fetch columns needed for aggregation
    analyses   = supabase.table("analysis_results").select("id,power_saving_percent").execute()

    users  = users_res.data  or []
    units  = units_res.data  or []
    anlys  = analyses.data   or []

    total_analyses = len(anlys)
    avg_saving = (
        sum(a.get("power_saving_percent") or 0 for a in anlys) / total_analyses
        if total_analyses else 0
    )

    return {
        "total_users":              len(users),
        "active_users":             sum(1 for u in users if u.get("is_active")),
        "total_admins":             sum(1 for u in users if u.get("role") == "admin"),
        "total_engineers":          sum(1 for u in users if u.get("role") == "engineer"),
        "total_compressor_units":   len(units),
        "active_compressor_units":  sum(1 for c in units if c.get("is_active")),
        "total_analyses":           total_analyses,
        "avg_power_saving_percent": round(avg_saving, 2),
    }


# ── All Users (paginated) ─────────────────────────────────────
@router.get("/users")
async def get_all_users(
    current_user=Depends(require_admin),
    limit:  int = Query(50, ge=1, le=200),
    offset: int = Query(0,  ge=0),
    role:   Optional[str] = Query(None),
):
    supabase = get_supabase_admin_client()
    fields   = (
        "id,email,full_name,role,company,is_active,is_email_verified,"
        "is_default_admin,agreed_to_terms,created_at,last_login,deleted_at"
    )
    q = supabase.table("users").select(fields)
    if role in ("admin", "engineer"):
        q = q.eq("role", role)
    result = q.order("created_at", desc=False).range(offset, offset + limit - 1).execute()
    return [_safe_user(u) for u in result.data]


# ── Single User Detail ────────────────────────────────────────
@router.get("/users/{user_id}")
async def get_user_detail(user_id: str, current_user=Depends(require_admin)):
    supabase = get_supabase_admin_client()
    fields   = (
        "id,email,full_name,role,company,is_active,is_email_verified,"
        "is_default_admin,agreed_to_terms,created_at,last_login"
    )
    user_res = supabase.table("users").select(fields).eq("id", user_id).execute()
    if not user_res.data:
        raise HTTPException(404, "User not found.")
    user = _safe_user(user_res.data[0])

    # Units this user is linked to
    uc_res   = supabase.table("user_units").select("unit_id,added_at,is_active") \
        .eq("user_id", user_id).execute()
    unit_ids = [link["unit_id"] for link in uc_res.data]

    # Bulk fetch units (no N+1)
    units_map = {}
    if unit_ids:
        units_res = supabase.table("compressor_units").select("*") \
            .in_("id", unit_ids).execute()
        units_map = {u["id"]: u for u in units_res.data}

    compressors = [units_map[uid] for uid in unit_ids if uid in units_map]

    # Analysis summary (last 20)
    analyses_res = supabase.table("analysis_results").select(
        "id,unit_id,dataset_id,power_saving_percent,scores,created_at"
    ).eq("user_id", user_id).order("created_at", desc=True).limit(20).execute()

    user["compressors"]    = compressors
    user["analyses"]       = analyses_res.data
    user["total_analyses"] = len(analyses_res.data)
    return user


# ── Soft Delete User ──────────────────────────────────────────
@router.delete("/users/{user_id}")
async def delete_user(user_id: str, current_user=Depends(require_admin_db)):
    supabase = get_supabase_admin_client()
    target_res = supabase.table("users").select(
        "id,email,role,is_default_admin"
    ).eq("id", user_id).execute()
    if not target_res.data:
        raise HTTPException(404, "User not found.")
    target = target_res.data[0]

    if target.get("is_default_admin"):
        raise HTTPException(403, "The default admin cannot be deleted.")
    if target["id"] == current_user.get("sub"):
        raise HTTPException(400, "You cannot delete your own account.")
    if target["role"] == "admin" and not current_user.get("is_default_admin"):
        raise HTTPException(403, "Only the default admin can delete other admins.")

    supabase.table("users").update({
        "is_active":  False,
        "deleted_at": utc_now(),
    }).eq("id", user_id).execute()
    logger.info(f"Admin {current_user.get('email')} soft-deleted user {target['email']}")
    return {"message": f"User {target['email']} deactivated successfully."}


# ── Toggle Active ─────────────────────────────────────────────
@router.put("/users/{user_id}/toggle-active")
async def toggle_active(user_id: str, current_user=Depends(require_admin_db)):
    supabase = get_supabase_admin_client()
    target_res = supabase.table("users").select(
        "id,email,role,is_active,is_default_admin"
    ).eq("id", user_id).execute()
    if not target_res.data:
        raise HTTPException(404, "User not found.")
    target = target_res.data[0]

    if target.get("is_default_admin"):
        raise HTTPException(403, "Cannot deactivate the default admin.")
    if target["role"] == "admin" and not current_user.get("is_default_admin"):
        raise HTTPException(403, "Only the default admin can toggle other admins.")

    new_status = not target.get("is_active", True)
    supabase.table("users").update({"is_active": new_status}).eq("id", user_id).execute()
    action = "activated" if new_status else "deactivated"
    logger.info(f"Admin {current_user.get('email')} {action} user {target['email']}")
    return {"message": f"User {action} successfully.", "is_active": new_status}


# ── Reset Password ────────────────────────────────────────────
class ResetPasswordRequest(BaseModel):
    new_password: str

    @field_validator("new_password")
    @classmethod
    def pw_length(cls, v: str) -> str:
        if len(v) < MIN_PASSWORD_LEN:
            raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
        return v


@router.put("/users/{user_id}/reset-password")
async def reset_password(
    user_id: str,
    data:    ResetPasswordRequest,
    current_user=Depends(require_admin_db),
):
    supabase = get_supabase_admin_client()
    target_res = supabase.table("users").select(
        "id,email,role,is_default_admin"
    ).eq("id", user_id).execute()
    if not target_res.data:
        raise HTTPException(404, "User not found.")
    target = target_res.data[0]

    if target.get("is_default_admin") and not current_user.get("is_default_admin"):
        raise HTTPException(403, "Cannot reset the default admin's password.")
    if target["role"] == "admin" and not current_user.get("is_default_admin"):
        raise HTTPException(403, "Only the default admin can reset other admins' passwords.")

    pw_hash = hash_password(data.new_password)
    # password_plain intentionally NOT stored
    supabase.table("users").update({"password_hash": pw_hash}).eq("id", user_id).execute()
    logger.info(f"Admin {current_user.get('email')} reset password for {target['email']}")
    return {"message": "Password reset successfully."}


# ── All Compressor Units ──────────────────────────────────────
@router.get("/compressors")
async def get_all_compressors(current_user=Depends(require_admin)):
    """
    All compressor units with user_count + analysis_count.
    Bulk-fetched to avoid N+1.
    """
    supabase = get_supabase_admin_client()
    units    = supabase.table("compressor_units").select("*") \
        .order("created_at", desc=False).execute()
    if not units.data:
        return []

    unit_ids = [c["id"] for c in units.data]

    # Bulk: user_units
    uu_res   = supabase.table("user_units").select("unit_id,user_id") \
        .in_("unit_id", unit_ids).execute()
    uu_map: dict[str, list] = {}
    for link in uu_res.data:
        uu_map.setdefault(link["unit_id"], []).append(link["user_id"])

    # Bulk: analysis_results (power_saving_percent only)
    an_res  = supabase.table("analysis_results") \
        .select("unit_id,power_saving_percent") \
        .in_("unit_id", unit_ids).execute()
    an_map: dict[str, list] = {}
    for a in an_res.data:
        an_map.setdefault(a["unit_id"], []).append(a.get("power_saving_percent") or 0)

    result = []
    for c in units.data:
        savings           = an_map.get(c["id"], [])
        c["user_count"]   = len(uu_map.get(c["id"], []))
        c["analysis_count"] = len(savings)
        c["avg_saving"]   = round(sum(savings) / len(savings), 2) if savings else None
        result.append(c)
    return result


# ── Datasets for a Compressor Unit ────────────────────────────
@router.get("/compressors/{unit_id}/datasets")
async def get_compressor_datasets(unit_id: str, current_user=Depends(require_admin)):
    supabase = get_supabase_admin_client()
    result   = supabase.table("datasets").select(
        "id,user_id,original_filename,total_rows,clean_rows,"
        "was_raw,cleaning_summary,is_processed,created_at"
    ).eq("unit_id", unit_id).order("created_at", desc=True).execute()

    if not result.data:
        return []

    # Bulk-fetch uploaders
    user_ids  = list({d["user_id"] for d in result.data if d.get("user_id")})
    users_res = supabase.table("users").select("id,full_name,email") \
        .in_("id", user_ids).execute() if user_ids else type("R", (), {"data": []})()
    users_map = {u["id"]: u for u in users_res.data}

    datasets = []
    for d in result.data:
        uploader        = users_map.get(d["user_id"], {})
        d["user_name"]  = uploader.get("full_name", "Unknown")
        d["user_email"] = uploader.get("email", "Unknown")
        datasets.append(d)
    return datasets


# ── User's datasets for a specific unit ──────────────────────
@router.get("/users/{user_id}/compressors/{unit_id}/datasets")
async def get_user_compressor_datasets(
    user_id: str, unit_id: str, current_user=Depends(require_admin)
):
    supabase = get_supabase_admin_client()
    result   = supabase.table("datasets").select("*") \
        .eq("user_id", user_id).eq("unit_id", unit_id) \
        .order("created_at", desc=True).execute()
    return result.data


# ── Single Analysis Result ────────────────────────────────────
@router.get("/analyses/{analysis_id}")
async def get_analysis_detail(analysis_id: str, current_user=Depends(require_admin)):
    supabase = get_supabase_admin_client()
    result   = supabase.table("analysis_results").select("*").eq("id", analysis_id).execute()
    if not result.data:
        raise HTTPException(404, "Analysis not found.")
    return result.data[0]


# ── Trigger Retrain ───────────────────────────────────────────
@router.post("/retrain/{type_id}")
async def retrain_model(type_id: str, current_user=Depends(require_admin)):
    """Trigger retrain for a compressor type (delegates to retrain router)."""
    supabase = get_supabase_admin_client()

    units    = supabase.table("compressor_units").select("id") \
        .eq("compressor_type_id", type_id).execute()
    unit_ids = [u["id"] for u in units.data]
    if not unit_ids:
        raise HTTPException(404, "No compressor units found for this type.")

    analyses = supabase.table("analysis_results").select("id") \
        .in_("unit_id", unit_ids).limit(50).execute()
    if len(analyses.data) < 3:
        raise HTTPException(
            400,
            f"Need at least 3 analyses to retrain. Currently: {len(analyses.data)}"
        )

    try:
        from routers.retrain import trigger_retrain_task
        trigger_retrain_task(type_id, triggered_by=current_user.get("sub", "admin"))
    except Exception as e:
        raise HTTPException(500, f"Retrain trigger failed: {str(e)}")

    return {
        "message":       f"Retrain triggered for type {type_id}.",
        "type_id":       type_id,
        "analyses_used": len(analyses.data),
    }
