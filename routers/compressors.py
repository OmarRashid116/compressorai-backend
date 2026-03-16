"""
Compressors Router — CompressorAI v5

Structure:
  compressor_types  → "Siera SH-250"  (brand/model — global registry)
  compressor_units  → "CIK1001-A"     (physical unit, belongs to a type)
  user_units        → engineer ↔ unit link

Rules:
  - Anyone can create a type or unit
  - Creating a unit auto-links the engineer to it
  - If unit already exists → just link the user
  - Engineers see only their linked units
  - Admins see everything
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone
from config import get_supabase_client
from deps import get_current_user, require_admin

router = APIRouter()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class CompressorTypeCreate(BaseModel):
    name:               str                     # "Siera SH-250"
    manufacturer:       Optional[str]  = None
    rated_power_kw:     Optional[float]= None
    rated_pressure_bar: Optional[float]= None
    description:        Optional[str]  = None

class CompressorTypeUpdate(BaseModel):
    name:               Optional[str]  = None
    manufacturer:       Optional[str]  = None
    rated_power_kw:     Optional[float]= None
    rated_pressure_bar: Optional[float]= None
    description:        Optional[str]  = None
    is_active:          Optional[bool] = None

class CompressorUnitCreate(BaseModel):
    compressor_type_id: str                     # UUID of "Siera SH-250"
    unit_id:            str                     # "CIK1001-A"
    serial_number:      Optional[str]  = None
    location:           Optional[str]  = None
    notes:              Optional[str]  = None

class CompressorUnitUpdate(BaseModel):
    unit_id:       Optional[str]  = None
    serial_number: Optional[str]  = None
    location:      Optional[str]  = None
    notes:         Optional[str]  = None
    is_active:     Optional[bool] = None


# ══════════════════════════════════════════════════════════════
# COMPRESSOR TYPES
# ══════════════════════════════════════════════════════════════

@router.get("/types/search")
async def search_types(q: str = "", current_user=Depends(get_current_user)):
    """Search compressor types globally — everyone can search."""
    supabase = get_supabase_client()
    if q:
        res = supabase.table("compressor_types").select("*") \
            .ilike("name", f"%{q}%").eq("is_active", True) \
            .order("name").limit(20).execute()
    else:
        res = supabase.table("compressor_types").select("*") \
            .eq("is_active", True).order("name").limit(20).execute()

    results = []
    for t in res.data:
        uc = supabase.table("compressor_units").select("id", count="exact") \
            .eq("compressor_type_id", t["id"]).eq("is_active", True).execute()
        t["unit_count"] = uc.count or 0
        results.append(t)
    return {"results": results}


@router.get("/types")
async def list_types(current_user=Depends(get_current_user)):
    """List all active compressor types with unit count and ML model info."""
    supabase = get_supabase_client()
    res = supabase.table("compressor_types").select("*") \
        .eq("is_active", True).order("name").execute()

    results = []
    for t in res.data:
        uc = supabase.table("compressor_units").select("id", count="exact") \
            .eq("compressor_type_id", t["id"]).execute()
        t["unit_count"] = uc.count or 0

        ml = supabase.table("ml_models").select("id,r2_score,trained_at,trained_on_units") \
            .eq("compressor_type_id", t["id"]).eq("is_active", True) \
            .order("trained_at", desc=True).limit(1).execute()
        t["ml_model"] = ml.data[0] if ml.data else None
        results.append(t)
    return results


@router.post("/types")
async def create_type(data: CompressorTypeCreate, current_user=Depends(get_current_user)):
    """
    Create a new compressor type.
    Rejects duplicate names (case-insensitive).
    """
    supabase = get_supabase_client()
    existing = supabase.table("compressor_types").select("id") \
        .ilike("name", data.name.strip()).execute()
    if existing.data:
        raise HTTPException(400, f"Compressor type '{data.name}' already exists.")

    row = {
        "name":               data.name.strip(),
        "manufacturer":       data.manufacturer,
        "rated_power_kw":     data.rated_power_kw,
        "rated_pressure_bar": data.rated_pressure_bar,
        "description":        data.description,
        "created_by":         current_user["sub"],
    }
    res = supabase.table("compressor_types").insert(row).execute()
    return res.data[0]


@router.get("/types/{type_id}")
async def get_type(type_id: str, current_user=Depends(get_current_user)):
    """Get a single compressor type with all its units and ML model info."""
    supabase = get_supabase_client()
    t = supabase.table("compressor_types").select("*").eq("id", type_id).execute()
    if not t.data:
        raise HTTPException(404, "Compressor type not found.")

    result = t.data[0]
    units = supabase.table("compressor_units").select("*") \
        .eq("compressor_type_id", type_id).eq("is_active", True).execute()
    result["units"] = units.data

    ml = supabase.table("ml_models").select("*") \
        .eq("compressor_type_id", type_id).eq("is_active", True) \
        .order("trained_at", desc=True).limit(1).execute()
    result["ml_model"] = ml.data[0] if ml.data else None
    return result


@router.put("/types/{type_id}")
async def update_type(type_id: str, data: CompressorTypeUpdate,
                      current_user=Depends(require_admin)):
    """Update compressor type — admin only."""
    supabase = get_supabase_client()
    update = {k: v for k, v in data.dict().items() if v is not None}
    if not update:
        raise HTTPException(400, "Nothing to update.")
    supabase.table("compressor_types").update(update).eq("id", type_id).execute()
    return {"message": "Updated successfully."}


# ══════════════════════════════════════════════════════════════
# COMPRESSOR UNITS
# ══════════════════════════════════════════════════════════════

@router.post("/units")
async def create_unit(data: CompressorUnitCreate, current_user=Depends(get_current_user)):
    """
    Add a new unit (e.g. CIK1001-A) under a compressor type.
    If unit already exists → just link this engineer to it.
    Auto-links the creating engineer to the unit.
    """
    supabase = get_supabase_client()

    # Verify type exists
    t = supabase.table("compressor_types").select("id,name") \
        .eq("id", data.compressor_type_id).execute()
    if not t.data:
        raise HTTPException(404, "Compressor type not found.")

    # Check duplicate unit_id under same type
    dup = supabase.table("compressor_units").select("id") \
        .eq("compressor_type_id", data.compressor_type_id) \
        .eq("unit_id", data.unit_id.strip().upper()).execute()

    if dup.data:
        # Unit already exists — just link user and return
        unit = supabase.table("compressor_units").select("*") \
            .eq("id", dup.data[0]["id"]).execute().data[0]
        _link_user_to_unit(supabase, current_user["sub"], unit["id"])
        unit["already_existed"] = True
        unit["type_name"] = t.data[0]["name"]
        unit["message"] = "Unit already exists — linked to your account."
        return unit

    # Create new unit
    row = {
        "compressor_type_id": data.compressor_type_id,
        "unit_id":            data.unit_id.strip().upper(),
        "serial_number":      data.serial_number,
        "location":           data.location,
        "notes":              data.notes,
        "created_by":         current_user["sub"],
    }
    res  = supabase.table("compressor_units").insert(row).execute()
    unit = res.data[0]
    unit["type_name"]       = t.data[0]["name"]
    unit["already_existed"] = False
    unit["message"]         = "New unit created and linked to your account."

    # Auto-link creating engineer
    _link_user_to_unit(supabase, current_user["sub"], unit["id"])
    return unit


@router.get("/units/my")
async def get_my_units(current_user=Depends(get_current_user)):
    """Engineer: list own linked units with type info and latest stats."""
    supabase = get_supabase_client()
    links = supabase.table("user_units").select("unit_id,added_at") \
        .eq("user_id", current_user["sub"]).eq("is_active", True).execute()

    result = []
    for link in links.data:
        unit = supabase.table("compressor_units").select("*") \
            .eq("id", link["unit_id"]).execute()
        if not unit.data:
            continue
        u = unit.data[0]
        u["linked_at"] = link["added_at"]

        # Type info
        ct = supabase.table("compressor_types") \
            .select("name,manufacturer,rated_power_kw,rated_pressure_bar") \
            .eq("id", u["compressor_type_id"]).execute()
        u["type"] = ct.data[0] if ct.data else {}

        # This user's dataset count for this unit
        ds = supabase.table("datasets").select("id", count="exact") \
            .eq("unit_id", u["id"]).eq("user_id", current_user["sub"]).execute()
        u["dataset_count"]    = ds.count or 0
        u["my_dataset_count"] = ds.count or 0  # keep both for compat

        # Latest analysis result
        ar = supabase.table("analysis_results") \
            .select("power_saving_percent,created_at") \
            .eq("unit_id", u["id"]).eq("user_id", current_user["sub"]) \
            .order("created_at", desc=True).limit(1).execute()
        u["latest_saving"] = ar.data[0]["power_saving_percent"] if ar.data else None
        result.append(u)
    return result


@router.get("/units/search")
async def search_units(q: str = "", type_id: str = "",
                       current_user=Depends(get_current_user)):
    """Search units globally by unit_id string or filter by type."""
    supabase = get_supabase_client()
    query = supabase.table("compressor_units").select("*").eq("is_active", True)
    if q:
        query = query.ilike("unit_id", f"%{q}%")
    if type_id:
        query = query.eq("compressor_type_id", type_id)
    res = query.order("unit_id").limit(20).execute()

    results = []
    for u in res.data:
        ct = supabase.table("compressor_types").select("name,manufacturer") \
            .eq("id", u["compressor_type_id"]).execute()
        u["type"] = ct.data[0] if ct.data else {}
        results.append(u)
    return {"results": results}


@router.get("/units/{unit_uuid}")
async def get_unit(unit_uuid: str, current_user=Depends(get_current_user)):
    """
    Get a single unit.
    Engineers: only if they are linked to it.
    Admins: any unit.
    """
    supabase = get_supabase_client()
    unit = supabase.table("compressor_units").select("*").eq("id", unit_uuid).execute()
    if not unit.data:
        raise HTTPException(404, "Unit not found.")
    u = unit.data[0]

    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", current_user["sub"]).eq("unit_id", unit_uuid).execute()
        if not link.data:
            raise HTTPException(403, "You are not linked to this unit.")

    ct = supabase.table("compressor_types").select("*") \
        .eq("id", u["compressor_type_id"]).execute()
    u["type"] = ct.data[0] if ct.data else {}
    return u


@router.post("/units/{unit_uuid}/link")
async def link_unit(unit_uuid: str, current_user=Depends(get_current_user)):
    """Link current user to an existing unit."""
    supabase = get_supabase_client()
    unit = supabase.table("compressor_units").select("id") \
        .eq("id", unit_uuid).execute()
    if not unit.data:
        raise HTTPException(404, "Unit not found.")
    _link_user_to_unit(supabase, current_user["sub"], unit_uuid)
    return {"message": "Linked successfully."}


@router.delete("/units/{unit_uuid}/unlink")
async def unlink_unit(unit_uuid: str, current_user=Depends(get_current_user)):
    """Unlink current user from a unit."""
    supabase = get_supabase_client()
    supabase.table("user_units").update({"is_active": False}) \
        .eq("user_id", current_user["sub"]).eq("unit_id", unit_uuid).execute()
    return {"message": "Unlinked successfully."}


@router.put("/units/{unit_uuid}")
async def update_unit(unit_uuid: str, data: CompressorUnitUpdate,
                      current_user=Depends(require_admin)):
    """Update unit details — admin only."""
    supabase = get_supabase_client()
    update = {k: v for k, v in data.dict().items() if v is not None}
    if not update:
        raise HTTPException(400, "Nothing to update.")
    supabase.table("compressor_units").update(update).eq("id", unit_uuid).execute()
    return {"message": "Updated successfully."}


# ══════════════════════════════════════════════════════════════
# UNIT STATS  (replaces old /compressor_id/stats)
# ══════════════════════════════════════════════════════════════

@router.get("/units/{unit_uuid}/stats")
async def get_unit_stats(unit_uuid: str, current_user=Depends(get_current_user)):
    """
    Stats for a unit.
    Engineers: own analysis results only.
    Admins: all results for the unit.
    """
    supabase = get_supabase_client()

    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", current_user["sub"]).eq("unit_id", unit_uuid).execute()
        if not link.data:
            raise HTTPException(403, "You are not linked to this unit.")

    q = supabase.table("analysis_results").select(
        "id,best_electrical_power,best_mechanical_power,power_saving_percent,scores,created_at"
    ).eq("unit_id", unit_uuid)

    if current_user.get("role") == "engineer":
        q = q.eq("user_id", current_user["sub"])

    res = q.order("created_at", desc=True).limit(50).execute()
    if not res.data:
        return {"total_analyses": 0, "message": "No analyses yet."}

    data    = res.data
    savings = [d.get("power_saving_percent") or 0 for d in data]
    return {
        "total_analyses":        len(data),
        "best_power_saving_pct":     round(max(savings), 2),
        "avg_power_saving_pct":      round(sum(savings) / len(savings), 2),
        "best_power_saving_percent": round(max(savings), 2),
        "avg_power_saving_percent":  round(sum(savings) / len(savings), 2),
        "latest_analysis":       data[0],
        "trend": [
            {"date":       d["created_at"][:10],
             "saving_pct": d.get("power_saving_percent"),
             "best_elec":  d.get("best_electrical_power")}
            for d in data[:20]
        ]
    }


# ══════════════════════════════════════════════════════════════
# ADMIN — FULL OVERVIEW
# ══════════════════════════════════════════════════════════════

@router.get("/admin/overview")
async def admin_overview(current_user=Depends(require_admin)):
    """
    Admin: all types → all units → dataset counts + analysis stats.
    """
    supabase = get_supabase_client()
    types = supabase.table("compressor_types").select("*").order("name").execute()

    result = []
    for t in types.data:
        units = supabase.table("compressor_units").select("*") \
            .eq("compressor_type_id", t["id"]).execute()

        t_units = []
        for u in units.data:
            # All datasets for this unit (all users)
            ds = supabase.table("datasets") \
                .select("id,user_id,total_rows,clean_rows,created_at,is_processed") \
                .eq("unit_id", u["id"]).order("created_at", desc=True).execute()

            # Latest analysis
            ar = supabase.table("analysis_results") \
                .select("power_saving_percent,created_at") \
                .eq("unit_id", u["id"]).order("created_at", desc=True).limit(1).execute()

            # Linked engineers
            uu = supabase.table("user_units").select("user_id") \
                .eq("unit_id", u["id"]).eq("is_active", True).execute()

            u["datasets"]      = ds.data
            u["dataset_count"] = len(ds.data)
            u["latest_saving"] = ar.data[0]["power_saving_percent"] if ar.data else None
            u["linked_users"]  = [x["user_id"] for x in uu.data]
            t_units.append(u)

        ml = supabase.table("ml_models").select("*") \
            .eq("compressor_type_id", t["id"]).eq("is_active", True) \
            .order("trained_at", desc=True).limit(1).execute()

        t["units"]      = t_units
        t["unit_count"] = len(t_units)
        t["ml_model"]   = ml.data[0] if ml.data else None
        result.append(t)
    return result


# ── Also keep a flat /  endpoint for admin dashboard compatibility ─
@router.get("/")
async def get_all(current_user=Depends(get_current_user)):
    """
    Returns all active compressor types (flat list).
    Kept for admin dashboard compatibility.
    """
    supabase = get_supabase_client()
    res = supabase.table("compressor_types").select("*") \
        .eq("is_active", True).order("name").execute()
    return res.data


# ══════════════════════════════════════════════════════════════
# INTERNAL HELPER
# ══════════════════════════════════════════════════════════════

def _link_user_to_unit(supabase, user_id: str, unit_id: str):
    """Link a user to a unit — upsert (reactivates if previously unlinked)."""
    existing = supabase.table("user_units").select("id,is_active") \
        .eq("user_id", user_id).eq("unit_id", unit_id).execute()
    if existing.data:
        if not existing.data[0]["is_active"]:
            supabase.table("user_units").update({"is_active": True}) \
                .eq("id", existing.data[0]["id"]).execute()
    else:
        supabase.table("user_units").insert({
            "user_id": user_id, "unit_id": unit_id
        }).execute()

# ── Short aliases — frontend calls /compressors/my and /compressors/search ──
@router.get("/my", include_in_schema=False)
async def get_my_units_alias(current_user=Depends(get_current_user)):
    """Alias for /units/my — for frontend compatibility."""
    return await get_my_units(current_user=current_user)


@router.get("/search", include_in_schema=False)
async def search_units_alias(
    q: str = "",
    type_id: str = "",
    current_user=Depends(get_current_user),
):
    """Alias for /units/search — for frontend compatibility."""
    return await search_units(q=q, type_id=type_id, current_user=current_user)
