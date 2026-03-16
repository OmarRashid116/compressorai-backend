"""
Datasets Router — CompressorAI v5
Production changes vs original:
  - File size validated from Content-Length BEFORE reading into memory
  - Upload failure recovery: if processed upload fails, raw file is deleted (no orphans)
  - Bulk user fetch in admin endpoints (no N+1)
  - Pagination on list endpoints
  - Filename sanitisation before storage
  - Allow admin to upload on behalf of any linked user
"""
import io
import re
import logging
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Request, Query
from typing import Optional
from datetime import datetime, timezone
from config import get_supabase_client
from deps import get_current_user, require_admin
from storage import (
    upload_raw_dataset, upload_processed_dataset,
    get_dataset_download_url, delete_dataset_file,
)
from ml.engine import validate_dataset, auto_clean

router = APIRouter()
logger = logging.getLogger("compressorai.datasets")

MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB
ALLOWED_EXTS   = (".xlsx", ".xls", ".csv")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_filename(name: str) -> str:
    name = re.sub(r"[^\w\s\-.]", "", name).strip()
    return name or "dataset"


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════

@router.post("/upload/{unit_uuid}")
async def upload_dataset(
    request:    Request,
    unit_uuid:  str,
    file:       UploadFile = File(...),
    v_voltage:  Optional[float] = Form(None),
    cos_phi:    Optional[float] = Form(None),
    current_user=Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id  = current_user["sub"]

    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_FILE_BYTES + 1024:
        raise HTTPException(413, f"File too large. Maximum {MAX_FILE_BYTES // (1024*1024)} MB allowed.")

    unit = supabase.table("compressor_units") \
        .select("id,unit_id,compressor_type_id") \
        .eq("id", unit_uuid).eq("is_active", True).execute()
    if not unit.data:
        raise HTTPException(404, "Compressor unit not found.")

    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", user_id).eq("unit_id", unit_uuid) \
            .eq("is_active", True).execute()
        if not link.data:
            raise HTTPException(403, "You are not linked to this unit.")

    fname = _safe_filename(file.filename or "dataset.xlsx")
    ext   = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Only Excel ({', '.join(ALLOWED_EXTS)}) files accepted.")

    raw_bytes = await file.read()
    if len(raw_bytes) > MAX_FILE_BYTES:
        raise HTTPException(413, f"File too large. Maximum {MAX_FILE_BYTES // (1024*1024)} MB allowed.")
    if len(raw_bytes) == 0:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        df_raw = pd.read_csv(io.BytesIO(raw_bytes)) if ext == ".csv" \
            else pd.read_excel(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(400, f"Could not read file: {str(e)}")

    user_params: dict = {}
    if v_voltage is not None:
        user_params["voltage"]      = float(v_voltage)
    if cos_phi is not None:
        user_params["power_factor"] = float(cos_phi)

    validation = validate_dataset(df_raw, user_params)
    if not validation["valid"]:
        raise HTTPException(422, {
            "message": "Dataset validation failed.",
            "errors":  validation["errors"],
            "hint":    "Ensure the file has required columns: "
                       "Loading Pressure, Unloading Pressure, Inlet Pressure, "
                       "Discharge Pressure, Current (Amp).",
        })

    was_raw  = validation.get("was_raw", False)
    raw_path = upload_raw_dataset(unit_uuid, user_id, raw_bytes, fname)

    try:
        clean_result     = auto_clean(df_raw, user_params)
        df_clean         = clean_result["df"]
        cleaning_summary = clean_result["summary"]
        proc_path        = upload_processed_dataset(
            unit_uuid, user_id, _df_to_bytes(df_clean), fname
        )
    except Exception as e:
        logger.error(f"Processed upload failed for unit {unit_uuid}: {e} — rolling back raw.")
        delete_dataset_file(raw_path)
        raise HTTPException(500, f"Failed to process dataset: {str(e)}")

    record = {
        "unit_id":             unit_uuid,
        "user_id":             user_id,
        "original_filename":   fname,
        "raw_file_path":       raw_path,
        "processed_file_path": proc_path,
        "total_rows":          len(df_raw),
        "clean_rows":          len(df_clean),
        "was_raw":             was_raw,
        "cleaning_summary":    cleaning_summary,
        "is_processed":        True,
    }
    ds_res  = supabase.table("datasets").insert(record).execute()
    dataset = ds_res.data[0]

    type_id           = unit.data[0]["compressor_type_id"]
    retrain_triggered = _check_auto_retrain(supabase, type_id, dataset["id"])

    return {
        "dataset":           dataset,
        "validation":        validation,
        "cleaning_summary":  cleaning_summary,
        "retrain_triggered": retrain_triggered,
        "message":           "Dataset uploaded and processed successfully.",
    }


# ══════════════════════════════════════════════════════════════
# READ
# ══════════════════════════════════════════════════════════════

@router.get("/my/{unit_uuid}")
async def get_my_datasets(
    unit_uuid:   str,
    limit:       int = Query(20, ge=1, le=100),
    offset:      int = Query(0,  ge=0),
    current_user=Depends(get_current_user),
):
    supabase = get_supabase_client()
    user_id  = current_user["sub"]

    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", user_id).eq("unit_id", unit_uuid).execute()
        if not link.data:
            raise HTTPException(403, "Not linked to this unit.")

    datasets = supabase.table("datasets").select("*") \
        .eq("unit_id", unit_uuid).eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .range(offset, offset + limit - 1).execute()
    return datasets.data


# ── Admin endpoints MUST come before /{dataset_id} ───────────

@router.get("/admin/unit/{unit_uuid}")
async def admin_unit_datasets(unit_uuid: str, current_user=Depends(require_admin)):
    supabase = get_supabase_client()
    unit     = supabase.table("compressor_units").select("*").eq("id", unit_uuid).execute()
    if not unit.data:
        raise HTTPException(404, "Unit not found.")

    datasets_res = supabase.table("datasets").select("*") \
        .eq("unit_id", unit_uuid).order("created_at", desc=True).execute()

    user_ids  = list({d["user_id"] for d in datasets_res.data if d.get("user_id")})
    users_map = {}
    if user_ids:
        ur = supabase.table("users").select("id,full_name,email").in_("id", user_ids).execute()
        users_map = {u["id"]: u for u in ur.data}

    enriched = []
    for ds in datasets_res.data:
        ds["uploader"] = users_map.get(ds["user_id"], {})
        enriched.append(ds)

    ct = supabase.table("compressor_types").select("name,manufacturer") \
        .eq("id", unit.data[0]["compressor_type_id"]).execute()
    return {
        "unit":     unit.data[0],
        "type":     ct.data[0] if ct.data else {},
        "datasets": enriched,
        "total":    len(enriched),
    }


@router.get("/admin/type/{type_id}")
async def admin_type_datasets(type_id: str, current_user=Depends(require_admin)):
    supabase = get_supabase_client()
    ct       = supabase.table("compressor_types").select("*").eq("id", type_id).execute()
    if not ct.data:
        raise HTTPException(404, "Compressor type not found.")

    units    = supabase.table("compressor_units") \
        .select("id,unit_id,location") \
        .eq("compressor_type_id", type_id).execute()
    unit_ids = [u["id"] for u in units.data]

    all_datasets = []
    if unit_ids:
        ds_res       = supabase.table("datasets").select("*") \
            .in_("unit_id", unit_ids).order("created_at", desc=True).execute()
        all_datasets = ds_res.data

    user_ids  = list({d["user_id"] for d in all_datasets if d.get("user_id")})
    users_map = {}
    if user_ids:
        ur = supabase.table("users").select("id,full_name,email").in_("id", user_ids).execute()
        users_map = {u["id"]: u for u in ur.data}

    ds_by_unit: dict[str, list] = {}
    for ds in all_datasets:
        ds["uploader"] = users_map.get(ds["user_id"], {})
        ds_by_unit.setdefault(ds["unit_id"], []).append(ds)

    result_units = []
    for u in units.data:
        u["datasets"]      = ds_by_unit.get(u["id"], [])
        u["dataset_count"] = len(u["datasets"])
        result_units.append(u)

    return {
        "type":           ct.data[0],
        "units":          result_units,
        "total_datasets": len(all_datasets),
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    ds       = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset  = ds.data[0]
    if current_user.get("role") == "engineer" and dataset["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")
    return dataset


# ══════════════════════════════════════════════════════════════
# DOWNLOAD — signed URLs (valid 1 hour)
# ══════════════════════════════════════════════════════════════

@router.get("/{dataset_id}/download/raw")
async def download_raw(dataset_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    ds       = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset  = ds.data[0]
    if current_user.get("role") == "engineer" and dataset["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")
    if not dataset.get("raw_file_path"):
        raise HTTPException(404, "Raw file not available.")
    url = get_dataset_download_url(dataset["raw_file_path"], expires_in=3600)
    return {"url": url, "filename": dataset["original_filename"], "expires_in_seconds": 3600}


@router.get("/{dataset_id}/download/processed")
async def download_processed(dataset_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    ds       = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset  = ds.data[0]
    if current_user.get("role") == "engineer" and dataset["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")
    if not dataset.get("processed_file_path"):
        raise HTTPException(404, "Processed file not ready yet.")
    base        = dataset["original_filename"].rsplit(".", 1)[0]
    dl_filename = f"{base}_processed.xlsx"
    url         = get_dataset_download_url(dataset["processed_file_path"], expires_in=3600)
    return {"url": url, "filename": dl_filename, "expires_in_seconds": 3600}


@router.get("/{dataset_id}/download/{file_type}")
async def download_by_type(dataset_id: str, file_type: str, current_user=Depends(get_current_user)):
    """Generic download: file_type='raw'|'csv' → raw file, 'processed'|'excel'|'xlsx' → processed."""
    supabase = get_supabase_client()
    ds       = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset  = ds.data[0]
    if current_user.get("role") == "engineer" and dataset["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")

    if file_type in ("processed", "excel", "xlsx"):
        if not dataset.get("processed_file_path"):
            raise HTTPException(404, "Processed file not ready yet.")
        base        = dataset["original_filename"].rsplit(".", 1)[0]
        dl_filename = f"{base}_processed.xlsx"
        url         = get_dataset_download_url(dataset["processed_file_path"], expires_in=3600)
    else:
        if not dataset.get("raw_file_path"):
            raise HTTPException(404, "Raw file not available.")
        dl_filename = dataset["original_filename"]
        url         = get_dataset_download_url(dataset["raw_file_path"], expires_in=3600)

    return {"url": url, "filename": dl_filename, "expires_in_seconds": 3600}


# ══════════════════════════════════════════════════════════════
# DELETE
# ══════════════════════════════════════════════════════════════

@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    ds       = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset  = ds.data[0]
    if current_user.get("role") == "engineer" and dataset["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")

    if dataset.get("raw_file_path"):
        delete_dataset_file(dataset["raw_file_path"])
    if dataset.get("processed_file_path"):
        delete_dataset_file(dataset["processed_file_path"])

    supabase.table("analysis_results").delete().eq("dataset_id", dataset_id).execute()
    supabase.table("datasets").delete().eq("id", dataset_id).execute()
    logger.info(f"Dataset {dataset_id} deleted by user {current_user.get('sub')}")
    return {"message": "Dataset deleted successfully."}


# ══════════════════════════════════════════════════════════════
# AUTO-RETRAIN HELPER
# ══════════════════════════════════════════════════════════════

def _check_auto_retrain(supabase, compressor_type_id: str, new_dataset_id: str) -> bool:
    ml = supabase.table("ml_models") \
        .select("id,auto_retrain,retrain_threshold") \
        .eq("compressor_type_id", compressor_type_id) \
        .eq("is_active", True) \
        .order("trained_at", desc=True).limit(1).execute()
    if not ml.data:
        return False

    model = ml.data[0]
    if not model.get("auto_retrain", True):
        return False

    threshold = model.get("retrain_threshold", 100)

    units    = supabase.table("compressor_units").select("id") \
        .eq("compressor_type_id", compressor_type_id).execute()
    unit_ids = [u["id"] for u in units.data]
    if not unit_ids:
        return False

    pending        = supabase.table("datasets").select("clean_rows") \
        .in_("unit_id", unit_ids).eq("contributed_to_model", False).execute()
    total_new_rows = sum(d.get("clean_rows") or 0 for d in pending.data)

    if total_new_rows >= threshold:
        try:
            from routers.retrain import trigger_retrain_task
            trigger_retrain_task(compressor_type_id, triggered_by="auto")
            logger.info(
                f"Auto-retrain triggered for type {compressor_type_id} "
                f"({total_new_rows} new rows >= threshold {threshold})"
            )
            return True
        except Exception as e:
            logger.error(f"Auto-retrain trigger failed: {e}")
            return False
    return False
