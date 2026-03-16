"""
Retrain Router — CompressorAI v5

Production changes vs original:
  - Threading lock prevents concurrent retrains for the same type (race condition fix)
  - In-progress placeholder cleaned up properly on any failure path
  - Logging upgraded to structured logger (not print)
  - train_model user_params not hardcoded — uses sensible defaults
  - Pagination on history endpoint
  - Status endpoint shows in-progress state more accurately
"""
import io
import pickle
import threading
import logging
import pandas as pd

from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime, timezone

from config import get_supabase_client
from deps import require_default_admin, require_admin
from storage import upload_ml_model, get_dataset_bytes, delete_ml_model_file
from ml.engine import train_model

router = APIRouter()
logger = logging.getLogger("compressorai.retrain")

# ── Prevent concurrent retrains per compressor type ──────────
_retrain_locks: dict[str, threading.Lock] = {}
_locks_guard   = threading.Lock()


def _get_lock(type_id: str) -> threading.Lock:
    with _locks_guard:
        if type_id not in _retrain_locks:
            _retrain_locks[type_id] = threading.Lock()
        return _retrain_locks[type_id]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ══════════════════════════════════════════════════════════════
# TRIGGER RETRAIN
# ══════════════════════════════════════════════════════════════

@router.post("/{type_id}")
async def trigger_retrain(type_id: str, current_user=Depends(require_default_admin)):
    """
    Default admin manually triggers retraining for a compressor type.
    Runs in a background thread — returns immediately.
    Returns 409 if a retrain is already in progress for this type.
    """
    supabase = get_supabase_client()
    ct       = supabase.table("compressor_types").select("id,name") \
        .eq("id", type_id).execute()
    if not ct.data:
        raise HTTPException(404, "Compressor type not found.")

    lock = _get_lock(type_id)
    if not lock.acquire(blocking=False):
        raise HTTPException(409, f"Retraining is already in progress for '{ct.data[0]['name']}'. Please wait.")

    thread = threading.Thread(
        target=_run_retrain_with_lock,
        args=(type_id, current_user["sub"], lock),
        daemon=True,
    )
    thread.start()

    return {
        "message":   f"Retraining started for '{ct.data[0]['name']}'. "
                     f"Poll GET /api/retrain/{type_id}/status for progress.",
        "type_id":   type_id,
        "type_name": ct.data[0]["name"],
    }


def trigger_retrain_task(compressor_type_id: str, triggered_by: str = "auto"):
    """
    Called internally by datasets.py auto-retrain logic.
    Silently skips if a retrain is already running.
    """
    lock = _get_lock(compressor_type_id)
    if not lock.acquire(blocking=False):
        logger.info(f"[RETRAIN] Auto-retrain skipped for {compressor_type_id} — already in progress.")
        return

    thread = threading.Thread(
        target=_run_retrain_with_lock,
        args=(compressor_type_id, triggered_by, lock),
        daemon=True,
    )
    thread.start()


def _run_retrain_with_lock(type_id: str, triggered_by: str, lock: threading.Lock):
    """Wrapper that releases lock after retrain finishes or fails."""
    try:
        _run_retrain(type_id, triggered_by)
    finally:
        try:
            lock.release()
        except RuntimeError:
            pass  # already released


# ══════════════════════════════════════════════════════════════
# BACKGROUND WORKER
# ══════════════════════════════════════════════════════════════

def _run_retrain(type_id: str, triggered_by: str):
    """
    Background retrain worker:
      1. Create in-progress placeholder
      2. Gather all processed datasets for all units of this type
      3. Download + combine into one DataFrame
      4. Train ML model via engine.train_model()
      5. Serialize + upload model.pkl to Supabase Storage
      6. Deactivate old model versions (+ delete old storage files)
      7. Activate new model with training stats
      8. Mark all contributing datasets as contributed_to_model=True
    """
    supabase        = get_supabase_client()
    placeholder_id  = None

    try:
        # ── 1. In-progress placeholder ────────────────────────
        placeholder = supabase.table("ml_models").insert({
            "compressor_type_id": type_id,
            "is_active":          False,   # False = training in progress
            "trained_by":         triggered_by if triggered_by != "auto" else None,
            "trained_at":         utc_now(),
        }).execute()
        placeholder_id = placeholder.data[0]["id"]
        logger.info(f"[RETRAIN] Started for type_id={type_id}, triggered_by={triggered_by}")

        # ── 2. All units for this type ─────────────────────────
        units    = supabase.table("compressor_units").select("id") \
            .eq("compressor_type_id", type_id).execute()
        unit_ids = [u["id"] for u in units.data]
        if not unit_ids:
            raise Exception("No compressor units found for this type.")

        # ── 3. All processed datasets ──────────────────────────
        datasets = supabase.table("datasets") \
            .select("id,processed_file_path,clean_rows") \
            .in_("unit_id", unit_ids) \
            .eq("is_processed", True).execute()
        if not datasets.data:
            raise Exception("No processed datasets available for training.")

        # ── 4. Download + combine ──────────────────────────────
        dfs              = []
        used_dataset_ids = []
        total_rows       = 0

        for ds in datasets.data:
            if not ds.get("processed_file_path"):
                continue
            try:
                file_bytes = get_dataset_bytes(ds["processed_file_path"])
                df         = pd.read_excel(io.BytesIO(file_bytes))
                dfs.append(df)
                used_dataset_ids.append(ds["id"])
                total_rows += len(df)
            except Exception as e:
                logger.warning(f"[RETRAIN] Skipping dataset {ds['id']}: {e}")

        if not dfs:
            raise Exception("Could not read any dataset files from storage.")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"[RETRAIN] Combined {len(dfs)} datasets → {total_rows} rows")

        # ── 5. Train ───────────────────────────────────────────
        train_result = train_model(combined_df, compressor_id=type_id)
        model_obj    = train_result["model"]
        scores       = train_result.get("scores", {})
        logger.info(f"[RETRAIN] Training complete. Scores: {scores}")

        # ── 6. Upload model.pkl ────────────────────────────────
        model_bytes = pickle.dumps(model_obj)
        model_path  = upload_ml_model(type_id, model_bytes)
        logger.info(f"[RETRAIN] Model uploaded to: {model_path}")

        # ── 7. Deactivate old models + delete old files ────────
        old_models = supabase.table("ml_models").select("id,model_path") \
            .eq("compressor_type_id", type_id) \
            .eq("is_active", True).execute()
        for old in old_models.data:
            if old.get("model_path"):
                delete_ml_model_file(old["model_path"])
            supabase.table("ml_models").update({"is_active": False}) \
                .eq("id", old["id"]).execute()

        # ── 8. Activate new model ──────────────────────────────
        # Normalise score keys
        ga_convergence = scores.get("ga_convergence") or scores.get("convergence")
        supabase.table("ml_models").update({
            "model_path":       model_path,
            "is_active":        True,
            "trained_on_rows":  total_rows,
            "trained_on_units": len(unit_ids),
            "silhouette_score": scores.get("silhouette"),
            "r2_score":         scores.get("r2"),
            "f1_score":         scores.get("f1"),
            "ga_convergence":   ga_convergence,
            "trained_at":       utc_now(),
        }).eq("id", placeholder_id).execute()

        # ── 9. Mark datasets as contributed ───────────────────
        if used_dataset_ids:
            supabase.table("datasets") \
                .update({"contributed_to_model": True}) \
                .in_("id", used_dataset_ids).execute()

        logger.info(
            f"[RETRAIN] Completed. type_id={type_id}, "
            f"rows={total_rows}, units={len(unit_ids)}, datasets={len(used_dataset_ids)}"
        )

    except Exception as e:
        logger.error(f"[RETRAIN ERROR] type_id={type_id}: {e}", exc_info=True)
        # Clean up placeholder on any failure
        if placeholder_id:
            try:
                supabase.table("ml_models").delete() \
                    .eq("id", placeholder_id).execute()
            except Exception as cleanup_err:
                logger.error(f"[RETRAIN] Placeholder cleanup failed: {cleanup_err}")


# ══════════════════════════════════════════════════════════════
# STATUS + HISTORY
# ══════════════════════════════════════════════════════════════

@router.get("/{type_id}/status")
async def retrain_status(type_id: str, current_user=Depends(require_admin)):
    """Current ML model status for a compressor type."""
    supabase = get_supabase_client()
    ct       = supabase.table("compressor_types").select("id,name").eq("id", type_id).execute()
    if not ct.data:
        raise HTTPException(404, "Compressor type not found.")

    active = supabase.table("ml_models").select("*") \
        .eq("compressor_type_id", type_id).eq("is_active", True) \
        .order("trained_at", desc=True).limit(1).execute()

    # is_active=False + model_path=NULL → training in progress
    in_progress = supabase.table("ml_models").select("id,trained_at") \
        .eq("compressor_type_id", type_id).eq("is_active", False) \
        .is_("model_path", "null") \
        .order("trained_at", desc=True).limit(1).execute()

    # Check live lock state
    lock           = _get_lock(type_id)
    lock_active    = not lock.acquire(blocking=False)
    if not lock_active:
        lock.release()

    # Pending uncontributed rows
    units    = supabase.table("compressor_units").select("id") \
        .eq("compressor_type_id", type_id).execute()
    unit_ids = [u["id"] for u in units.data]
    pending_rows = 0
    if unit_ids:
        pending      = supabase.table("datasets").select("clean_rows") \
            .in_("unit_id", unit_ids).eq("contributed_to_model", False).execute()
        pending_rows = sum(d.get("clean_rows") or 0 for d in pending.data)

    return {
        "type_id":                type_id,
        "type_name":              ct.data[0]["name"],
        "active_model":           active.data[0] if active.data else None,
        "training_in_progress":   lock_active or bool(in_progress.data),
        "pending_new_rows":       pending_rows,
        "auto_retrain_threshold": (
            active.data[0].get("retrain_threshold", 100) if active.data else 100
        ),
    }


@router.get("/{type_id}/history")
async def retrain_history(
    type_id:      str,
    current_user=Depends(require_admin),
    limit:        int = Query(20, ge=1, le=100),
    offset:       int = Query(0,  ge=0),
):
    """All model versions for a compressor type (newest first), paginated."""
    supabase = get_supabase_client()
    models   = supabase.table("ml_models").select("*") \
        .eq("compressor_type_id", type_id) \
        .order("trained_at", desc=True) \
        .range(offset, offset + limit - 1).execute()

    # Bulk-fetch trainers
    trainer_ids = list({
        m["trained_by"] for m in models.data
        if m.get("trained_by") and m["trained_by"] != "auto"
    })
    trainers_map = {}
    if trainer_ids:
        tr_res = supabase.table("users").select("id,full_name,email") \
            .in_("id", trainer_ids).execute()
        trainers_map = {u["id"]: u for u in tr_res.data}

    result = []
    for m in models.data:
        trainer_id = m.get("trained_by")
        if trainer_id and trainer_id != "auto":
            m["trainer"] = trainers_map.get(trainer_id, {"full_name": "Unknown"})
        else:
            m["trainer"] = {"full_name": "Auto-retrain", "email": "system"}
        result.append(m)
    return result


@router.put("/{type_id}/config")
async def update_retrain_config(
    type_id:           str,
    auto_retrain:      bool = True,
    retrain_threshold: int  = 100,
    current_user=Depends(require_default_admin),
):
    """Default admin: update auto-retrain settings."""
    if retrain_threshold < 10:
        raise HTTPException(400, "Retrain threshold must be at least 10 rows.")

    supabase = get_supabase_client()
    active   = supabase.table("ml_models").select("id") \
        .eq("compressor_type_id", type_id).eq("is_active", True) \
        .order("trained_at", desc=True).limit(1).execute()
    if not active.data:
        raise HTTPException(404, "No active model found. Trigger a retrain first.")

    supabase.table("ml_models").update({
        "auto_retrain":      auto_retrain,
        "retrain_threshold": retrain_threshold,
    }).eq("id", active.data[0]["id"]).execute()

    return {
        "message":           "Retrain config updated.",
        "auto_retrain":      auto_retrain,
        "retrain_threshold": retrain_threshold,
    }