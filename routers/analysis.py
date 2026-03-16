"""
Analysis Router — CompressorAI v5

Flow:
  1. Engineer selects an already-uploaded + processed dataset
  2. System loads the active ML model for that compressor type
  3. Runs full pipeline (DBSCAN → GBR → GA optimisation)
  4. Saves result to analysis_results table

Production fixes vs original:
  - FIXED: pretrained_model warm-start (was checking hasattr on a dict → always False)
  - FIXED: engine._save_model() writes to local disk — disabled in production
    (models are persisted only in Supabase Storage via retrain router)
  - Added pagination to history endpoint
  - Added dataset ownership double-check (engineer can't run on another's dataset)
  - user_params validated / clamped to safe ranges
  - numpy_clean handles nested None safely
"""
import io
import json
import math
import pickle
import logging

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, timezone

from config import get_supabase_client
from deps import get_current_user, require_admin
from storage import get_dataset_bytes, get_ml_model_bytes
from ml.engine import (
    CompressorMLEngine, validate_dataset,
    REQUIRED_COLUMNS, OPTIONAL_COLUMNS, enrich_dataframe,
)

router = APIRouter()
logger = logging.getLogger("compressorai.analysis")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def numpy_clean(obj):
    """Recursively convert numpy types → native Python for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: numpy_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_clean(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return numpy_clean(obj.tolist())
    return obj


# ── Safe param defaults + clamping ───────────────────────────
PARAM_DEFAULTS = {
    "voltage":               415.0,
    "power_factor":          0.9,
    "compression_stages":    2,
    "p_low":                 7.0,
    "p_high":                10.0,
    "q_low":                 45.23,
    "q_high":                35.47,
    "hours_per_day":         24.0,
    "cost_per_kwh":          0.0,
    "operating_days":        365,
    "total_lifetime_hours":  0.0,   # ← NEW: raw lifetime hours input
    "total_lifetime_days":   0,     # ← NEW: raw lifetime days input
}

PARAM_RANGES = {
    "voltage":               (100, 11000),
    "power_factor":          (0.5, 1.0),
    "compression_stages":    (1, 6),
    "p_low":                 (1.0, 50.0),
    "p_high":                (1.0, 50.0),
    "q_low":                 (1.0, 500.0),
    "q_high":                (1.0, 500.0),
    "hours_per_day":         (1.0, 24.0),
    "cost_per_kwh":          (0.0, 100000.0),
    "operating_days":        (1, 365),
    "total_lifetime_hours":  (0.0, 1_000_000.0),
    "total_lifetime_days":   (0, 36500),
}


def sanitize_user_params(raw: dict) -> dict:
    """Apply defaults and clamp to safe ranges.
    Passes cost params (hours_per_day, cost_per_kwh, operating_days) to engine.
    If total_lifetime_hours + total_lifetime_days are both provided and non-zero,
    derives hours_per_day from them (overrides any explicit hours_per_day).
    """
    params = {**PARAM_DEFAULTS}
    for key, (lo, hi) in PARAM_RANGES.items():
        if key in raw and raw[key] is not None and raw[key] != '':
            try:
                val = float(raw[key])
                params[key] = max(lo, min(hi, val))
            except (TypeError, ValueError):
                pass  # keep default

    # Ensure p_high > p_low
    if params["p_high"] <= params["p_low"]:
        params["p_high"] = params["p_low"] + 1.0

    # Derive hours_per_day from lifetime usage if both fields are filled
    lt_hours = params.get("total_lifetime_hours", 0)
    lt_days  = params.get("total_lifetime_days", 0)
    if lt_hours > 0 and lt_days > 0:
        derived = lt_hours / lt_days
        # Clamp to valid range (1–24)
        params["hours_per_day"] = max(1.0, min(24.0, derived))

    return params


class AnalysisParams(BaseModel):
    voltage:            Optional[float] = 415.0
    power_factor:       Optional[float] = 0.9
    compression_stages: Optional[int]   = 2
    p_low:              Optional[float] = 7.0
    p_high:             Optional[float] = 10.0
    q_low:              Optional[float] = 45.23
    q_high:             Optional[float] = 35.47


# ══════════════════════════════════════════════════════════════
# VALIDATE DATASET (pre-upload check — no DB writes)
# ══════════════════════════════════════════════════════════════

@router.post("/validate-dataset")
async def validate_upload(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    content = await file.read()
    fname   = file.filename or "dataset"
    try:
        if fname.lower().endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Cannot read file: {str(e)}")
    return validate_dataset(df)


# ══════════════════════════════════════════════════════════════
# RUN ANALYSIS
# ══════════════════════════════════════════════════════════════

@router.post("/run/{dataset_id}")
async def run_analysis_endpoint(
    dataset_id:  str,
    params:      str = Form("{}"),
    current_user=Depends(get_current_user),
):
    """
    Full ML pipeline on an already-uploaded + processed dataset.

    Steps:
      1. Load dataset record + verify ownership
      2. Load compressor unit + type
      3. Load active ML model for this type (if exists) → warm-start
      4. Download processed file from Supabase Storage
      5. Run CompressorMLEngine (DBSCAN → GBR → GA)
      6. Save result to analysis_results
    """
    supabase = get_supabase_client()
    user_id  = current_user["sub"]

    # ── Parse + sanitize params ──────────────────────────────
    try:
        raw_params = json.loads(params)
        if not isinstance(raw_params, dict):
            raw_params = {}
    except Exception:
        raw_params = {}
    user_params = sanitize_user_params(raw_params)

    # ── 1. Load dataset ──────────────────────────────────────
    ds = supabase.table("datasets").select("*").eq("id", dataset_id).execute()
    if not ds.data:
        raise HTTPException(404, "Dataset not found.")
    dataset = ds.data[0]

    if current_user.get("role") == "engineer" and dataset["user_id"] != user_id:
        raise HTTPException(403, "Access denied.")

    if not dataset.get("processed_file_path"):
        raise HTTPException(422, "Dataset not processed yet. Please re-upload.")

    # ── 2. Load compressor unit + type ───────────────────────
    unit_res = supabase.table("compressor_units") \
        .select("id,unit_id,compressor_type_id") \
        .eq("id", dataset["unit_id"]).execute()
    if not unit_res.data:
        raise HTTPException(404, "Compressor unit not found.")
    u = unit_res.data[0]

    # ── 3. Load active ML model (warm-start) ─────────────────
    ml = supabase.table("ml_models").select("*") \
        .eq("compressor_type_id", u["compressor_type_id"]) \
        .eq("is_active", True) \
        .order("trained_at", desc=True).limit(1).execute()

    pretrained_data = None   # will be a dict (from pickle.loads)
    ml_model_id     = None
    if ml.data and ml.data[0].get("model_path"):
        try:
            model_bytes     = get_ml_model_bytes(ml.data[0]["model_path"])
            pretrained_data = pickle.loads(model_bytes)  # this is a dict, NOT an engine instance
            ml_model_id     = ml.data[0]["id"]
            logger.info(f"Loaded pretrained model {ml_model_id} for warm-start.")
        except Exception as e:
            logger.warning(f"Could not load pretrained model: {e} — will train fresh.")
            pretrained_data = None

    # ── 4. Download processed dataset ────────────────────────
    try:
        file_bytes = get_dataset_bytes(dataset["processed_file_path"])
        df         = pd.read_excel(io.BytesIO(file_bytes))
    except Exception as e:
        raise HTTPException(500, f"Could not load processed dataset: {str(e)}")

    if len(df) < 20:
        raise HTTPException(400, "Dataset too small — need at least 20 rows after cleaning.")

    # ── 5. Run ML pipeline ───────────────────────────────────
    try:
        engine = CompressorMLEngine(dataset["unit_id"])

        # FIXED: pretrained_data is a dict (from _save_model / pickle.dumps)
        # Use isinstance check — not hasattr — because it's a plain dict
        if (
            pretrained_data is not None
            and isinstance(pretrained_data, dict)
            and all(k in pretrained_data for k in ("model_elec", "model_mech", "model_spc", "scaler"))
        ):
            engine.model_elec = pretrained_data["model_elec"]
            engine.model_mech = pretrained_data["model_mech"]
            engine.model_spc  = pretrained_data["model_spc"]
            engine.scaler     = pretrained_data["scaler"]
            logger.info("Warm-start: pre-trained sub-models loaded into engine.")

        results = engine.train(df, user_params)

    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Analysis pipeline failed")
        raise HTTPException(500, f"Analysis failed: {str(e)}")

    # ── 6. Save result ───────────────────────────────────────
    try:
        r = numpy_clean(results)
        # Normalise score keys: engine returns 'convergence', DB stores 'ga_convergence'
        scores = r.get("scores", {})
        if "convergence" in scores and "ga_convergence" not in scores:
            scores["ga_convergence"] = scores.pop("convergence")

        record = {
            "dataset_id":                dataset_id,
            "unit_id":                   dataset["unit_id"],
            "user_id":                   user_id,
            "ml_model_id":               ml_model_id,
            "scores":                    scores,
            "optimal_parameters":        r.get("optimal_parameters"),
            "best_electrical_power":     r.get("best_electrical_power"),
            "best_mechanical_power":     r.get("best_mechanical_power"),
            "best_spc":                  r.get("best_spc"),
            "baseline_electrical_power": r.get("baseline_electrical_power"),
            "power_saving_percent":      r.get("power_saving_percent"),
            "kw_saved":                  r.get("kw_saved"),
            "energy_saved_kwh":          r.get("energy_saved_kwh"),
            "cost_saved_annual":         r.get("cost_saved_annual"),
            "cost_saved_monthly":        r.get("cost_saved_monthly"),
            "user_params":               user_params,
            "feature_importance":        r.get("feature_importance"),
            "scatter_data":              r.get("scatter_data"),
            "cluster_data":              r.get("cluster_data"),
            "histogram_data":            r.get("histogram_data"),
            "training_curve":            r.get("training_curve"),
            "cluster_stats":             r.get("cluster_stats"),
        }
        saved = supabase.table("analysis_results").insert(record).execute()
        results["result_id"] = saved.data[0]["id"] if saved.data else None
    except Exception as e:
        logger.error(f"Failed to save analysis result to DB: {e}")
        results["result_id"] = None  # result still returned to client

    results["unit_label"]            = u["unit_id"]
    results["used_pretrained_model"] = pretrained_data is not None
    return numpy_clean(results)


# ══════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════

@router.get("/history/{unit_uuid}")
async def get_analysis_history(
    unit_uuid:   str,
    limit:       int = 10,
    offset:      int = 0,
    current_user=Depends(get_current_user),
):
    """
    Analysis history for a unit.
    Engineers: own results only. Admins: all results.
    Supports pagination via limit/offset.
    """
    limit  = min(max(1, limit), 100)   # clamp 1-100
    offset = max(0, offset)

    supabase = get_supabase_client()

    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", current_user["sub"]).eq("unit_id", unit_uuid).execute()
        if not link.data:
            raise HTTPException(403, "You are not linked to this unit.")

    q = supabase.table("analysis_results").select(
        "id,created_at,power_saving_percent,best_electrical_power,"
        "best_mechanical_power,best_spc,scores,optimal_parameters,dataset_id"
    ).eq("unit_id", unit_uuid)

    if current_user.get("role") == "engineer":
        q = q.eq("user_id", current_user["sub"])

    res = q.order("created_at", desc=True).range(offset, offset + limit - 1).execute()
    return {
        "data":   res.data or [],
        "limit":  limit,
        "offset": offset,
        "count":  len(res.data or []),
    }


# ══════════════════════════════════════════════════════════════
# SINGLE RESULT
# ══════════════════════════════════════════════════════════════

@router.get("/result/{result_id}")
async def get_analysis_result(result_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    res      = supabase.table("analysis_results").select("*").eq("id", result_id).execute()
    if not res.data:
        raise HTTPException(404, "Result not found.")
    result = res.data[0]

    if current_user.get("role") == "engineer" and result["user_id"] != current_user["sub"]:
        raise HTTPException(403, "Access denied.")

    # Enrich with unit + type info
    unit = supabase.table("compressor_units") \
        .select("unit_id,compressor_type_id").eq("id", result["unit_id"]).execute()
    if unit.data:
        u  = unit.data[0]
        ct = supabase.table("compressor_types") \
            .select("name,manufacturer").eq("id", u["compressor_type_id"]).execute()
        result["unit_label"] = u["unit_id"]
        result["type_name"]  = ct.data[0]["name"] if ct.data else ""

    return result


# ══════════════════════════════════════════════════════════════
# ADMIN — ALL RESULTS FOR A UNIT
# ══════════════════════════════════════════════════════════════

@router.get("/admin/unit/{unit_uuid}")
async def admin_unit_results(unit_uuid: str, current_user=Depends(require_admin)):
    """Admin: all analysis results for a unit across all engineers."""
    supabase = get_supabase_client()

    unit = supabase.table("compressor_units").select("*").eq("id", unit_uuid).execute()
    if not unit.data:
        raise HTTPException(404, "Unit not found.")

    results = supabase.table("analysis_results").select("*") \
        .eq("unit_id", unit_uuid).order("created_at", desc=True).execute()

    # Bulk-fetch users to avoid N+1
    user_ids   = list({r["user_id"] for r in results.data if r.get("user_id")})
    users_res  = supabase.table("users").select("id,full_name,email") \
        .in_("id", user_ids).execute() if user_ids else type("R", (), {"data": []})()
    users_map  = {u["id"]: u for u in users_res.data}

    enriched = []
    for r in results.data:
        r["analyst"] = users_map.get(r["user_id"], {})
        enriched.append(r)

    return {
        "unit":    unit.data[0],
        "results": enriched,
        "total":   len(enriched),
    }