"""
storage.py — Supabase Storage helpers for CompressorAI v5
Handles upload/download of datasets, ML model files, and generated reports.
"""
import io
from datetime import datetime, timezone
from config import supabase_admin, BUCKET_DATASETS, BUCKET_ML_MODELS

BUCKET_REPORTS = "reports"


def utc_now():
    return datetime.now(timezone.utc)


# ── File path helpers ─────────────────────────────────────────
# Pattern: datasets/{unit_id}/{user_id}/{timestamp}_{type}_{filename}
def _dataset_path(unit_id: str, user_id: str, filename: str, file_type: str) -> str:
    ts        = utc_now().strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace(" ", "_")
    return f"{unit_id}/{user_id}/{ts}_{file_type}_{safe_name}"


# Pattern: ml-models/{compressor_type_id}/{timestamp}_model.pkl
def _model_path(compressor_type_id: str) -> str:
    ts = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"{compressor_type_id}/{ts}_model.pkl"


# ═══════════════════════════════════════════════════════════════
# DATASET STORAGE
# ═══════════════════════════════════════════════════════════════

def upload_raw_dataset(unit_id: str, user_id: str,
                       file_bytes: bytes, filename: str) -> str:
    """Upload original (raw) dataset file. Returns storage path."""
    path = _dataset_path(unit_id, user_id, filename, "raw")
    supabase_admin.storage.from_(BUCKET_DATASETS).upload(
        path=path,
        file=file_bytes,
        file_options={
            "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "upsert": "true"
        }
    )
    return path


def upload_processed_dataset(unit_id: str, user_id: str,
                              df_bytes: bytes, original_filename: str) -> str:
    """Upload cleaned/processed dataset as xlsx. Returns storage path."""
    base      = original_filename.rsplit(".", 1)[0]
    proc_name = f"{base}_processed.xlsx"
    path      = _dataset_path(unit_id, user_id, proc_name, "processed")
    supabase_admin.storage.from_(BUCKET_DATASETS).upload(
        path=path,
        file=df_bytes,
        file_options={
            "content-type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "upsert": "true"
        }
    )
    return path


def get_dataset_download_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Generate a signed URL for dataset download.
    expires_in: seconds (default 1 hour).
    Handles both old and new Supabase Python SDK response shapes.
    """
    res = supabase_admin.storage.from_(BUCKET_DATASETS).create_signed_url(
        path=storage_path,
        expires_in=expires_in
    )
    # SDK v1 returns dict, SDK v2 may return object
    if isinstance(res, dict):
        return res.get("signedURL") or res.get("signed_url") or ""
    return getattr(res, "signed_url", "") or getattr(res, "signedURL", "") or ""


def get_dataset_bytes(storage_path: str) -> bytes:
    """Download dataset file bytes from storage (used by analysis + retrain)."""
    return supabase_admin.storage.from_(BUCKET_DATASETS).download(storage_path)


def delete_dataset_file(storage_path: str) -> None:
    """Delete a dataset file from storage. Non-critical — errors are swallowed."""
    try:
        supabase_admin.storage.from_(BUCKET_DATASETS).remove([storage_path])
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# ML MODEL STORAGE
# ═══════════════════════════════════════════════════════════════

def upload_ml_model(compressor_type_id: str, model_bytes: bytes) -> str:
    """Upload trained model.pkl to storage. Returns storage path."""
    path = _model_path(compressor_type_id)
    supabase_admin.storage.from_(BUCKET_ML_MODELS).upload(
        path=path,
        file=model_bytes,
        file_options={
            "content-type": "application/octet-stream",
            "upsert": "true"
        }
    )
    return path


def get_ml_model_bytes(storage_path: str) -> bytes:
    """Download model.pkl bytes from storage."""
    return supabase_admin.storage.from_(BUCKET_ML_MODELS).download(storage_path)


def delete_ml_model_file(storage_path: str) -> None:
    """Delete old model file (called after successful retrain). Non-critical."""
    try:
        supabase_admin.storage.from_(BUCKET_ML_MODELS).remove([storage_path])
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
# REPORTS STORAGE
# ═══════════════════════════════════════════════════════════════

def upload_report_file(path: str, content: bytes,
                       content_type: str = "application/octet-stream") -> None:
    """
    Upload a generated report (PDF or Excel) to the 'reports' bucket.

    Args:
        path:         Storage path, e.g. "reports/{user_id}/{filename}"
        content:      Raw file bytes
        content_type: MIME type — "application/pdf" or
                      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    """
    try:
        supabase_admin.storage.from_(BUCKET_REPORTS).upload(
            path=path,
            file=content,
            file_options={
                "content-type": content_type,
                "upsert": "true"
            }
        )
    except Exception as e:
        # Non-fatal: log but don't crash the download response
        import logging
        logging.getLogger("compressorai.storage").warning(
            f"Report upload to storage failed (path={path}): {e}"
        )


def get_report_download_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Generate a signed URL for a stored report.
    expires_in: seconds (default 1 hour).
    """
    res = supabase_admin.storage.from_(BUCKET_REPORTS).create_signed_url(
        path=storage_path,
        expires_in=expires_in
    )
    if isinstance(res, dict):
        return res.get("signedURL") or res.get("signed_url") or ""
    return getattr(res, "signed_url", "") or getattr(res, "signedURL", "") or ""


def delete_report_file(storage_path: str) -> None:
    """Delete a report file from storage. Non-critical."""
    try:
        supabase_admin.storage.from_(BUCKET_REPORTS).remove([storage_path])
    except Exception:
        pass