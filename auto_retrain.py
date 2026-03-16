"""
Auto-retraining service — CompressorAI v5
Runs as a separate process to periodically retrain models.

Production changes vs original:
  - Uses APScheduler (production-grade) instead of 'schedule' (toy library)
  - retrain_compressor() now actually calls trigger_retrain_task()
    (original only logged — never trained!)
  - Graceful shutdown on SIGTERM/SIGINT
  - Structured logging with rotation
  - Configurable schedule via env vars

Run: python auto_retrain.py
"""
import os
import sys
import signal
import logging
import logging.handlers
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from config import get_supabase_client

# ── Logging ───────────────────────────────────────────────────
log_handler = logging.handlers.RotatingFileHandler(
    "auto_retrain.log", maxBytes=5 * 1024 * 1024, backupCount=3
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[log_handler, logging.StreamHandler()],
)
logger = logging.getLogger("auto_retrain")

# ── Config ────────────────────────────────────────────────────
RETRAIN_HOUR     = int(os.getenv("RETRAIN_HOUR",     "2"))    # 2 AM daily
RETRAIN_INTERVAL = int(os.getenv("RETRAIN_INTERVAL", "24"))   # also every N hours


def run_retraining():
    """Run retraining check for all active compressor types."""
    logger.info("=" * 60)
    logger.info(f"Retraining cycle started at {datetime.now(timezone.utc).isoformat()}")

    try:
        from routers.retrain import trigger_retrain_task

        supabase = get_supabase_client()

        # Get all active compressor types that have a model with auto_retrain=True
        models = supabase.table("ml_models") \
            .select("compressor_type_id,retrain_threshold,auto_retrain") \
            .eq("is_active", True) \
            .eq("auto_retrain", True) \
            .execute()

        if not models.data:
            logger.info("No active auto-retrain models found.")
            return

        # Deduplicate by type
        processed_types = set()
        for model in models.data:
            type_id   = model["compressor_type_id"]
            threshold = model.get("retrain_threshold", 100)

            if type_id in processed_types:
                continue
            processed_types.add(type_id)

            try:
                # Check how many uncontributed rows exist for this type
                units    = supabase.table("compressor_units").select("id") \
                    .eq("compressor_type_id", type_id).execute()
                unit_ids = [u["id"] for u in units.data]
                if not unit_ids:
                    continue

                pending = supabase.table("datasets") \
                    .select("clean_rows") \
                    .in_("unit_id", unit_ids) \
                    .eq("contributed_to_model", False) \
                    .eq("is_processed", True).execute()
                pending_rows = sum(d.get("clean_rows") or 0 for d in pending.data)

                logger.info(
                    f"  type_id={type_id}: {pending_rows} pending rows "
                    f"(threshold={threshold})"
                )

                if pending_rows >= threshold:
                    logger.info(f"  → Triggering retrain for type_id={type_id}")
                    trigger_retrain_task(type_id, triggered_by="auto_scheduler")
                else:
                    logger.info(f"  → Skipping (below threshold)")

            except Exception as e:
                logger.error(f"  Error checking type_id={type_id}: {e}")

    except Exception as e:
        logger.error(f"Retraining cycle failed: {e}", exc_info=True)

    logger.info("Retraining cycle complete.")
    logger.info("=" * 60)


# ── Graceful shutdown ─────────────────────────────────────────
scheduler: BlockingScheduler = None


def _shutdown(signum, frame):
    logger.info("Shutdown signal received — stopping scheduler.")
    if scheduler:
        scheduler.shutdown(wait=False)
    sys.exit(0)


signal.signal(signal.SIGTERM, _shutdown)
signal.signal(signal.SIGINT,  _shutdown)


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("Auto-retrain service starting…")
    logger.info(f"  Daily run at: {RETRAIN_HOUR:02d}:00 UTC")
    logger.info(f"  Interval:     every {RETRAIN_INTERVAL} hours")

    scheduler = BlockingScheduler(timezone="UTC")

    # Daily at configured hour
    scheduler.add_job(
        run_retraining,
        CronTrigger(hour=RETRAIN_HOUR, minute=0),
        id="daily_retrain",
        name="Daily retraining",
        misfire_grace_time=3600,
    )

    # Also every N hours (catches high-upload periods)
    scheduler.add_job(
        run_retraining,
        IntervalTrigger(hours=RETRAIN_INTERVAL),
        id="interval_retrain",
        name=f"Every {RETRAIN_INTERVAL}h retraining",
        misfire_grace_time=3600,
    )

    # Run once immediately on start
    logger.info("Running initial check on startup…")
    run_retraining()

    logger.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Auto-retrain service stopped.")