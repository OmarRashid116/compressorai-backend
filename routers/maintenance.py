"""
Maintenance Router — CompressorAI v6
PM Plan Compliance & Work Order Analysis

Base logic: Final_Code2.ipynb (verbatim)
Improvements over notebook:
  1. Deduplication — same task + same date = 1 physical event (not multiple SAP WOs)
  2. Interval calc  — notebook formula: [first_event_hrs] + consecutive diffs
  3. Auto start_date — min(WO dates) if user doesn't provide one
  4. Validation     — catches wrong start_date (all hrs = 0 error)
  5. Raw WO count   — shows both raw matched + deduped count in results
  6. Cost accuracy  — sums cost across all WOs for same physical event
"""

import io
import math
import logging
import re
from datetime import datetime, timezone, date
from typing import Optional, List

import pandas as pd
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form

from config import get_supabase_client
from deps import get_current_user

router = APIRouter()
logger = logging.getLogger("compressorai.maintenance")


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_file(content: bytes, filename: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(content)) \
             if (filename or "").lower().endswith(".csv") \
             else pd.read_excel(io.BytesIO(content))
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        raise HTTPException(400, f"Cannot read '{filename}': {e}")


# ── Column detection (notebook detect_columns — verbatim) ─────
def _detect_wo_columns(df: pd.DataFrame) -> dict:
    desc_col = date_col = cost_col = None

    for col in df.columns:
        c = col.lower()
        if any(x in c for x in ['description', 'task', 'job', 'work']):
            desc_col = col
        if "actual" in c and ("date" in c or "start" in c):
            date_col = col

    if not date_col:
        for col in df.columns:
            if any(x in col.lower() for x in ['date', 'start']):
                date_col = col
                break

    # Cost: actual cost preferred
    for col in df.columns:
        c = col.lower()
        if "cost" in c and ("act" in c or "actual" in c):
            cost_col = col
            break

    return {"desc": desc_col, "date": date_col, "cost": cost_col}


# ── PM Plan parsing (notebook process_pm_plan — verbatim) ─────
def _parse_pm_plan(df: pd.DataFrame) -> list:
    df = df.iloc[:, 0:3].copy()
    df.columns = ['Machine', 'Task', 'Frequency']
    df['Task'] = df['Task'].astype(str).str.lower().str.strip()

    tasks = []
    for _, row in df.iterrows():
        task_raw = row['Task']
        freq_raw = str(row['Frequency']).lower()
        if not task_raw or task_raw in ('nan', 'none', ''):
            continue
        if "hour" in freq_raw or "hr" in freq_raw:
            num = ''.join(ch for ch in freq_raw if ch.isdigit())
            if num:
                tasks.append({
                    "task":           task_raw,
                    "task_display":   task_raw,
                    "interval_hours": int(num),
                    "machine":        str(row['Machine']).strip(),
                })
    return tasks


# ── Running hours (notebook calculate_running_hours — verbatim)
def _assign_running_hours(
    df: pd.DataFrame,
    date_col: str,
    yearly_hours: list,
    start_date: date,
) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    start_ts  = pd.to_datetime(start_date)
    days_list = (df[date_col] - start_ts).dt.days

    running_hours = []
    for d in days_list:
        total = 0.0
        remaining = float(d)
        for yr_hours in yearly_hours:
            daily_rate = yr_hours / 365.0
            use_days   = min(365.0, remaining)
            total     += use_days * daily_rate
            remaining -= use_days
            if remaining <= 0:
                break
        running_hours.append(total)

    df["Running_Hours"] = running_hours
    return df


# ── Action detection (notebook detect_action — verbatim) ──────
def _detect_action(text: str) -> str:
    t = text.lower()
    if any(x in t for x in ["replace", "replacement", "changed"]):
        return "replacement"
    if any(x in t for x in ["clean", "cleaning"]):
        return "cleaning"
    if any(x in t for x in ["inspect", "inspection", "check"]):
        return "inspection"
    return "unknown"


# ── NLP matching (notebook match_pm_tasks — verbatim) ─────────
def _match_pm_tasks(
    wo_df: pd.DataFrame,
    desc_col: str,
    pm_tasks: list,
) -> pd.DataFrame:
    """
    Notebook thresholds: score > 0.45, keyword_ratio > 0.5, wo_action == pm_action.
    unknown != unknown → reject (strict, same as notebook).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    task_list = [t["task"] for t in pm_tasks]
    if not task_list:
        wo_df = wo_df.copy()
        wo_df["PM_Task"]       = "Unmatched"
        wo_df["PM_Interval"]   = None
        wo_df["Match_Score"]   = 0.0
        wo_df["Keyword_Ratio"] = 0.0
        return wo_df

    wo_descs = wo_df[desc_col].astype(str).str.lower().fillna("").tolist()

    # Notebook stop words (verbatim)
    stop_words = [
        "of", "the", "and", "to", "for",
        "cleaning", "inspection", "replacement", "change", "check",
    ]

    def extract_keywords(task: str) -> list:
        return [w for w in task.split() if w not in stop_words and len(w) > 2]

    task_keywords = {task: extract_keywords(task) for task in task_list}

    try:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        pm_vectors = vectorizer.fit_transform(task_list)
        wo_vectors = vectorizer.transform(wo_descs)
        similarity = cosine_similarity(wo_vectors, pm_vectors)
    except Exception as e:
        logger.warning(f"TF-IDF failed: {e}")
        wo_df = wo_df.copy()
        wo_df["PM_Task"]       = "Unmatched"
        wo_df["PM_Interval"]   = None
        wo_df["Match_Score"]   = 0.0
        wo_df["Keyword_Ratio"] = 0.0
        return wo_df

    matches, intervals_out, scores, kw_ratios = [], [], [], []

    for i, sim_row in enumerate(similarity):
        idx            = int(sim_row.argmax())
        score          = float(sim_row[idx])
        predicted_task = task_list[idx]
        text           = wo_descs[i]

        keywords  = task_keywords[predicted_task]
        kw_hits   = sum(1 for w in keywords if w in text)
        kw_ratio  = kw_hits / len(keywords) if keywords else 0.0

        wo_action = _detect_action(text)
        pm_action = _detect_action(predicted_task)

        # Notebook condition (verbatim): score > 0.45, kw > 0.5, actions equal
        if score > 0.45 and kw_ratio > 0.5 and wo_action == pm_action:
            matches.append(pm_tasks[idx]["task_display"])
            intervals_out.append(pm_tasks[idx]["interval_hours"])
        else:
            matches.append("Unmatched")
            intervals_out.append(None)

        scores.append(round(score, 4))
        kw_ratios.append(round(kw_ratio, 2))

    wo_df = wo_df.copy()
    wo_df["PM_Task"]       = matches
    wo_df["PM_Interval"]   = intervals_out
    wo_df["Match_Score"]   = scores
    wo_df["Keyword_Ratio"] = kw_ratios

    return wo_df[wo_df["PM_Task"] != "Unmatched"].copy()


# ── Compliance engine (notebook + improvements) ───────────────
def _calculate_compliance(
    matched_df:  pd.DataFrame,
    pm_tasks:    list,
    total_hours: float,
    cost_col:    Optional[str],
) -> list:
    """
    Notebook logic (verbatim):
      expected  = floor(total_hours / target)
      intervals = [first_event_hrs] + consecutive_diffs
      recency   = total_hours - max(Running_Hours)
      tolerance = 2.5%

    Improvement over notebook:
      Dedup same task + same date before computing intervals/compliance
      (multiple SAP WOs on same day = 1 physical maintenance event)
      Cost is summed across all WOs for same physical event.
    """
    TOLERANCE = 0.025
    results   = []

    for task_info in pm_tasks:
        task_name = task_info["task_display"]
        target    = float(task_info["interval_hours"])

        subset_raw = matched_df[matched_df["PM_Task"] == task_name].copy()
        raw_count  = len(subset_raw)

        # ── IMPROVEMENT: dedup same task + same date ──────────
        # Multiple SAP WOs on same day for same task = 1 physical job
        # Sum costs before dedup so we don't lose money data
        if cost_col and cost_col in subset_raw.columns:
            cost_by_date = (
                subset_raw
                .assign(_cost=pd.to_numeric(subset_raw[cost_col], errors="coerce").fillna(0))
                .groupby("date_str")["_cost"]
                .sum()
                .reset_index()
                .rename(columns={"_cost": "_summed_cost"})
            )
        else:
            cost_by_date = None

        subset = (
            subset_raw
            .drop_duplicates(subset=["date_str"])
            .sort_values("Running_Hours")
            .reset_index(drop=True)
        )
        dedup_count = len(subset)

        # Merge summed costs back
        if cost_by_date is not None:
            subset = subset.merge(cost_by_date, on="date_str", how="left")

        # ── Never performed ───────────────────────────────────
        if subset.empty:
            expected = math.floor(total_hours / target) if target > 0 else 0
            last_int = total_hours
            results.append({
                "task":              task_name,
                "interval_hours":    int(target),
                "expected_pm":       expected,
                "actual_pm":         0,
                "raw_wo_count":      raw_count,
                "compliance_pct":    0.0,
                "avg_interval":      None,
                "interval_ratio":    None,
                "interval_status":   "Never Performed",
                "last_interval":     round(last_int, 1),
                "recency_status":    "Overdue" if last_int > target
                                     else "Due Soon" if last_int > 0.8 * target
                                     else "OK",
                "delay_hours":       round(max(0, last_int - target), 1),
                "over_maintenance":  0,
                "under_maintenance": expected,
                "total_cost":        0.0,
                "avg_cost_per_pm":   0.0,
                "over_maint_cost":   0.0,
                "events":            [],
            })
            continue

        # ── Compliance ────────────────────────────────────────
        actual_events   = dedup_count
        expected_events = math.floor(total_hours / target) if target > 0 else 0
        compliance      = round((actual_events / expected_events * 100), 1) \
                          if expected_events > 0 else 0.0

        # ── Cost (summed per physical event) ──────────────────
        if cost_by_date is not None and "_summed_cost" in subset.columns:
            total_cost = float(subset["_summed_cost"].sum())
        elif cost_col and cost_col in subset.columns:
            total_cost = float(
                pd.to_numeric(subset[cost_col], errors="coerce").fillna(0).sum()
            )
        else:
            total_cost = 0.0

        over_maint      = max(0, actual_events - expected_events)
        avg_cost        = round(total_cost / actual_events, 1) if actual_events > 0 else 0.0
        over_maint_cost = round(over_maint * avg_cost, 1)

        # ── Interval (notebook formula verbatim) ──────────────
        # intervals = [first_event_hrs, diff_1_2, diff_2_3, ...]
        hrs_list  = subset["Running_Hours"].tolist()
        intervals = [hrs_list[0]] + \
                    [hrs_list[i] - hrs_list[i-1] for i in range(1, len(hrs_list))]
        avg_interval   = round(sum(intervals) / len(intervals), 1)
        interval_ratio = round(avg_interval / target, 2) if target > 0 else None

        lo, hi = 1 - TOLERANCE, 1 + TOLERANCE
        if interval_ratio is None:    interval_status = "Unknown"
        elif interval_ratio < lo:     interval_status = "Over"
        elif interval_ratio > hi:     interval_status = "Under"
        else:                         interval_status = "OK"

        # ── Recency (notebook formula verbatim) ───────────────
        last_interval  = round(total_hours - float(subset["Running_Hours"].max()), 1)
        delay_hours    = round(max(0.0, last_interval - target), 1)
        if last_interval > target:
            recency_status = "Overdue"
        elif last_interval > 0.8 * target:
            recency_status = "Due Soon"
        else:
            recency_status = "OK"

        # ── Events list ───────────────────────────────────────
        events = []
        for _, ev in subset.iterrows():
            ev_cost = float(ev.get("_summed_cost", 0) or 0) \
                      if "_summed_cost" in subset.columns \
                      else float(pd.to_numeric(ev.get(cost_col, 0), errors="coerce") or 0) \
                           if cost_col else 0.0
            events.append({
                "running_hours": round(float(ev["Running_Hours"]), 1),
                "date":          str(ev.get("date_str", "")),
                "description":   str(ev.get("Description", ""))[:120],
                "cost":          round(ev_cost, 2),
                "wo_count":      int(
                    subset_raw[subset_raw["date_str"] == ev.get("date_str", "")].shape[0]
                ),
            })

        results.append({
            "task":              task_name,
            "interval_hours":    int(target),
            "expected_pm":       expected_events,
            "actual_pm":         actual_events,
            "raw_wo_count":      raw_count,
            "compliance_pct":    compliance,
            "avg_interval":      avg_interval,
            "interval_ratio":    interval_ratio,
            "interval_status":   interval_status,
            "last_interval":     last_interval,
            "recency_status":    recency_status,
            "delay_hours":       delay_hours,
            "over_maintenance":  over_maint,
            "under_maintenance": max(0, expected_events - actual_events),
            "total_cost":        round(total_cost, 0),
            "avg_cost_per_pm":   avg_cost,
            "over_maint_cost":   over_maint_cost,
            "events":            events,
        })

    return results


def _build_summary(results: list, total_hours: float) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "total_tasks":          n,
        "tasks_performed":      sum(1 for r in results if r["actual_pm"] > 0),
        "tasks_never_done":     sum(1 for r in results if r["actual_pm"] == 0),
        "on_track_count":       sum(1 for r in results if r["interval_status"] == "OK"),
        "over_maint_count":     sum(1 for r in results if r["interval_status"] == "Over"),
        "under_maint_count":    sum(1 for r in results if r["interval_status"] == "Under"),
        "overdue_count":        sum(1 for r in results if r["recency_status"] == "Overdue"),
        "due_soon_count":       sum(1 for r in results if r["recency_status"] == "Due Soon"),
        "avg_compliance_pct":   round(sum(r["compliance_pct"] for r in results) / n, 1),
        "total_cost":           round(sum(r["total_cost"] for r in results), 0),
        "over_maint_cost":      round(sum(r["over_maint_cost"] for r in results), 0),
        "total_hours_analyzed": round(total_hours, 0),
    }


# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════

@router.post("/validate-workorder")
async def validate_workorder(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    content = await file.read()
    df      = _read_file(content, file.filename or "")
    cols    = _detect_wo_columns(df)
    issues  = []
    if not cols["desc"]: issues.append("No description column detected.")
    if not cols["date"]: issues.append("No date column detected.")

    auto_start = None
    if cols["date"]:
        try:
            dates      = pd.to_datetime(df[cols["date"]], errors="coerce").dropna()
            auto_start = str(dates.min().date()) if len(dates) > 0 else None
        except Exception:
            pass

    return {
        "valid":       len(issues) == 0,
        "issues":      issues,
        "rows":        len(df),
        "columns":     list(df.columns),
        "detected":    cols,
        "auto_start":  auto_start,
        "sample_desc": df[cols["desc"]].dropna().head(5).tolist() if cols["desc"] else [],
    }


@router.post("/upload-pm/{unit_id}")
async def upload_pm_plan(
    unit_id: str,
    file:    UploadFile = File(...),
    current_user=Depends(get_current_user),
):
    supabase = get_supabase_client()
    unit = supabase.table("compressor_units").select("id,unit_id").eq("id", unit_id).execute()
    if not unit.data:
        raise HTTPException(404, "Compressor unit not found.")
    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", current_user["sub"]).eq("unit_id", unit_id).execute()
        if not link.data:
            raise HTTPException(403, "Not linked to this unit.")

    content = await file.read()
    df      = _read_file(content, file.filename or "pm_plan.xlsx")
    tasks   = _parse_pm_plan(df)
    if not tasks:
        raise HTTPException(422,
            "No valid PM tasks found. Ensure columns: Machine, Task/Description, "
            "Frequency (e.g. '700 Hours').")

    machine_tag = tasks[0].get("machine", "") if tasks else ""
    supabase.table("pm_plans").update({"is_active": False}).eq("unit_id", unit_id).execute()
    res = supabase.table("pm_plans").insert({
        "unit_id":           unit_id,
        "user_id":           current_user["sub"],
        "original_filename": file.filename,
        "tasks":             tasks,
        "machine_tag":       machine_tag,
        "is_active":         True,
    }).execute()

    return {
        "message":     f"PM plan uploaded: {len(tasks)} tasks parsed.",
        "pm_plan_id":  res.data[0]["id"],
        "tasks":       tasks,
        "machine_tag": machine_tag,
    }


@router.get("/pm-plan/{unit_id}")
async def get_pm_plan(unit_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    res = supabase.table("pm_plans").select("*") \
        .eq("unit_id", unit_id).eq("is_active", True) \
        .order("created_at", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(404, "No active PM plan found.")
    return res.data[0]


@router.get("/pm-plans/{unit_id}")
async def list_pm_plans(unit_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    res = supabase.table("pm_plans") \
        .select("id,original_filename,machine_tag,is_active,created_at") \
        .eq("unit_id", unit_id).order("created_at", desc=True).execute()
    return res.data or []


@router.post("/analyze/{unit_id}")
async def run_analysis(
    unit_id:      str,
    wo_file:      UploadFile    = File(...),
    yearly_hours: str           = Form(...),
    start_date:   Optional[str] = Form(None),
    pm_plan_id:   Optional[str] = Form(None),
    current_user=Depends(get_current_user),
):
    """
    Full PM compliance analysis.

    yearly_hours : JSON list — one value per year.
                   e.g. "[1866,1866,1866]" for CIK (9 yrs → "[1866]*9")
                   e.g. "[2175,2175,2175]" for CSK
    start_date   : Compressor commissioning date (YYYY-MM-DD).
                   Leave blank → auto-set to earliest date in WO file.
    """
    import json as _json
    supabase = get_supabase_client()

    # ── Parse yearly_hours ────────────────────────────────────
    try:
        yearly_list = _json.loads(yearly_hours)
        yearly_list = [float(h) for h in yearly_list]
        assert len(yearly_list) > 0
    except Exception:
        raise HTTPException(400, "yearly_hours must be JSON list e.g. '[1866,1866]'")

    total_hours = sum(yearly_list)
    if total_hours <= 0:
        raise HTTPException(400, "Total running hours must be > 0.")

    # ── Verify unit access ────────────────────────────────────
    unit = supabase.table("compressor_units").select("id,unit_id").eq("id", unit_id).execute()
    if not unit.data:
        raise HTTPException(404, "Compressor unit not found.")
    if current_user.get("role") == "engineer":
        link = supabase.table("user_units").select("id") \
            .eq("user_id", current_user["sub"]).eq("unit_id", unit_id).execute()
        if not link.data:
            raise HTTPException(403, "Not linked to this unit.")

    # ── Load PM Plan ──────────────────────────────────────────
    if pm_plan_id:
        pm_rec = supabase.table("pm_plans").select("*").eq("id", pm_plan_id).execute()
    else:
        pm_rec = supabase.table("pm_plans").select("*") \
            .eq("unit_id", unit_id).eq("is_active", True) \
            .order("created_at", desc=True).limit(1).execute()
    if not pm_rec.data:
        raise HTTPException(404, "No PM plan found. Upload one first.")

    pm_tasks        = pm_rec.data[0]["tasks"]
    pm_plan_id_used = pm_rec.data[0]["id"]

    # ── Read Work Order file ──────────────────────────────────
    content = await wo_file.read()
    wo_df   = _read_file(content, wo_file.filename or "workorder.xlsx")
    cols    = _detect_wo_columns(wo_df)
    wo_rows = len(wo_df)

    if not cols["desc"]:
        raise HTTPException(422, "No description column found in work order file.")
    if not cols["date"]:
        raise HTTPException(422, "No date column found in work order file.")

    # ── Parse dates ───────────────────────────────────────────
    wo_df[cols["date"]] = pd.to_datetime(wo_df[cols["date"]], errors="coerce")
    wo_df = wo_df.dropna(subset=[cols["date"]]).reset_index(drop=True)

    # ── Auto start_date = min(WO dates) if not provided ───────
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date).date()
        except Exception:
            raise HTTPException(400, f"Invalid start_date '{start_date}'. Use YYYY-MM-DD.")
    else:
        start_dt = wo_df[cols["date"]].min().date()

    # ── All WOs — no order-type filter (notebook uses all) ────
    wo_df = _assign_running_hours(wo_df, cols["date"], yearly_list, start_dt)

    # ── Validate ──────────────────────────────────────────────
    if (wo_df["Running_Hours"] > 0).sum() == 0:
        min_date = wo_df[cols["date"]].min()
        raise HTTPException(422,
            f"All running hours = 0. start_date ({start_dt}) is after all WO dates "
            f"(earliest WO: {min_date.date() if pd.notna(min_date) else 'unknown'}). "
            f"Leave start_date blank for auto-detection."
        )

    wo_df["date_str"] = wo_df[cols["date"]].dt.strftime("%Y-%m-%d")

    # ── NLP match ─────────────────────────────────────────────
    wo_df[cols["desc"]] = wo_df[cols["desc"]].astype(str).str.lower()
    matched_df   = _match_pm_tasks(wo_df, cols["desc"], pm_tasks)
    matched_rows = len(matched_df)

    # ── Compliance ────────────────────────────────────────────
    results = _calculate_compliance(matched_df, pm_tasks, total_hours, cols["cost"])
    summary = _build_summary(results, total_hours)

    # ── Persist ───────────────────────────────────────────────
    analysis_id = None
    try:
        saved = supabase.table("maintenance_analyses").insert({
            "unit_id":      unit_id,
            "user_id":      current_user["sub"],
            "pm_plan_id":   pm_plan_id_used,
            "start_date":   str(start_dt),
            "yearly_hours": yearly_list,
            "total_hours":  total_hours,
            "wo_filename":  wo_file.filename,
            "wo_rows":      wo_rows,
            "matched_rows": matched_rows,
            "results":      results,
            "summary":      summary,
        }).execute()
        analysis_id = saved.data[0]["id"] if saved.data else None
    except Exception as e:
        logger.error(f"DB save failed: {e}")

    return {
        "analysis_id":    analysis_id,
        "unit_id":        unit_id,
        "unit_label":     unit.data[0]["unit_id"],
        "total_hours":    total_hours,
        "yearly_hours":   yearly_list,
        "start_date":     str(start_dt),
        "wo_rows":        wo_rows,
        "matched_rows":   matched_rows,
        "pm_tasks_count": len(pm_tasks),
        "summary":        summary,
        "results":        results,
    }


@router.get("/results/{unit_id}")
async def get_latest_result(unit_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    q = supabase.table("maintenance_analyses").select("*").eq("unit_id", unit_id)
    if current_user.get("role") == "engineer":
        q = q.eq("user_id", current_user["sub"])
    res = q.order("created_at", desc=True).limit(1).execute()
    if not res.data:
        raise HTTPException(404, "No analysis found.")
    return res.data[0]


@router.get("/history/{unit_id}")
async def get_history(
    unit_id: str,
    limit:   int = 10,
    current_user=Depends(get_current_user),
):
    supabase = get_supabase_client()
    q = supabase.table("maintenance_analyses").select(
        "id,unit_id,start_date,total_hours,wo_filename,wo_rows,matched_rows,summary,created_at"
    ).eq("unit_id", unit_id)
    if current_user.get("role") == "engineer":
        q = q.eq("user_id", current_user["sub"])
    res = q.order("created_at", desc=True).limit(limit).execute()
    return res.data or []


@router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    res = supabase.table("maintenance_analyses").select("*").eq("id", analysis_id).execute()
    if not res.data:
        raise HTTPException(404, "Analysis not found.")
    record = res.data[0]
    if (current_user.get("role") == "engineer"
            and record["user_id"] != current_user["sub"]):
        raise HTTPException(403, "Access denied.")
    return record


@router.delete("/pm-plan/{pm_plan_id}")
async def delete_pm_plan(pm_plan_id: str, current_user=Depends(get_current_user)):
    supabase = get_supabase_client()
    res = supabase.table("pm_plans").select("id,user_id").eq("id", pm_plan_id).execute()
    if not res.data:
        raise HTTPException(404, "PM plan not found.")
    if (current_user.get("role") == "engineer"
            and res.data[0]["user_id"] != current_user["sub"]):
        raise HTTPException(403, "Access denied.")
    supabase.table("pm_plans").delete().eq("id", pm_plan_id).execute()
    return {"message": "PM plan deleted."}