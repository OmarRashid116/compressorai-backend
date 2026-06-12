-- ============================================================
-- CompressorAI Database Schema v6 — FINAL COMPLETE
-- Roles: admin | engineer ONLY
-- Run in: Supabase Dashboard → SQL Editor → New Query
--
-- ✅ Includes ALL v5 + v6 columns
-- ✅ Includes pm_plans + maintenance_analyses tables
-- ✅ Safe ADD COLUMN IF NOT EXISTS for existing DBs
-- ✅ Fresh install: run SECTION A (DROP + CREATE)
-- ✅ Existing DB upgrade: run SECTION B (ALTER only)
-- ============================================================


-- ============================================================
-- SECTION A — FRESH INSTALL (drops everything and recreates)
-- If you already have data you want to keep, SKIP to SECTION B
-- ============================================================

DROP TABLE IF EXISTS maintenance_analyses CASCADE;
DROP TABLE IF EXISTS pm_plans            CASCADE;
DROP TABLE IF EXISTS reports             CASCADE;
DROP TABLE IF EXISTS analyses            CASCADE;
DROP TABLE IF EXISTS user_compressors    CASCADE;
DROP TABLE IF EXISTS compressors         CASCADE;
DROP TABLE IF EXISTS email_verifications CASCADE;
DROP TABLE IF EXISTS users               CASCADE;

DROP TABLE IF EXISTS analysis_results  CASCADE;
DROP TABLE IF EXISTS datasets          CASCADE;
DROP TABLE IF EXISTS user_units        CASCADE;
DROP TABLE IF EXISTS compressor_units  CASCADE;
DROP TABLE IF EXISTS ml_models         CASCADE;
DROP TABLE IF EXISTS compressor_types  CASCADE;


-- ── USERS ──────────────────────────────────────────────────
CREATE TABLE users (
  id                UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  email             TEXT        UNIQUE NOT NULL,
  password_hash     TEXT        NOT NULL,
  full_name         TEXT        NOT NULL,
  role              TEXT        NOT NULL DEFAULT 'engineer'
                                  CHECK (role IN ('admin','engineer')),
  company           TEXT,
  is_active         BOOLEAN     DEFAULT TRUE,
  is_email_verified BOOLEAN     DEFAULT FALSE,
  is_default_admin  BOOLEAN     DEFAULT FALSE,
  agreed_to_terms   BOOLEAN     DEFAULT FALSE,
  deleted_at        TIMESTAMPTZ,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  last_login        TIMESTAMPTZ
);

-- ── EMAIL VERIFICATIONS ───────────────────────────────────
CREATE TABLE email_verifications (
  id          UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id     UUID        REFERENCES users(id) ON DELETE CASCADE,
  email       TEXT        NOT NULL,
  code        TEXT        NOT NULL,
  expires_at  TIMESTAMPTZ NOT NULL,
  is_used     BOOLEAN     DEFAULT FALSE,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── COMPRESSOR TYPES ──────────────────────────────────────
CREATE TABLE compressor_types (
  id                 UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  name               TEXT        NOT NULL UNIQUE,
  manufacturer       TEXT,
  rated_power_kw     FLOAT,
  rated_pressure_bar FLOAT,
  description        TEXT,
  is_active          BOOLEAN     DEFAULT TRUE,
  created_by         UUID        REFERENCES users(id),
  created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ── ML MODELS ─────────────────────────────────────────────
CREATE TABLE ml_models (
  id                  UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  compressor_type_id  UUID        REFERENCES compressor_types(id) ON DELETE CASCADE,
  model_path          TEXT,
  trained_on_rows     INTEGER,
  trained_on_units    INTEGER,
  silhouette_score    FLOAT,
  r2_score            FLOAT,
  f1_score            FLOAT,
  ga_convergence      FLOAT,
  auto_retrain        BOOLEAN     DEFAULT TRUE,
  retrain_threshold   INTEGER     DEFAULT 100,
  is_active           BOOLEAN     DEFAULT TRUE,
  trained_by          UUID        REFERENCES users(id),
  trained_at          TIMESTAMPTZ DEFAULT NOW(),
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── COMPRESSOR UNITS ──────────────────────────────────────
CREATE TABLE compressor_units (
  id                   UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  compressor_type_id   UUID        REFERENCES compressor_types(id) ON DELETE CASCADE,
  unit_id              TEXT        NOT NULL,
  serial_number        TEXT,
  location             TEXT,
  notes                TEXT,
  is_active            BOOLEAN     DEFAULT TRUE,
  created_by           UUID        REFERENCES users(id),
  created_at           TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(compressor_type_id, unit_id)
);

-- ── USER_UNITS ────────────────────────────────────────────
CREATE TABLE user_units (
  id        UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id   UUID        REFERENCES users(id)             ON DELETE CASCADE,
  unit_id   UUID        REFERENCES compressor_units(id)  ON DELETE CASCADE,
  added_at  TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN     DEFAULT TRUE,
  UNIQUE(user_id, unit_id)
);

-- ── DATASETS ──────────────────────────────────────────────
CREATE TABLE datasets (
  id                   UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id              UUID        REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id              UUID        REFERENCES users(id),
  original_filename    TEXT        NOT NULL,
  raw_file_path        TEXT,
  processed_file_path  TEXT,
  total_rows           INTEGER,
  clean_rows           INTEGER,
  was_raw              BOOLEAN     DEFAULT FALSE,
  cleaning_summary     JSONB,
  is_processed         BOOLEAN     DEFAULT FALSE,
  contributed_to_model BOOLEAN     DEFAULT FALSE,
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ── ANALYSIS RESULTS ──────────────────────────────────────
-- Contains ALL columns used by both v5 and v6 routers
CREATE TABLE analysis_results (
  id                        UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id                UUID        REFERENCES datasets(id)          ON DELETE CASCADE,
  unit_id                   UUID        REFERENCES compressor_units(id)  ON DELETE CASCADE,
  user_id                   UUID        REFERENCES users(id),
  ml_model_id               UUID        REFERENCES ml_models(id),

  -- Core scores / metrics
  scores                    JSONB,
  r2_score                  NUMERIC,
  mae                       NUMERIC,
  rmse                      NUMERIC,

  -- Optimization outputs
  optimal_parameters        JSONB,
  best_electrical_power     NUMERIC,
  best_mechanical_power     NUMERIC,
  best_spc                  NUMERIC,
  baseline_electrical_power NUMERIC,
  power_saving_percent      NUMERIC,

  -- Savings (v5 extra columns)
  cost_saved_annual         NUMERIC     DEFAULT 0,
  energy_saved_kwh          NUMERIC     DEFAULT 0,

  -- Input parameters used
  user_params               JSONB,

  -- Visualisation data
  feature_importance        JSONB,
  scatter_data              JSONB,
  cluster_data              JSONB,      -- v6 name
  clusters                  JSONB,      -- v5 alias (kept for compatibility)
  histogram_data            JSONB,
  training_curve            JSONB,
  cluster_stats             JSONB,

  -- Report file paths
  report_pdf_path           TEXT,
  report_excel_path         TEXT,

  created_at                TIMESTAMPTZ DEFAULT NOW()
);

-- ── PM PLANS ──────────────────────────────────────────────
-- Preventive Maintenance plans per compressor unit
CREATE TABLE pm_plans (
  id                UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id           UUID        REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id           UUID        REFERENCES users(id),
  original_filename TEXT,
  tasks             JSONB       NOT NULL,  -- [{task, task_display, interval_hours, machine}]
  machine_tag       TEXT,
  is_active         BOOLEAN     DEFAULT TRUE,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ── MAINTENANCE ANALYSES ──────────────────────────────────
-- PM compliance analysis runs
CREATE TABLE maintenance_analyses (
  id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id       UUID        REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id       UUID        REFERENCES users(id),
  pm_plan_id    UUID        REFERENCES pm_plans(id),
  start_date    DATE        NOT NULL,
  yearly_hours  JSONB       NOT NULL,   -- e.g. [1866, 1866, 1866]
  total_hours   FLOAT       NOT NULL,
  wo_filename   TEXT,
  wo_rows       INTEGER,
  matched_rows  INTEGER,
  results       JSONB       NOT NULL,   -- full compliance results array
  summary       JSONB,                  -- KPI summary object
  created_at    TIMESTAMPTZ DEFAULT NOW()
);


-- ── INDEXES ───────────────────────────────────────────────
CREATE INDEX idx_users_email           ON users(email);
CREATE INDEX idx_users_role            ON users(role);
CREATE INDEX idx_users_active          ON users(is_active);
CREATE INDEX idx_users_deleted         ON users(deleted_at) WHERE deleted_at IS NOT NULL;
CREATE INDEX idx_verif_user            ON email_verifications(user_id);
CREATE INDEX idx_verif_used            ON email_verifications(is_used) WHERE is_used = FALSE;
CREATE INDEX idx_types_name            ON compressor_types(name);
CREATE INDEX idx_types_active          ON compressor_types(is_active);
CREATE INDEX idx_ml_type               ON ml_models(compressor_type_id);
CREATE INDEX idx_ml_active             ON ml_models(is_active);
CREATE INDEX idx_units_type            ON compressor_units(compressor_type_id);
CREATE INDEX idx_units_unit_id         ON compressor_units(unit_id);
CREATE INDEX idx_units_active          ON compressor_units(is_active);
CREATE INDEX idx_uu_user               ON user_units(user_id);
CREATE INDEX idx_uu_unit               ON user_units(unit_id);
CREATE INDEX idx_uu_active             ON user_units(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_datasets_unit         ON datasets(unit_id);
CREATE INDEX idx_datasets_user         ON datasets(user_id);
CREATE INDEX idx_datasets_created      ON datasets(created_at DESC);
CREATE INDEX idx_datasets_contributed  ON datasets(contributed_to_model);
CREATE INDEX idx_datasets_processed    ON datasets(is_processed) WHERE is_processed = TRUE;
CREATE INDEX idx_analysis_dataset      ON analysis_results(dataset_id);
CREATE INDEX idx_analysis_unit         ON analysis_results(unit_id);
CREATE INDEX idx_analysis_user         ON analysis_results(user_id);
CREATE INDEX idx_analysis_created      ON analysis_results(created_at DESC);
CREATE INDEX idx_pm_plans_unit         ON pm_plans(unit_id);
CREATE INDEX idx_pm_plans_active       ON pm_plans(unit_id, is_active);
CREATE INDEX idx_maint_unit            ON maintenance_analyses(unit_id);
CREATE INDEX idx_maint_user            ON maintenance_analyses(user_id);
CREATE INDEX idx_maint_created         ON maintenance_analyses(created_at DESC);


-- ── DISABLE RLS ────────────────────────────────────────────
-- Backend uses service_role key which bypasses RLS anyway.
ALTER TABLE users                DISABLE ROW LEVEL SECURITY;
ALTER TABLE email_verifications  DISABLE ROW LEVEL SECURITY;
ALTER TABLE compressor_types     DISABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models            DISABLE ROW LEVEL SECURITY;
ALTER TABLE compressor_units     DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_units           DISABLE ROW LEVEL SECURITY;
ALTER TABLE datasets             DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results     DISABLE ROW LEVEL SECURITY;
ALTER TABLE pm_plans             DISABLE ROW LEVEL SECURITY;
ALTER TABLE maintenance_analyses DISABLE ROW LEVEL SECURITY;


-- ── STORAGE BUCKETS ────────────────────────────────────────
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'datasets', 'datasets', FALSE, 10485760,
    ARRAY[
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.ms-excel',
        'text/csv',
        'application/octet-stream'
    ]
)
ON CONFLICT (id) DO UPDATE SET
    public             = EXCLUDED.public,
    file_size_limit    = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'ml-models', 'ml-models', FALSE, 104857600,
    ARRAY['application/octet-stream']
)
ON CONFLICT (id) DO UPDATE SET
    public             = EXCLUDED.public,
    file_size_limit    = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'reports', 'reports', FALSE, 52428800,
    ARRAY[
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ]
)
ON CONFLICT (id) DO UPDATE SET
    public             = EXCLUDED.public,
    file_size_limit    = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;


-- ============================================================
-- SECTION B — SAFE UPGRADE (existing DB, keeps your data)
-- Run ONLY these ALTERs if you already have data and skipped
-- the DROP/CREATE above.
-- ============================================================

-- analysis_results: add any missing columns safely
ALTER TABLE analysis_results
    ADD COLUMN IF NOT EXISTS r2_score                 NUMERIC,
    ADD COLUMN IF NOT EXISTS mae                      NUMERIC,
    ADD COLUMN IF NOT EXISTS rmse                     NUMERIC,
    ADD COLUMN IF NOT EXISTS cost_saved_annual        NUMERIC DEFAULT 0,
    ADD COLUMN IF NOT EXISTS energy_saved_kwh         NUMERIC DEFAULT 0,
    ADD COLUMN IF NOT EXISTS clusters                 JSONB,
    ADD COLUMN IF NOT EXISTS best_electrical_power    NUMERIC,
    ADD COLUMN IF NOT EXISTS best_mechanical_power    NUMERIC,
    ADD COLUMN IF NOT EXISTS best_spc                 NUMERIC,
    ADD COLUMN IF NOT EXISTS optimal_parameters       JSONB,
    ADD COLUMN IF NOT EXISTS feature_importance       JSONB,
    ADD COLUMN IF NOT EXISTS scatter_data             JSONB,
    ADD COLUMN IF NOT EXISTS cluster_data             JSONB,
    ADD COLUMN IF NOT EXISTS histogram_data           JSONB,
    ADD COLUMN IF NOT EXISTS training_curve           JSONB,
    ADD COLUMN IF NOT EXISTS cluster_stats            JSONB,
    ADD COLUMN IF NOT EXISTS report_pdf_path          TEXT,
    ADD COLUMN IF NOT EXISTS report_excel_path        TEXT;

-- pm_plans: create only if missing
CREATE TABLE IF NOT EXISTS pm_plans (
  id                UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id           UUID        REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id           UUID        REFERENCES users(id),
  original_filename TEXT,
  tasks             JSONB       NOT NULL,
  machine_tag       TEXT,
  is_active         BOOLEAN     DEFAULT TRUE,
  created_at        TIMESTAMPTZ DEFAULT NOW()
);

-- maintenance_analyses: create only if missing
CREATE TABLE IF NOT EXISTS maintenance_analyses (
  id            UUID        DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id       UUID        REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id       UUID        REFERENCES users(id),
  pm_plan_id    UUID        REFERENCES pm_plans(id),
  start_date    DATE        NOT NULL,
  yearly_hours  JSONB       NOT NULL,
  total_hours   FLOAT       NOT NULL,
  wo_filename   TEXT,
  wo_rows       INTEGER,
  matched_rows  INTEGER,
  results       JSONB       NOT NULL,
  summary       JSONB,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for new tables (IF NOT EXISTS — safe to re-run)
CREATE INDEX IF NOT EXISTS idx_pm_plans_unit    ON pm_plans(unit_id);
CREATE INDEX IF NOT EXISTS idx_pm_plans_active  ON pm_plans(unit_id, is_active);
CREATE INDEX IF NOT EXISTS idx_maint_unit       ON maintenance_analyses(unit_id);
CREATE INDEX IF NOT EXISTS idx_maint_user       ON maintenance_analyses(user_id);
CREATE INDEX IF NOT EXISTS idx_maint_created    ON maintenance_analyses(created_at DESC);

-- RLS off for new tables
ALTER TABLE pm_plans             DISABLE ROW LEVEL SECURITY;
ALTER TABLE maintenance_analyses DISABLE ROW LEVEL SECURITY;


-- ── VERIFY — should list all 10 tables ────────────────────
SELECT table_name
FROM   information_schema.tables
WHERE  table_schema = 'public'
  AND  table_name IN (
         'users', 'email_verifications', 'compressor_types', 'ml_models',
         'compressor_units', 'user_units', 'datasets', 'analysis_results',
         'pm_plans', 'maintenance_analyses'
       )
ORDER BY table_name;

-- ============================================================
-- After running this schema:
--   1. Make sure .env has all required variables
--   2. Run:  python init_admin.py
--      OR just double-click Start_CompressorAI_Local.bat
--      (it auto-runs init_admin.py if no admin found)
-- ============================================================