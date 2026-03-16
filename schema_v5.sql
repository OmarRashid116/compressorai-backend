-- ============================================================
-- CompressorAI Database Schema v5 — Production
-- Roles: admin | engineer ONLY
-- Run in: Supabase Dashboard → SQL Editor → New Query
-- ============================================================

-- ── DROP ALL (clean migration) ─────────────────────────────
DROP TABLE IF EXISTS reports          CASCADE;
DROP TABLE IF EXISTS analyses         CASCADE;
DROP TABLE IF EXISTS user_compressors CASCADE;
DROP TABLE IF EXISTS compressors      CASCADE;
DROP TABLE IF EXISTS email_verifications CASCADE;
DROP TABLE IF EXISTS users            CASCADE;

DROP TABLE IF EXISTS analysis_results  CASCADE;
DROP TABLE IF EXISTS datasets          CASCADE;
DROP TABLE IF EXISTS user_units        CASCADE;
DROP TABLE IF EXISTS compressor_units  CASCADE;
DROP TABLE IF EXISTS ml_models         CASCADE;
DROP TABLE IF EXISTS compressor_types  CASCADE;

-- ── USERS ──────────────────────────────────────────────────
-- NOTE: password_plain column REMOVED in v5 production.
--       Plain-text passwords are NEVER stored.
CREATE TABLE users (
  id                UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  email             TEXT UNIQUE NOT NULL,
  password_hash     TEXT NOT NULL,
  full_name         TEXT NOT NULL,
  role              TEXT NOT NULL DEFAULT 'engineer'
                      CHECK (role IN ('admin','engineer')),
  company           TEXT,
  is_active         BOOLEAN DEFAULT TRUE,
  is_email_verified BOOLEAN DEFAULT FALSE,
  is_default_admin  BOOLEAN DEFAULT FALSE,
  agreed_to_terms   BOOLEAN DEFAULT FALSE,
  deleted_at        TIMESTAMPTZ,
  created_at        TIMESTAMPTZ DEFAULT NOW(),
  last_login        TIMESTAMPTZ
);

-- ── EMAIL VERIFICATIONS ───────────────────────────────────
CREATE TABLE email_verifications (
  id          UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
  email       TEXT NOT NULL,
  code        TEXT NOT NULL,
  expires_at  TIMESTAMPTZ NOT NULL,
  is_used     BOOLEAN DEFAULT FALSE,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── COMPRESSOR TYPES ──────────────────────────────────────
CREATE TABLE compressor_types (
  id                 UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  name               TEXT NOT NULL UNIQUE,
  manufacturer       TEXT,
  rated_power_kw     FLOAT,
  rated_pressure_bar FLOAT,
  description        TEXT,
  is_active          BOOLEAN DEFAULT TRUE,
  created_by         UUID REFERENCES users(id),
  created_at         TIMESTAMPTZ DEFAULT NOW()
);

-- ── ML MODELS ─────────────────────────────────────────────
CREATE TABLE ml_models (
  id                  UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  compressor_type_id  UUID REFERENCES compressor_types(id) ON DELETE CASCADE,
  model_path          TEXT,
  trained_on_rows     INTEGER,
  trained_on_units    INTEGER,
  silhouette_score    FLOAT,
  r2_score            FLOAT,
  f1_score            FLOAT,
  ga_convergence      FLOAT,
  auto_retrain        BOOLEAN DEFAULT TRUE,
  retrain_threshold   INTEGER DEFAULT 100,
  is_active           BOOLEAN DEFAULT TRUE,
  trained_by          UUID REFERENCES users(id),
  trained_at          TIMESTAMPTZ DEFAULT NOW(),
  created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- ── COMPRESSOR UNITS ──────────────────────────────────────
CREATE TABLE compressor_units (
  id                   UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  compressor_type_id   UUID REFERENCES compressor_types(id) ON DELETE CASCADE,
  unit_id              TEXT NOT NULL,
  serial_number        TEXT,
  location             TEXT,
  notes                TEXT,
  is_active            BOOLEAN DEFAULT TRUE,
  created_by           UUID REFERENCES users(id),
  created_at           TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(compressor_type_id, unit_id)
);

-- ── USER_UNITS ────────────────────────────────────────────
CREATE TABLE user_units (
  id        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id   UUID REFERENCES users(id) ON DELETE CASCADE,
  unit_id   UUID REFERENCES compressor_units(id) ON DELETE CASCADE,
  added_at  TIMESTAMPTZ DEFAULT NOW(),
  is_active BOOLEAN DEFAULT TRUE,
  UNIQUE(user_id, unit_id)
);

-- ── DATASETS ──────────────────────────────────────────────
CREATE TABLE datasets (
  id                   UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  unit_id              UUID REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id              UUID REFERENCES users(id),
  original_filename    TEXT NOT NULL,
  raw_file_path        TEXT,
  processed_file_path  TEXT,
  total_rows           INTEGER,
  clean_rows           INTEGER,
  was_raw              BOOLEAN DEFAULT FALSE,
  cleaning_summary     JSONB,
  is_processed         BOOLEAN DEFAULT FALSE,
  contributed_to_model BOOLEAN DEFAULT FALSE,
  created_at           TIMESTAMPTZ DEFAULT NOW()
);

-- ── ANALYSIS RESULTS ──────────────────────────────────────
CREATE TABLE analysis_results (
  id                        UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id                UUID REFERENCES datasets(id) ON DELETE CASCADE,
  unit_id                   UUID REFERENCES compressor_units(id) ON DELETE CASCADE,
  user_id                   UUID REFERENCES users(id),
  ml_model_id               UUID REFERENCES ml_models(id),
  scores                    JSONB,
  optimal_parameters        JSONB,
  best_electrical_power     FLOAT,
  best_mechanical_power     FLOAT,
  best_spc                  FLOAT,
  baseline_electrical_power FLOAT,
  power_saving_percent      FLOAT,
  user_params               JSONB,
  feature_importance        JSONB,
  scatter_data              JSONB,
  cluster_data              JSONB,
  histogram_data            JSONB,
  training_curve            JSONB,
  cluster_stats             JSONB,
  report_pdf_path           text,
  report_excel_path         text,
  created_at                TIMESTAMPTZ DEFAULT NOW()
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

-- ── DISABLE RLS ────────────────────────────────────────────
-- Backend uses service_role key which bypasses RLS.
-- RLS is disabled here for simplicity; in a multi-tenant SaaS
-- you would enable RLS and add per-user policies as an
-- additional defense-in-depth layer.
ALTER TABLE users                DISABLE ROW LEVEL SECURITY;
ALTER TABLE email_verifications  DISABLE ROW LEVEL SECURITY;
ALTER TABLE compressor_types     DISABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models            DISABLE ROW LEVEL SECURITY;
ALTER TABLE compressor_units     DISABLE ROW LEVEL SECURITY;
ALTER TABLE user_units           DISABLE ROW LEVEL SECURITY;
ALTER TABLE datasets             DISABLE ROW LEVEL SECURITY;
ALTER TABLE analysis_results     DISABLE ROW LEVEL SECURITY;

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
    'reports',
    'reports',
    false,
    52428800,  -- 50 MB
    ARRAY[
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    ]
)
ON CONFLICT (id) DO UPDATE SET
    public             = EXCLUDED.public,
    file_size_limit    = EXCLUDED.file_size_limit,
    allowed_mime_types = EXCLUDED.allowed_mime_types;

SELECT id, name, public, file_size_limit FROM storage.buckets
WHERE id IN ('datasets', 'ml-models', 'reports');

-- ============================================================
-- After running this schema:
--   1. Create .env with all required variables
--   2. python init_admin.py
-- ============================================================