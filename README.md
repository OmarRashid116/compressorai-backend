# CompressorAI v5 — Backend

FastAPI backend for Industrial Air Compressor Optimizer.
**Stack:** FastAPI + Supabase (PostgreSQL) + Supabase Storage + ML (DBSCAN + GBR + GA)

---

## 🚀 Free Deployment: Replit + Supabase

### Step 1 — Supabase Setup (already done if you have a project)
1. Go to [supabase.com](https://supabase.com) → Your project → SQL Editor
2. Run `schema_v5.sql` to create all tables
3. Get your keys from Settings → API

### Step 2 — Upload to GitHub
```bash
git init
git add .
git commit -m "Initial backend"
git remote add origin https://github.com/YOUR_USERNAME/compressorai-backend.git
git push -u origin main
```

### Step 3 — Replit Setup
1. Go to [replit.com](https://replit.com) → Create Repl → Import from GitHub
2. Select your backend repo → Python template
3. Go to **Tools → Secrets** and add ALL these secrets:

| Secret Key | Value |
|---|---|
| `SUPABASE_URL` | `https://xxxx.supabase.co` |
| `SUPABASE_ANON_KEY` | your anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | your service role key |
| `JWT_SECRET` | `Ali_FYP_2026_MySecretKey_XYZ` |
| `APP_ENV` | `production` |
| `CORS_ORIGINS` | `https://your-frontend.vercel.app` |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USER` | your gmail |
| `SMTP_PASS` | your app password |
| `DEFAULT_ADMIN_EMAIL` | `ali.rashid.fyp@gmail.com` |
| `DEFAULT_ADMIN_PASSWORD` | `Me-22032` |

### Step 4 — Run on Replit
Click **Run** — Replit will install dependencies and start the server.

Your backend URL will be: `https://your-repl-name.your-username.repl.co`

### Step 5 — Initialize Admin (first time only)
In Replit Shell:
```bash
python init_admin.py
```

---

## ⚠️ Important Notes

- **Replit Free Tier**: Server sleeps after ~30min inactivity. First request after sleep takes ~10s.
- **No auto_retrain.py on Replit free**: Run it manually from Shell if needed.
- **Models**: Stored in Supabase Storage (not local disk) — works fine on Replit.

---

## 📁 Project Structure
```
backend/
├── main.py              # FastAPI app entry point
├── config.py            # Settings + Supabase clients
├── deps.py              # Auth dependency helpers
├── storage.py           # Supabase Storage helpers
├── init_admin.py        # Run once: creates default admin
├── auto_retrain.py      # Background retraining scheduler
├── schema_v5.sql        # Supabase database schema
├── requirements.txt     # Python dependencies
├── .replit              # Replit config
├── routers/
│   ├── auth.py
│   ├── admin.py
│   ├── compressors.py
│   ├── datasets.py
│   ├── analysis.py
│   ├── retrain.py
│   └── reports.py
└── ml/
    └── engine.py        # DBSCAN + GBR + Genetic Algorithm
```
