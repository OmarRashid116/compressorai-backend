"""
main.py — CompressorAI v5
Replit-compatible FastAPI entry point.

Fixes for Replit:
  - os import added (was missing — caused crash in __main__ block)
  - PORT reads from env (Replit sets this automatically)
  - workers=1 always (Replit free tier is single-process)
  - host="0.0.0.0" required for Replit to expose the port
"""
import os
import logging
import time
import uuid

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

from config import CORS_ORIGINS, IS_PROD, get_supabase_client
from routers import auth, compressors, datasets, analysis, retrain, reports, admin

logger = logging.getLogger("compressorai")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Rate limiter ──────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


# ── Lifespan ──────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 CompressorAI v5 starting…")
    logger.info(f"   Environment : {'PRODUCTION' if IS_PROD else 'development'}")
    logger.info(f"   CORS origins: {CORS_ORIGINS}")

    try:
        sb = get_supabase_client()
        sb.table("users").select("id").limit(1).execute()
        logger.info("   Database    : ✅ connected")
    except Exception as e:
        logger.error(f"   Database    : ❌ connection failed — {e}")
        if IS_PROD:
            raise RuntimeError("Cannot start: database connection failed.") from e

    yield
    logger.info("👋 CompressorAI v5 shutting down.")


# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title="Industrial Air Compressor Optimizer API",
    description=(
        "AI-powered optimization for Industrial Air Compressors — "
        "DBSCAN + GBR + Genetic Algorithm"
    ),
    version="5.0.0",
    lifespan=lifespan,
    # Always show docs (useful for testing on Replit)
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── Rate limiter state ────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Security headers ──────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]          = "DENY"
    response.headers["X-XSS-Protection"]         = "1; mode=block"
    response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    if IS_PROD:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# ── Request logging ───────────────────────────────────────────
@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start      = time.perf_counter()
    response   = await call_next(request)
    duration   = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} "
        f"[{duration:.1f}ms] id={request_id}"
    )
    return response


# ── Global exception handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.exception(f"Unhandled exception on {request.method} {request.url.path}")
    detail = str(exc) if not IS_PROD else "An internal server error occurred."
    return JSONResponse(status_code=500, content={"detail": detail})


# ── Routers ───────────────────────────────────────────────────
# Dual-mount: both /api/... and /... work
# Frontend may call /auth/login OR /api/auth/login depending on version
for _prefix in ("/api", ""):
    app.include_router(auth.router,        prefix=f"{_prefix}/auth",        tags=["Authentication"])
    app.include_router(admin.router,       prefix=f"{_prefix}/admin",       tags=["Admin"])
    app.include_router(compressors.router, prefix=f"{_prefix}/compressors", tags=["Compressors"])
    app.include_router(datasets.router,    prefix=f"{_prefix}/datasets",    tags=["Datasets"])
    app.include_router(analysis.router,    prefix=f"{_prefix}/analysis",    tags=["Analysis"])
    app.include_router(retrain.router,     prefix=f"{_prefix}/retrain",     tags=["Retrain"])
    app.include_router(reports.router,     prefix=f"{_prefix}/reports",     tags=["Reports"])

@app.get("/health", include_in_schema=False)
async def health_alias():
    return await health_check()


# ── Health / root ─────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "CompressorAI v5 API is running", "version": "5.0.0"}


@app.get("/api/health", tags=["System"])
async def health_check():
    try:
        sb = get_supabase_client()
        sb.table("users").select("id").limit(1).execute()
        db_ok = True
    except Exception:
        db_ok = False

    status = "healthy" if db_ok else "degraded"
    return JSONResponse(
        status_code=200 if db_ok else 503,
        content={"status": status, "version": "5.0.0", "database": "ok" if db_ok else "error"},
    )


if __name__ == "__main__":
    # Replit sets PORT env var automatically
    # Always bind 0.0.0.0 — localhost won't work on Replit
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting on http://0.0.0.0:{port}")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        workers=1,
        log_level="info",
    )
