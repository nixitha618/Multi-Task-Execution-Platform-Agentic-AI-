import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.core.config import settings
from backend.app.core.database import init_db
from backend.app.api.router import api_router
from backend.app.services.scheduler import scheduler_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: initialize database tables and background scheduler
    await init_db()
    scheduler_service.start()
    await scheduler_service.load_jobs_from_db()
    yield
    # Shutdown
    scheduler_service.shutdown()

app = FastAPI(
    title=settings.PROJECT_NAME,
    version="1.0.0",
    description="General-Purpose Agentic AI System for Autonomous Multi-Task Execution",
    lifespan=lifespan
)

# CORS middleware for open local access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strict anti-cache headers for all HTML pages to prevent browser stale redirects
NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0",
    "Pragma": "no-cache",
    "Expires": "0",
    "Clear-Site-Data": '"cache"',
}

@app.middleware("http")
async def no_cache_html_middleware(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if (
        path in ["/", "/home", "/home.html", "/login", "/login.html", "/signup", "/signup.html", "/dashboard", "/dashboard.html"]
        or "text/html" in response.headers.get("content-type", "")
        or path.endswith(".js")
        or path.endswith(".css")
    ):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["Clear-Site-Data"] = '"cache"'
    return response

# Include API v1 Router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve generated outputs (charts, reports)
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")

# Frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend"))

if os.path.exists(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    def serve_html_file(file_name: str) -> HTMLResponse:
        file_path = os.path.join(FRONTEND_DIR, file_name)
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content=content, headers=NO_CACHE_HEADERS)
        return HTMLResponse(content="<h1>Page Not Found</h1>", status_code=404, headers=NO_CACHE_HEADERS)

    @app.get("/", include_in_schema=False)
    @app.get("/home", include_in_schema=False)
    @app.get("/home.html", include_in_schema=False)
    async def serve_home():
        return serve_html_file("home.html")

    @app.get("/signup", include_in_schema=False)
    @app.get("/signup.html", include_in_schema=False)
    async def serve_signup():
        return serve_html_file("signup.html")

    @app.get("/login", include_in_schema=False)
    @app.get("/login.html", include_in_schema=False)
    async def serve_login():
        return serve_html_file("login.html")

    @app.get("/dashboard", include_in_schema=False)
    @app.get("/dashboard.html", include_in_schema=False)
    async def serve_dashboard():
        return serve_html_file("dashboard.html")


