import os
import sys

# Ensure backend directory and project root are always in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)

if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from dotenv import load_dotenv

# Load .env from backend directory and project root
load_dotenv(os.path.join(_current_dir, ".env"))
load_dotenv(os.path.join(_parent_dir, ".env"))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.database import database
from app.models import models
from app.api.routes import chat, case_studies, hearings, drafts, evidence

# Create database tables
models.Base.metadata.create_all(bind=database.engine)

app = FastAPI(title="AI Legal Assistant API")

@app.on_event("startup")
def on_startup():
    from app.services import vector_service
    db = database.SessionLocal()
    try:
        # Seed initial case studies if empty
        case_studies.seed_db_case_studies(db)
        # Sync all entities to ChromaDB collection
        vector_service.sync_all_from_db(db)
    finally:
        db.close()

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(case_studies.router, prefix="/api/case-studies", tags=["case-studies"])
app.include_router(hearings.router, prefix="/api/hearings", tags=["hearings"])
app.include_router(drafts.router, prefix="/api/drafts", tags=["drafts"])
app.include_router(evidence.router, prefix="/api/evidence", tags=["evidence"])

# Serve static files from Next.js export
frontend_build_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "out")

if os.path.exists(frontend_build_path):
    app.mount("/_next", StaticFiles(directory=os.path.join(frontend_build_path, "_next")), name="next_assets")
    
    # Static files catch-all for Next.js routing (GET + HEAD for prefetch)
    @app.api_route("/{path_name:path}", methods=["GET", "HEAD"])
    async def catch_all(request: Request, path_name: str):
        if path_name.startswith("api/"):
            return None  # Let FastAPI handle API routes

        if not path_name or path_name == "/":
            return FileResponse(os.path.join(frontend_build_path, "index.html"))

        # Check if the requested file exists (favicon.ico, images, etc.)
        file_path = os.path.join(frontend_build_path, path_name)
        if os.path.isfile(file_path):
            return FileResponse(file_path)

        # For nested routes like cases/[id], try index.html inside the folder
        folder_index = os.path.join(frontend_build_path, path_name, "index.html")
        if os.path.isfile(folder_index):
            return FileResponse(folder_index)

        # For Next.js App Router static export, paths correspond to .html files
        html_path = os.path.join(frontend_build_path, path_name + ".html")
        if os.path.isfile(html_path):
            return FileResponse(html_path)

        # Fallback
        index = os.path.join(frontend_build_path, "index.html")
        return FileResponse(index)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=_current_dir)

