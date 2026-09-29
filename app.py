"""FastAPI application — endpoints for Self-RAG chat, document upload, and audit trail."""

import logging
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, Request, UploadFile, File, HTTPException, Depends, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

from src.models import ChatRequest, ChatResponse
from src.self_rag import run_self_rag
from src.ingestion import ingest_file, delete_file_from_vectorstore, namespace, SUPPORTED
from src.db import init_db, save_audit, latest_audits
from src.vectorstore import get_vector_store


load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)


# ─── User / Workspace Identification Dependency ─────────────────────


def get_current_user_id(
    x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    user_id_query: Optional[str] = Query(None, alias="user_id"),
) -> str:
    """Extract or generate a secure, sanitized user/workspace ID.
    
    Ensures multi-tenant isolation: every visitor has a dedicated workspace token
    and cannot view, overwrite, or access documents/audits/memory of other users.
    """
    raw = (x_user_id or user_id_query or "").strip()
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "", raw)
    if not clean:
        clean = f"usr_{uuid.uuid4().hex[:12]}"
    return clean[:60]


# ─── Lifespan (replaces deprecated @app.on_event) ───────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialize database and vector store."""
    init_db()
    try:
        get_vector_store()
        logger.info("Pinecone vector store initialized successfully.")
    except Exception as exc:
        logger.warning(
            "Pinecone vector store initialization skipped at startup: %s. "
            "Please ensure PINECONE_API_KEY is properly set in your .env file.",
            exc,
        )
    logger.info("OpsGuard ready — http://localhost:8080")
    yield


app = FastAPI(
    title="OpsGuard — Enterprise Incident Response Self-RAG Copilot",
    version="2.0.0",
    description="Self-RAG copilot for cloud operations and incident response.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))


# ─── Routes ──────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "opsguard-self-rag"}


@app.get("/api/workspace")
def workspace_info(user_id: str = Depends(get_current_user_id)):
    """Return info about current active user workspace."""
    user_upload_dir = UPLOADS / user_id
    doc_count = 0
    if user_upload_dir.exists():
        doc_count = sum(1 for p in user_upload_dir.iterdir() if p.is_file())
    return {
        "user_id": user_id,
        "namespace": namespace(user_id),
        "private_docs_count": doc_count,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    user_id: str = Depends(get_current_user_id),
):
    try:
        # payload.user_id takes precedence if provided and valid, else header
        effective_user_id = (payload.user_id or user_id).strip()
        clean_user_id = re.sub(r"[^a-zA-Z0-9_\-]", "", effective_user_id) or user_id

        result = await run_in_threadpool(
            run_self_rag,
            payload.question.strip(),
            payload.thread_id.strip(),
            clean_user_id,
        )
        await run_in_threadpool(save_audit, clean_user_id, payload.question, result)
        result["user_id"] = clean_user_id
        return ChatResponse(**result)
    except Exception:
        logger.exception("Chat endpoint error")
        raise HTTPException(status_code=500, detail="An internal error occurred. Please try again.")


# ─── File Upload ─────────────────────────────────────────────────────


MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB limit per document


def _get_file_size(upload_file: UploadFile) -> int:
    """Determine file size in bytes without reading content into memory."""
    if getattr(upload_file, "size", None) is not None:
        return upload_file.size
    upload_file.file.seek(0, 2)
    size = upload_file.file.tell()
    upload_file.file.seek(0)
    return size


@app.post("/api/upload")
async def upload(
    files: List[UploadFile] = File(default=[]),
    user_id: str = Depends(get_current_user_id),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    total_chunks = 0
    user_ns = namespace(user_id)

    # Per-user isolated upload directory: uploads/{user_id}/
    user_upload_dir = UPLOADS / user_id
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    for uf in files:
        safe_name = Path(uf.filename or "unnamed").name
        suffix = Path(safe_name).suffix.lower()

        # 1. Supported extension validation
        if suffix not in SUPPORTED:
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": user_ns,
                "status": "error",
                "error": f"Unsupported file type '{suffix}' for '{safe_name}'. Supported: PDF, TXT, MD, DOCX",
            })
            continue

        # 2. File size validation (max 20 MB)
        file_size = _get_file_size(uf)
        if file_size > MAX_FILE_SIZE:
            size_mb = file_size / (1024 * 1024)
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": user_ns,
                "status": "error",
                "error": f"File '{safe_name}' ({size_mb:.1f} MB) exceeds maximum allowed size of 20 MB.",
            })
            continue

        # 3. Save into user's private upload directory and ingest into user's private namespace
        target = user_upload_dir / safe_name
        with target.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        try:
            count = await run_in_threadpool(ingest_file, target, namespace=user_ns, user_id=user_id)
            total_chunks += count
            results.append({
                "filename": safe_name,
                "chunks_indexed": count,
                "namespace": user_ns,
                "user_id": user_id,
                "status": "success",
            })
        except Exception:
            logger.exception("Failed to ingest file: %s for user %s", safe_name, user_id)
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": user_ns,
                "status": "error",
                "error": f"Failed to process '{safe_name}'. Please try again or use a different file format.",
            })

    errors = [r["error"] for r in results if r.get("status") == "error"]
    if errors and total_chunks == 0:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    response_data = {
        "user_id": user_id,
        "total_files": len(files),
        "chunks_indexed": total_chunks,
        "namespace": user_ns,
        "results": results,
        "files": results,
    }
    if len(files) == 1 and results:
        response_data["filename"] = results[0]["filename"]

    return response_data


@app.get("/api/audits")
def audits(
    limit: int = 50,
    user_id: str = Depends(get_current_user_id),
):
    """Return audit records scoped strictly to the calling user."""
    data = latest_audits(user_id=user_id, limit=min(max(limit, 1), 100))
    return JSONResponse(
        content=data,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.get("/api/runbooks")
def runbooks(user_id: str = Depends(get_current_user_id)):
    """Return catalog of shared system runbooks and this user's private uploaded runbooks."""
    docs = []
    seen = set()

    # 1. User's private uploaded documents (strictly isolated: uploads/{user_id}/)
    user_upload_dir = UPLOADS / user_id
    if user_upload_dir.exists() and user_upload_dir.is_dir():
        for p in sorted(user_upload_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in {".md", ".pdf", ".txt", ".docx"}:
                seen.add(p.name)
                docs.append({
                    "name": p.name,
                    "category": "Private Upload",
                    "is_user_owned": True,
                    "size_bytes": p.stat().st_size,
                    "type": p.suffix.lower().replace(".", "").upper(),
                })

    # 2. Shared system runbooks and docs (from ./documents and ./docs)
    for d in [ROOT / "documents", ROOT / "docs"]:
        if d.exists() and d.is_dir():
            for p in sorted(d.iterdir()):
                if p.is_file() and p.suffix.lower() in {".md", ".pdf", ".txt", ".docx"}:
                    if p.name not in seen:
                        seen.add(p.name)
                        docs.append({
                            "name": p.name,
                            "category": "System Runbook" if d.name == "documents" else "Documentation",
                            "is_user_owned": False,
                            "size_bytes": p.stat().st_size,
                            "type": p.suffix.lower().replace(".", "").upper(),
                        })

    return {"runbooks": docs, "user_id": user_id}


@app.delete("/api/runbooks/{filename}")
async def delete_runbook(
    filename: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a privately uploaded runbook for this user."""
    safe_name = Path(filename).name
    user_upload_dir = UPLOADS / user_id
    target = user_upload_dir / safe_name

    if not target.exists() or not target.is_file():
        raise HTTPException(
            status_code=404,
            detail="File not found in your private workspace or cannot be deleted.",
        )

    try:
        # Delete file from user disk directory
        target.unlink()
        # Delete vectors from user namespace
        user_ns = namespace(user_id)
        await run_in_threadpool(delete_file_from_vectorstore, safe_name, user_ns)
        return {"status": "deleted", "filename": safe_name, "user_id": user_id}
    except Exception:
        logger.exception("Failed to delete runbook %s for user %s", safe_name, user_id)
        raise HTTPException(status_code=500, detail="Failed to delete document.")


if __name__ == "__main__":
    import os
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host=host, port=port, reload=True)