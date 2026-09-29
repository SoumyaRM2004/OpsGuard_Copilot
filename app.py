"""FastAPI application — endpoints for Self-RAG chat, document upload, and audit trail."""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

from src.models import ChatRequest, ChatResponse
from src.self_rag import run_self_rag
from src.ingestion import ingest_file, namespace, SUPPORTED
from src.db import init_db, save_audit, latest_audits
from src.vectorstore import get_vector_store


load_dotenv()

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)


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


@app.post("/api/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest):
    try:
        result = await run_in_threadpool(run_self_rag, payload.question.strip(), payload.thread_id.strip())
        await run_in_threadpool(save_audit, payload.question, result)
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
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    total_chunks = 0
    current_namespace = namespace()

    for uf in files:
        safe_name = Path(uf.filename or "unnamed").name
        suffix = Path(safe_name).suffix.lower()

        # 1. Supported extension validation
        if suffix not in SUPPORTED:
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": current_namespace,
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
                "namespace": current_namespace,
                "status": "error",
                "error": f"File '{safe_name}' ({size_mb:.1f} MB) exceeds maximum allowed size of 20 MB.",
            })
            continue

        # 3. Save and ingest valid file
        target = UPLOADS / safe_name
        with target.open("wb") as f:
            shutil.copyfileobj(uf.file, f)
        try:
            count = await run_in_threadpool(ingest_file, target)
            total_chunks += count
            results.append({
                "filename": safe_name,
                "chunks_indexed": count,
                "namespace": current_namespace,
                "status": "success",
            })
        except Exception:
            logger.exception("Failed to ingest file: %s", safe_name)
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": current_namespace,
                "status": "error",
                "error": f"Failed to process '{safe_name}'. Please try again or use a different file format.",
            })

    errors = [r["error"] for r in results if r.get("status") == "error"]
    if errors and total_chunks == 0:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    response_data = {
        "total_files": len(files),
        "chunks_indexed": total_chunks,
        "namespace": current_namespace,
        "results": results,
        "files": results,
    }
    if len(files) == 1 and results:
        response_data["filename"] = results[0]["filename"]

    return response_data


@app.get("/api/audits")
def audits(limit: int = 20):
    return latest_audits(min(max(limit, 1), 100))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)