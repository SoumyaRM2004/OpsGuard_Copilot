from pathlib import Path
import shutil
from typing import List
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.concurrency import run_in_threadpool


from src.models import ChatRequest, ChatResponse, UploadResponse
from src.self_rag import run_self_rag
from src.ingestion import ingest_file, namespace, SUPPORTED
from src.db import init_db, save_audit, latest_audits
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

ROOT = Path(__file__).resolve().parent
UPLOADS = ROOT / "uploads"
UPLOADS.mkdir(exist_ok=True)


app = FastAPI(
    title="OpsGuard — Enterprise Incident Response Self-RAG Copilot",
    version="2.0.0",
    description="Self-RAG copilot for cloud operations, production troubleshooting, and incident-response runbooks.",
)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))



@app.on_event("startup")
def startup():
    init_db()
    print("\n  Local:   http://localhost:8080")
    print("  Network: http://127.0.0.1:8080\n")


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
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    file: List[UploadFile] = File(default=[]),
):
    upload_files = list(files) + list(file)
    if not upload_files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    results = []
    total_chunks = 0
    current_namespace = namespace()

    for uf in upload_files:
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
        except Exception as e:
            results.append({
                "filename": safe_name,
                "chunks_indexed": 0,
                "namespace": current_namespace,
                "status": "error",
                "error": str(e),
            })

    errors = [r["error"] for r in results if r.get("status") == "error"]
    if errors and total_chunks == 0:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    response_data = {
        "total_files": len(upload_files),
        "chunks_indexed": total_chunks,
        "namespace": current_namespace,
        "results": results,
        "files": results,
    }
    if len(upload_files) == 1 and results:
        response_data["filename"] = results[0]["filename"]

    return response_data


@app.get("/api/audits")
def audits(limit: int = 20):
    return latest_audits(min(max(limit, 1), 100))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)