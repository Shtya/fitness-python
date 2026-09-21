import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from analyzer import analyze_body, ensure_model

app = FastAPI(title="So7baFit AI Body Scan", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED = {"image/jpeg", "image/jpg", "image/png", "image/webp"}


@app.on_event("startup")
def startup():
    ensure_model()


@app.get("/health")
def health():
    return {"ok": True, "service": "ai-body-scan"}


def _save_upload(upload: UploadFile, prefix: str) -> str:
    if upload.content_type and upload.content_type.lower() not in ALLOWED:
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {upload.content_type}")
    suffix = os.path.splitext(upload.filename or "")[1] or ".jpg"
    fd, path = tempfile.mkstemp(prefix=f"{prefix}-", suffix=suffix)
    os.close(fd)
    with open(path, "wb") as out:
        out.write(upload.file.read())
    if os.path.getsize(path) < 1024:
        os.unlink(path)
        raise HTTPException(status_code=400, detail=f"{prefix} photo is empty")
    return path


@app.post("/analyze")
def analyze(
    height: float = Form(...),
    front: UploadFile = File(...),
    side: UploadFile = File(...),
):
    front_path: Optional[str] = None
    side_path: Optional[str] = None
    try:
        front_path = _save_upload(front, "front")
        side_path = _save_upload(side, "side")
        return analyze_body(front_path, side_path, float(height))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Measurement analysis failed") from exc
    finally:
        for path in (front_path, side_path):
            if path:
                try:
                    os.unlink(path)
                except OSError:
                    pass
