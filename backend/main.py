from __future__ import annotations

from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.answer import LLMUnavailableError, answer_question
from backend.ingest import build_corpus
from backend.retrieval import get_retriever

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CORPUS_PATH = DATA_DIR / "processed" / "corpus.json"
MAX_QUESTION_LENGTH = 1000

load_dotenv(ROOT / ".env")

DEFAULT_ALLOWED_ORIGINS = [
    "https://localhost",
    "http://localhost",
    "capacitor://localhost",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

extra_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
ALLOWED_ORIGINS = list(dict.fromkeys([*DEFAULT_ALLOWED_ORIGINS, *extra_origins]))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Generate corpus from PDFs if missing on a fresh deployment, then preload retriever
    if not CORPUS_PATH.exists():
        try:
            build_corpus()
        except Exception:
            pass
    if CORPUS_PATH.exists():
        try:
            get_retriever()
        except Exception:
            pass
    yield


app = FastAPI(title="Ask the Economic Survey", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|capacitor://localhost)$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)


class AskRequest(BaseModel):
    question: str = Field(..., description="User question about Employment in India (Economic Survey 2025-26)")


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    if not CORPUS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Corpus file is missing. Please run `python -m backend.ingest` first.",
        )

    try:
        data = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        records = data.get("records")
        table_metadata = data.get("table_metadata")
        if not isinstance(records, list) or not isinstance(table_metadata, dict):
            raise ValueError("Invalid corpus structure")

        chapter_chunks = sum(1 for r in records if r.get("kind") == "chapter_chunk")
        table_rows = sum(1 for r in records if r.get("kind") == "table_row")
        table_8_10 = sum(
            1 for r in records if r.get("kind") == "table_row" and r.get("table_id") == "8.10"
        )
        table_8_12 = sum(
            1 for r in records if r.get("kind") == "table_row" and r.get("table_id") == "8.12"
        )

        if chapter_chunks <= 0 or table_rows != 46 or table_8_10 != 23 or table_8_12 != 23:
            raise ValueError("Invalid record counts in corpus")

        return {
            "status": "ok",
            "chapter_chunks": chapter_chunks,
            "table_rows": table_rows,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Corpus is invalid ({exc}). Please run `python -m backend.ingest` to regenerate it.",
        ) from exc


@app.post("/api/ask")
def ask_endpoint(payload: AskRequest) -> dict[str, Any]:
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question must not be blank.",
        )
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds maximum length of {MAX_QUESTION_LENGTH} characters.",
        )

    if not CORPUS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Corpus file is missing. Please run `python -m backend.ingest` first.",
        )

    try:
        retriever = get_retriever()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Corpus index could not be loaded. Please run `python -m backend.ingest`.",
        ) from exc

    try:
        result = answer_question(question, retriever=retriever)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    response: dict[str, Any] = {
        "status": result["status"],
        "answer": result["answer"],
        "sources": result["sources"],
    }
    if "comparison" in result and result["comparison"] is not None:
        response["comparison"] = result["comparison"]
    return response
