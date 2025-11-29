from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Any, Dict
from pathlib import Path
import shutil
import subprocess
import json

from context_generator.retriever import ContextRetriever

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

retriever: ContextRetriever | None = None
try:
    retriever = ContextRetriever(top_k=5)
    print("[INFO] ContextRetriever loaded successfully.")
except Exception as e:
    print(f"[WARN] ContextRetriever not available: {e}")
    retriever = None


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Endpoint for frontend integration.
    - Saves the uploaded video to runs/<job_id>/
    - Calls scripts.llm_eval_dummy to generate evaluation_llava.json
    - Uses ContextRetriever to compute scores + neighbors from summary text
    - Returns a response matching the frontend contract.
    """

    # 1) Save uploaded video into a job-specific directory
    job_id = uuid4().hex
    job_dir = RUNS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / file.filename
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # 2) Run dummy evaluator
    output_json_name = "evaluation_llava.json"
    output_json_path = job_dir / output_json_name

    cmd = [
        "python",
        "-m",
        "scripts.llm_eval_dummy",
        "--video",
        str(video_path),
        "--out",
        str(job_dir),
        "--output-json",
        output_json_name,
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )

    except subprocess.CalledProcessError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Dummy evaluator failed: {e.stderr or e.stdout}",
        )

    if not output_json_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Dummy evaluator did not produce evaluation_llava.json",
        )

    # 3) Load dummy evaluation JSON
    try:
        with output_json_path.open("r", encoding="utf-8") as f:
            eval_json = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="Could not parse evaluation_llava.json",
        )

    summary_text = eval_json.get("summary", "")
    notes = eval_json.get("notes", [])

    # 4) Use ContextRetriever to compute scores + neighbors from the summary
    context_scores: Dict[str, Any] | None = None
    neighbors: list[Dict[str, Any]] = []

    if retriever is not None and summary_text:
        try:
            ctx = retriever.build_context_from_text(summary_text)
            context_scores = ctx.get("scores")
            neighbors = ctx.get("neighbors", [])
            print("[CTX] scores from retriever:", context_scores)
            print("[CTX] num neighbors:", len(neighbors))
        except Exception as e:
            print(f"[WARN] Failed to build context: {e}")

    # Fallback scores if context generator isn't available
    fallback_scores = {
        "overall": 75,
        "posture": 0.7,
        "gaze": 0.85,
        "gestures": 0.4,
        "facial_expression": 0.6,
    }

    response: Dict[str, Any] = {
        "status": "ok",
        "job_id": job_id,
        "summary": summary_text,
        # Use context_scores if we got them; otherwise fallback
        "scores": context_scores or fallback_scores,
        "strengths": [
            "Maintains strong eye contact with the camera.",
            "Background is clean and not distracting.",
        ],
        "opportunities": [
            "Posture can appear slightly rigid; adding small shifts can feel more natural.",
            "Increase hand gestures to emphasize key points.",
        ],
        # Use retrieved neighbors
        "neighbors": neighbors,
        "artifacts": {
            "annotated_video_path": None,
            "features_path": None,
            "context_path": None,
        },
        "raw": {
            "context": context_scores,
            "eval": eval_json,
            "notes": notes,
        },
    }

    return response
