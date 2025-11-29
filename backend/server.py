# backend/server.py

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
from typing import Any, Dict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Stub endpoint for frontend integration.
    - Accepts a video file upload.
    - Ignores the file for now.
    - Returns a hardcoded response that matches our contract.
    """

    # For now, we just generate a fake job_id and ignore the file contents.
    job_id = uuid4().hex

    response = {
        "status": "ok",
        "job_id": job_id,
        "summary": (
            "You maintain solid eye contact and a clear, steady delivery. "
            "Posture is generally good but a bit rigid, and hand gestures are underused."
        ),
        "scores": {
            "overall": 75,
            "posture": 0.7,
            "gaze": 0.85,
            "gestures": 0.4,
            "facial_expression": 0.6,
        },
        "strengths": [
            "Maintains strong eye contact with the camera.",
            "Background is clean and not distracting.",
            "Voice and pacing are steady and easy to follow.",
        ],
        "opportunities": [
            "Posture can appear slightly rigid; adding small shifts can feel more natural.",
            "Increase hand gestures to emphasize key points.",
            "Use more varied facial expressions to match your message.",
        ],
        "neighbors": [
            {
                "total_points": 72,
                "description": "Rigid posture, strong gaze, minimal gestures, calm delivery.",
            },
            {
                "total_points": 68,
                "description": "Good clarity, low movement, limited expressiveness.",
            },
        ],
        "artifacts": {
            "annotated_video_path": None,
            "features_path": None,
            "context_path": None,
        },
        "raw": {
            "context": None,
            "eval": None,
        },
    }

    return response
