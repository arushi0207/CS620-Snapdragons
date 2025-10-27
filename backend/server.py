# server.py (replace your /process-video route with this)

import os
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://127.0.0.1:5173",
        "http://localhost:3000", "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-video")
async def process_video(
    video: UploadFile = File(...),
    max_frames: Optional[int] = Form(None),
    display: Optional[bool] = Form(False),
):
    # Absolute path to run_video_face.py (sibling file)
    script_path = Path(__file__).with_name("run_video_face.py")
    if not script_path.exists():
        return JSONResponse(
            status_code=500,
            content={"error": f"run_video_face.py not found at {str(script_path)}"}
        )

    # Create temp dir WITHOUT context manager; we’ll clean after sending the file
    td = tempfile.mkdtemp()
    in_path = os.path.join(td, "input.mp4")
    out_path = os.path.join(td, "annotated.mp4")

    try:
        # Save uploaded file
        data = await video.read()
        if not data:
            return JSONResponse(status_code=400, content={"error": "Empty upload"})
        with open(in_path, "wb") as f:
            f.write(data)

        # Build command; use absolute script path
        cmd = [
            os.sys.executable, str(script_path),
            "--video", in_path,
            "--out", out_path,
        ]
        if display:
            cmd.append("--display")
        if max_frames:
            cmd += ["--max-frames", str(max_frames)]

        # Run and capture output for debugging
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            cwd=str(script_path.parent),  # ensure cwd=backend folder
        )

        # Log to server console for quick debugging
        print("=== run_video_face.py STDOUT ===")
        print(proc.stdout)
        print("=== run_video_face.py STDERR ===")
        print(proc.stderr)

        if proc.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Processing failed",
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                    "cmd": cmd,
                },
            )

        if not os.path.exists(out_path):
            return JSONResponse(
                status_code=500,
                content={
                    "error": "No output produced",
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )

        # Stream annotated mp4 then cleanup
        cleanup = BackgroundTask(lambda: shutil.rmtree(td, ignore_errors=True))
        return FileResponse(
            out_path,
            media_type="video/mp4",
            filename="annotated.mp4",
            background=cleanup,
        )

    except Exception as e:
        # Keep temp dir for post-mortem if something unexpected happens
        return JSONResponse(status_code=500, content={"error": str(e), "tmpdir": td})
