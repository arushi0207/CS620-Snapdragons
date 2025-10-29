# app.py

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
import uvicorn

# ---------------------------
# Import your model function
# ---------------------------
def run_face_tracking(video_path: str):
    """
    Temporary mock function for demo purposes.
    Replace this with your actual model logic.
    """
    # Example output
    output_points = [
        {"frame": 1, "points": [[100, 120], [110, 130], [115, 125]]},
        {"frame": 2, "points": [[101, 121], [111, 131], [116, 126]]},
    ]
    return {"status": "success", "data": output_points}


# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(title="Speech Feedback Backend", version="1.0")

# Allow your React Native frontend to make requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Health Check Endpoint
# ---------------------------
@app.get("/api/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend running"}


# ---------------------------
# Main Video Upload & Analysis Endpoint
# ---------------------------
@app.post("/api/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Receives video from frontend, runs model, and returns AI feedback.
    """
    try:
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, file.filename)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        model_output = run_face_tracking(temp_file_path)

        os.remove(temp_file_path)

        return JSONResponse(content=model_output)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)},
        )


# ---------------------------
# Optional Endpoint — for testing text-based feedback
# ---------------------------
@app.post("/api/get_text_feedback")
async def get_text_feedback(data: dict):
    """
    Example endpoint if you want to send transcript text instead of video.
    """
    transcript = data.get("transcript", "")
    return {"status": "success", "message": "Text feedback generated", "transcript": transcript}


# ---------------------------
# Run the server (for local dev)
# ---------------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
