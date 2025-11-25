"""
SpeakEasy API - Public Speaking Coaching Backend
A comprehensive FastAPI backend for video analysis and speech feedback.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime
import os
import uvicorn
import logging
import uuid
import json
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------
UPLOAD_DIR = "temp_uploads"
RESULTS_DIR = "results"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
LOG_DIR = "logs"

# Create necessary directories
for directory in [UPLOAD_DIR, RESULTS_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'api.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------------
# Data Models
# ---------------------------

class GoalType(str, Enum):
    """Available coaching goals"""
    JOB_INTERVIEW = "job-interview"
    CLASS_PRESENTATION = "class-presentation"
    NETWORKING_PITCH = "networking-pitch"
    GENERAL_CONFIDENCE = "general-confidence"


class AnalysisRequest(BaseModel):
    """Request model for video analysis"""
    goal: GoalType = Field(..., description="The coaching goal type")
    session_name: Optional[str] = Field(None, description="Optional session name")


class TextFeedbackRequest(BaseModel):
    """Request model for text-based feedback"""
    transcript: str = Field(..., min_length=10, description="Speech transcript text")
    goal: GoalType = Field(..., description="The coaching goal type")
    session_name: Optional[str] = None


class MetricResult(BaseModel):
    """Individual metric result"""
    metric_name: str
    value: Any
    unit: Optional[str] = None
    status: str = Field(..., description="good, warning, or needs_improvement")
    message: str
    tip: Optional[str] = None


class AnalysisResult(BaseModel):
    """Complete analysis result"""
    session_id: str
    goal: GoalType
    session_name: Optional[str] = None
    timestamp: datetime
    metrics: List[MetricResult]
    overall_score: float = Field(..., ge=0, le=100, description="Overall score out of 100")
    summary: str
    recommendations: List[str]
    processing_time: float


class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    goal: GoalType
    session_name: Optional[str]
    created_at: datetime
    status: str = Field(..., description="processing, completed, or failed")
    has_results: bool = False


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    message: str
    version: str
    timestamp: datetime


# ---------------------------
# In-Memory Storage (Replace with database in production)
# ---------------------------
sessions_db: Dict[str, Dict[str, Any]] = {}
results_db: Dict[str, AnalysisResult] = {}

# Constants for efficiency
FILLER_WORDS = frozenset(["um", "uh", "like", "you know", "so", "well", "actually", "basically"])
FILLER_WORDS_LIST = list(FILLER_WORDS)
GOAL_RECOMMENDATIONS = {
    GoalType.JOB_INTERVIEW: "For interviews, maintain a confident but approachable demeanor.",
    GoalType.CLASS_PRESENTATION: "For presentations, use more gestures to emphasize key points.",
    GoalType.NETWORKING_PITCH: "For networking, focus on warm facial expressions and open body language.",
    GoalType.GENERAL_CONFIDENCE: "Continue practicing to build your confidence and speaking skills."
}


# ---------------------------
# Model Functions (Replace with actual implementations)
# ---------------------------

def run_face_tracking(video_path: str, goal: GoalType) -> Dict[str, Any]:
    """
    Analyze video for face tracking, eye gaze, posture, and facial expressions.
    Replace this with your actual model logic.
    """
    logger.info(f"Running face tracking analysis for goal: {goal}")
    
    # Mock implementation - replace with actual model
    import random
    import time
    
    # Simulate processing time
    time.sleep(0.5)
    
    # Pre-generate tracking points more efficiently
    tracking_points = [
        {"frame": i, "points": [[100 + i, 120 + i], [110 + i, 130 + i], [115 + i, 125 + i]]}
        for i in range(1, 11)
    ]
    
    return {
        "eye_gaze_percentage": random.uniform(75, 95),
        "posture_score": random.uniform(60, 90),
        "facial_expression_score": random.uniform(70, 95),
        "face_tracking_points": tracking_points
    }


def analyze_audio(video_path: str, goal: GoalType) -> Dict[str, Any]:
    """
    Analyze audio for pace, filler words, and vocal variety.
    Replace this with your actual audio analysis logic.
    """
    logger.info(f"Running audio analysis for goal: {goal}")
    
    # Mock implementation
    import random
    
    # Use cached filler words list
    filler_counts = {word: random.randint(0, 5) for word in FILLER_WORDS_LIST[:6]}
    total_fillers = sum(filler_counts.values())
    
    return {
        "pace_wpm": random.randint(130, 170),
        "filler_words": {
            "total": total_fillers,
            "breakdown": filler_counts
        },
        "vocal_variety_score": random.uniform(70, 95),
        "pauses": random.randint(5, 15)
    }


def generate_feedback(metrics: Dict[str, Any], goal: GoalType) -> Dict[str, Any]:
    """
    Generate comprehensive feedback based on metrics and goal.
    """
    recommendations = []
    summary_parts = []
    
    # Analyze pace
    pace = metrics.get("pace_wpm", 150)
    if pace < 140:
        recommendations.append("Try speaking slightly faster to maintain audience engagement.")
        summary_parts.append(f"Your pace of {pace} WPM is a bit slow.")
    elif pace > 160:
        recommendations.append("Slow down slightly to allow your audience to process your points.")
        summary_parts.append(f"Your pace of {pace} WPM is a bit fast.")
    else:
        summary_parts.append(f"Your pace of {pace} WPM is excellent.")
    
    # Analyze filler words
    total_fillers = metrics.get("filler_words", {}).get("total", 0)
    if total_fillers > 5:
        recommendations.append("Practice replacing filler words with brief pauses.")
        summary_parts.append(f"You used {total_fillers} filler words.")
    elif total_fillers > 0:
        summary_parts.append(f"You used {total_fillers} filler words, which is acceptable.")
    else:
        summary_parts.append("You avoided filler words completely - great job!")
    
    # Analyze eye gaze
    eye_gaze = metrics.get("eye_gaze_percentage", 80)
    if eye_gaze < 70:
        recommendations.append("Make more eye contact with your audience to build connection.")
        summary_parts.append(f"Eye contact was {eye_gaze:.0f}%.")
    elif eye_gaze >= 85:
        summary_parts.append(f"Excellent eye contact at {eye_gaze:.0f}%.")
    else:
        summary_parts.append(f"Good eye contact at {eye_gaze:.0f}%.")
    
    # Analyze posture
    posture = metrics.get("posture_score", 75)
    if posture < 70:
        recommendations.append("Focus on maintaining an upright, confident posture.")
        summary_parts.append("Posture needs improvement.")
    else:
        summary_parts.append("Your posture was generally good.")
    
    # Goal-specific recommendations (using cached dictionary)
    goal_recommendation = GOAL_RECOMMENDATIONS.get(goal)
    if goal_recommendation:
        recommendations.append(goal_recommendation)
    
    # Calculate overall score
    score = (
        (min(pace, 160) / 160) * 20 +  # Pace component (max 20)
        (max(0, 10 - total_fillers) / 10) * 20 +  # Filler words (max 20)
        (eye_gaze / 100) * 30 +  # Eye gaze (max 30)
        (posture / 100) * 30  # Posture (max 30)
    )
    
    summary = " ".join(summary_parts)
    if not summary:
        summary = "Overall, you delivered a solid performance with room for improvement."
    
    return {
        "summary": summary,
        "recommendations": recommendations,
        "overall_score": round(score, 1)
    }


def _create_metric_result(metric_name: str, value: Any, unit: str, status: str, 
                          message: str, tip: str) -> MetricResult:
    """Helper function to create metric results efficiently"""
    return MetricResult(
        metric_name=metric_name,
        value=value,
        unit=unit,
        status=status,
        message=message,
        tip=tip
    )


def process_video_analysis(video_path: str, goal: GoalType, session_id: str) -> AnalysisResult:
    """
    Main function to process video and generate complete analysis.
    """
    start_time = datetime.now()
    
    try:
        # Run face tracking and audio analysis (could be parallelized in production)
        face_data = run_face_tracking(video_path, goal)
        audio_data = analyze_audio(video_path, goal)
        
        # Combine metrics efficiently
        combined_metrics = {**face_data, **audio_data}
        
        # Generate feedback
        feedback = generate_feedback(combined_metrics, goal)
        
        # Build metric results using helper function
        pace_wpm = combined_metrics["pace_wpm"]
        filler_total = combined_metrics["filler_words"]["total"]
        eye_gaze = combined_metrics["eye_gaze_percentage"]
        posture = combined_metrics["posture_score"]
        facial_expr = combined_metrics["facial_expression_score"]
        
        metrics = [
            _create_metric_result(
                "Pace", pace_wpm, "WPM",
                "good" if 140 <= pace_wpm <= 160 else "warning",
                f"Your speaking pace is {pace_wpm} WPM. Ideal range is 140-160 WPM.",
                "Try pausing for 2 seconds after each main point."
            ),
            _create_metric_result(
                "Filler Words", filler_total, "count",
                "good" if filler_total <= 3 else "needs_improvement",
                f"You used {filler_total} filler words.",
                "Practice with a silent pause instead of 'um'."
            ),
            _create_metric_result(
                "Eye Gaze", f"{eye_gaze:.0f}%", "percentage",
                "good" if eye_gaze >= 85 else "warning",
                f"You maintained {eye_gaze:.0f}% eye contact with the audience.",
                "Avoid looking down at your notes too frequently."
            ),
            _create_metric_result(
                "Posture", f"{posture:.0f}/100", "score",
                "good" if posture >= 80 else "needs_improvement",
                f"Your posture score is {posture:.0f}/100.",
                "Practice keeping your back straight and shoulders relaxed."
            ),
            _create_metric_result(
                "Facial Expression", f"{facial_expr:.0f}/100", "score",
                "good" if facial_expr >= 80 else "warning",
                f"Your facial expression score is {facial_expr:.0f}/100.",
                "Align your facial expressions with the emotion of the topic."
            )
        ]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AnalysisResult(
            session_id=session_id,
            goal=goal,
            timestamp=datetime.now(),
            metrics=metrics,
            overall_score=feedback["overall_score"],
            summary=feedback["summary"],
            recommendations=feedback["recommendations"],
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        raise


# ---------------------------
# FastAPI Setup
# ---------------------------
app = FastAPI(
    title="SpeakEasy API",
    description="AI-powered public speaking coaching backend",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Utility Functions
# ---------------------------

def validate_video_file(file: UploadFile) -> None:
    """Validate uploaded video file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    # Check file size (if available)
    if hasattr(file, 'size') and file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB"
        )


async def save_uploaded_file(file: UploadFile) -> str:
    """Save uploaded file and return path (async for better performance)"""
    validate_video_file(file)
    
    file_ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    # Read file content once
    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    logger.info(f"Saved uploaded file: {file_path}")
    return file_path


def cleanup_file(file_path: str) -> None:
    """Remove temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to cleanup file {file_path}: {str(e)}")


# ---------------------------
# API Endpoints
# ---------------------------

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="ok",
        message="SpeakEasy API is running",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/api/analyze_video", response_model=AnalysisResult)
async def analyze_video(
    file: UploadFile = File(...),
    goal: GoalType = Query(..., description="The coaching goal type"),
    session_name: Optional[str] = Query(None, description="Optional session name")
):
    """
    Main endpoint: Upload video, analyze it, and return comprehensive feedback.
    """
    file_path = None
    session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"Received video upload request: {file.filename}, goal: {goal}")
        
        # Create session first with timestamp tracking
        now = datetime.now()
        sessions_db[session_id] = {
            "session_id": session_id,
            "goal": goal,
            "session_name": session_name,
            "created_at": now.isoformat(),
            "updated_at": now.isoformat(),
            "status": "processing"
        }
        
        # Save uploaded file (async)
        file_path = await save_uploaded_file(file)
        
        # Process video
        result = process_video_analysis(file_path, goal, session_id)
        result.session_name = session_name
        
        # Store result atomically
        results_db[session_id] = result
        sessions_db[session_id].update({
            "status": "completed",
            "has_results": True,
            "updated_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        })
        
        logger.info(f"Analysis completed for session: {session_id}")
        return result
        
    except HTTPException:
        if session_id in sessions_db:
            sessions_db[session_id].update({
                "status": "failed",
                "updated_at": datetime.now().isoformat()
            })
        raise
    except Exception as e:
        logger.error(f"Error in analyze_video: {str(e)}", exc_info=True)
        if session_id in sessions_db:
            sessions_db[session_id].update({
                "status": "failed",
                "updated_at": datetime.now().isoformat()
            })
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )
    finally:
        # Cleanup uploaded file
        if file_path:
            cleanup_file(file_path)


@app.post("/api/get_text_feedback", response_model=Dict[str, Any])
async def get_text_feedback(request: TextFeedbackRequest):
    """
    Generate feedback from text transcript (no video required).
    """
    try:
        logger.info(f"Received text feedback request for goal: {request.goal}")
        
        # Optimize text analysis - use lower() once and split once
        transcript_lower = request.transcript.lower()
        words = request.transcript.split()
        word_count = len(words)
        
        # Estimate pace (assuming average reading speed)
        estimated_wpm = word_count * 2  # Rough estimate
        
        # Count filler words more efficiently using cached set
        filler_counts = {word: transcript_lower.count(word) for word in FILLER_WORDS_LIST}
        total_fillers = sum(filler_counts.values())
        
        # Generate feedback
        feedback = generate_feedback({
            "pace_wpm": estimated_wpm,
            "filler_words": {"total": total_fillers, "breakdown": filler_counts}
        }, request.goal)
        
        return {
            "status": "success",
            "goal": request.goal,
            "session_name": request.session_name,
            "transcript_length": word_count,
            "estimated_pace": estimated_wpm,
            "filler_words": {
                "total": total_fillers,
                "breakdown": filler_counts
            },
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error in get_text_feedback: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing text feedback: {str(e)}"
        )


@app.get("/api/sessions", response_model=List[SessionInfo])
async def get_sessions(
    goal: Optional[GoalType] = Query(None, description="Filter by goal type"),
    session_name: Optional[str] = Query(None, description="Search sessions by name (case-insensitive partial match)"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of sessions to return")
):
    """
    Get list of all analysis sessions with optional filtering by goal and session name.
    """
    try:
        # Use list comprehension for better performance
        sessions = []
        for session_data in sessions_db.values():
            # Filter by goal
            if goal is not None and session_data.get("goal") != goal:
                continue
            
            # Filter by session name (case-insensitive partial match)
            if session_name is not None:
                session_name_value = session_data.get("session_name", "")
                if not session_name_value or session_name.lower() not in session_name_value.lower():
                    continue
            
            sessions.append(SessionInfo(
                session_id=session_data["session_id"],
                goal=session_data["goal"],
                session_name=session_data.get("session_name"),
                created_at=datetime.fromisoformat(session_data["created_at"]),
                status=session_data["status"],
                has_results=session_data.get("has_results", False)
            ))
        
        # Sort by creation date (newest first) and limit
        sessions.sort(key=lambda x: x.created_at, reverse=True)
        return sessions[:limit]

    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving sessions: {str(e)}"
        )


@app.get("/api/sessions/{session_id}", response_model=AnalysisResult)
async def get_session_result(session_id: str):
    """
    Get detailed analysis result for a specific session.
    """
    if session_id not in results_db:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return results_db[session_id]


@app.get("/api/sessions/{session_id}/export")
async def export_session_result(session_id: str):
    """
    Export analysis result as JSON file download.
    """
    if session_id not in results_db:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    try:
        result = results_db[session_id]
        
        # Convert to dict and make it JSON serializable
        result_dict = result.model_dump()
        result_dict["timestamp"] = result_dict["timestamp"].isoformat()
        
        export_filename = f"speakeasy-analysis-{session_id}.json"
        export_path = os.path.join(RESULTS_DIR, export_filename)
        
        with open(export_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Exported session {session_id} to {export_path}")
        
        return FileResponse(
            export_path,
            media_type="application/json",
            filename=export_filename,
            headers={"Content-Disposition": f"attachment; filename={export_filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error exporting session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error exporting session: {str(e)}"
        )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and its results.
    """
    if session_id not in sessions_db:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    try:
        del sessions_db[session_id]
        if session_id in results_db:
            del results_db[session_id]
        
        logger.info(f"Deleted session: {session_id}")
        return {"status": "success", "message": f"Session {session_id} deleted"}

    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting session: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """
    Get overall statistics about all sessions.
    """
    try:
        # Optimize stats calculation with single pass
        total_sessions = len(sessions_db)
        completed_count = 0
        processing_count = 0
        goal_distribution = {}
        
        for session in sessions_db.values():
            status = session.get("status")
            if status == "completed":
                completed_count += 1
            elif status == "processing":
                processing_count += 1
            
            goal = session.get("goal", "unknown")
            goal_distribution[goal] = goal_distribution.get(goal, 0) + 1
        
        # Calculate average score more efficiently
        avg_score = None
        if results_db:
            total_score = sum(r.overall_score for r in results_db.values())
            avg_score = round(total_score / len(results_db), 2)
        
        # Calculate processing times if available
        avg_processing_time = None
        if results_db:
            processing_times = [r.processing_time for r in results_db.values() if hasattr(r, 'processing_time')]
            if processing_times:
                avg_processing_time = round(sum(processing_times) / len(processing_times), 2)
        
        return {
            "total_sessions": total_sessions,
            "completed_sessions": completed_count,
            "processing_sessions": processing_count,
            "failed_sessions": total_sessions - completed_count - processing_count,
            "goal_distribution": goal_distribution,
            "average_score": avg_score,
            "average_processing_time_seconds": avg_processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving stats: {str(e)}"
        )


# ---------------------------
# Run the server
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
