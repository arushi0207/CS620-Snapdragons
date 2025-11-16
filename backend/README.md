# SpeakEasy API Backend

A comprehensive FastAPI backend for the SpeakEasy public speaking coaching application.

## Features

- **Video Analysis**: Upload videos for comprehensive speech and presentation analysis
- **Text Feedback**: Get feedback from text transcripts without video
- **Session Management**: Track and manage analysis sessions
- **Multiple Goals**: Support for job interviews, presentations, networking, and general confidence
- **Comprehensive Metrics**: Pace, filler words, eye gaze, posture, and facial expression analysis
- **RESTful API**: Well-documented endpoints with automatic OpenAPI documentation

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python api.py
```

Or using uvicorn directly:
```bash
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
- `GET /api/health` - Check API status

### Video Analysis
- `POST /api/analyze_video` - Upload video and get comprehensive analysis
  - Parameters:
    - `file`: Video file (mp4, mov, avi, mkv, webm)
    - `goal`: Goal type (job-interview, class-presentation, networking-pitch, general-confidence)
    - `session_name`: Optional session name

### Text Feedback
- `POST /api/get_text_feedback` - Get feedback from text transcript
  - Body: JSON with `transcript`, `goal`, and optional `session_name`

### Session Management
- `GET /api/sessions` - List all sessions (with optional goal filter)
- `GET /api/sessions/{session_id}` - Get detailed results for a session
- `DELETE /api/sessions/{session_id}` - Delete a session

### Statistics
- `GET /api/stats` - Get overall statistics about all sessions

## API Documentation

Once the server is running, visit:
- Swagger UI: http://localhost:8000/api/docs
- ReDoc: http://localhost:8000/api/redoc

## Configuration

The API uses in-memory storage by default. For production, replace the `sessions_db` and `results_db` dictionaries with a proper database (e.g., PostgreSQL, MongoDB).

## Model Integration

Replace the mock functions in `api.py` with your actual models:
- `run_face_tracking()` - Face tracking, eye gaze, posture analysis
- `analyze_audio()` - Audio analysis for pace and filler words
- `generate_feedback()` - AI feedback generation (can be enhanced with LLMs)

## File Structure

```
backend/
├── api.py              # Main API application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Notes

- Temporary uploaded files are stored in `temp_uploads/` and automatically cleaned up
- Results are stored in `results/` directory
- Logs are written to `logs/api.log`
- Maximum file size: 100MB
- Supported video formats: mp4, mov, avi, mkv, webm

