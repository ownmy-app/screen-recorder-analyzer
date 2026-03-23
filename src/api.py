"""
FastAPI server for screen recording analysis.

POST /process-video/   { "video_path": "...", "config": {...} }
"""
import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field

from .processor import VideoProcessor, extract_actions


class ProcessingConfig(BaseModel):
    whisper_model_size: str = Field("base", description="tiny | base | small | medium | large")
    frame_skip: int = Field(29, description="Analyze every N+1 frames")
    max_frames: Optional[int] = Field(100, description="Max frames to OCR (null = no limit)")
    ocr_lang: str = Field("eng", description="Tesseract language code(s), e.g. 'eng+fra'")
    openai_model: str = Field("gpt-4o", description="OpenAI model for action extraction")


class VideoRequest(BaseModel):
    video_path: str
    config: ProcessingConfig = ProcessingConfig()


app = FastAPI(
    title="Screen Recorder Analyzer",
    description="Upload a screen recording path → get structured user actions via Whisper + OCR + GPT.",
    version="0.1.0",
)


@app.get("/")
async def status():
    return {"status": "ok", "message": "Screen Recorder Analyzer is running."}


@app.post("/process-video/")
async def process_video(request: VideoRequest = Body(...)) -> Dict[str, Any]:
    """
    Process a video from the local filesystem.

    Returns transcript, OCR frame summary, and AI-extracted structured actions.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured.")

    if not os.path.exists(request.video_path):
        raise HTTPException(status_code=400, detail=f"Video path not found: {request.video_path}")

    cfg = request.config
    processor = VideoProcessor(
        whisper_model_size=cfg.whisper_model_size,
        frame_skip=cfg.frame_skip,
        max_frames=cfg.max_frames,
        ocr_lang=cfg.ocr_lang,
    )

    try:
        pipeline_results = processor.process(request.video_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

    try:
        actions = extract_actions(pipeline_results, api_key=api_key)
        pipeline_results["structured_actions"] = actions
        pipeline_results["ai_status"] = "ok"
    except Exception as e:
        pipeline_results["structured_actions"] = []
        pipeline_results["ai_status"] = f"error: {e}"

    # Slim down frame_analysis in response (remove raw OCR text to reduce payload)
    pipeline_results["frame_analysis_summary"] = [
        {"timestamp_sec": f.get("timestamp_sec"), "status": f.get("status")}
        for f in pipeline_results.pop("frame_analysis", [])
    ]

    return pipeline_results
