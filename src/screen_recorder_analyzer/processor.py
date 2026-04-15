"""
Screen recording processor: extract audio -> transcribe with Whisper -> OCR keyframes -> LLM action extraction.

Supports multiple LLM backends (OpenAI, Anthropic, LiteLLM) via the ``llm``
module, and two Whisper backends:

    WHISPER_BACKEND=local   (default)  Uses openai-whisper local model
    WHISPER_BACKEND=api                Uses OpenAI Whisper API (no local model download)
"""
import json
import os
import subprocess
from typing import Any, Dict, List, Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import whisper
    import torch
except ImportError:
    whisper = None
    torch = None

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

try:
    import numpy as np
except ImportError:
    np = None

DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
DEFAULT_WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "local").lower().strip()
DEFAULT_FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "29"))
DEFAULT_MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "100"))
DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "eng")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
# Minimum mean-squared-error between frames before we consider them different
DEFAULT_FRAME_DIFF_THRESHOLD = float(os.environ.get("FRAME_DIFF_THRESHOLD", "500"))


def _is_dark_theme(gray_frame) -> bool:
    """Detect if a frame is dark-themed (light text on dark background)."""
    if np is None:
        return False
    mean_val = float(np.mean(gray_frame))
    return mean_val < 100  # below ~40% brightness


def _preprocess_for_ocr(gray_frame):
    """Apply adaptive preprocessing to improve OCR accuracy.

    Handles dark themes (inverts), applies denoising, and uses adaptive
    thresholding for better text extraction.
    """
    if cv2 is None or np is None:
        return gray_frame

    processed = gray_frame.copy()

    # Dark theme: invert so text becomes dark on light background
    if _is_dark_theme(processed):
        processed = cv2.bitwise_not(processed)

    # Denoise with bilateral filter (preserves edges)
    processed = cv2.bilateralFilter(processed, 9, 75, 75)

    # Adaptive thresholding for better text/background separation
    processed = cv2.adaptiveThreshold(
        processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )

    return processed


def _frames_are_similar(frame_a, frame_b, threshold: float = DEFAULT_FRAME_DIFF_THRESHOLD) -> bool:
    """Check if two frames are visually similar enough to skip OCR on the second."""
    if np is None or frame_a is None or frame_b is None:
        return False
    if frame_a.shape != frame_b.shape:
        return False
    mse = float(np.mean((frame_a.astype(float) - frame_b.astype(float)) ** 2))
    return mse < threshold


class VideoProcessor:
    def __init__(
        self,
        whisper_model_size: str = DEFAULT_WHISPER_MODEL,
        whisper_backend: str = DEFAULT_WHISPER_BACKEND,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
        ocr_lang: str = DEFAULT_OCR_LANG,
        preprocess_ocr: bool = True,
        dedup_frames: bool = True,
    ):
        self.whisper_model_size = whisper_model_size
        self.whisper_backend = whisper_backend  # "local" or "api"
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.ocr_lang = ocr_lang
        self.preprocess_ocr = preprocess_ocr
        self.dedup_frames = dedup_frames
        self._whisper_model = None  # lazy-loaded only when backend == "local"
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.tesseract_ok = False
        if pytesseract:
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_ok = True
            except Exception:
                pass

    def _load_whisper(self):
        """Lazy-load the local Whisper model (only needed for backend='local')."""
        if self._whisper_model is None and whisper and torch:
            self._whisper_model = whisper.load_model(self.whisper_model_size, device=self.device)

    def extract_audio(self, video_path: str) -> str:
        audio_path = video_path + ".audio.wav"
        if os.path.exists(audio_path):
            os.remove(audio_path)
        subprocess.run(
            ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", audio_path],
            check=True, capture_output=True, timeout=120,
        )
        return audio_path

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio to text.

        When ``WHISPER_BACKEND=api`` (or ``whisper_backend='api'`` in the
        constructor), uses the OpenAI Whisper API -- no local model download
        required.  Otherwise falls back to the local ``openai-whisper`` package.
        """
        if self.whisper_backend == "api":
            return self._transcribe_api(audio_path)
        return self._transcribe_local(audio_path)

    def _transcribe_local(self, audio_path: str) -> str:
        if not whisper or not torch:
            return "[Skipped: whisper/torch not installed]"
        self._load_whisper()
        if not self._whisper_model:
            return "[Skipped: model failed to load]"
        result = self._whisper_model.transcribe(audio_path, fp16=(self.device != "cpu"))
        return result["text"]

    def _transcribe_api(self, audio_path: str) -> str:
        """Transcribe via the OpenAI Whisper API (requires OPENAI_API_KEY)."""
        try:
            import openai as _openai
        except ImportError:
            return "[Skipped: openai package not installed for Whisper API]"

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return "[Skipped: OPENAI_API_KEY not set for Whisper API]"

        client = _openai.OpenAI(api_key=api_key)
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
        return result.text

    def get_metadata(self, video_path: str) -> Dict[str, Any]:
        if not VideoFileClip:
            return {}
        with VideoFileClip(video_path) as clip:
            return {"duration_seconds": clip.duration, "fps": clip.fps, "width": clip.w, "height": clip.h}

    def analyze_frames(self, video_path: str) -> List[Dict[str, Any]]:
        if not cv2:
            return []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        results, frame_num, analyzed = [], 0, 0
        prev_gray = None
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % (self.frame_skip + 1) == 0:
                    # Convert to grayscale for comparison/OCR
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if cv2 else None

                    # Frame deduplication: skip near-identical frames
                    if self.dedup_frames and prev_gray is not None and _frames_are_similar(gray, prev_gray):
                        frame_num += 1
                        continue

                    results.append(self._analyze_frame(frame, frame_num, fps, gray=gray))
                    prev_gray = gray
                    analyzed += 1
                    if self.max_frames and analyzed >= self.max_frames:
                        break
                frame_num += 1
        finally:
            cap.release()
        return results

    def _analyze_frame(self, frame, frame_num: int, fps: float, gray=None) -> Dict[str, Any]:
        ts = round(frame_num / fps, 2)
        result = {"frame_number": frame_num, "timestamp_sec": ts, "status": "pending", "text": None}
        if not pytesseract or not self.tesseract_ok:
            result["status"] = "skipped"
            return result
        try:
            if gray is None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply preprocessing if enabled
            ocr_input = _preprocess_for_ocr(gray) if self.preprocess_ocr else gray

            text = pytesseract.image_to_string(ocr_input, config=f"--psm 6 -l {self.ocr_lang}", timeout=10)
            result["text"] = text
            result["status"] = "ok"
        except Exception as e:
            result["status"] = f"error: {e}"
        return result

    def process(self, video_path: str) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        results["metadata"] = self.get_metadata(video_path)

        audio_path = None
        try:
            audio_path = self.extract_audio(video_path)
            results["transcript"] = self.transcribe(audio_path)
        except Exception as e:
            results["transcript"] = f"[Error: {e}]"
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        results["frame_analysis"] = self.analyze_frames(video_path)
        return results


def _parse_llm_actions(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of action dicts.

    Handles:
    - Bare JSON arrays: [{"id": "1", ...}]
    - Wrapped in markdown code fences: ```json ... ```
    - Wrapped in a JSON object: {"actions": [...]}
    - Extra whitespace/newlines
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)

    # If the LLM wrapped the list in an object, try to extract it
    if isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("actions", "results", "data", "items"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
        # If there's only one key and its value is a list, use that
        values = list(parsed.values())
        if len(values) == 1 and isinstance(values[0], list):
            return values[0]
        raise ValueError(
            f"LLM returned a JSON object but no recognizable action list. Keys: {list(parsed.keys())}"
        )

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array from LLM, got {type(parsed).__name__}")

    return parsed


def extract_actions(results: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Use an LLM to extract a chronological list of user actions from transcript + OCR.

    Routes through the multi-LLM abstraction (``llm.py``).  The provider is
    selected via the ``LLM_PROVIDER`` env var; *api_key* is accepted for
    backwards compatibility but is no longer required when using the llm module.
    """
    from .llm import ask_llm

    transcript = results.get("transcript", "")
    frames = results.get("frame_analysis", [])

    sections = [
        "Analyze screen recording data to identify user actions chronologically.",
        f"\n## Transcript:\n{transcript[:30000]}" if transcript else "\n## Transcript:\nNot available.",
    ]

    ocr_lines = []
    for f in frames[:50]:
        if f.get("status") == "ok" and f.get("text"):
            preview = f["text"][:500].replace("\n", " ")
            ocr_lines.append(f"- At {f['timestamp_sec']}s: {preview}")
    if ocr_lines:
        sections.append("\n## OCR Keyframes:\n" + "\n".join(ocr_lines))

    sections.append(
        "\n## Task:\nReturn a JSON list of distinct actions in chronological order. "
        "For each action: infer the tool(s) used (e.g. excel, gmail, hubspot) and the action performed. "
        "Output ONLY the JSON list.\n"
        'Example: [{"id":"1","tools":["excel"],"action":["viewing data"]},...]'
    )

    prompt = "\n".join(sections)

    raw = ask_llm(
        prompt=prompt,
        system="You analyze screen recordings. Output only JSON.",
        max_tokens=16000,
        temperature=0.3,
        response_json=True,
    ).strip()

    return _parse_llm_actions(raw)
