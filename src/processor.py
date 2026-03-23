"""
Screen recording processor: extract audio → transcribe with Whisper → OCR keyframes → GPT action extraction.
"""
import json
import os
import subprocess
import tempfile
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
    import openai
except ImportError:
    openai = None

try:
    from moviepy import VideoFileClip
except ImportError:
    VideoFileClip = None

DEFAULT_WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")
DEFAULT_FRAME_SKIP = int(os.environ.get("FRAME_SKIP", "29"))
DEFAULT_MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "100"))
DEFAULT_OCR_LANG = os.environ.get("OCR_LANG", "eng")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")


class VideoProcessor:
    def __init__(
        self,
        whisper_model_size: str = DEFAULT_WHISPER_MODEL,
        frame_skip: int = DEFAULT_FRAME_SKIP,
        max_frames: Optional[int] = DEFAULT_MAX_FRAMES,
        ocr_lang: str = DEFAULT_OCR_LANG,
    ):
        self.whisper_model_size = whisper_model_size
        self.frame_skip = frame_skip
        self.max_frames = max_frames
        self.ocr_lang = ocr_lang
        self._whisper_model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.tesseract_ok = False
        if pytesseract:
            try:
                pytesseract.get_tesseract_version()
                self.tesseract_ok = True
            except Exception:
                pass

    def _load_whisper(self):
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
        if not whisper or not torch:
            return "[Skipped: whisper/torch not installed]"
        self._load_whisper()
        if not self._whisper_model:
            return "[Skipped: model failed to load]"
        result = self._whisper_model.transcribe(audio_path, fp16=(self.device != "cpu"))
        return result["text"]

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
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_num % (self.frame_skip + 1) == 0:
                    results.append(self._analyze_frame(frame, frame_num, fps))
                    analyzed += 1
                    if self.max_frames and analyzed >= self.max_frames:
                        break
                frame_num += 1
        finally:
            cap.release()
        return results

    def _analyze_frame(self, frame, frame_num: int, fps: float) -> Dict[str, Any]:
        ts = round(frame_num / fps, 2)
        result = {"frame_number": frame_num, "timestamp_sec": ts, "status": "pending", "text": None}
        if not pytesseract or not self.tesseract_ok:
            result["status"] = "skipped"
            return result
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config=f"--psm 6 -l {self.ocr_lang}", timeout=10)
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


def extract_actions(results: Dict[str, Any], api_key: Optional[str] = None) -> List[Dict[str, Any]]:
    """Use GPT to extract a chronological list of user actions from transcript + OCR."""
    if not openai:
        raise RuntimeError("openai package not installed")

    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

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
    client = openai.OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        messages=[
            {"role": "system", "content": "You analyze screen recordings. Output only JSON."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=16000,
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
