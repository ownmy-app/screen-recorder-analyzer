"""
Benchmark: Processing pipeline speed and keyframe extraction quality.
"""
import os
import sys
import time
import json
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from screen_recorder_analyzer.processor import VideoProcessor


def test_frame_skip_configuration():
    """Different frame_skip values should control analysis density."""
    configs = [
        (0, "every frame"),
        (14, "every 15th frame"),
        (29, "every 30th frame (default)"),
        (59, "every 60th frame"),
    ]
    for skip, desc in configs:
        p = VideoProcessor(frame_skip=skip, max_frames=1000)
        assert p.frame_skip == skip
        expected_count = 300 // (skip + 1)
        print(f"  frame_skip={skip:3d} ({desc:30s}): ~{expected_count} frames in 10s@30fps")


def test_max_frames_limit():
    """max_frames should cap the number of analyzed frames."""
    p = VideoProcessor(frame_skip=0, max_frames=5)
    assert p.max_frames == 5

    mock_cap = mock.MagicMock()
    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(20)]
    call_count = [0]

    def mock_read():
        if call_count[0] < len(frames):
            f = frames[call_count[0]]
            call_count[0] += 1
            return True, f
        return False, None

    mock_cap.read = mock_read
    mock_cap.get.return_value = 30.0

    with mock.patch("screen_recorder_analyzer.processor.cv2") as mock_cv2:
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.cvtColor = lambda f, _: f[:, :, 0]
        mock_cv2.COLOR_BGR2GRAY = 6
        results = p.analyze_frames("fake_video.mp4")

    assert len(results) <= 5, f"max_frames=5 but got {len(results)} results"


def test_processor_init_speed():
    """VideoProcessor initialization should be fast (lazy loading)."""
    times = []
    for _ in range(50):
        start = time.perf_counter()
        p = VideoProcessor(whisper_model_size="tiny", frame_skip=29)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    avg_ms = (sum(times) / len(times)) * 1000
    print(f"\n  Processor init speed: avg={avg_ms:.2f}ms ({len(times)} iterations)")
    assert avg_ms < 100, f"Init too slow: {avg_ms:.2f}ms"
    assert p._whisper_model is None, "Whisper model should be lazy-loaded"


def test_analyze_frame_timestamp_calculation():
    """Frame timestamps should be correctly calculated from frame number and FPS."""
    p = VideoProcessor()
    test_cases = [
        (0, 30.0, 0.0),
        (30, 30.0, 1.0),
        (150, 30.0, 5.0),
        (60, 24.0, 2.5),
        (0, 60.0, 0.0),
        (120, 60.0, 2.0),
    ]
    for frame_num, fps, expected_ts in test_cases:
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = p._analyze_frame(frame, frame_num, fps)
        assert result["timestamp_sec"] == expected_ts, (
            f"frame={frame_num} fps={fps}: expected {expected_ts}, got {result['timestamp_sec']}"
        )


def test_pipeline_result_structure():
    """Full pipeline should return well-structured results dict."""
    p = VideoProcessor()

    with mock.patch.object(p, "extract_audio", return_value="/tmp/fake.wav"):
        with mock.patch.object(p, "transcribe", return_value="Test transcript"):
            with mock.patch.object(p, "get_metadata", return_value={"duration_seconds": 10.0}):
                with mock.patch.object(p, "analyze_frames", return_value=[
                    {"frame_number": 0, "timestamp_sec": 0.0, "status": "ok", "text": "Hello"},
                ]):
                    with mock.patch("os.path.exists", return_value=False):
                        results = p.process("fake_video.mp4")

    assert "metadata" in results
    assert "transcript" in results
    assert "frame_analysis" in results
    assert results["transcript"] == "Test transcript"
    assert len(results["frame_analysis"]) == 1


def test_whisper_backend_selection():
    """Whisper backend should be selectable between local and API."""
    p_local = VideoProcessor(whisper_backend="local")
    assert p_local.whisper_backend == "local"

    p_api = VideoProcessor(whisper_backend="api")
    assert p_api.whisper_backend == "api"


def test_whisper_api_transcription_mock():
    """Whisper API backend should call OpenAI API correctly."""
    p = VideoProcessor(whisper_backend="api")

    mock_client = mock.MagicMock()
    mock_result = mock.MagicMock()
    mock_result.text = "Transcribed text from API"
    mock_client.audio.transcriptions.create.return_value = mock_result

    with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
        with mock.patch("openai.OpenAI", return_value=mock_client):
            with mock.patch("builtins.open", mock.mock_open(read_data=b"fake audio")):
                result = p.transcribe("/tmp/fake_audio.wav")

    assert result == "Transcribed text from API"


def test_ocr_lang_configuration():
    """OCR language should be configurable."""
    p_eng = VideoProcessor(ocr_lang="eng")
    assert p_eng.ocr_lang == "eng"

    p_multi = VideoProcessor(ocr_lang="eng+fra")
    assert p_multi.ocr_lang == "eng+fra"


def test_device_selection():
    """Device should be cpu when CUDA is not available."""
    p = VideoProcessor()
    assert p.device in ("cpu", "cuda")


def test_extract_audio_cleanup():
    """Process should clean up temporary audio file."""
    p = VideoProcessor()
    audio_created = []

    def fake_extract(path):
        audio_path = path + ".audio.wav"
        audio_created.append(audio_path)
        return audio_path

    with mock.patch.object(p, "extract_audio", side_effect=fake_extract):
        with mock.patch.object(p, "transcribe", return_value="text"):
            with mock.patch.object(p, "get_metadata", return_value={}):
                with mock.patch.object(p, "analyze_frames", return_value=[]):
                    with mock.patch("os.path.exists", return_value=True):
                        with mock.patch("os.remove") as mock_remove:
                            p.process("test.mp4")
    assert mock_remove.called, "Temporary audio file was not cleaned up"
