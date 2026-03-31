"""Tests for screen-recorder-analyzer — no GPU/video files required."""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_processor_imports_cleanly():
    """Module must import without heavy deps being installed."""
    from screen_recorder_analyzer import processor  # noqa: F401


def test_video_processor_init():
    from screen_recorder_analyzer.processor import VideoProcessor
    p = VideoProcessor(whisper_model_size="tiny", frame_skip=5, max_frames=10)
    assert p.whisper_model_size == "tiny"
    assert p.frame_skip == 5
    assert p.max_frames == 10


def test_video_processor_missing_file():
    from screen_recorder_analyzer.processor import VideoProcessor
    p = VideoProcessor()
    with pytest.raises(Exception):
        p.extract_audio("/nonexistent/video.mp4")


def test_extract_actions_raises_without_api_key(monkeypatch):
    """extract_actions must raise when no API key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    from screen_recorder_analyzer.processor import extract_actions
    with pytest.raises(Exception):
        extract_actions({"transcript": "hello", "frame_analysis": []})


def test_action_prompt_structure():
    """The action extraction prompt should include the transcript."""
    from screen_recorder_analyzer.processor import extract_actions
    import unittest.mock as mock

    fake_actions = [{"id": "1", "tools": ["excel"], "action": ["viewing data"]}]

    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as mock_ask:
        mock_ask.return_value = json.dumps(fake_actions)

        result = extract_actions(
            {"transcript": "I opened Excel and sorted column A.", "frame_analysis": []},
        )
        assert isinstance(result, list)
        assert result[0]["tools"] == ["excel"]
        # Verify the prompt included the transcript
        call_args = mock_ask.call_args
        assert "Excel" in call_args[0][0] or "Excel" in call_args[1].get("prompt", "")


def test_api_app_creates():
    """FastAPI app must be importable."""
    try:
        from screen_recorder_analyzer.api import app
        assert app is not None
    except ImportError:
        pytest.skip("fastapi not installed")
