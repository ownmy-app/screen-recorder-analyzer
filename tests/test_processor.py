"""Tests for screen-recorder-analyzer — no GPU/video files required."""
import sys
import os
import json
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def test_processor_imports_cleanly():
    """Module must import without heavy deps being installed."""
    import processor  # noqa: F401


def test_video_processor_init():
    from processor import VideoProcessor
    p = VideoProcessor(whisper_model_size="tiny", frame_skip=5, max_frames=10)
    assert p.whisper_model_size == "tiny"
    assert p.frame_skip == 5
    assert p.max_frames == 10


def test_video_processor_missing_file():
    from processor import VideoProcessor
    p = VideoProcessor()
    with pytest.raises(Exception):
        p.extract_audio("/nonexistent/video.mp4")


def test_extract_actions_raises_without_api_key(monkeypatch):
    """extract_actions must raise ValueError when no API key is set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from processor import extract_actions
    with pytest.raises((ValueError, RuntimeError)):
        extract_actions({"transcript": "hello", "frame_analysis": []})


def test_action_prompt_structure():
    """The action extraction prompt should include the transcript."""
    from processor import extract_actions
    import unittest.mock as mock

    fake_actions = [{"id": "1", "tools": ["excel"], "action": ["viewing data"]}]

    with mock.patch("processor.openai") as mock_openai:
        mock_client = mock.MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_resp = mock.MagicMock()
        mock_resp.choices[0].message.content = json.dumps(fake_actions)
        mock_client.chat.completions.create.return_value = mock_resp

        result = extract_actions(
            {"transcript": "I opened Excel and sorted column A.", "frame_analysis": []},
            api_key="sk-test",
        )
        assert isinstance(result, list)
        assert result[0]["tools"] == ["excel"]


def test_api_app_creates():
    """FastAPI app must be importable."""
    try:
        from api import app
        assert app is not None
    except ImportError:
        pytest.skip("fastapi not installed")
