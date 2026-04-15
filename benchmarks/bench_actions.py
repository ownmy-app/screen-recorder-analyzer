"""
Benchmark: Action categorization accuracy using mock LLM responses.
"""
import os
import sys
import json
import time
from unittest import mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from screen_recorder_analyzer.processor import extract_actions


def _make_results(transcript="", ocr_frames=None):
    """Build a minimal pipeline results dict."""
    frames = []
    if ocr_frames:
        for i, text in enumerate(ocr_frames):
            frames.append({"frame_number": i * 30, "timestamp_sec": i, "status": "ok", "text": text})
    return {"transcript": transcript, "frame_analysis": frames}


def test_action_extraction_excel(mock_llm_responses):
    """Excel browsing scenario should return exactly 2 actions with tool=excel."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = mock_llm_responses["excel_browsing"]
        results = _make_results("I opened Excel and sorted the data by column A")
        actions = extract_actions(results)
    assert len(actions) == 2
    assert all(a["tools"] == ["excel"] for a in actions)


def test_action_extraction_multi_tool(mock_llm_responses):
    """Multi-tool scenario should correctly identify distinct tools."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = mock_llm_responses["multi_tool"]
        results = _make_results(
            "Let me check the dashboard, then search contacts in HubSpot, and send an email",
            ocr_frames=["HubSpot CRM - Contacts", "Gmail - Compose"],
        )
        actions = extract_actions(results)
    assert len(actions) == 4
    tools_seen = set()
    for a in actions:
        tools_seen.update(a["tools"])
    assert "chrome" in tools_seen
    assert "hubspot" in tools_seen
    assert "gmail" in tools_seen


def test_action_extraction_coding(mock_llm_responses):
    """Coding session should identify IDE, terminal, and git tools."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = mock_llm_responses["coding_session"]
        results = _make_results("Let me fix this bug and open a PR")
        actions = extract_actions(results)
    assert len(actions) == 3
    tools_seen = set()
    for a in actions:
        tools_seen.update(a["tools"])
    assert "vscode" in tools_seen
    assert "terminal" in tools_seen
    assert "github" in tools_seen


def test_action_extraction_markdown_fence(mock_llm_responses):
    """LLM response wrapped in markdown code fences should be parsed correctly."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = mock_llm_responses["malformed_markdown"]
        results = _make_results("Some transcript")
        actions = extract_actions(results)
    assert isinstance(actions, list)
    assert len(actions) >= 1


def test_action_extraction_empty(mock_llm_responses):
    """Empty action list from LLM should return empty list without error."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = mock_llm_responses["empty_actions"]
        results = _make_results("")
        actions = extract_actions(results)
    assert actions == []


def test_action_extraction_with_ocr_context():
    """OCR text should be included in the prompt sent to the LLM."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([{"id": "1", "tools": ["slack"], "action": ["reading messages"]}])
        results = _make_results(
            "checking slack",
            ocr_frames=["Slack - #engineering", "John: deploy v2.1 today?"],
        )
        actions = extract_actions(results)
        args, kwargs = m.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "Slack" in prompt
        assert "engineering" in prompt


def test_action_extraction_truncation():
    """Very long transcripts should be truncated to 30000 chars in the prompt."""
    long_transcript = "A" * 50000
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([{"id": "1", "tools": ["app"], "action": ["something"]}])
        results = _make_results(long_transcript)
        extract_actions(results)
        args, kwargs = m.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert len(prompt) < 35000


def test_action_prompt_includes_system_message():
    """extract_actions should send a system message to the LLM."""
    with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
        m.return_value = json.dumps([])
        extract_actions(_make_results("test"))
        _, kwargs = m.call_args
        assert kwargs.get("system"), "No system message sent to LLM"
        assert "JSON" in kwargs["system"]


def test_action_extraction_speed(mock_llm_responses):
    """Action extraction pipeline (excluding LLM latency) should be fast."""
    times = []
    for _ in range(20):
        with mock.patch("screen_recorder_analyzer.llm.ask_llm") as m:
            m.return_value = mock_llm_responses["multi_tool"]
            results = _make_results(
                "User opened chrome, searched contacts, composed email",
                ocr_frames=["Chrome tab", "HubSpot CRM", "Gmail compose"] * 10,
            )
            start = time.perf_counter()
            extract_actions(results)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
    avg_ms = (sum(times) / len(times)) * 1000
    print(f"\n  Action extraction speed: avg={avg_ms:.1f}ms ({len(times)} iterations)")
    assert avg_ms < 500, f"Action extraction too slow: {avg_ms:.1f}ms"
